"""Tests for live ingest.

The vision models are stubbed, so these run without a GPU, a video or a
network. What they pin is the session machinery that makes a feed trustworthy:
that counts do not accumulate across replays, that a slow subscriber cannot
stall the detector, and that throughput is reported honestly.

    python -m pytest tests/test_live.py -q
"""
from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from urbantrace.ingest import live as L  # noqa: E402


class FakeTrack:
    def __init__(self, tid, cls="car", box=(100, 100, 260, 230)):
        self.track_id, self.cls_name, self.confidence, self.box = tid, cls, 0.9, box


class FakeTracker:
    """Emits one steadily moving car so the counter has something real to gate."""

    def __init__(self):
        self.n = 0

    def update(self, _frame):
        self.n += 1
        x = 100 + self.n * 14
        return [FakeTrack(1, "car", (x, 300, x + 160, 430))]

    def reset(self):
        self.n = 0


class NoPlates:
    def detect(self, _frame):
        return []


class FakeCapture:
    """A capture that yields `frames` frames, then reports end-of-stream."""

    def __init__(self, frames=6, w=1280, h=720, fps=24.0):
        self.total, self.i = frames, 0
        self.w, self.h, self.fps = w, h, fps
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        return {cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_FRAME_WIDTH: self.w,
                cv2.CAP_PROP_FRAME_HEIGHT: self.h}.get(prop, 0)

    def read(self):
        if self.i >= self.total:
            return False, None
        self.i += 1
        return True, np.zeros((self.h, self.w, 3), np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "get_ocr", lambda: None)
    s = L.LiveSession("CAM_01", "fake", realtime=False, evidence_dir=tmp_path,
                      vehicle_tracker=FakeTracker(), plate_detector=NoPlates())
    return s


def drain(q: queue.Queue, timeout=4.0) -> list:
    """Collect events until the session says it stopped."""
    out, deadline = [], time.time() + timeout
    while time.time() < deadline:
        try:
            ev = q.get(timeout=0.25)
        except queue.Empty:
            continue
        out.append(ev)
        if ev["type"] == "stopped":
            break
    return out


def test_session_runs_and_reports_lifecycle(session, monkeypatch):
    monkeypatch.setattr(L.cv2, "VideoCapture", lambda *_: FakeCapture(frames=8))
    session.loop = False
    q = session.subscribe()
    session.start()
    evs = drain(q)
    kinds = [e["type"] for e in evs]
    assert "started" in kinds
    assert "ended" in kinds        # a non-looping source announces the end
    assert "stopped" in kinds
    assert session.frames == 8


def test_counts_reset_each_replay(session, monkeypatch):
    """A looping file must not count the same vehicles again on every pass.

    Left unreset, the running total climbed forever and meant nothing.
    """
    monkeypatch.setattr(L.cv2, "VideoCapture", lambda *_: FakeCapture(frames=14))
    session.loop = True
    session.start()
    # Let it complete at least two passes.
    deadline = time.time() + 6
    while session.cycles < 2 and time.time() < deadline:
        time.sleep(0.05)
    cycles = session.cycles
    total = session.counter.counts("traffic")["total"] if session.counter else 0
    session.stop()

    assert cycles >= 2, "the source should have replayed"
    # One synthetic car per pass, so a reset counter never exceeds a small
    # number however many times the clip repeats.
    assert total <= 2, f"counts accumulated across replays: {total}"


def test_non_looping_source_stops_itself(session, monkeypatch):
    monkeypatch.setattr(L.cv2, "VideoCapture", lambda *_: FakeCapture(frames=4))
    session.loop = False
    session.start()
    deadline = time.time() + 5
    while session.state == "running" and time.time() < deadline:
        time.sleep(0.05)
    assert session.state == "stopped"


def test_bad_source_reports_error_and_does_not_raise(session, monkeypatch):
    class Dead:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(L.cv2, "VideoCapture", lambda *_: Dead())
    q = session.subscribe()
    session.start()
    evs = drain(q)
    assert session.state == "error"
    assert "Cannot open" in (session.error or "")
    assert any(e["type"] == "error" for e in evs)


def test_slow_subscriber_never_blocks_the_detector(session, monkeypatch):
    """A stalled detector stops being live, so stats are dropped instead."""
    monkeypatch.setattr(L.cv2, "VideoCapture", lambda *_: FakeCapture(frames=40))
    session.loop = False
    q = session.subscribe()
    # Fill the subscriber's queue and never read it.
    for i in range(L.SUBSCRIBER_QUEUE):
        try:
            q.put_nowait({"type": "filler", "i": i})
        except queue.Full:
            break
    session.start()
    deadline = time.time() + 6
    while session.state == "running" and time.time() < deadline:
        time.sleep(0.05)
    assert session.state in ("stopped", "error")
    assert session.frames == 40, "the detector kept running despite a full queue"


def test_status_reports_throughput_against_the_source(session, monkeypatch):
    monkeypatch.setattr(L.cv2, "VideoCapture", lambda *_: FakeCapture(frames=10, fps=24.0))
    session.loop = False
    session.start()
    deadline = time.time() + 5
    while session.state == "running" and time.time() < deadline:
        time.sleep(0.05)
    st = session.status()
    assert st["source_fps"] == 24.0
    assert st["realtime_factor"] is not None
    assert st["keeping_up"] in (True, False)
    assert st["detect_every"] == 1


def test_subscriber_gets_recent_history_on_connect(session, monkeypatch):
    """A dashboard opened late should not face a blank screen."""
    monkeypatch.setattr(L.cv2, "VideoCapture", lambda *_: FakeCapture(frames=10))
    session.loop = False
    session.start()
    deadline = time.time() + 5
    while session.state == "running" and time.time() < deadline:
        time.sleep(0.05)
    q = session.subscribe()
    assert not q.empty(), "backlog should be replayed to a new subscriber"


def test_manager_is_idempotent_per_camera(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "get_ocr", lambda: None)
    monkeypatch.setattr(L.cv2, "VideoCapture", lambda *_: FakeCapture(frames=200))
    m = L.LiveManager()
    monkeypatch.setattr(m, "_models", lambda: (FakeTracker(), NoPlates()))
    a = m.start("CAM_01", "fake", realtime=False)
    b = m.start("CAM_01", "fake", realtime=False)
    assert a is b, "starting twice must not spawn a second session"
    assert m.status()["any_running"] is True
    m.stop_all()
    assert m.status()["any_running"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
