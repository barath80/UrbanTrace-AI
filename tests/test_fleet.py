"""Fleet pieces that need no GPU, database or network."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from urbantrace.scale import capture
from urbantrace.scale.adaptive import RateController
from urbantrace.scale.cluster import (Membership, affinity_key, assign, owner_of)
from urbantrace.scale.config import SCALE
from urbantrace.scale.registry import CameraRegistry


# ------------------------------------------------------------- assignment

def _keys(n):
    return [(f"CAM_{i}", f"cell:{i}:0") for i in range(n)]


def test_every_camera_has_exactly_one_owner():
    workers = ["w1", "w2", "w3"]
    owned = [assign(_keys(300), workers, w) for w in workers]
    assert sum(len(o) for o in owned) == 300
    assert not (owned[0] & owned[1]) and not (owned[1] & owned[2])


def test_load_is_roughly_even():
    workers = [f"w{i}" for i in range(4)]
    sizes = [len(assign(_keys(1000), workers, w)) for w in workers]
    assert min(sizes) > 180 and max(sizes) < 320


def test_clumped_cameras_still_spread_across_workers():
    # 12 busy junctions of very different sizes: plain hashing leaves workers idle.
    keys = [(f"C{i}", f"cell:{i % 12}:0") for i in range(40)] +            [(f"D{i}", "cell:99:0") for i in range(15)]
    workers = ["w1", "w2", "w3", "w4"]
    sizes = [len(assign(keys, workers, w)) for w in workers]
    assert sum(sizes) == 55
    assert min(sizes) > 0 and max(sizes) <= 18


def test_losing_a_worker_moves_only_its_cameras():
    workers = ["w1", "w2", "w3", "w4"]
    def owners(ws):
        out = {}
        for w in ws:
            for c in assign(_keys(500), ws, w):
                out[c] = w
        return out
    before, after = owners(workers), owners(["w1", "w2", "w3"])
    # Everything w4 held must move; most of the rest should stay where it was.
    stayed = sum(1 for c in before if before[c] != "w4" and after[c] == before[c])
    kept = sum(1 for c in before if before[c] != "w4")
    assert stayed / kept > 0.6


def test_a_worker_not_in_the_list_owns_nothing():
    assert assign(_keys(10), ["w1"], "ghost") == set()


def test_neighbouring_cameras_share_an_affinity():
    a = affinity_key("", 13.0694, 80.1948, cell_km=8)
    b = affinity_key("", 13.0720, 80.1990, cell_km=8)
    far = affinity_key("", 13.60, 80.90, cell_km=8)
    assert a == b and a != far
    assert affinity_key("north", 1, 1) == affinity_key("north", 50, 50)


class MemoryStore:
    def __init__(self):
        self.rows = {}
        self.fail = False

    def upsert(self, worker_id, host, stats):
        if self.fail:
            raise RuntimeError("db down")
        self.rows[worker_id] = {"worker_id": worker_id}

    def live(self, ttl_s):
        if self.fail:
            raise RuntimeError("db down")
        return list(self.rows.values())

    def remove(self, worker_id):
        self.rows.pop(worker_id, None)


def test_membership_keeps_last_view_when_database_is_down():
    store = MemoryStore()
    a, b = Membership(store, "a"), Membership(store, "b")
    b.beat({})
    assert a.beat({}) == ["a", "b"]
    store.fail = True
    assert a.beat({}) == ["a", "b"]          # does not abandon its cameras
    assert a.failures == 1


def test_leaving_removes_the_worker():
    store = MemoryStore()
    m = Membership(store, "a")
    m.beat({})
    m.leave()
    assert "a" not in store.rows


# --------------------------------------------------------------- adaptive

def test_rate_drops_under_overload_and_recovers():
    rc = RateController(target_fps=6.0)
    rc.tick(0, 0)
    for i in range(1, 6):
        rc.tick(100 * i, 30 * i)             # 30% dropped each interval
    assert rc.fps() < 6.0
    low = rc.fps()
    n, d = 500, 150
    for _ in range(40):
        n += 100
        rc.tick(n, d)                         # nothing dropped any more
    assert rc.fps() > low
    assert rc.fps() <= 6.0
    assert rc.fps() >= rc.floor_fps


# ----------------------------------------------------------------- capture

def test_redact_url_hides_credentials():
    assert "secret" not in capture.redact_url("rtsp://admin:secret@10.0.0.5:554/stream1")
    assert capture.redact_url("rtsp://10.0.0.5/stream") == "rtsp://10.0.0.5/stream"


def test_motion_gate_ignores_a_still_scene_and_sees_movement():
    gate = capture.MotionGate(pixel_delta=14, min_fraction=0.004)
    still = np.full((360, 640, 3), 90, np.uint8)
    assert gate.moved(still)                  # first frame is always sent
    assert not gate.moved(still.copy())
    moved = still.copy()
    cv2.rectangle(moved, (100, 100), (300, 250), (255, 255, 255), -1)
    assert gate.moved(moved)


def _write_video(path: Path, frames: int = 60, fps: int = 25):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(frames):
        img = np.full((240, 320, 3), 40, np.uint8)
        cv2.rectangle(img, (5 * i % 250, 80), (5 * i % 250 + 60, 160), (255, 255, 255), -1)
        w.write(img)
    w.release()


def test_reader_samples_a_file_at_the_analysis_rate(tmp_path):
    video = tmp_path / "cam.mp4"
    _write_video(video)
    got = []
    sink = lambda cid, frame, **kw: got.append((cid, frame.shape)) or True
    gate = threading.Semaphore(2)
    r = capture.StreamReader("CAM_T", str(video), sink, lambda: 5.0, gate)
    r.start()
    r.join(timeout=15)
    assert not r.is_alive()
    assert r.health.frames_read == 60
    # 60 frames at 25 fps is 2.4 s of video; at 5 fps that is about a dozen
    # frames, far fewer than were read.
    assert 3 <= len(got) <= 30
    assert len(got) < r.health.frames_read
    assert r.health.state == "stopped"


def test_reader_reconnects_with_backoff_on_a_dead_source():
    gate = threading.Semaphore(2)
    r = capture.StreamReader("CAM_X", "rtsp://127.0.0.1:1/none", lambda *a, **k: True,
                             lambda: 5.0, gate)
    r.start()
    time.sleep(3.0)
    r.stop()
    r.join(timeout=15)
    assert r.health.reconnects >= 1
    assert r.health.last_error


def test_manager_reconcile_starts_and_stops_only_the_difference(tmp_path):
    video = tmp_path / "a.mp4"
    _write_video(video, frames=400)
    mgr = capture.CaptureManager(lambda *a, **k: True, lambda: 5.0)
    d = mgr.reconcile({"A": str(video), "B": str(video)})
    assert d["started"] == 2
    d = mgr.reconcile({"A": str(video)})
    assert d == {"started": 0, "stopped": 1, "running": 1}
    mgr.stop()


# ----------------------------------------------------------------- registry

def test_registry_replace_never_exposes_an_empty_estate():
    reg = CameraRegistry()
    reg.load([{"camera_id": "A", "name": "A", "latitude": 13.0, "longitude": 80.0}])
    seen_empty = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            if len(reg) == 0:
                seen_empty.append(1)

    t = threading.Thread(target=watch)
    t.start()
    for _ in range(200):
        reg.replace([{"camera_id": "A", "name": "A", "latitude": 13.0, "longitude": 80.0},
                     {"camera_id": "B", "name": "B", "latitude": 13.01, "longitude": 80.0}])
    stop.set()
    t.join()
    assert not seen_empty
    assert len(reg) == 2


# ----------------------------------------------------------------- database

def test_connect_uses_database_url_when_set(monkeypatch):
    from urbantrace.db import client
    seen = []

    class Boom(Exception):
        pass

    def fake_connect(dsn, **kw):
        seen.append((dsn, kw["sslmode"]))
        raise Boom()

    import psycopg2
    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@dbhost:5432/x")
    monkeypatch.setenv("UT_DB_SSLMODE", "disable")
    with pytest.raises(ConnectionError):
        client.connect()
    assert seen == [("postgresql://u:p@dbhost:5432/x", "disable")]


# ------------------------------------------------ multi-worker correctness

def test_cluster_ids_do_not_collide_across_workers():
    """Two workers (or one worker after a restart) must never mint the same
    cluster id: it is the primary key of net_vehicles."""
    from datetime import datetime, timezone

    from urbantrace.scale.identity import IdentityIndex, Sighting

    def one(plate):
        idx = IdentityIndex()
        return idx.add(Sighting(plate, "CAM_A", datetime.now(timezone.utc), 0.9, 1, 0.0, 0.0))

    assert len({one("TN01AB1234") for _ in range(20)}) == 20


def test_admission_tolerates_paced_frames_with_clock_jitter():
    from urbantrace.scale.inference import FrameBus

    bus = FrameBus(target_fps=10.0)
    assert bus.admit("CAM_A", now=100.000)
    assert bus.admit("CAM_A", now=100.096)        # 4 ms early: still on schedule
    assert not bus.admit("CAM_A", now=100.150)    # genuinely too soon


def test_video_endpoint_rejects_unsafe_camera_id():
    from urbantrace.api.main import _SAFE_ID

    assert _SAFE_ID.fullmatch("CAM_01")
    for bad in ("../secret", "a/b", "a b", "", "x" * 65, "a\b"):
        assert not _SAFE_ID.fullmatch(bad)


def test_db_pool_queues_instead_of_failing_when_exhausted(monkeypatch):
    """More concurrent callers than connections must wait, not raise PoolError."""
    import threading
    import time

    from urbantrace.db import client

    class Conn:
        def commit(self): pass
        def rollback(self): pass

    class FakePool:
        """Like psycopg2's: raises when empty rather than waiting."""
        def __init__(self, n):
            self.free = [Conn() for _ in range(n)]
            self.lock = threading.Lock()

        def getconn(self):
            with self.lock:
                if not self.free:
                    raise RuntimeError("connection pool exhausted")
                return self.free.pop()

        def putconn(self, conn, close=False):
            with self.lock:
                self.free.append(conn)

    pool = FakePool(2)
    monkeypatch.setattr(client, "_pool", lambda: pool)
    monkeypatch.setattr(client, "_SLOTS", threading.BoundedSemaphore(2))
    errors, held = [], []

    def worker():
        try:
            with client.pooled():
                held.append(1)
                time.sleep(0.05)
        except Exception as e:                     # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker) for _ in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and len(held) == 12
