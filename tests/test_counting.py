"""Tests for unique-vehicle counting.

These drive VehicleCounter with synthetic tracks, so they pin the counting
logic without loading a model or a video. Each test states the real-world
failure it guards against.

    python -m pytest tests/test_counting.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from urbantrace.vision.counting import (  # noqa: E402
    CountConfig, VehicleCounter)

W, H = 1280, 720


class T:
    """A track detection, shaped like detectors.VehicleTrack."""

    def __init__(self, tid, cls="car", conf=0.9, box=(0, 0, 100, 100)):
        self.track_id, self.cls_name, self.confidence, self.box = tid, cls, conf, box


def box_at(cx, cy, w=100, h=80):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def drive(counter, tid, cls, start, n, x0, y0, dx=0.0, dy=0.0, w=100, h=80):
    """Feed a track moving linearly from (x0,y0) for n frames."""
    for i in range(n):
        counter.update(
            [T(tid, cls, 0.9, box_at(x0 + dx * i, y0 + dy * i, w, h))],
            start + i)


def test_moving_vehicle_counted_once():
    c = VehicleCounter(W, H)
    drive(c, 1, "car", 0, 40, 100, 400, dx=20)
    assert c.counts("traffic")["total"] == 1
    assert c.counts("traffic")["car"] == 1
    assert c.counts("presence")["total"] == 1


def test_fragmented_track_is_one_vehicle():
    """The core bug: a tracker losing a vehicle and re-acquiring it under a new
    id must not produce two vehicles."""
    c = VehicleCounter(W, H)
    drive(c, 1, "car", 0, 20, 100, 400, dx=20)     # ends near x=480
    drive(c, 7, "car", 26, 20, 600, 400, dx=20)    # reappears where it would be
    assert c.stitches == 1
    assert c.counts("traffic")["total"] == 1


def test_distinct_vehicles_not_merged():
    """Stitching must not swallow a genuinely different vehicle elsewhere."""
    c = VehicleCounter(W, H)
    drive(c, 1, "car", 0, 20, 100, 200, dx=20)
    drive(c, 7, "car", 26, 20, 600, 650, dx=20)    # far away, other side
    assert c.stitches == 0
    assert c.counts("traffic")["total"] == 2


def test_concurrent_tracks_never_stitched():
    """Two tracks alive in the same frame are two vehicles, however close."""
    c = VehicleCounter(W, H)
    for i in range(30):
        c.update([T(1, "car", 0.9, box_at(100 + 10 * i, 400)),
                  T(2, "car", 0.9, box_at(140 + 10 * i, 400))], i)
    assert c.stitches == 0
    assert c.counts("traffic")["total"] == 2


def test_parked_vehicle_excluded_from_traffic_included_in_presence():
    c = VehicleCounter(W, H)
    drive(c, 1, "car", 0, 60, 300, 400)            # never moves
    assert c.counts("traffic")["total"] == 0
    assert c.counts("presence")["total"] == 1


def test_flicker_rejected():
    """A two-frame spurious detection is not a vehicle."""
    c = VehicleCounter(W, H)
    drive(c, 1, "car", 0, 2, 300, 400, dx=30)
    assert c.counts("presence")["total"] == 0
    v = c.vehicles()[0]
    assert "floor" in v.reject_reason


def test_far_field_speck_rejected():
    """A 20x20 blob is below the size floor no matter how it moves."""
    c = VehicleCounter(W, H)
    drive(c, 1, "car", 0, 20, 300, 100, dx=20, w=20, h=20)
    assert c.counts("presence")["total"] == 0


def test_small_but_very_persistent_track_is_counted():
    """Guards a real false negative: a distant motorcycle held for 174 frames
    was rejected purely on size when size and persistence were ANDed."""
    c = VehicleCounter(W, H)
    drive(c, 1, "motorcycle", 0, 174, 300, 120, dx=2, w=34, h=28)
    assert c.counts("presence")["total"] == 1


def test_large_but_brief_track_is_counted():
    """Guards the other half: a truck entering just before the clip ends."""
    c = VehicleCounter(W, H)
    drive(c, 1, "truck", 0, 5, 300, 400, dx=40, w=180, h=140)
    assert c.counts("presence")["total"] == 1


def test_class_vote_is_area_weighted():
    """A distant blob's guess must not outvote the close-up frames. Here the
    track is called 'truck' on many tiny frames and 'car' on a few big ones."""
    c = VehicleCounter(W, H)
    for i in range(30):
        c.update([T(1, "truck", 0.9, box_at(100 + 10 * i, 300, 40, 34))], i)
    for i in range(30, 40):
        c.update([T(1, "car", 0.9, box_at(100 + 10 * i, 300, 300, 240))], i)
    assert c.counts("traffic")["car"] == 1
    assert c.counts("traffic")["truck"] == 0


def test_counting_line_requires_crossing():
    line = ((0.5, 0.0), (0.5, 1.0))            # vertical line at x = 640
    c = VehicleCounter(W, H, CountConfig(line=line))
    drive(c, 1, "car", 0, 40, 100, 400, dx=30)       # 100 -> 1270, crosses
    drive(c, 2, "car", 0, 40, 100, 200, dx=5)        # 100 -> 295, never crosses
    assert c.counts("traffic")["total"] == 1
    assert c.counts("presence")["total"] == 2


def test_line_crossing_counts_once_not_per_frame():
    line = ((0.5, 0.0), (0.5, 1.0))
    c = VehicleCounter(W, H, CountConfig(line=line))
    drive(c, 1, "car", 0, 60, 100, 400, dx=20)
    assert c.counts("traffic")["total"] == 1


def test_report_is_self_describing():
    c = VehicleCounter(W, H)
    drive(c, 1, "car", 0, 40, 100, 400, dx=20)
    drive(c, 2, "car", 0, 2, 900, 400)
    r = c.report()
    assert r["traffic"]["total"] == 1
    assert r["raw_tracker_ids"] == 2
    assert r["rejected"] == 1
    assert r["config"]["min_frames"] == CountConfig.min_frames


def test_rejected_vehicles_carry_a_reason():
    """A count you cannot audit is a count you cannot trust."""
    c = VehicleCounter(W, H)
    drive(c, 1, "car", 0, 60, 300, 400)
    v = c.vehicles()[0]
    assert v.counted_presence and not v.counted_traffic
    assert "stationary" in v.reject_reason


def test_empty_video_counts_zero():
    c = VehicleCounter(W, H)
    for i in range(10):
        c.update([], i)
    assert c.counts("traffic")["total"] == 0
    assert c.report()["raw_tracker_ids"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
