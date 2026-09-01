"""Tests for the city-scale layer.

These cover the properties that only break at scale and are therefore easy to
regress without noticing: identity precision, sub-quadratic resolution, graph
size, shard stability, queue back-pressure, and the ByteTrack id-counter bug
that appears only when a second camera comes online.

No GPU, no database and no network are required.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from urbantrace.scale.identity import (  # noqa: E402
    IdentityIndex, Sighting, blocking_keys, canonical_key, plate_similarity, same_vehicle)
from urbantrace.scale.registry import CameraRegistry, haversine_km  # noqa: E402
from urbantrace.scale.service import shard_of  # noqa: E402
from urbantrace.scale.tracking import Detections, MultiCameraTracker  # noqa: E402

T0 = datetime(2026, 8, 31, 9, 0, 0, tzinfo=timezone.utc)


def sighting(plate, camera, minutes=0.0, conf=0.9, track=0, lat=13.0, lon=80.0):
    return Sighting(plate=plate, camera_id=camera,
                    observed_at=T0 + timedelta(minutes=minutes),
                    confidence=conf, track_id=track, latitude=lat, longitude=lon)


# ------------------------------------------------------------- geometry

def test_haversine_matches_known_distance():
    # Koyambedu -> Ambattur, Chennai. Straight-line ~6.6 km.
    d = haversine_km(13.0694, 80.1948, 13.1143, 80.1548)
    assert 6.0 < d < 7.2


def test_haversine_is_zero_for_same_point():
    assert haversine_km(13.0, 80.0, 13.0, 80.0) == pytest.approx(0.0, abs=1e-9)


# -------------------------------------------------------- plate identity

@pytest.mark.parametrize("a,b", [
    ("TN01AB1234", "TNO1AB1234"),      # O read as 0
    ("TN01AB1234", "TN01A81234"),      # B read as 8
    ("TN01AB1234", "TNOIAB1234"),      # two confusions at once
    ("TN01AB1234", "TN01AB123"),       # a character dropped
    ("TN01AB1234", "TN01AB12345"),     # a spurious character
])
def test_same_vehicle_accepts_ocr_confusion(a, b):
    assert same_vehicle(a, b)


@pytest.mark.parametrize("a,b", [
    ("TN01AB1234", "TN01AB1235"),      # different last digit
    ("TN01AB1234", "TN01AB1334"),      # different middle digit
    ("TN01AB1234", "TN02AB1234"),      # different RTO district
    ("TN01AB1234", "KA05MM9999"),      # unrelated
    ("TN01AB1234", ""),
])
def test_same_vehicle_rejects_real_differences(a, b):
    """The costly error is merging two vehicles, so this is the important half.

    Similarity alone would accept several of these: TN01AB1234 vs TN01AB1235
    scores 0.90, above the 0.82 threshold.
    """
    assert not same_vehicle(a, b)


def test_similarity_still_ranks_candidates():
    assert plate_similarity("TN01AB1234", "TNO1AB1234") > 0.9
    assert plate_similarity("TN01AB1234", "KA05MM9999") < 0.3


def test_canonical_key_folds_confusable_characters():
    assert canonical_key("TNO1AB1234") == canonical_key("TN01AB1234")
    assert canonical_key("TN01AB1234") != canonical_key("TN01AB1235")


def test_blocking_keys_include_canonical_and_deletions():
    keys = blocking_keys("TN01AB1234")
    assert canonical_key("TN01AB1234") in keys
    assert len(keys) > 1
    # A dropped character must still share a key, or recall is lost.
    assert set(blocking_keys("TN01AB123")) & set(keys)


# ----------------------------------------------------- identity index

def test_index_merges_variants_and_separates_vehicles():
    idx = IdentityIndex()
    a = idx.add(sighting("TN01AB1234", "CAM_A", 0, 0.94, 1))
    b = idx.add(sighting("TNO1AB1234", "CAM_B", 12, 0.81, 2))
    c = idx.add(sighting("TN01A81234", "CAM_C", 25, 0.88, 3))
    other = idx.add(sighting("KA05MM9999", "CAM_B", 5, 0.77, 4))

    assert a == b == c
    assert other != a
    cluster = idx.get(a)
    assert cluster.plate == "TN01AB1234"          # most total evidence wins
    assert cluster.cameras() == ["CAM_A", "CAM_B", "CAM_C"]


def test_canonical_spelling_follows_total_evidence_not_one_lucky_frame():
    idx = IdentityIndex()
    for i in range(5):
        idx.add(sighting("TN01AB1234", f"CAM_{i}", i, 0.80, i))
    cid = idx.add(sighting("TNO1AB1234", "CAM_9", 9, 0.99, 99))
    assert idx.get(cid).plate == "TN01AB1234"


def test_repeated_sighting_of_same_track_does_not_duplicate_identity():
    idx = IdentityIndex()
    first = idx.add(sighting("TN01AB1234", "CAM_A", 0, 0.70, 7))
    again = idx.add(sighting("TN01AB1234", "CAM_A", 1, 0.95, 7))
    assert first == again
    assert len(idx.get(first).sightings) == 1


def test_feasibility_callback_can_veto_a_merge():
    """Two identical plates seen 1 minute and 500 km apart are not one car."""
    idx = IdentityIndex(feasible=lambda a, b: False)
    a = idx.add(sighting("TN01AB1234", "CAM_A", 0))
    b = idx.add(sighting("TN01AB1234", "CAM_FAR", 1))
    assert a != b


def test_resolution_is_sub_quadratic():
    """Blocking must keep the comparison count far below all-pairs.

    A deliberately low-entropy plate generator is used, because that is the
    adversarial case for blocking: many plates share a deletion key. Even so
    the work must stay a tiny fraction of the naive comparison count, and the
    per-insert cost must stay bounded by a small constant rather than growing
    with the corpus.
    """
    def measure(n):
        idx = IdentityIndex()
        for i in range(n):
            idx.add(sighting(f"TN{i % 90:02d}XY{i:05d}", f"CAM_{i % 50}", i * 0.1, track=i))
        st = idx.stats()
        return st["comparisons_per_add"], st["candidate_comparisons"]

    for n in (500, 5000):
        per_insert, total = measure(n)
        all_pairs = n * (n - 1) // 2
        assert total < all_pairs / 100, f"n={n}: {total} vs all-pairs {all_pairs}"
        assert per_insert < 10.0, f"n={n}: {per_insert} comparisons per insert"


def test_resolution_is_near_free_on_realistic_plates():
    """On a realistic plate distribution the candidate set is essentially empty."""
    import random
    random.seed(5)
    letters, digits = "ABCDEFGHJKMNPRSTUVWXYZ", "0123456789"

    def plate():
        return ("".join(random.choice(letters) for _ in range(2))
                + "".join(random.choice(digits) for _ in range(2))
                + "".join(random.choice(letters) for _ in range(2))
                + "".join(random.choice(digits) for _ in range(4)))

    idx = IdentityIndex()
    for i in range(5000):
        idx.add(sighting(plate(), f"CAM_{i % 1000}", i * 0.05, track=i))
    assert idx.stats()["comparisons_per_add"] < 1.0


def test_eviction_bounds_the_index():
    idx = IdentityIndex()
    for i in range(20):
        idx.add(sighting(f"TN01AB{i:04d}", "CAM_A", i, track=i))
    assert idx.stats()["clusters"] == 20
    idx.evict_before(T0 + timedelta(minutes=10))
    assert idx.stats()["clusters"] == 10
    # The blocking index must shrink too, or memory leaks silently.
    assert idx.stats()["blocking_keys"] > 0


# ---------------------------------------------------------- registry

def _grid_registry(n=400):
    reg = CameraRegistry(cell_km=2.0)
    reg.load([{"camera_id": f"CAM_{i:04d}", "name": f"N{i}",
               "latitude": 13.0 + (i % 20) * 0.01,
               "longitude": 80.0 + (i // 20) * 0.01} for i in range(n)])
    return reg


def test_graph_is_linear_not_all_pairs():
    n = 400
    reg = _grid_registry(n)
    info = reg.build_graph(k=6)
    all_pairs = n * (n - 1) // 2
    assert info["edges"] < n * 25
    assert info["edges"] < all_pairs / 10


def test_near_returns_sorted_and_bounded_results():
    reg = _grid_registry(400)
    found = reg.near(13.10, 80.10, radius_km=3.0, limit=5)
    assert len(found) <= 5
    assert found == sorted(found, key=lambda p: p[1])


def test_edge_exists_for_non_adjacent_pair_and_is_labelled():
    reg = _grid_registry(100)
    reg.build_graph(k=3)
    e = reg.edge("CAM_0000", "CAM_0099")
    assert e is not None
    assert e.road_distance_km > 0
    assert "nonadjacent" in e.distance_source or e.distance_source == "estimate"


def test_registry_rejects_bad_coordinates():
    reg = CameraRegistry()
    with pytest.raises(ValueError):
        reg.load([{"camera_id": "X", "latitude": 999.0, "longitude": 0.0}])


# ------------------------------------------------------------ sharding

def test_shard_assignment_is_stable_and_balanced():
    ids = [f"CAM_{i:05d}" for i in range(4000)]
    counts = {}
    for c in ids:
        counts[shard_of(c, 8)] = counts.get(shard_of(c, 8), 0) + 1
    assert len(counts) == 8
    assert max(counts.values()) < min(counts.values()) * 1.3
    # Stable across calls, so a restart does not reshuffle the estate.
    assert all(shard_of(c, 8) == shard_of(c, 8) for c in ids)


def test_single_shard_owns_everything():
    assert all(shard_of(f"CAM_{i}", 1) == 0 for i in range(100))


# ------------------------------------------------------------ tracking

def _dets(boxes, cls=2.0):
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return Detections(b, np.full(len(b), 0.9, np.float32), np.full(len(b), cls, np.float32))


def test_detections_converts_xyxy_to_xywh():
    d = _dets([[10, 20, 30, 60]])
    assert d.xywh[0].tolist() == [20.0, 40.0, 20.0, 40.0]
    assert len(d) == 1


def test_detections_supports_boolean_indexing():
    d = _dets([[0, 0, 10, 10], [20, 20, 30, 30]])
    sub = d[np.array([True, False])]
    assert len(sub) == 1


def test_new_camera_does_not_reissue_ids_of_a_running_camera():
    """Regression test for a real ByteTrack defect.

    Ultralytics allocates track ids from a CLASS-level counter and resets it in
    `BYTETracker.__init__`. Bringing a second camera online therefore rewound
    the counter, and a camera already running began reissuing ids it was still
    using -- two vehicles sharing one identity.
    """
    from ultralytics.trackers.basetrack import BaseTrack

    mct = MultiCameraTracker()
    for i in range(10):
        mct.update("CAM_A", _dets([[100 + i * 5, 100, 160 + i * 5, 160],
                                   [300 + i * 5, 300, 360 + i * 5, 360]]))
    before = BaseTrack._count
    assert before >= 2

    mct.update("CAM_NEW", _dets([[10, 10, 70, 70]]))
    assert BaseTrack._count >= before, "creating a camera rewound the global id counter"


def test_trackers_are_independent_per_camera():
    mct = MultiCameraTracker()
    for i in range(12):
        mct.update("CAM_A", _dets([[100 + i * 6, 100, 160 + i * 6, 160]], cls=2.0))
        mct.update("CAM_B", _dets([[700 - i * 6, 400, 760 - i * 6, 460]], cls=7.0))
    a = mct.update("CAM_A", _dets([[172, 100, 232, 160]], cls=2.0))
    b = mct.update("CAM_B", _dets([[628, 400, 688, 460]], cls=7.0))
    assert a and b
    assert a[0].cls_name == "car" and b[0].cls_name == "truck"
    assert a[0].track_id != b[0].track_id
    assert mct.stats()["cameras_tracked"] == 2


def test_tracker_eviction_releases_state():
    # Negative idle window puts the cutoff in the future, so the entry
    # created a moment ago is already stale.
    mct = MultiCameraTracker(idle_evict_seconds=-1.0)
    mct.update("CAM_A", _dets([[0, 0, 10, 10]]))
    assert mct.stats()["cameras_tracked"] == 1
    assert mct.evict_idle() == 1
    assert mct.stats()["cameras_tracked"] == 0


# ------------------------------------------------------- back-pressure

def test_bus_drops_oldest_and_never_blocks():
    from urbantrace.scale.inference import FrameBus, FrameItem

    bus = FrameBus(depth=2, target_fps=0)
    frame = np.zeros((4, 4, 3), np.uint8)
    for i in range(5):
        bus.submit(FrameItem("CAM_A", frame, T0, 0.0, i))
    st = bus.stats()
    assert st["frames_pending"] == 2
    assert st["dropped"] == 3
    # The frames kept must be the NEWEST, which is what live monitoring wants.
    batch = bus.drain_batch(10, 0.01)
    assert [f.frame_number for f in batch] == [3]


def test_bus_is_fair_across_cameras():
    from urbantrace.scale.inference import FrameBus, FrameItem

    bus = FrameBus(depth=4, target_fps=0)
    frame = np.zeros((4, 4, 3), np.uint8)
    for cam in ("A", "B", "C"):
        for i in range(4):
            bus.submit(FrameItem(cam, frame, T0, 0.0, i))
    batch = bus.drain_batch(3, 0.01)
    assert sorted(f.camera_id for f in batch) == ["A", "B", "C"]


def test_admission_rate_limits_per_camera():
    from urbantrace.scale.inference import FrameBus

    bus = FrameBus(target_fps=10.0)
    assert bus.admit("CAM_A", now=100.0)
    assert not bus.admit("CAM_A", now=100.01)     # inside the 0.1s period
    assert bus.admit("CAM_A", now=100.2)
    assert bus.admit("CAM_B", now=100.01)         # a different camera is unaffected


# ------------------------------------------------- leg / journey continuity
# These pin the edge cases in speed measurement between sparse cameras. Each
# one produced a wrong number on real data before it was guarded.

def _leg_registry():
    reg = CameraRegistry(cell_km=5.0)
    reg.load([
        {"camera_id": "CAM_A", "name": "A", "latitude": 13.0694, "longitude": 80.1948},
        {"camera_id": "CAM_B", "name": "B", "latitude": 13.1143, "longitude": 80.1548},
    ])
    reg.build_graph()
    return reg


def _sight(cam, minutes, lat, lon):
    return {"camera_id": cam, "observed_at": T0 + timedelta(minutes=minutes),
            "latitude": lat, "longitude": lon, "plate": "TN01AB1234"}


def _pair(minutes):
    return (_sight("CAM_A", 0, 13.0694, 80.1948),
            _sight("CAM_B", minutes, 13.1143, 80.1548))


def test_leg_rejects_zero_and_negative_time():
    """E6: simultaneous or out-of-order sightings must not divide by zero."""
    from urbantrace.scale.analytics import _leg
    reg = _leg_registry()
    for minutes in (0.0, -5.0):
        a, b = _pair(minutes)
        leg = _leg(a, b, reg)
        assert leg["verdict"] == "IMPLAUSIBLE"
        # A missing measurement, never a measured zero.
        assert leg["implied_speed_kmh"] is None


def test_leg_flags_physically_impossible_speed():
    """E8: the same plate in two places at once means a clone or a misread."""
    from urbantrace.scale.analytics import _leg
    reg = _leg_registry()
    a, b = _pair(1.0)                      # ~8 km in 1 minute
    leg = _leg(a, b, reg)
    assert leg["verdict"] == "IMPLAUSIBLE"
    assert leg["implied_speed_kmh"] > 140


def test_leg_flags_slow_but_still_possible():
    """A genuinely slow crossing is SLOW, not impossible."""
    from urbantrace.scale.analytics import _leg
    reg = _leg_registry()
    a, b = _pair(45.0)
    leg = _leg(a, b, reg)
    assert leg["verdict"] == "SLOW"
    assert 0 < leg["implied_speed_kmh"] < 30


def test_normal_crossing_is_plausible():
    from urbantrace.scale.analytics import _leg
    reg = _leg_registry()
    a, b = _pair(14.0)
    leg = _leg(a, b, reg)
    assert leg["verdict"] == "PLAUSIBLE"
    assert 25 < leg["implied_speed_kmh"] < 50


def test_long_gap_becomes_a_break_not_a_leg(monkeypatch):
    """E9: a vehicle seen days apart made two trips, not one four-day crawl.

    Before this guard the pair produced a SLOW leg at 0.1 km/h which was
    averaged into the origin-destination matrix and the congestion baseline.
    """
    import urbantrace.scale.analytics as A
    reg = _leg_registry()
    monkeypatch.setattr(A, "REGISTRY", reg, raising=False)
    monkeypatch.setitem(sys.modules, "urbantrace.scale.registry",
                        sys.modules["urbantrace.scale.registry"])
    sys.modules["urbantrace.scale.registry"].REGISTRY = reg

    sightings = [_sight("CAM_A", 0, 13.0694, 80.1948),
                 _sight("CAM_B", 4 * 24 * 60, 13.1143, 80.1548)]
    legs, breaks = A._legs_from(sightings)
    assert legs == []
    assert len(breaks) == 1
    assert breaks[0]["after_camera"] == "CAM_A"
    assert breaks[0]["before_camera"] == "CAM_B"
    assert breaks[0]["gap_minutes"] == pytest.approx(5760.0)


def test_short_gap_still_produces_a_leg(monkeypatch):
    import urbantrace.scale.analytics as A
    reg = _leg_registry()
    sys.modules["urbantrace.scale.registry"].REGISTRY = reg
    sightings = [_sight("CAM_A", 0, 13.0694, 80.1948),
                 _sight("CAM_B", 14, 13.1143, 80.1548)]
    legs, breaks = A._legs_from(sightings)
    assert len(legs) == 1
    assert breaks == []


def test_same_camera_repeats_produce_no_leg(monkeypatch):
    """E7: a vehicle idling in view is not a zero-kilometre journey."""
    import urbantrace.scale.analytics as A
    sys.modules["urbantrace.scale.registry"].REGISTRY = _leg_registry()
    sightings = [_sight("CAM_A", m, 13.0694, 80.1948) for m in (0, 1, 2, 3)]
    legs, breaks = A._legs_from(sightings)
    assert legs == []
    assert breaks == []


def test_service_does_not_write_over_gap_legs():
    """The live path must not persist a leg it would have to exclude later."""
    from urbantrace.scale.identity import Cluster
    from urbantrace.scale.service import IngestService

    svc = IngestService(registry=_leg_registry(), persist=False)
    c = Cluster(cluster_id="GV_TEST", plate="TN01AB1234")
    c.sightings = [
        sighting("TN01AB1234", "CAM_A", 0, track=1, lat=13.0694, lon=80.1948),
        sighting("TN01AB1234", "CAM_B", 4 * 24 * 60, track=2, lat=13.1143, lon=80.1548),
    ]
    assert svc._legs_for(c) == []

    c.sightings[1] = sighting("TN01AB1234", "CAM_B", 14, track=2,
                              lat=13.1143, lon=80.1548)
    assert len(svc._legs_for(c)) == 1
