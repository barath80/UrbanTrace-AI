"""Tests for real-road routing and per-vehicle path building.

No network and no API key are used: the GraphHopper call is stubbed. What these
pin down is the behaviour that matters when routing is NOT available, because
that is when a dashboard is most likely to quietly show a straight line as if
it were a measured path.

    python -m pytest tests/test_routing.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from urbantrace.routing import graphhopper as gh  # noqa: E402
from urbantrace.routing.graphhopper import RoadRoute, road_route  # noqa: E402

KOYAMBEDU = (13.0694, 80.1948)
AMBATTUR = (13.1143, 80.1548)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never read or write the project's real route cache during tests."""
    monkeypatch.setattr(gh, "CACHE_DIR", tmp_path / "routes")
    yield


def _fake_response(distance_m=9182.0, time_ms=1_446_000, n=5):
    coords = [[80.1948 + i * 0.001, 13.0694 + i * 0.001] for i in range(n)]
    return {"paths": [{"distance": distance_m, "time": time_ms,
                       "points": {"coordinates": coords}}]}


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fallback_when_no_api_key(monkeypatch):
    """No key must not raise, and must not claim to be a real road."""
    monkeypatch.setattr(gh, "_load_key", lambda: "")
    r = road_route(*KOYAMBEDU, *AMBATTUR)
    assert isinstance(r, RoadRoute)
    assert r.source == "fallback_estimate"
    assert r.is_real_road is False
    assert "GRAPHHOPPER_API_KEY" in r.error
    assert r.distance_km > 0
    # The straight line is just the endpoints, so a caller drawing it gets a
    # visibly different shape from a road.
    assert r.coordinates == [list(KOYAMBEDU), list(AMBATTUR)]


def test_fallback_when_network_fails(monkeypatch):
    monkeypatch.setattr(gh, "_load_key", lambda: "k")
    monkeypatch.setattr(gh.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no route to host")))
    r = road_route(*KOYAMBEDU, *AMBATTUR)
    assert r.source == "fallback_estimate"
    assert r.is_real_road is False
    assert "OSError" in r.error


def test_fallback_distance_is_not_the_straight_line():
    """The estimate applies a road factor; a road is longer than the crow flies."""
    gc = gh.haversine_km(*KOYAMBEDU, *AMBATTUR)
    r = gh._fallback([KOYAMBEDU, AMBATTUR], "test")
    # The module rounds to 3 decimals, so compare at that precision.
    assert r.distance_km == pytest.approx(gc * gh.FALLBACK_ROAD_FACTOR, abs=5e-4)
    assert r.distance_km > gc


def test_successful_route_is_marked_real(monkeypatch):
    monkeypatch.setattr(gh, "_load_key", lambda: "k")
    monkeypatch.setattr(gh.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(_fake_response()))
    r = road_route(*KOYAMBEDU, *AMBATTUR)
    assert r.source == "graphhopper"
    assert r.is_real_road is True
    assert r.distance_km == pytest.approx(9.182)
    assert r.duration_min == pytest.approx(24.1)
    # GraphHopper returns [lon, lat]; Leaflet needs [lat, lon].
    assert r.coordinates[0] == [pytest.approx(13.0694), pytest.approx(80.1948)]


def test_second_call_is_served_from_cache(monkeypatch):
    calls = {"n": 0}

    def _once(*a, **k):
        calls["n"] += 1
        return _Resp(_fake_response())

    monkeypatch.setattr(gh, "_load_key", lambda: "k")
    monkeypatch.setattr(gh.urllib.request, "urlopen", _once)

    first = road_route(*KOYAMBEDU, *AMBATTUR)
    second = road_route(*KOYAMBEDU, *AMBATTUR)
    assert calls["n"] == 1, "the second call must not hit the network"
    assert first.source == "graphhopper"
    assert second.source == "cache"
    assert second.is_real_road is True
    assert second.distance_km == first.distance_km


def test_empty_paths_response_falls_back(monkeypatch):
    monkeypatch.setattr(gh, "_load_key", lambda: "k")
    monkeypatch.setattr(gh.urllib.request, "urlopen",
                        lambda *a, **k: _Resp({"paths": [], "message": "quota"}))
    r = road_route(*KOYAMBEDU, *AMBATTUR)
    assert r.source == "fallback_estimate"
    assert "quota" in r.error


def test_corrupt_cache_entry_is_refetched(monkeypatch):
    monkeypatch.setattr(gh, "_load_key", lambda: "k")
    monkeypatch.setattr(gh.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(_fake_response()))
    gh.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gh._cache_path([KOYAMBEDU, AMBATTUR], "car").write_text("{not json")
    r = road_route(*KOYAMBEDU, *AMBATTUR)
    assert r.is_real_road is True


# ------------------------------------------------------------- paths

def test_vehicle_path_marks_inferred_road(monkeypatch):
    """The path between two sightings is inferred; it must say so."""
    from urbantrace.core import cameras as C
    from urbantrace.trajectory import paths as P

    monkeypatch.setattr(P, "road_route", lambda *a, **k: RoadRoute(
        distance_km=9.18, duration_min=24.1,
        coordinates=[[13.07, 80.19], [13.09, 80.17], [13.11, 80.15]],
        source="graphhopper"))

    if len(C.CAMERAS) < 2:
        pytest.skip("needs at least two configured cameras")
    a, b = list(C.CAMERAS)[:2]

    out = P.build_vehicle_paths([{
        "global_vehicle_id": "GV_0001", "plate_number": "TN01AB1234",
        "route": [a, b], "total_duration_minutes": 14.0}])

    p = out["GV_0001"]
    assert p["is_real_road"] is True
    assert p["road_distance_km"] == pytest.approx(9.18)
    assert p["measured_duration_min"] == 14.0
    assert len(p["points"]) == 2
    assert len(p["coordinates"]) == 3
    assert "inferred" in p["path_note"]


def test_single_camera_vehicle_has_no_drawn_path(monkeypatch):
    from urbantrace.core import cameras as C
    from urbantrace.trajectory import paths as P
    if not C.CAMERAS:
        pytest.skip("needs a configured camera")
    a = list(C.CAMERAS)[0]
    out = P.build_vehicle_paths([{
        "global_vehicle_id": "GV_0009", "plate_number": "TN09XX0000",
        "route": [a], "total_duration_minutes": 0.0}])
    p = out["GV_0009"]
    assert p["legs"] == []
    assert p["coordinates"] == []
    # One sighting is not a route, so it must not be badged as a real road.
    assert p["is_real_road"] is False


def test_legs_join_without_duplicating_the_shared_vertex(monkeypatch):
    from urbantrace.core import cameras as C
    from urbantrace.trajectory import paths as P
    if len(C.CAMERAS) < 2:
        pytest.skip("needs at least two configured cameras")
    a, b = list(C.CAMERAS)[:2]
    monkeypatch.setattr(P, "road_route", lambda *args, **k: RoadRoute(
        distance_km=1.0, duration_min=2.0,
        coordinates=[[1.0, 1.0], [2.0, 2.0]], source="graphhopper"))
    out = P.build_vehicle_paths([{
        "global_vehicle_id": "GV_1", "plate_number": "P",
        "route": [a, b, a], "total_duration_minutes": 4.0}])
    # Two legs of 2 points each -> 3 points, not 4.
    assert out["GV_1"]["coordinates"] == [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
