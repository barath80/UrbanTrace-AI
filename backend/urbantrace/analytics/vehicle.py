"""Everything known about one vehicle, in one payload.

The tracking view needs a plate's sightings, its road path, its per-leg
verdicts, its own heat surface and its alerts. Fetching those as five separate
round trips makes the client stitch them together and invites the halves to
disagree - which is exactly how the tracking view came to show "2 cameras" and
"1 sighting" at the same time.

This module assembles one consistent dossier server-side, from the database and
the routing cache, with provenance on every derived number.
"""
from __future__ import annotations

import math

# Heat weighting. A camera sighting is a MEASURED presence, so it dominates the
# surface; the road between two sightings is inferred, so it contributes a low
# baseline that shows the corridor without claiming the vehicle was observed
# along it.
SIGHTING_WEIGHT = 1.0
PATH_WEIGHT = 0.22
# Sample the road path at roughly this spacing so the heat layer is even
# regardless of how densely the router returned geometry.
PATH_SAMPLE_M = 120.0


def _haversine_m(a, b) -> float:
    r = 6371008.8
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def path_heat_points(path: dict, timeline: list) -> list:
    """[[lat, lon, weight], ...] - this vehicle's own heat surface.

    Sightings are measured and weighted heavily. The road between them is
    inferred and weighted low, so the map shows the corridor the vehicle used
    without implying it was seen at every point along it.
    """
    pts = []
    for t in timeline:
        if t.get("latitude") is None:
            continue
        pts.append([float(t["latitude"]), float(t["longitude"]), SIGHTING_WEIGHT])

    coords = (path or {}).get("coordinates") or []
    if len(coords) > 1:
        acc = 0.0
        last = coords[0]
        pts.append([float(last[0]), float(last[1]), PATH_WEIGHT])
        for c in coords[1:]:
            acc += _haversine_m(last, c)
            if acc >= PATH_SAMPLE_M:
                pts.append([float(c[0]), float(c[1]), PATH_WEIGHT])
                acc = 0.0
            last = c
    return pts


def journey_stats(timeline: list, legs: list, path: dict) -> dict:
    """Measured journey figures, kept separate from inferred ones."""
    measured_min = None
    if len(timeline) > 1:
        try:
            from datetime import datetime

            def _p(v):
                return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))

            measured_min = round(
                (_p(timeline[-1]["observed_at"]) -
                 _p(timeline[0]["observed_at"])).total_seconds() / 60.0, 2)
        except Exception:
            measured_min = None

    road_km = (path or {}).get("road_distance_km")
    typical_min = (path or {}).get("typical_duration_min")
    speeds = [l.get("implied_speed_kmh") for l in legs
              if l.get("implied_speed_kmh") is not None]
    mean_speed = round(sum(speeds) / len(speeds), 1) if speeds else None

    delay = None
    if measured_min is not None and typical_min:
        delay = round(measured_min - typical_min, 1)

    return {
        "sightings": len(timeline),
        "cameras": len({t.get("camera_id") for t in timeline}),
        "measured_duration_min": measured_min,
        "road_distance_km": road_km,
        "typical_duration_min": typical_min,
        # Negative = faster than the road's typical time, positive = slower.
        "delay_vs_typical_min": delay,
        "mean_implied_speed_kmh": mean_speed,
        "verdicts": {v: sum(1 for l in legs if l.get("verdict") == v)
                     for v in {l.get("verdict") for l in legs} if v},
        "provenance": {
            "sightings": "measured (camera + video frame index)",
            "wall_clock": "derived (configured camera start + measured offset)",
            "road_distance_km": ((path or {}).get("is_real_road")
                                 and "GraphHopper road route"
                                 or "great-circle estimate"),
            "path_between_sightings": "inferred, not observed",
        },
    }
