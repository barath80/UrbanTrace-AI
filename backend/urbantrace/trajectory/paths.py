"""The road path each vehicle actually took, for drawing on a real map.

A vehicle's route is a list of cameras it was seen at, in order. That is a
sequence of *sightings*, not a path: what happened between two sightings is not
observed. This module fills that gap the only defensible way - by asking a
routing engine for the road a driver would take between them - and labels the
result as inferred, because it is.

WHAT IS MEASURED AND WHAT IS NOT

  measured  : the camera a plate was read at, and when (video_time_s)
  inferred  : the roads travelled between two consecutive sightings
  estimated : the road distance itself, IF GraphHopper was unavailable

Every path carries `is_real_road` and each leg carries its own `source`, so a
map can draw an inferred road path differently from a straight-line guess and
never imply the vehicle was tracked continuously.
"""
from __future__ import annotations

from ..core.cameras import CAMERAS
from ..routing import road_route


def _as_dicts(vehicles):
    return [v if isinstance(v, dict) else v.to_dict() for v in vehicles]


def build_vehicle_paths(vehicles, skip_routing: bool = False) -> dict:
    """{global_vehicle_id: path} for every vehicle with a route.

    A path holds one entry per leg (camera -> next camera), each with the real
    road geometry between them, plus a flattened `coordinates` list ready to
    hand to Leaflet as a single polyline.
    """
    out = {}
    for v in _as_dicts(vehicles):
        route = v.get("route", [])
        gid = v.get("global_vehicle_id")
        if not gid:
            continue

        # Sighting points are measured: the camera positions, in visit order.
        points = [{"camera_id": c,
                   "name": CAMERAS[c].name if c in CAMERAS else c,
                   "latitude": CAMERAS[c].latitude if c in CAMERAS else None,
                   "longitude": CAMERAS[c].longitude if c in CAMERAS else None}
                  for c in route if c in CAMERAS]

        legs, coords, total_km, total_min = [], [], 0.0, 0.0
        all_real = True
        for a, b in zip(route, route[1:]):
            if a not in CAMERAS or b not in CAMERAS:
                continue
            ca, cb = CAMERAS[a], CAMERAS[b]
            if skip_routing:
                from ..routing.graphhopper import _fallback
                r = _fallback([(ca.latitude, ca.longitude),
                               (cb.latitude, cb.longitude)],
                              "road routing disabled for this run")
            else:
                r = road_route(ca.latitude, ca.longitude,
                               cb.latitude, cb.longitude)
            all_real = all_real and r.is_real_road
            total_km += r.distance_km
            total_min += r.duration_min
            legs.append({"from_camera": a, "to_camera": b,
                         "distance_km": r.distance_km,
                         "typical_duration_min": r.duration_min,
                         "source": r.source, "is_real_road": r.is_real_road,
                         "error": r.error, "coordinates": r.coordinates})
            # Avoid duplicating the shared vertex between consecutive legs.
            coords.extend(r.coordinates if not coords else r.coordinates[1:])

        out[gid] = {
            "global_vehicle_id": gid,
            "plate_number": v.get("plate_number"),
            "route": route,
            "points": points,
            "legs": legs,
            "coordinates": coords,
            "road_distance_km": round(total_km, 3),
            "typical_duration_min": round(total_min, 1),
            "measured_duration_min": v.get("total_duration_minutes"),
            "is_real_road": bool(legs) and all_real,
            # Said plainly so no caller can present this as continuous tracking.
            "path_note": ("Camera sightings are measured. The road between two "
                          "sightings is the route a driver would take, from "
                          "GraphHopper - inferred, not observed."),
        }
    return out
