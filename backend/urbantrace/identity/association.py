"""Cross-camera vehicle association and trajectory reconstruction.

Local track IDs are per-video and meaningless across cameras. This module binds
observations from CAM_01/02/03 into GLOBAL vehicles and reconstructs the route.

Association is never plain string equality. Each candidate pair is scored on:
    plate similarity (OCR-confusion aware) | OCR confidence
    travel-time plausibility vs the camera graph | direction consistency

Every transition carries an explicit verdict and a human-readable reason, so an
operator can see WHY two sightings were judged the same vehicle.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from itertools import combinations

from ..core.cameras import CAMERAS, get_edge

# Characters that OCR genuinely confuses. Substituting within a group is cheap;
# any other substitution is expensive.
CONFUSION_GROUPS = [set("0OQD"), set("1IL"), set("2Z"), set("5S"), set("8B"),
                    set("6G"), set("7T"), set("4A"), set("9P")]

MIN_SIMILARITY = 0.82
MAX_PLAUSIBLE_SPEED_KMH = 140.0
WARN_SPEED_KMH = 100.0


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    for g in CONFUSION_GROUPS:
        if a in g and b in g:
            return 0.35          # visually ambiguous -> cheap
    return 1.0


def plate_similarity(p1: str, p2: str) -> float:
    """Confusion-weighted normalised edit similarity in [0,1]."""
    if not p1 or not p2:
        return 0.0
    if p1 == p2:
        return 1.0
    n, m = len(p1), len(p2)
    prev = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [float(i)] + [0.0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1.0,                       # deletion
                         cur[j - 1] + 1.0,                    # insertion
                         prev[j - 1] + _sub_cost(p1[i - 1], p2[j - 1]))
        prev = cur
    return max(0.0, 1.0 - prev[m] / max(n, m))


@dataclass
class Leg:
    """One camera-to-camera transition within a trajectory."""
    from_camera: str
    to_camera: str
    departed: str
    arrived: str
    travel_minutes: float
    road_distance_km: float
    implied_speed_kmh: float
    expected_min_minutes: float
    expected_max_minutes: float
    verdict: str          # PLAUSIBLE | SLOW | FAST | IMPLAUSIBLE | UNKNOWN_EDGE
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GlobalVehicle:
    global_vehicle_id: str
    plate_number: str
    observations: list = field(default_factory=list)
    route: list = field(default_factory=list)
    legs: list = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    total_duration_minutes: float = 0.0
    camera_count: int = 0
    merged_variants: list = field(default_factory=list)
    confidence: float = 0.0
    route_status: str = ""      # COMPLETE_CORRIDOR | PARTIAL_ROUTE | DROPPED_AT_ORIGIN
    dropped_at: str | None = None
    status_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["route_display"] = " -> ".join(self.route)
        return d


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def validate_transition(a: dict, b: dict) -> Leg:
    """Spatio-temporal feasibility of moving from sighting `a` to sighting `b`."""
    t1, t2 = _ts(a["timestamp"]), _ts(b["timestamp"])
    minutes = (t2 - t1).total_seconds() / 60.0
    edge = get_edge(a["camera_id"], b["camera_id"])

    if edge is None:
        return Leg(a["camera_id"], b["camera_id"], a["timestamp"], b["timestamp"],
                   round(minutes, 2), 0.0, 0.0, 0.0, 0.0, "UNKNOWN_EDGE",
                   f"No configured road link between {a['camera_id']} and {b['camera_id']}.")

    dist = edge.road_distance_km
    speed = (dist / (minutes / 60.0)) if minutes > 0 else float("inf")

    if minutes <= 0:
        verdict = "IMPLAUSIBLE"
        reason = (f"Arrival at {b['camera_id']} is not after departure from "
                  f"{a['camera_id']} ({minutes:.2f} min).")
    elif speed > MAX_PLAUSIBLE_SPEED_KMH:
        verdict = "IMPLAUSIBLE"
        reason = (f"{dist:.2f} km covered in {minutes:.2f} min implies "
                  f"{speed:.0f} km/h, above the {MAX_PLAUSIBLE_SPEED_KMH:.0f} km/h feasibility limit.")
    elif speed > WARN_SPEED_KMH:
        verdict = "FAST"
        reason = (f"{dist:.2f} km in {minutes:.2f} min implies {speed:.0f} km/h - "
                  f"possible but above the {WARN_SPEED_KMH:.0f} km/h expectation for this corridor.")
    elif minutes > edge.expected_max_minutes * 1.5:
        verdict = "SLOW"
        reason = (f"Took {minutes:.1f} min against an expected "
                  f"{edge.expected_min_minutes:.1f}-{edge.expected_max_minutes:.1f} min - "
                  f"congestion or an intermediate stop.")
    else:
        verdict = "PLAUSIBLE"
        reason = (f"{dist:.2f} km in {minutes:.1f} min ({speed:.0f} km/h), within the "
                  f"expected {edge.expected_min_minutes:.1f}-{edge.expected_max_minutes:.1f} min window.")

    return Leg(a["camera_id"], b["camera_id"], a["timestamp"], b["timestamp"],
               round(minutes, 2), dist, round(speed, 1),
               edge.expected_min_minutes, edge.expected_max_minutes, verdict, reason)


def _merge_similar_plates(plates: list, observations_by_plate: dict) -> dict:
    """Union plate strings that are near-identical AND temporally compatible.

    Guard: two plates are only merged if at least one supporting pair of
    sightings forms a physically plausible transition. This stops genuinely
    different vehicles with similar plates from collapsing into one identity.
    """
    parent = {p: p for p in plates}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the plate with the stronger evidence as the canonical form.
            wa = max(o["ocr_confidence"] for o in observations_by_plate[ra])
            wb = max(o["ocr_confidence"] for o in observations_by_plate[rb])
            parent[rb if wa >= wb else ra] = ra if wa >= wb else rb

    for p1, p2 in combinations(plates, 2):
        if plate_similarity(p1, p2) < MIN_SIMILARITY:
            continue
        ok = False
        for o1 in observations_by_plate[p1]:
            for o2 in observations_by_plate[p2]:
                if o1["camera_id"] == o2["camera_id"]:
                    continue
                a, b = sorted((o1, o2), key=lambda o: o["timestamp"])
                if validate_transition(a, b).verdict in ("PLAUSIBLE", "SLOW", "FAST"):
                    ok = True
                    break
            if ok:
                break
        if ok:
            union(p1, p2)

    return {p: find(p) for p in plates}


def associate(observations: list, min_confidence: float = 0.0) -> list:
    """Group observations from all cameras into global vehicles with routes."""
    obs = [o if isinstance(o, dict) else o.to_dict() for o in observations]
    obs = [o for o in obs if o.get("plate_number") and o["ocr_confidence"] >= min_confidence]

    by_plate: dict = {}
    for o in obs:
        by_plate.setdefault(o["plate_number"], []).append(o)

    canonical = _merge_similar_plates(list(by_plate), by_plate)

    grouped: dict = {}
    for plate, root in canonical.items():
        grouped.setdefault(root, []).extend(by_plate[plate])

    vehicles = []
    for idx, (plate, group) in enumerate(sorted(grouped.items()), start=1):
        group.sort(key=lambda o: o["timestamp"])
        variants = sorted({o["plate_number"] for o in group} - {plate})

        legs = [validate_transition(group[i], group[i + 1]) for i in range(len(group) - 1)]
        # A transition between two sightings at the SAME camera is not a route hop.
        route, seen_last = [], None
        for o in group:
            if o["camera_id"] != seen_last:
                route.append(o["camera_id"])
                seen_last = o["camera_id"]

        first, last = _ts(group[0]["timestamp"]), _ts(group[-1]["timestamp"])
        conf = sum(o["ocr_confidence"] for o in group) / len(group)

        # Route completeness. A vehicle seen at only one camera did not
        # continue along the corridor - it is reported as dropped there, not
        # as a route.
        n_cams = len({o["camera_id"] for o in group})
        if len(CAMERAS) <= 1:
            # A one-camera network has no corridor to traverse, so calling this
            # a "complete route" would overstate what the data shows.
            status = "SINGLE_CAMERA_NETWORK"
            dropped = route[0] if route else None
            reason = (f"Only one camera is configured, so no route can be "
                      f"reconstructed. Vehicle observed at {route[0]} "
                      f"({CAMERAS[route[0]].name}).") if route else \
                     "Only one camera is configured; no route can be reconstructed."
        elif n_cams >= len(CAMERAS):
            status = "COMPLETE_CORRIDOR"
            dropped = None
            reason = f"Observed at all {n_cams} cameras; full corridor traversal."
        elif n_cams > 1:
            status = "PARTIAL_ROUTE"
            dropped = route[-1]
            reason = (f"Observed at {n_cams} of {len(CAMERAS)} cameras; last seen at "
                      f"{route[-1]} ({CAMERAS[route[-1]].name}) and not observed downstream.")
        else:
            status = "DROPPED_AT_ORIGIN"
            dropped = route[0]
            reason = (f"Observed only at {route[0]} ({CAMERAS[route[0]].name}); "
                      f"never reached a downstream camera.")

        vehicles.append(GlobalVehicle(
            global_vehicle_id=f"GV_{idx:04d}",
            plate_number=plate,
            observations=group,
            route=route,
            legs=[l.to_dict() for l in legs],
            first_seen=group[0]["timestamp"],
            last_seen=group[-1]["timestamp"],
            total_duration_minutes=round((last - first).total_seconds() / 60.0, 2),
            camera_count=len({o["camera_id"] for o in group}),
            merged_variants=variants,
            confidence=round(conf, 4),
            route_status=status, dropped_at=dropped, status_reason=reason,
        ))

    # Multi-camera vehicles first - those are the ones with an actual route.
    vehicles.sort(key=lambda v: (-v.camera_count, -v.confidence))
    return vehicles


def summarize(vehicles: list) -> dict:
    multi = [v for v in vehicles if v.camera_count > 1]
    by_status = {}
    for v in vehicles:
        by_status[v.route_status] = by_status.get(v.route_status, 0) + 1
    dropped = {}
    for v in vehicles:
        if v.route_status == "DROPPED_AT_ORIGIN" and v.dropped_at:
            dropped[v.dropped_at] = dropped.get(v.dropped_at, 0) + 1
    return {
        "global_vehicles": len(vehicles),
        "multi_camera_vehicles": len(multi),
        "single_camera_vehicles": len(vehicles) - len(multi),
        "by_route_status": by_status,
        "dropped_by_camera": dropped,
        "routes": [{"plate": v.plate_number, "route": " -> ".join(v.route),
                    "cameras": v.camera_count,
                    "duration_min": v.total_duration_minutes} for v in multi],
    }
