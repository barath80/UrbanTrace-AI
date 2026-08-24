"""Dynamic camera network.

The network is NOT fixed at three cameras. It is built at run time from
whatever videos are present in `data/videos/`, one camera per video, in any
number.

Location mapping, in priority order:
  1. `data/cameras.json`  - your explicit mapping (video file -> place, lat/lon)
  2. `DEFAULT_LOCATIONS`  - a Chennai-corridor fallback pool, used in order for
                            any video without an entry, so the system works
                            out of the box and can be corrected later.

Road links between cameras are derived from the coordinates: cameras are
ordered along the corridor and consecutive pairs get an edge, plus skip-edges
so a vehicle missed by an intermediate camera still forms a valid transition.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VIDEO_DIR = ROOT / "data" / "videos"
CONFIG_FILE = ROOT / "data" / "cameras.json"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg")

# Fallback locations along the real Chennai western corridor, used in order for
# any video that has no explicit entry in cameras.json. Edit cameras.json to
# map your footage to its true locations.
DEFAULT_LOCATIONS = [
    ("Koyambedu Junction",   "100 Feet Road / Jawaharlal Nehru Salai", 13.0694, 80.1948, "WEST"),
    ("Ambattur Estate",      "Chennai-Tiruvallur High Road",           13.1143, 80.1548, "WEST"),
    ("Avadi Checkpost",      "Chennai-Tiruvallur High Road",           13.1147, 80.1098, "WEST"),
    ("Thiruninravur",        "Chennai-Tiruvallur High Road",           13.1180, 80.0300, "WEST"),
    ("Tiruvallur Bypass",    "NH-716 Tiruvallur Bypass",               13.1430, 79.9080, "WEST"),
    ("Poonamallee Bypass",   "NH-48 Poonamallee Bypass",               13.0480, 80.0950, "WEST"),
    ("Porur Junction",       "Mount-Poonamallee Road",                 13.0358, 80.1565, "WEST"),
    ("Guindy Junction",      "Anna Salai / GST Road",                  13.0067, 80.2206, "SOUTH"),
    ("Tambaram",             "GST Road (NH-45)",                       12.9249, 80.1000, "SOUTH"),
    ("Chengalpattu",         "GST Road (NH-45)",                       12.6819, 79.9760, "SOUTH"),
    ("Sriperumbudur",        "NH-48 Chennai-Bengaluru Highway",        12.9675, 79.9430, "WEST"),
    ("Kanchipuram",          "NH-48 / Chennai-Bengaluru Highway",      12.8342, 79.7036, "WEST"),
]

# Road distance is ~1.25x great-circle for an urban corridor.
ROAD_FACTOR = 1.25
# Speed band used to derive the expected travel-time window for an edge.
FREEFLOW_KMH = 55.0
CONGESTED_KMH = 22.0


@dataclass(frozen=True)
class Camera:
    camera_id: str
    name: str
    road: str
    latitude: float
    longitude: float
    direction: str
    video_file: str
    start_time: str = ""          # ISO capture start (configured, not measured)
    location_source: str = "default"   # "config" | "default"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CameraEdge:
    source: str
    destination: str
    road_distance_km: float
    expected_min_minutes: float
    expected_max_minutes: float
    road_segment: str

    def to_dict(self) -> dict:
        return asdict(self)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# Populated by load_network(); mutated IN PLACE so `from .cameras import CAMERAS`
# elsewhere keeps working after a reload.
CAMERAS: dict = {}
CAMERA_EDGES: list = []
EDGE_INDEX: dict = {}
CAMERA_ORDER: list = []


def _natural_key(name: str):
    """Sort CAMERA_2 before CAMERA_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def discover_videos(video_dir: Path | None = None) -> list:
    d = Path(video_dir) if video_dir else VIDEO_DIR
    if not d.exists():
        return []
    vids = [p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(vids, key=lambda p: _natural_key(p.name))


def _load_config() -> dict:
    """cameras.json -> {video_filename: {...}} (also keyed by camera_id)."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        raw = json.loads(CONFIG_FILE.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{CONFIG_FILE} is not valid JSON: {e}") from e
    entries = raw.get("cameras", raw) if isinstance(raw, dict) else raw
    out = {}
    if isinstance(entries, dict):
        entries = [{"camera_id": k, **v} for k, v in entries.items()]
    for e in entries or []:
        if e.get("video_file"):
            out[e["video_file"]] = e
        if e.get("camera_id"):
            out.setdefault(e["camera_id"], e)
    return out


def load_network(video_dir: Path | None = None,
                 base_start: datetime | None = None,
                 stagger_minutes: float | None = None) -> dict:
    """Build the camera network from the videos present. Returns CAMERAS.

    One camera per video file, numbered in natural filename order.
    `stagger_minutes` sets each camera's configured capture start relative to
    the previous one; when None it is derived from the expected travel time
    along the corridor.
    """
    videos = discover_videos(video_dir)
    cfg = _load_config()

    CAMERAS.clear()
    CAMERA_ORDER.clear()
    base = base_start or datetime(2026, 8, 24, 10, 0, 0)

    for i, vid in enumerate(videos):
        entry = cfg.get(vid.name) or cfg.get(f"CAM_{i+1:02d}") or {}
        cam_id = entry.get("camera_id") or f"CAM_{i+1:02d}"
        if entry:
            name = entry.get("name", f"Camera {i+1}")
            road = entry.get("road", "")
            lat = float(entry["latitude"]); lon = float(entry["longitude"])
            direction = entry.get("direction", "UNKNOWN")
            src = "config"
        else:
            d = DEFAULT_LOCATIONS[i % len(DEFAULT_LOCATIONS)]
            name, road, lat, lon, direction = d
            src = "default"

        CAMERAS[cam_id] = Camera(
            camera_id=cam_id, name=name, road=road,
            latitude=lat, longitude=lon, direction=direction,
            video_file=vid.name, start_time=entry.get("start_time", ""),
            location_source=src)
        CAMERA_ORDER.append(cam_id)

    _build_edges()
    _assign_start_times(base, stagger_minutes)
    return CAMERAS


def _build_edges() -> None:
    """Consecutive edges along the corridor plus skip-edges (missed cameras)."""
    CAMERA_EDGES.clear()
    EDGE_INDEX.clear()
    order = CAMERA_ORDER
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = CAMERAS[order[i]], CAMERAS[order[j]]
            gc = haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
            road = round(gc * ROAD_FACTOR, 2)
            if road <= 0:
                road = 0.01
            seg = f"{a.name} -> {b.name}"
            if j > i + 1:
                seg += f" (skips {j - i - 1} camera(s))"
            e = CameraEdge(order[i], order[j], road,
                           round(road / FREEFLOW_KMH * 60, 1),
                           round(road / CONGESTED_KMH * 60, 1), seg)
            CAMERA_EDGES.append(e)
            EDGE_INDEX[(e.source, e.destination)] = e


def _assign_start_times(base: datetime, stagger_minutes: float | None) -> None:
    """Configured capture start per camera (videos carry no wall clock)."""
    t = base
    for idx, cam_id in enumerate(CAMERA_ORDER):
        cam = CAMERAS[cam_id]
        if cam.start_time:
            try:
                t = datetime.fromisoformat(cam.start_time)
            except ValueError:
                pass
        elif idx > 0:
            if stagger_minutes is not None:
                t = t + timedelta(minutes=stagger_minutes)
            else:
                prev = CAMERA_ORDER[idx - 1]
                e = EDGE_INDEX.get((prev, cam_id))
                gap = ((e.expected_min_minutes + e.expected_max_minutes) / 2) if e else 10.0
                t = t + timedelta(minutes=round(gap, 1))
        CAMERAS[cam_id] = Camera(**{**cam.to_dict(), "start_time": t.isoformat()})


def get_edge(src: str, dst: str):
    if (src, dst) in EDGE_INDEX:
        return EDGE_INDEX[(src, dst)]
    rev = EDGE_INDEX.get((dst, src))
    if rev is not None:
        return CameraEdge(src, dst, rev.road_distance_km, rev.expected_min_minutes,
                          rev.expected_max_minutes, rev.road_segment + " (reverse)")
    return None


def camera_start(camera_id: str) -> datetime:
    return datetime.fromisoformat(CAMERAS[camera_id].start_time)


def network_summary() -> dict:
    return {"camera_count": len(CAMERAS),
            "cameras": [c.to_dict() for c in CAMERAS.values()],
            "edges": [e.to_dict() for e in CAMERA_EDGES]}


def write_config_template(path: Path | None = None) -> Path:
    """Emit a cameras.json reflecting the current network, for you to edit."""
    path = Path(path) if path else CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": ("Map each video to its real location. Delete entries to fall "
                     "back to DEFAULT_LOCATIONS. start_time is the configured "
                     "wall-clock capture start; leave blank to auto-stagger."),
        "cameras": [{"camera_id": c.camera_id, "video_file": c.video_file,
                     "name": c.name, "road": c.road,
                     "latitude": c.latitude, "longitude": c.longitude,
                     "direction": c.direction, "start_time": c.start_time}
                    for c in CAMERAS.values()],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


# Build on import so `from .cameras import CAMERAS` is populated when videos exist.
load_network()


if __name__ == "__main__":
    print(f"videos found: {len(discover_videos())}")
    print(f"cameras     : {len(CAMERAS)}")
    for c in CAMERAS.values():
        print(f"  {c.camera_id}  {c.name:24} {c.latitude:8.4f},{c.longitude:8.4f}  "
              f"[{c.location_source}]  start={c.start_time}  <- {c.video_file}")
    print(f"edges       : {len(CAMERA_EDGES)}")
    for e in CAMERA_EDGES:
        print(f"  {e.source}->{e.destination}  {e.road_distance_km:6.2f} km  "
              f"{e.expected_min_minutes:5.1f}-{e.expected_max_minutes:5.1f} min  {e.road_segment}")
