"""Real road routes between cameras, via the GraphHopper Routing API.

Until now a "route" between two cameras was a straight line on the map and a
great-circle distance multiplied by a 1.25 fudge factor. That is a guess: it
ignores which roads actually connect the two points, and the travel-time window
derived from it drives the PLAUSIBLE / FAST / SLOW verdicts. On the Koyambedu ->
Ambattur leg the guess is 8.26 km; the road is 9.18 km.

This module fetches the real driven path - the road geometry, its length, and
GraphHopper's own duration estimate for it.

THREE THINGS THIS MODULE IS CAREFUL ABOUT

  1. The API key never reaches the browser. It is read from `.env` and used
     server-side only; the frontend asks the FastAPI app for geometry, which
     is why `/api/road-route` exists.
  2. Every result is cached on disk under `data/cache/routes/`, keyed by the
     coordinates and profile. Camera positions do not move, so a corridor is
     fetched once and never again - free-tier quota is not spent on redraws.
  3. A route always comes back, even with no key, no network, or a quota
     error. The fallback is the old great-circle estimate, and `source` says
     which you got, so an estimate is never presented as a measurement.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = ROOT / ".env"
CACHE_DIR = ROOT / "data" / "cache" / "routes"

API_URL = "https://graphhopper.com/api/1/route"
TIMEOUT_S = 30

# Road distance as a multiple of great-circle, used only when the real road is
# unavailable. An urban corridor is rarely straight.
FALLBACK_ROAD_FACTOR = 1.25
FALLBACK_KMH = 32.0


def _load_key() -> str:
    """GRAPHHOPPER_API_KEY from the environment, else from .env."""
    key = os.environ.get("GRAPHHOPPER_API_KEY", "").strip()
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("GRAPHHOPPER_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


@dataclass
class RoadRoute:
    """A path along real roads, or an honest admission that it is a guess."""
    distance_km: float
    duration_min: float
    # [[lat, lon], ...] along the road. For a fallback this is just the two
    # endpoints, so a caller that draws it gets a straight line and the
    # `source` field tells the UI to say so.
    coordinates: list = field(default_factory=list)
    source: str = "graphhopper"     # graphhopper | cache | fallback_estimate
    profile: str = "car"
    error: str | None = None        # why the fallback was used, if it was

    @property
    def is_real_road(self) -> bool:
        return self.source in ("graphhopper", "cache")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_real_road"] = self.is_real_road
        return d


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _fallback(points: list, reason: str) -> RoadRoute:
    total = sum(haversine_km(*points[i], *points[i + 1])
                for i in range(len(points) - 1)) * FALLBACK_ROAD_FACTOR
    return RoadRoute(
        distance_km=round(total, 3),
        duration_min=round(total / FALLBACK_KMH * 60, 1),
        coordinates=[[float(a), float(b)] for a, b in points],
        source="fallback_estimate", error=reason)


def _cache_path(points: list, profile: str) -> Path:
    key = json.dumps({"p": [[round(a, 6), round(b, 6)] for a, b in points],
                      "v": profile}, sort_keys=True)
    return CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.json"


def road_route_multi(points: list, profile: str = "car",
                     use_cache: bool = True) -> RoadRoute:
    """Real road route through `points` = [(lat, lon), ...] (2 or more).

    Never raises: on any failure it returns a great-circle estimate whose
    `source` is "fallback_estimate" and whose `error` says what went wrong.
    """
    points = [(float(a), float(b)) for a, b in points]
    if len(points) < 2:
        return _fallback(points * 2 if points else [(0, 0), (0, 0)],
                         "need at least two points")

    cache_file = _cache_path(points, profile)
    if use_cache and cache_file.exists():
        try:
            d = json.loads(cache_file.read_text())
            d["source"] = "cache"
            return RoadRoute(**d)
        except (json.JSONDecodeError, TypeError):
            cache_file.unlink(missing_ok=True)   # corrupt entry; refetch

    key = _load_key()
    if not key:
        return _fallback(points, "GRAPHHOPPER_API_KEY not set in .env")

    q = [("profile", profile), ("points_encoded", "false"),
         ("instructions", "false"), ("calc_points", "true"), ("key", key)]
    q = [("point", f"{lat},{lon}") for lat, lon in points] + q
    url = f"{API_URL}?{urllib.parse.urlencode(q)}"

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.load(e).get("message", "")
        except Exception:
            pass
        return _fallback(points, f"HTTP {e.code} from GraphHopper: {detail}"[:200])
    except Exception as e:                     # network down, DNS, timeout
        return _fallback(points, f"{type(e).__name__}: {e}"[:200])

    paths = payload.get("paths") or []
    if not paths:
        return _fallback(points, f"GraphHopper returned no path: "
                                 f"{payload.get('message', '')}"[:200])

    p = paths[0]
    # GraphHopper GeoJSON is [lon, lat]; Leaflet wants [lat, lon].
    coords = [[float(c[1]), float(c[0])]
              for c in p.get("points", {}).get("coordinates", [])]
    route = RoadRoute(
        distance_km=round(p["distance"] / 1000.0, 3),
        duration_min=round(p["time"] / 60000.0, 1),
        coordinates=coords, source="graphhopper", profile=profile)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    d = asdict(route)
    d.pop("source", None)                      # re-read as "cache" next time
    cache_file.write_text(json.dumps(d))
    return route


def road_route(lat1, lon1, lat2, lon2, profile: str = "car",
               use_cache: bool = True) -> RoadRoute:
    """Real road route between two points."""
    return road_route_multi([(lat1, lon1), (lat2, lon2)], profile, use_cache)


def clear_cache() -> int:
    """Delete cached routes. Returns how many were removed."""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)


def describe() -> dict:
    """Routing availability, for the run record and the dashboard."""
    return {
        "provider": "graphhopper",
        "api_key_configured": bool(_load_key()),
        "cached_routes": len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0,
        "fallback": "great-circle x 1.25 when unavailable",
    }
