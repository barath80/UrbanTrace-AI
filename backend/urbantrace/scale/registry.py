"""City-scale camera registry.

Holds the camera network for a whole city -- thousands of nodes -- and answers
the two questions the rest of the platform asks of it:

    "which cameras are near this point?"        -> spatial lookup
    "could a vehicle have driven A to B?"       -> graph edge with a travel window

Two things here are scale decisions rather than style:

1. The spatial index is a uniform lat/lon grid. Lookups touch only the cells
   that intersect the query radius, so finding a camera's neighbours costs the
   same at 10,000 cameras as at 10. A linear scan would make every ingest
   sighting an O(n) operation.

2. Edges connect each camera to its k nearest neighbours within a radius,
   NOT every pair. All-pairs on 1000 cameras is 499,500 edges, almost all of
   them between cameras on opposite sides of the city, which describe no road a
   vehicle would take between two consecutive sightings. Neighbour edges keep
   the graph linear in camera count.

Cameras are loaded from a JSON file or from any iterable of dicts, so a
deployment can source them from its own asset database without changing code.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator

from .config import SCALE

EARTH_R_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(min(1.0, a)))


@dataclass(frozen=True)
class CameraNode:
    camera_id: str
    name: str
    latitude: float
    longitude: float
    road: str = ""
    direction: str = "UNKNOWN"
    sector: str = ""
    source: str = ""            # rtsp:// URL, device index, file, or "" for push-only
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    """A plausible road link between two cameras.

    `road_distance_km` is a great-circle distance scaled by a road factor
    unless a router has supplied a real one; `distance_source` always says
    which, because every travel-time verdict rests on this number.
    """
    source: str
    destination: str
    road_distance_km: float
    expected_min_minutes: float
    expected_max_minutes: float
    distance_source: str = "estimate"

    def to_dict(self) -> dict:
        return asdict(self)


class SpatialGrid:
    """Uniform lat/lon bucket index.

    Cell size is expressed in kilometres and converted to degrees at the
    equator for longitude, which over-covers at higher latitudes. Over-covering
    is safe: it returns a superset of candidates and the caller filters by true
    distance. Under-covering would silently lose cameras.
    """

    def __init__(self, cell_km: float = 5.0):
        self.cell_km = max(0.1, cell_km)
        self._deg = self.cell_km / 111.32
        self._cells: dict[tuple[int, int], list[str]] = {}

    def _key(self, lat: float, lon: float) -> tuple[int, int]:
        return (int(math.floor(lat / self._deg)), int(math.floor(lon / self._deg)))

    def add(self, camera_id: str, lat: float, lon: float) -> None:
        self._cells.setdefault(self._key(lat, lon), []).append(camera_id)

    def clear(self) -> None:
        self._cells.clear()

    def candidates(self, lat: float, lon: float, radius_km: float) -> list[str]:
        """Camera ids in every cell touching the radius. A superset."""
        span = max(1, int(math.ceil(radius_km / self.cell_km)))
        r, c = self._key(lat, lon)
        out: list[str] = []
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                out.extend(self._cells.get((r + dr, c + dc), ()))
        return out


class CameraRegistry:
    """The camera network. Safe to hold thousands of nodes."""

    def __init__(self, cell_km: float = 5.0):
        self._nodes: dict[str, CameraNode] = {}
        self._grid = SpatialGrid(cell_km)
        self._edges: dict[tuple[str, str], GraphEdge] = {}
        self._neighbours: dict[str, list[str]] = {}
        self._graph_built = False

    # ---------------------------------------------------------- loading

    def add(self, node: CameraNode) -> None:
        self._nodes[node.camera_id] = node
        self._grid.add(node.camera_id, node.latitude, node.longitude)
        self._graph_built = False

    def load(self, entries: Iterable[dict]) -> int:
        """Load camera definitions. Returns the number accepted.

        A malformed entry raises rather than being skipped: a camera silently
        missing from a city network is a far worse failure than a loud one at
        startup.
        """
        n = 0
        for e in entries:
            try:
                cam = CameraNode(
                    camera_id=str(e["camera_id"]),
                    name=str(e.get("name") or e["camera_id"]),
                    latitude=float(e["latitude"]),
                    longitude=float(e["longitude"]),
                    road=str(e.get("road") or ""),
                    direction=str(e.get("direction") or "UNKNOWN"),
                    sector=str(e.get("sector") or ""),
                    source=str(e.get("source") or ""),
                    enabled=bool(e.get("enabled", True)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid camera entry {e!r}: {exc}") from exc
            if not (-90.0 <= cam.latitude <= 90.0 and -180.0 <= cam.longitude <= 180.0):
                raise ValueError(f"Camera {cam.camera_id} has out-of-range coordinates")
            self.add(cam)
            n += 1
        return n

    def load_file(self, path: str | Path) -> int:
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        entries = raw.get("cameras", raw) if isinstance(raw, dict) else raw
        if isinstance(entries, dict):
            entries = [{"camera_id": k, **v} for k, v in entries.items()]
        return self.load(entries)

    def clear(self) -> None:
        self._nodes.clear()
        self._grid.clear()
        self._edges.clear()
        self._neighbours.clear()
        self._graph_built = False

    # ---------------------------------------------------------- access

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, camera_id: object) -> bool:
        return camera_id in self._nodes

    def __iter__(self) -> Iterator[CameraNode]:
        return iter(self._nodes.values())

    def get(self, camera_id: str) -> CameraNode | None:
        return self._nodes.get(camera_id)

    def ids(self) -> list[str]:
        return list(self._nodes)

    def enabled_ids(self) -> list[str]:
        return [c.camera_id for c in self._nodes.values() if c.enabled]

    # --------------------------------------------------------- spatial

    def near(self, lat: float, lon: float, radius_km: float, limit: int = 0) -> list[tuple[str, float]]:
        """Cameras within `radius_km`, nearest first, as (camera_id, km).

        Candidates are filtered by a cheap bounding box before any haversine is
        computed, then sorted once. Inserting into a sorted list instead makes
        this quadratic in the candidate count, which is what dominates graph
        construction on a dense network.
        """
        # Degrees of latitude are constant; degrees of longitude shrink with
        # latitude, so the longitude window is widened by 1/cos(lat).
        dlat = radius_km / 110.574
        coslat = max(0.01, math.cos(math.radians(lat)))
        dlon = radius_km / (111.320 * coslat)
        lat_lo, lat_hi = lat - dlat, lat + dlat
        lon_lo, lon_hi = lon - dlon, lon + dlon

        out: list[tuple[float, str]] = []
        for cid in self._grid.candidates(lat, lon, radius_km):
            n = self._nodes[cid]
            if not (lat_lo <= n.latitude <= lat_hi and lon_lo <= n.longitude <= lon_hi):
                continue
            d = haversine_km(lat, lon, n.latitude, n.longitude)
            if d <= radius_km:
                out.append((d, cid))
        out.sort()
        pairs = [(cid, round(d, 4)) for d, cid in out]
        return pairs[:limit] if limit else pairs

    # ----------------------------------------------------------- graph

    def build_graph(self, k: int | None = None, radius_km: float | None = None) -> dict:
        """Connect every camera to its k nearest neighbours within a radius.

        Edges are stored in both directions with the same distance, so a
        lookup never depends on which camera was seen first.
        """
        k = SCALE.graph_neighbours if k is None else k
        radius_km = SCALE.graph_radius_km if radius_km is None else radius_km

        self._edges.clear()
        self._neighbours.clear()

        for node in self._nodes.values():
            # Expanding search. On a dense network the k nearest cameras are
            # almost always within a kilometre or two, so scanning the full
            # radius for every node wastes most of the work; on a sparse one
            # the radius still grows until neighbours are found.
            near: list = []
            r = min(radius_km, max(0.5, radius_km / 8.0))
            while True:
                near = self.near(node.latitude, node.longitude, r, limit=k + 1)
                if len(near) > k or r >= radius_km:
                    break
                r = min(radius_km, r * 2.0)
            neighbours = [cid for cid, _ in near if cid != node.camera_id][:k]
            self._neighbours[node.camera_id] = neighbours
            for cid, dist in near:
                if cid == node.camera_id or cid not in neighbours:
                    continue
                self._put_edge(node.camera_id, cid, dist)

        self._graph_built = True
        return {
            "cameras": len(self._nodes),
            "edges": len(self._edges),
            "neighbours_per_camera": k,
            "radius_km": radius_km,
        }

    def _put_edge(self, a: str, b: str, straight_km: float) -> None:
        km = max(0.01, round(straight_km * SCALE.road_factor, 3))
        edge = GraphEdge(
            source=a, destination=b, road_distance_km=km,
            expected_min_minutes=round(km / SCALE.freeflow_kmh * 60.0, 2),
            expected_max_minutes=round(km / SCALE.congested_kmh * 60.0, 2),
        )
        self._edges[(a, b)] = edge
        # Mirror, so B->A is answerable without a second pass.
        self._edges.setdefault((b, a), GraphEdge(
            source=b, destination=a, road_distance_km=km,
            expected_min_minutes=edge.expected_min_minutes,
            expected_max_minutes=edge.expected_max_minutes,
        ))

    def neighbours(self, camera_id: str) -> list[str]:
        if not self._graph_built:
            self.build_graph()
        return self._neighbours.get(camera_id, [])

    def edge(self, src: str, dst: str) -> GraphEdge | None:
        """The stored edge, or a distance-only edge for a non-adjacent pair.

        A vehicle can legitimately be seen at two cameras that are not graph
        neighbours, because intermediate cameras missed it. Rather than
        refusing to judge that transition, the distance is computed directly
        and the edge is marked so callers know it is not an adjacency.
        """
        if not self._graph_built:
            self.build_graph()
        e = self._edges.get((src, dst))
        if e is not None:
            return e
        a, b = self._nodes.get(src), self._nodes.get(dst)
        if a is None or b is None:
            return None
        km = max(0.01, round(haversine_km(a.latitude, a.longitude,
                                          b.latitude, b.longitude) * SCALE.road_factor, 3))
        return GraphEdge(
            source=src, destination=dst, road_distance_km=km,
            expected_min_minutes=round(km / SCALE.freeflow_kmh * 60.0, 2),
            expected_max_minutes=round(km / SCALE.congested_kmh * 60.0, 2),
            distance_source="estimate-nonadjacent",
        )

    def edges(self) -> list[GraphEdge]:
        if not self._graph_built:
            self.build_graph()
        # Deduplicate the mirrored pairs for reporting.
        seen: set[tuple[str, str]] = set()
        out = []
        for (a, b), e in self._edges.items():
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out

    def summary(self) -> dict:
        if not self._graph_built:
            self.build_graph()
        sectors: dict[str, int] = {}
        for n in self._nodes.values():
            sectors[n.sector or "unassigned"] = sectors.get(n.sector or "unassigned", 0) + 1
        return {
            "cameras": len(self._nodes),
            "enabled": len(self.enabled_ids()),
            "edges": len(self.edges()),
            "sectors": sectors,
        }


REGISTRY = CameraRegistry()
