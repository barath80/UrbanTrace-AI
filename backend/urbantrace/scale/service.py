"""The city-scale ingest service.

Wires the pieces into one runnable system and owns the sighting lifecycle:

    frame -> FrameBus -> InferencePool -> Sighting
          -> IdentityIndex (which vehicle is this?)
          -> legs (was that journey physically possible?)
          -> AlertEngine (does anyone care?)
          -> SightingSink (batched write to Postgres)

Everything above the sink is in-memory and bounded; everything the dashboard
reads comes back out of Postgres. That split is what lets the ingest process be
restarted, or run as several sharded processes, without losing history.

Sharding: `shard_of()` assigns each camera to a shard by a stable hash of its
id. Run N processes with `shard_index` 0..N-1 and each handles a disjoint slice
of the estate while writing to the same database. Nothing else has to change,
because identity resolution and analytics both happen against shared storage.
"""
from __future__ import annotations

import logging
import threading
import time
import zlib
from datetime import datetime, timedelta, timezone

from .alerts import AlertEngine, Watchlist
from .config import SCALE
from .identity import IdentityIndex, Sighting
from .inference import FrameBus, FrameItem, InferencePool, SharedModels
from .registry import REGISTRY, CameraRegistry
from .sink import SightingSink, VehicleUpsert
from .tracking import TRACKERS

logger = logging.getLogger("urbantrace.scale.service")


def shard_of(camera_id: str, shards: int) -> int:
    """Stable shard assignment for a camera.

    crc32 rather than Python's hash(): hash() is randomised per process, so the
    same camera would land on a different shard after every restart.
    """
    if shards <= 1:
        return 0
    return zlib.crc32(camera_id.encode("utf-8")) % shards


class IngestService:
    """One ingest process. Owns a slice of the camera estate."""

    def __init__(self, registry: CameraRegistry | None = None,
                 shard_index: int = 0, shards: int = 1,
                 persist: bool | None = None):
        self.registry = registry or REGISTRY
        self.shard_index = shard_index
        self.shards = shards

        self.bus = FrameBus()
        self.models = SharedModels()
        self.identity = IdentityIndex(feasible=self._feasible)
        self.sink = SightingSink(enabled=SCALE.sink_enabled if persist is None else persist)
        self.vehicles = VehicleUpsert()
        self.alerts = AlertEngine(Watchlist())
        self.pool = InferencePool(models=self.models, bus=self.bus,
                                  on_sighting=self._on_sighting)

        self._lock = threading.Lock()
        self._dirty: set = set()          # clusters needing a vehicle upsert
        self._legs: dict = {}             # cluster_id -> legs
        self._maintenance: threading.Thread | None = None
        self._stop = threading.Event()
        self.started_at: float | None = None
        self.sightings_seen = 0
        self.rejected_other_shard = 0

    # ------------------------------------------------------- lifecycle

    def start(self, workers: int | None = None) -> None:
        self.registry.build_graph()
        self.sink.start()
        self.pool.start(workers=workers)
        self.alerts.watchlist.load(force=True)
        self._stop.clear()
        self._maintenance = threading.Thread(target=self._maintain, name="scale-maint",
                                             daemon=True)
        self._maintenance.start()
        self.started_at = time.time()
        logger.info("ingest service started: shard %d/%d, %d camera(s)",
                    self.shard_index, self.shards, len(self.registry))

    def stop(self) -> None:
        self._stop.set()
        self.pool.stop()
        self.flush()
        self.sink.stop()
        if self._maintenance:
            self._maintenance.join(timeout=5.0)
            self._maintenance = None

    # ---------------------------------------------------------- ingest

    def owns(self, camera_id: str) -> bool:
        return shard_of(camera_id, self.shards) == self.shard_index

    def submit_frame(self, camera_id: str, frame, captured_at: datetime | None = None,
                     video_time_s: float = 0.0, frame_number: int = 0) -> bool:
        """Offer a frame for analysis.

        Returns False when the frame was not accepted -- wrong shard, unknown
        camera, or rate-limited. A caller that wants to know why should consult
        `stats()`; the counters distinguish the cases.
        """
        if not self.owns(camera_id):
            self.rejected_other_shard += 1
            return False
        if camera_id not in self.registry:
            return False
        if not self.bus.admit(camera_id):
            return False
        return self.bus.submit(FrameItem(
            camera_id=camera_id, frame=frame,
            captured_at=captured_at or datetime.now(timezone.utc),
            video_time_s=video_time_s, frame_number=frame_number))

    # ------------------------------------------------- sighting handling

    def _feasible(self, a: Sighting, b: Sighting) -> bool:
        """Could one vehicle have been at `a` and then `b`?"""
        minutes = (b.observed_at - a.observed_at).total_seconds() / 60.0
        if minutes < 0:
            return False
        if minutes == 0:
            return a.camera_id == b.camera_id
        edge = self.registry.edge(a.camera_id, b.camera_id)
        if edge is None:
            return True          # unknown geometry: do not veto on no evidence
        return (edge.road_distance_km / (minutes / 60.0)) <= SCALE.max_plausible_speed_kmh

    def _on_sighting(self, s: Sighting) -> None:
        self.sightings_seen += 1
        cluster_id = self.identity.add(s)
        cluster = self.identity.get(cluster_id)

        legs = self._legs_for(cluster) if cluster else []
        with self._lock:
            self._dirty.add(cluster_id)
            if legs:
                self._legs[cluster_id] = legs

        self.alerts.evaluate(s, cluster, legs[-1:] if legs else [])
        self.sink.submit(s, cluster_id=cluster_id)

    def _legs_for(self, cluster) -> list:
        """Measured hops between the cluster's first arrival at each camera."""
        first: dict = {}
        for s in cluster.sightings:
            first.setdefault(s.camera_id, s)
        visits = sorted(first.values(), key=lambda s: s.observed_at)
        out = []
        for a, b in zip(visits, visits[1:]):
            out.append(self._leg(a, b))
        return out

    def _leg(self, a: Sighting, b: Sighting) -> dict:
        minutes = (b.observed_at - a.observed_at).total_seconds() / 60.0
        edge = self.registry.edge(a.camera_id, b.camera_id)
        km = edge.road_distance_km if edge else 0.0
        source = edge.distance_source if edge else "unknown"
        lo = edge.expected_min_minutes if edge else None
        hi = edge.expected_max_minutes if edge else None
        speed = (km / (minutes / 60.0)) if minutes > 0 and km else None

        if minutes <= 0:
            verdict, reason = "IMPLAUSIBLE", (
                f"Arrival at {b.camera_id} is not after departure from {a.camera_id}.")
        elif speed is not None and speed > SCALE.max_plausible_speed_kmh:
            verdict, reason = "IMPLAUSIBLE", (
                f"{km:.2f} km in {minutes:.2f} min implies {speed:.0f} km/h, above the "
                f"{SCALE.max_plausible_speed_kmh:.0f} km/h feasibility limit.")
        elif speed is not None and speed > SCALE.warn_speed_kmh:
            verdict, reason = "FAST", (
                f"{km:.2f} km in {minutes:.2f} min implies {speed:.0f} km/h.")
        elif hi is not None and minutes > hi * 1.5:
            verdict, reason = "SLOW", (
                f"Took {minutes:.1f} min against an expected {lo:.1f}-{hi:.1f} min.")
        else:
            verdict, reason = "PLAUSIBLE", (
                f"{km:.2f} km in {minutes:.1f} min"
                + (f" ({speed:.0f} km/h)." if speed else "."))

        return {"from_camera": a.camera_id, "to_camera": b.camera_id,
                "departed": a.observed_at, "arrived": b.observed_at,
                "travel_minutes": round(minutes, 2), "road_distance_km": km,
                "implied_speed_kmh": round(speed, 1) if speed else None,
                "distance_source": source, "verdict": verdict, "reason": reason}

    # ------------------------------------------------------ maintenance

    def _maintain(self) -> None:
        """Periodic upkeep: persist identities, expire state, drain alerts."""
        while not self._stop.wait(SCALE.sink_flush_seconds):
            try:
                self.flush()
            except Exception:
                logger.exception("maintenance flush failed")

    def flush(self) -> dict:
        with self._lock:
            dirty = list(self._dirty)
            self._dirty.clear()
            legs = {c: self._legs.get(c, []) for c in dirty}

        saved = 0
        if dirty:
            clusters = [c for c in (self.identity.get(cid) for cid in dirty) if c]
            if clusters:
                saved = self.vehicles.save(clusters, legs)
        alerts = self.alerts.flush()
        written = self.sink.flush()
        return {"vehicles_saved": saved, "alerts_written": alerts,
                "sightings_written": written}

    def housekeeping(self) -> dict:
        """Bound memory. Safe to call on a timer; cheap when nothing is stale."""
        evicted_identity = self.identity.evict_before(
            datetime.now(timezone.utc) - timedelta(minutes=SCALE.identity_ttl_minutes))
        evicted_tracks = self.pool.evict_stale_tracks()
        evicted_cams = TRACKERS.evict_idle()
        with self._lock:
            for cid in list(self._legs):
                if self.identity.get(cid) is None:
                    del self._legs[cid]
        return {"identity_clusters_evicted": evicted_identity,
                "track_buffers_evicted": evicted_tracks,
                "camera_trackers_evicted": evicted_cams}

    # ---------------------------------------------------------- status

    def stats(self) -> dict:
        uptime = time.time() - self.started_at if self.started_at else 0.0
        return {
            "shard": {"index": self.shard_index, "count": self.shards,
                      "cameras_owned": sum(1 for c in self.registry.ids() if self.owns(c)),
                      "rejected_other_shard": self.rejected_other_shard},
            "uptime_s": round(uptime, 1),
            "registry": self.registry.summary(),
            "pipeline": self.pool.stats(),
            "identity": self.identity.stats(),
            "sink": self.sink.stats(),
            "vehicles": self.vehicles.stats(),
            "alerts": self.alerts.stats(),
            "sightings_seen": self.sightings_seen,
        }


SERVICE = IngestService()


def load_cameras_from_db(registry: CameraRegistry | None = None) -> int:
    """Populate the in-memory registry from `net_cameras`."""
    from ..db import repository as repo
    reg = registry or REGISTRY
    rows = repo._fetch("""select camera_id, name, road, sector, latitude, longitude,
                                 direction, source, enabled
                          from net_cameras order by camera_id""")
    reg.clear()
    n = reg.load(rows)
    reg.build_graph()
    return n


def save_cameras_to_db(registry: CameraRegistry | None = None) -> int:
    """Persist the in-memory registry and its edges."""
    from psycopg2.extras import execute_values

    from ..db.client import connect
    reg = registry or REGISTRY
    nodes = [(c.camera_id, SCALE.site_id, c.name, c.road, c.sector,
              c.latitude, c.longitude, c.direction, c.source, c.enabled)
             for c in reg]
    if not nodes:
        return 0
    edges = [(e.source, e.destination, e.road_distance_km,
              e.expected_min_minutes, e.expected_max_minutes, e.distance_source)
             for e in reg.edges()]
    with connect() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                insert into net_cameras
                  (camera_id, site_id, name, road, sector, latitude, longitude,
                   direction, source, enabled)
                values %s
                on conflict (camera_id) do update set
                  site_id=excluded.site_id, name=excluded.name, road=excluded.road,
                  sector=excluded.sector, latitude=excluded.latitude,
                  longitude=excluded.longitude, direction=excluded.direction,
                  source=excluded.source, enabled=excluded.enabled,
                  updated_at=now()
            """, nodes)
            if edges:
                execute_values(cur, """
                    insert into net_edges
                      (source_camera, destination_camera, road_distance_km,
                       expected_min_minutes, expected_max_minutes, distance_source)
                    values %s
                    on conflict (source_camera, destination_camera) do update set
                      road_distance_km=excluded.road_distance_km,
                      expected_min_minutes=excluded.expected_min_minutes,
                      expected_max_minutes=excluded.expected_max_minutes,
                      distance_source=excluded.distance_source
                """, edges)
        conn.commit()
    return len(nodes)
