"""Batched persistence for the live sighting stream.

At a thousand cameras the platform produces tens of millions of sightings a
day. Writing them one INSERT at a time is the single most common way an ANPR
backend falls over: each statement pays a network round trip and a transaction
commit, so the database becomes the bottleneck long before the GPU does.

This module buffers sightings and writes them in batches with `execute_values`,
which sends one statement per batch instead of one per row. A background thread
flushes on either a size or a time trigger, so a quiet network still persists
promptly and a busy one still writes efficiently.

Failure behaviour is deliberate. If the database is unreachable the buffer is
retained up to a bound and the failure is COUNTED and surfaced. It is never
silently discarded, and it is never allowed to grow until the process dies:
past the bound the oldest sightings are dropped and the drop is reported.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import date, datetime, timezone

from .config import SCALE
from .identity import Sighting, canonical_key

logger = logging.getLogger("urbantrace.scale.sink")

SIGHTING_COLUMNS = (
    "site_id", "camera_id", "plate", "plate_canonical", "cluster_id",
    "observed_at", "confidence", "support", "total_readings",
    "is_valid_format", "vehicle_type", "track_id", "latitude", "longitude",
)


class SightingSink:
    """Buffers sightings and writes them to Postgres in batches."""

    def __init__(self, batch_size: int | None = None,
                 flush_seconds: float | None = None,
                 max_pending: int | None = None,
                 enabled: bool | None = None):
        self.batch_size = batch_size or SCALE.sink_batch_size
        self.flush_seconds = flush_seconds or SCALE.sink_flush_seconds
        self.max_pending = max_pending or SCALE.sink_max_pending
        self.enabled = SCALE.sink_enabled if enabled is None else enabled

        self._buf: deque = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._partitions: set = set()

        self.written = 0
        self.dropped = 0
        self.batches = 0
        self.failures = 0
        self.last_error: str | None = None
        self.last_write_at: float | None = None

    # ------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread or not self.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sighting-sink", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        self.flush()          # final drain so a clean shutdown loses nothing

    # ---------------------------------------------------------- submit

    def submit(self, sighting: Sighting, cluster_id: str | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            if len(self._buf) >= self.max_pending:
                # Shed the OLDEST. Newer sightings describe where vehicles are
                # now, which is what the platform is for.
                self._buf.popleft()
                self.dropped += 1
            self._buf.append((sighting, cluster_id))
            if len(self._buf) >= self.batch_size:
                self._wake.set()

    # ----------------------------------------------------------- write

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self.flush_seconds)
            self._wake.clear()
            try:
                self.flush()
            except Exception:
                logger.exception("sink flush failed")

    def flush(self) -> int:
        """Write everything currently buffered. Returns rows written."""
        with self._lock:
            if not self._buf:
                return 0
            pending = list(self._buf)
            self._buf.clear()

        try:
            n = self._write(pending)
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning("sink write failed (%s); re-buffering %d sighting(s)",
                           self.last_error, len(pending))
            # Put them back at the FRONT so ordering is preserved, subject to
            # the same bound as normal submission.
            with self._lock:
                self._buf.extendleft(reversed(pending))
                while len(self._buf) > self.max_pending:
                    self._buf.popleft()
                    self.dropped += 1
            return 0

        self.written += n
        self.batches += 1
        self.last_write_at = time.time()
        return n

    def _write(self, pending: list) -> int:
        from psycopg2.extras import execute_values
        from ..db.client import pooled

        rows = []
        days: set[date] = set()
        for s, cluster_id in pending:
            observed = s.observed_at
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            days.add(observed.date())
            rows.append((
                s.site_id or SCALE.site_id, s.camera_id, s.plate,
                canonical_key(s.plate), cluster_id, observed,
                float(s.confidence), int(s.support or 0), int(s.total_readings or 0),
                bool(s.is_valid_format), s.vehicle_type, int(s.track_id),
                float(s.latitude), float(s.longitude),
            ))

        cols = ", ".join(SIGHTING_COLUMNS)
        with pooled() as conn:
            with conn.cursor() as cur:
                # Make sure a bounded daily partition exists before writing, so
                # rows land in a droppable partition rather than piling into
                # the catch-all default.
                new_days = [d for d in sorted(days) if d not in self._partitions]
                for d in new_days:
                    cur.execute("select ensure_sighting_partition(%s)", (d,))
                execute_values(
                    cur,
                    f"insert into net_sightings ({cols}) values %s",
                    rows, page_size=max(50, self.batch_size))
        # Cached only after the commit, so a rolled-back partition is retried.
        self._partitions.update(new_days)
        return len(rows)

    # ---------------------------------------------------------- status

    def stats(self) -> dict:
        with self._lock:
            pending = len(self._buf)
        return {
            "enabled": self.enabled,
            "pending": pending,
            "written": self.written,
            "batches": self.batches,
            "mean_batch": round(self.written / self.batches, 1) if self.batches else 0.0,
            "dropped": self.dropped,
            "failures": self.failures,
            "last_error": self.last_error,
            "seconds_since_write": round(time.time() - self.last_write_at, 1)
            if self.last_write_at else None,
        }


class VehicleUpsert:
    """Keeps `net_vehicles` and `net_legs` in step with resolved identities.

    Vehicle rows are rewritten rather than appended, because a cluster's plate,
    camera count and last-seen time all change as it accumulates sightings. The
    write is an upsert on cluster_id so repeated calls converge instead of
    duplicating.
    """

    def __init__(self):
        self.upserts = 0
        self.legs_written = 0
        self.failures = 0
        self.last_error: str | None = None

    def save(self, clusters: list, legs_by_cluster: dict | None = None) -> int:
        if not clusters:
            return 0
        import json

        from psycopg2.extras import execute_values
        from ..db.client import pooled

        rows = []
        for c in clusters:
            cams = c.cameras()
            rows.append((
                c.cluster_id, SCALE.site_id, c.plate, json.dumps(c.spellings),
                c.first_seen, c.last_seen, len(cams), len(c.sightings),
                round(c.mean_confidence(), 4),
                c.sightings[-1].vehicle_type if c.sightings else None,
            ))

        leg_rows = []
        for cid, legs in (legs_by_cluster or {}).items():
            for i, l in enumerate(legs):
                leg_rows.append((
                    cid, i, l["from_camera"], l["to_camera"], l["departed"], l["arrived"],
                    l["travel_minutes"], l["road_distance_km"], l.get("implied_speed_kmh"),
                    l.get("distance_source"), l["verdict"], l.get("reason"),
                ))

        try:
            with pooled() as conn:
                with conn.cursor() as cur:
                    execute_values(cur, """
                        insert into net_vehicles
                          (cluster_id, site_id, plate, spellings, first_seen, last_seen,
                           camera_count, sighting_count, confidence, vehicle_type)
                        values %s
                        on conflict (cluster_id) do update set
                          plate = excluded.plate,
                          spellings = excluded.spellings,
                          first_seen = least(net_vehicles.first_seen, excluded.first_seen),
                          last_seen = greatest(net_vehicles.last_seen, excluded.last_seen),
                          camera_count = excluded.camera_count,
                          sighting_count = excluded.sighting_count,
                          confidence = excluded.confidence,
                          vehicle_type = excluded.vehicle_type,
                          updated_at = now()
                    """, rows)
                    if leg_rows:
                        execute_values(cur, """
                            insert into net_legs
                              (cluster_id, seq, from_camera, to_camera, departed, arrived,
                               travel_minutes, road_distance_km, implied_speed_kmh,
                               distance_source, verdict, reason)
                            values %s
                            on conflict (cluster_id, seq) do update set
                              from_camera = excluded.from_camera,
                              to_camera = excluded.to_camera,
                              departed = excluded.departed,
                              arrived = excluded.arrived,
                              travel_minutes = excluded.travel_minutes,
                              road_distance_km = excluded.road_distance_km,
                              implied_speed_kmh = excluded.implied_speed_kmh,
                              distance_source = excluded.distance_source,
                              verdict = excluded.verdict,
                              reason = excluded.reason
                        """, leg_rows)
                        self.legs_written += len(leg_rows)
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning("vehicle upsert failed: %s", self.last_error)
            return 0

        self.upserts += len(rows)
        return len(rows)

    def stats(self) -> dict:
        return {"upserts": self.upserts, "legs_written": self.legs_written,
                "failures": self.failures, "last_error": self.last_error}
