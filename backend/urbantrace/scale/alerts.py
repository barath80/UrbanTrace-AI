"""Real-time alerting: blacklisted plates and route anomalies.

Alerts are raised on the ingest path, as each sighting is resolved, so a
watchlist hit surfaces within a frame or two of the plate being read rather
than on the next batch job.

Every alert carries a `reason` written for a human operator. An alert that
cannot explain itself cannot be acted on, and in this domain acting on one
means stopping a vehicle.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from .config import SCALE
from .identity import canonical_key

logger = logging.getLogger("urbantrace.scale.alerts")


class Watchlist:
    """Blacklisted plates, cached in memory and refreshed periodically.

    Every sighting is checked against this, so it cannot be a database query:
    at a thousand cameras that would be thousands of round trips a second. It
    is a set lookup on the OCR-folded key, refreshed on a timer.
    """

    def __init__(self, refresh_seconds: float = 60.0):
        self.refresh_seconds = refresh_seconds
        self._by_canonical: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._loaded_at = 0.0
        self.entries = 0

    def load(self, force: bool = False) -> int:
        if not force and time.time() - self._loaded_at < self.refresh_seconds:
            return self.entries
        try:
            from ..db import repository as repo
            rows = repo._fetch(
                "select plate, reason, severity from net_watchlist where active")
        except Exception as e:
            logger.warning("watchlist refresh failed: %s", e)
            self._loaded_at = time.time()      # do not hammer a failing database
            return self.entries

        table = {}
        for r in rows:
            table[canonical_key(r["plate"])] = {
                "plate": r["plate"],
                "reason": r.get("reason") or "Listed on the vehicle watchlist",
                "severity": r.get("severity") or "HIGH",
            }
        with self._lock:
            self._by_canonical = table
            self.entries = len(table)
            self._loaded_at = time.time()
        return self.entries

    def match(self, plate: str) -> dict | None:
        """Watchlist entry for a plate, tolerant of OCR confusion."""
        self.load()
        with self._lock:
            return self._by_canonical.get(canonical_key(plate))

    def add(self, plate: str, reason: str = "", severity: str = "HIGH") -> None:
        from ..db.client import connect
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    insert into net_watchlist (plate, reason, severity, active)
                    values (%s, %s, %s, true)
                    on conflict (plate) do update set
                      reason = excluded.reason,
                      severity = excluded.severity,
                      active = true
                """, (plate.upper().strip(), reason, severity))
            conn.commit()
        self.load(force=True)


class AlertEngine:
    """Evaluates each resolved sighting and writes any alerts it raises."""

    def __init__(self, watchlist: Watchlist | None = None):
        self.watchlist = watchlist or Watchlist()
        self._pending: list = []
        self._lock = threading.Lock()
        self.raised = 0
        self.written = 0
        self.failures = 0
        self.last_error: str | None = None
        self._seen_watchlist: dict[tuple, float] = {}
        # A watchlisted vehicle sitting in front of a camera must not raise an
        # alert every frame. One per plate per camera per this many seconds.
        self.dedupe_seconds = 120.0

    # ------------------------------------------------------- evaluation

    def evaluate(self, sighting, cluster=None, legs: list | None = None) -> list:
        """Return (and queue) alerts raised by this sighting."""
        out = []

        hit = self.watchlist.match(sighting.plate)
        if hit is not None and self._fresh(sighting):
            out.append({
                "alert_type": "WATCHLIST_MATCH",
                "severity": hit["severity"],
                "plate": sighting.plate,
                "cluster_id": getattr(cluster, "cluster_id", None),
                "camera_id": sighting.camera_id,
                "occurred_at": sighting.observed_at,
                "confidence": sighting.confidence,
                "reason": (f"{sighting.plate} matches watchlist entry {hit['plate']} "
                           f"({hit['reason']}). Seen at {sighting.camera_id} "
                           f"with OCR confidence {sighting.confidence:.2f}."),
                "evidence": {"matched_entry": hit["plate"],
                             "read_as": sighting.plate,
                             "support": sighting.support,
                             "total_readings": sighting.total_readings},
            })

        for leg in (legs or []):
            anomaly = self._route_anomaly(sighting, cluster, leg)
            if anomaly:
                out.append(anomaly)

        if out:
            with self._lock:
                self._pending.extend(out)
                self.raised += len(out)
        return out

    def _fresh(self, sighting) -> bool:
        key = (canonical_key(sighting.plate), sighting.camera_id)
        now = time.time()
        with self._lock:
            last = self._seen_watchlist.get(key, 0.0)
            if now - last < self.dedupe_seconds:
                return False
            self._seen_watchlist[key] = now
            if len(self._seen_watchlist) > 100_000:
                cutoff = now - self.dedupe_seconds
                self._seen_watchlist = {k: v for k, v in self._seen_watchlist.items()
                                        if v >= cutoff}
        return True

    def _route_anomaly(self, sighting, cluster, leg: dict) -> dict | None:
        """Flag a transition that is physically impossible or implausibly fast.

        IMPLAUSIBLE is the important one: it means the same plate was read in
        two places it could not have travelled between. That is either a
        cloned plate, or an OCR error that has merged two vehicles -- both
        worth an operator's attention, and the reason says which is suspected.
        """
        verdict = leg.get("verdict")
        if verdict not in ("IMPLAUSIBLE", "FAST"):
            return None
        speed = leg.get("implied_speed_kmh")
        if verdict == "IMPLAUSIBLE":
            reason = (f"{sighting.plate} was read at {leg['from_camera']} and "
                      f"{leg['to_camera']} {leg['travel_minutes']:.1f} min apart, "
                      f"{leg['road_distance_km']:.2f} km apart"
                      + (f", implying {speed:.0f} km/h. " if speed else ". ")
                      + "That journey is not physically possible: suspect a cloned "
                        "plate or a misread.")
            severity = "HIGH"
        else:
            reason = (f"{sighting.plate} covered {leg['road_distance_km']:.2f} km "
                      f"between {leg['from_camera']} and {leg['to_camera']} in "
                      f"{leg['travel_minutes']:.1f} min"
                      + (f" ({speed:.0f} km/h)" if speed else "")
                      + ", above the expected speed for this corridor.")
            severity = "MEDIUM"
        return {
            "alert_type": "ROUTE_ANOMALY",
            "severity": severity,
            "plate": sighting.plate,
            "cluster_id": getattr(cluster, "cluster_id", None),
            "camera_id": sighting.camera_id,
            "occurred_at": sighting.observed_at,
            "confidence": sighting.confidence,
            "reason": reason,
            "evidence": {k: leg.get(k) for k in
                         ("from_camera", "to_camera", "travel_minutes",
                          "road_distance_km", "implied_speed_kmh",
                          "distance_source", "verdict")},
        }

    # ---------------------------------------------------------- writing

    def flush(self) -> int:
        with self._lock:
            if not self._pending:
                return 0
            batch = self._pending
            self._pending = []
        try:
            import json

            from psycopg2.extras import execute_values

            from ..db.client import connect
            rows = [(SCALE.site_id, a["alert_type"], a["severity"], a["plate"],
                     a["cluster_id"], a["camera_id"], a["occurred_at"],
                     a["confidence"], a["reason"], json.dumps(a["evidence"]))
                    for a in batch]
            with connect() as conn:
                with conn.cursor() as cur:
                    execute_values(cur, """
                        insert into net_alerts
                          (site_id, alert_type, severity, plate, cluster_id, camera_id,
                           occurred_at, confidence, reason, evidence)
                        values %s
                    """, rows)
                conn.commit()
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning("alert flush failed: %s", self.last_error)
            with self._lock:
                self._pending = batch + self._pending
            return 0
        self.written += len(batch)
        return len(batch)

    def stats(self) -> dict:
        with self._lock:
            pending = len(self._pending)
        return {"raised": self.raised, "written": self.written, "pending": pending,
                "failures": self.failures, "last_error": self.last_error,
                "watchlist_entries": self.watchlist.entries}


def recent_alerts(hours: float = 24.0, limit: int = 200) -> list:
    from datetime import timedelta

    from ..db import repository as repo
    return repo._fetch("""
        select a.id, a.alert_type, a.severity, a.plate, a.cluster_id, a.camera_id,
               c.name as camera_name, a.occurred_at, a.confidence, a.reason,
               a.evidence, a.status
        from net_alerts a
        left join net_cameras c on c.camera_id = a.camera_id
        where a.occurred_at >= %s
        order by a.occurred_at desc
        limit %s
    """, (datetime.now(timezone.utc) - timedelta(hours=hours), limit))
