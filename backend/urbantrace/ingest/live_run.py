"""Persist the live multi-camera session as a run, so every tab sees it.

The problem this solves: the three-feed monitor kept its sightings in memory
only. The GIS map, Vehicle Tracking, Analytics and Observations tabs all read
runs from the database, so while the cameras were streaming those tabs showed
whatever batch run happened to be newest - stale data, presented as current.

So the live session writes itself into the same tables the batch pipeline uses.
One run per session, observations appended as they are confirmed, and the
reconstructed journeys refreshed alongside them. Nothing else in the API has to
change: `latest_run()` returns the live run, and every existing endpoint follows.

Design rules:

* **Never on the detection thread.** Writes are queued and flushed by one
  background thread, so a slow database cannot slow a camera down.
* **Never fatal.** No database, no credentials, an outage mid-session - the feeds
  keep running and the failure is counted and reported, not raised. Live
  monitoring working is more important than it being recorded.
* **Never silent.** `status()` reports what was written and what failed, so a
  dashboard showing stale data can say why.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

FLUSH_SECONDS = 4.0          # how often queued sightings reach the database
MAX_PENDING = 2000           # cap the queue rather than grow without bound


@dataclass
class _Camera:
    """The shape `repository.save_cameras` expects (attribute access)."""
    camera_id: str
    name: str
    road: str
    latitude: float
    longitude: float
    direction: str = "UNKNOWN"
    video_file: str = ""
    start_time: str | None = None
    location_source: str = "config"


@dataclass
class LiveRunStatus:
    enabled: bool = False
    run_id: str | None = None
    observations_written: int = 0
    vehicles_written: int = 0
    failures: int = 0
    last_error: str | None = None
    last_flush: str | None = None
    pending: int = 0
    dropped: int = 0

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled, "run_id": self.run_id,
            "observations_written": self.observations_written,
            "vehicles_written": self.vehicles_written,
            "failures": self.failures, "last_error": self.last_error,
            "last_flush": self.last_flush, "pending": self.pending,
            "dropped": self.dropped,
        }


class LiveRunWriter:
    """Streams live multi-camera sightings into a database run."""

    def __init__(self, journeys_fn=None):
        self._lock = threading.Lock()
        self._pending: list = []
        self._seen_ids: set = set()
        self._run_id: str | None = None
        self._cameras: dict = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._journeys_fn = journeys_fn      # callable returning REGISTRY.journeys()
        self.status = LiveRunStatus()

    # ------------------------------------------------------------ lifecycle

    def register_camera(self, camera_id: str, name: str, road: str,
                        latitude: float, longitude: float) -> None:
        self._cameras[camera_id] = _Camera(camera_id, name, road, latitude, longitude)

    def start(self) -> None:
        """Begin a session. Safe to call repeatedly; only the first one acts."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="live-run-writer")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._flush()                       # last write, so nothing is lost on stop
        if self._run_id:
            try:
                from ..db import repository as repo
                repo.finish_run(self._run_id, self.status.observations_written)
            except Exception as e:              # noqa: BLE001 - reported, never fatal
                self._note_failure(e)
            # The next session is a new run. Reusing a run already marked
            # finished would keep growing a record that claims to be complete.
            self._run_id = None
            self._seen_ids.clear()

    # ------------------------------------------------------------ ingest

    def record(self, sighting: dict) -> None:
        """Queue one confirmed sighting. Called from the detection thread."""
        with self._lock:
            if len(self._pending) >= MAX_PENDING:
                self.status.dropped += 1
                return
            self._pending.append(sighting)
            self.status.pending = len(self._pending)

    # ------------------------------------------------------------ internals

    def _note_failure(self, exc: Exception) -> None:
        self.status.failures += 1
        self.status.last_error = f"{type(exc).__name__}: {exc}"
        if self.status.failures in (1, 5, 25) or self.status.failures % 100 == 0:
            logger.warning("live run persistence failed %dx (latest: %s)",
                           self.status.failures, self.status.last_error)

    def _loop(self) -> None:
        while not self._stop.wait(FLUSH_SECONDS):
            self._flush()

    def _ensure_run(self, repo) -> str | None:
        if self._run_id:
            return self._run_id
        from ..core.runtime import describe
        from ..vision.detectors import PLATE_WEIGHTS, VEHICLE_WEIGHTS
        from ..vision.ocr_engine import DEFAULT_MODEL

        try:
            runtime = describe()
        except Exception:                       # runtime info is a nicety, not a blocker
            runtime = {}
        self._run_id = repo.start_run(
            runtime=runtime,
            models={"vehicle_model": VEHICLE_WEIGHTS.name,
                    "plate_backend": "local", "plate_model": PLATE_WEIGHTS.name,
                    "ocr_model": DEFAULT_MODEL},
            camera_count=len(self._cameras), video_count=0,
            notes="Live multi-camera session (written as it happens)",
        )
        self.status.run_id = self._run_id
        self.status.enabled = True
        if self._cameras:
            repo.save_cameras(self._run_id, self._cameras)
        logger.info("live multi-camera run started: %s", self._run_id)
        return self._run_id

    def _flush(self) -> None:
        with self._lock:
            batch, self._pending = self._pending, []
            self.status.pending = 0
        if not batch:
            return

        try:
            from ..db import repository as repo
        except Exception as e:                  # noqa: BLE001
            self._note_failure(e)
            return

        try:
            run_id = self._ensure_run(repo)
            if not run_id:
                return

            # One row per (camera, track). A track republishes as its plate
            # reading improves, so the existing row is replaced rather than
            # skipped - otherwise the database would keep the first, worst
            # reading while the live view showed the corrected one.
            observations = {o["observation_id"]: o
                            for o in (self._as_observation(s) for s in batch)}
            if observations:
                self._replace_observations(run_id, list(observations))
                written, _ = repo.save_observations(run_id, list(observations.values()))
                self.status.observations_written = len(
                    self._seen_ids | set(observations))
                self._seen_ids |= set(observations)

            self._flush_vehicles(repo, run_id)
            self.status.last_flush = datetime.now(timezone.utc).isoformat(timespec="seconds")
        except Exception as e:                  # noqa: BLE001 - the feeds must not stop
            self._note_failure(e)

    def _replace_observations(self, run_id: str, observation_ids: list) -> None:
        """Clear rows we are about to rewrite, so an improved reading wins."""
        known = [i for i in observation_ids if i in self._seen_ids]
        if not known:
            return
        from ..db.client import cursor
        with cursor() as cur:
            cur.execute("delete from plate_readings where run_id = %s "
                        "and observation_id = any(%s)", (run_id, known))
            cur.execute("delete from vehicle_observations where run_id = %s "
                        "and observation_id = any(%s)", (run_id, known))
            cur.execute("delete from observations where run_id = %s "
                        "and observation_id = any(%s)", (run_id, known))

    def _as_observation(self, s: dict) -> dict:
        """Map a live sighting onto the observation row the batch pipeline writes."""
        return {
            # Stable per camera+track+time, so a re-published track that improves
            # its reading does not create a second row for the same vehicle.
            "observation_id": f"LIVE_{s['camera_id']}_{s.get('track_id', 0)}",
            "camera_id": s["camera_id"],
            "plate_number": s["plate"],
            "ocr_confidence": s.get("confidence"),
            "plate_detection_confidence": None,
            "local_track_id": s.get("track_id"),
            "vehicle_type": s.get("vehicle_type", "vehicle"),
            "frame_number": None,
            "video_time_s": s.get("video_time_s"),
            "timestamp": s.get("observed_at"),
            "clock_source": "live wall clock at confirmation",
            "latitude": s.get("latitude"),
            "longitude": s.get("longitude"),
            "direction": None,
            "aggregation_method": "string-vote",
            "support": s.get("support"),
            "total_readings": s.get("total_readings"),
            "agreement": s.get("agreement"),
            "is_valid_format": s.get("is_valid_format"),
            "evidence_crop": None,
            "raw_readings": [],
        }

    def _flush_vehicles(self, repo, run_id: str) -> None:
        """Rewrite the reconstructed journeys for this run."""
        if not self._journeys_fn:
            return
        journeys = self._journeys_fn() or []
        if not journeys:
            return

        vehicles = []
        for i, j in enumerate(journeys, 1):
            vehicles.append({
                "global_vehicle_id": f"LIVE_GV_{i:04d}",
                "plate_number": j["plate"],
                "route": j.get("route") or [],
                "route_status": ("COMPLETE_CORRIDOR" if j.get("camera_count", 0) > 1
                                 else "SINGLE_CAMERA_NETWORK"),
                "dropped_at": None,
                "status_reason": (f"Seen at {j.get('camera_count', 0)} camera(s) "
                                  f"in the live session"),
                "first_seen": j.get("first_seen"),
                "last_seen": j.get("last_seen"),
                "total_duration_minutes": j.get("total_duration_minutes"),
                "camera_count": j.get("camera_count"),
                "confidence": j.get("confidence"),
                "merged_variants": j.get("variants") or [],
                "legs": j.get("legs") or [],
                # Observation links are omitted: the journey is rebuilt from
                # scratch on every flush, and pointing at rows that may be
                # rewritten would be a claim this cannot keep.
                "observations": [],
            })

        # Journeys are recomputed wholesale each flush, so replace rather than
        # append - otherwise the same vehicle accumulates duplicate rows.
        try:
            from ..db.client import cursor
            with cursor() as cur:
                cur.execute("delete from route_legs where run_id = %s", (run_id,))
                cur.execute("delete from vehicle_observations where run_id = %s", (run_id,))
                cur.execute("delete from global_vehicles where run_id = %s", (run_id,))
        except Exception as e:                  # noqa: BLE001
            self._note_failure(e)
            return
        written, _ = repo.save_global_vehicles(run_id, vehicles)
        self.status.vehicles_written = written
