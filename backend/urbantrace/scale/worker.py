"""Ingest worker: one process in the horizontally scaled fleet.

    python -m urbantrace.scale.worker                 # run
    python -m urbantrace.scale.worker --init-db       # apply schemas, then exit
    python -m urbantrace.scale.worker --import-cameras data/cameras.json

Every worker runs the same loop, with no coordinator:

    heartbeat -> read live workers -> compute my cameras -> open/close feeds
              -> tune analysis rate -> publish feed health

Start another worker and it takes its share of cameras within a heartbeat;
stop or lose one and the others take over its cameras once its TTL passes.

The worker serves its own probes and metrics (default port 9100):

    /healthz   the control loop is alive
    /readyz    models are loaded and the pool is running
    /metrics   Prometheus text
    /status    the full JSON picture
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .adaptive import RateController
from .capture import CaptureManager
from .cluster import Membership, affinity_key, assign
from .config import SCALE
from .registry import REGISTRY
from .service import IngestService, load_cameras_from_db, save_cameras_to_db

logger = logging.getLogger("urbantrace.scale.worker")

CAMERA_REFRESH_S = 15.0
HOUSEKEEPING_S = 60.0


class Worker:
    def __init__(self, persist: bool = True):
        self.membership = Membership()
        self.service = IngestService(persist=persist)
        self.service.assigned = set()
        self.rate = RateController()
        self.capture = CaptureManager(self.service.submit_frame, self.rate.fps)
        self._stop = threading.Event()
        self._signature = ""
        self._sources: dict[str, str] = {}
        self._affinity: list[tuple[str, str]] = []
        self.loop_ticks = 0
        self.last_tick_at = 0.0
        self.last_error: str | None = None
        self.health_failures = 0

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._refresh_cameras(force=True)
        self.service.start()
        self.membership.beat(self._stats())
        self._rebalance()

    def run(self) -> None:
        self.start()
        next_cams = time.monotonic() + CAMERA_REFRESH_S
        next_house = time.monotonic() + HOUSEKEEPING_S
        while not self._stop.wait(SCALE.worker_heartbeat_s):
            try:
                if time.monotonic() >= next_cams:
                    self._refresh_cameras()
                    next_cams = time.monotonic() + CAMERA_REFRESH_S
                self._rebalance()
                self._adapt()
                self._publish_health()
                if time.monotonic() >= next_house:
                    self.service.housekeeping()
                    self.service.alerts.watchlist.load()
                    next_house = time.monotonic() + HOUSEKEEPING_S
                self.loop_ticks += 1
                self.last_tick_at = time.time()
            except Exception as e:                      # noqa: BLE001 - the loop must survive
                self.last_error = f"{type(e).__name__}: {e}"
                logger.exception("worker loop error")
        self.shutdown()

    def request_stop(self) -> None:
        self._stop.set()

    def shutdown(self) -> None:
        logger.info("worker %s shutting down", self.membership.worker_id)
        self.capture.stop()
        self.service.stop()
        self.membership.leave()

    # ---------------------------------------------------------------- steps

    def _refresh_cameras(self, force: bool = False) -> None:
        """Re-read the estate; rebuild only when it actually changed."""
        try:
            from ..db import repository as repo
            rows = repo._fetch("""select camera_id, name, road, sector, latitude,
                                         longitude, direction, source, enabled
                                  from net_cameras order by camera_id""")
        except Exception as e:                          # noqa: BLE001
            self.last_error = f"camera refresh: {type(e).__name__}: {e}"
            logger.warning(self.last_error)
            return
        sig = hashlib.sha1(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
        if sig == self._signature and not force:
            return
        REGISTRY.replace(rows)
        self._signature = sig
        self._sources = {r["camera_id"]: (r.get("source") or "")
                         for r in rows if r.get("enabled", True)}
        self._affinity = [(r["camera_id"],
                           affinity_key(r.get("sector") or "", r["latitude"], r["longitude"]))
                          for r in rows if r.get("enabled", True)]
        logger.info("camera estate loaded: %d camera(s)", len(rows))

    def _rebalance(self) -> None:
        workers = self.membership.beat(self._stats())
        mine = assign(self._affinity, workers, self.membership.worker_id)
        self.service.assigned = mine
        wanted = {cid: self._sources.get(cid, "") for cid in mine}
        diff = self.capture.reconcile(wanted)
        if diff["started"] or diff["stopped"]:
            logger.info("rebalance: %d worker(s), owning %d camera(s) "
                        "(+%d started, -%d stopped)", len(workers), len(mine),
                        diff["started"], diff["stopped"])

    def _adapt(self) -> None:
        b = self.service.bus
        self.rate.tick(b.submitted, b.dropped)
        b.target_fps = self.rate.fps()

    def _publish_health(self) -> None:
        feeds = self.capture.feeds()
        if not feeds:
            return
        from psycopg2.extras import execute_values

        from ..db.client import cursor
        now = datetime.now(timezone.utc)
        wid = self.membership.worker_id
        rows = []
        for f in feeds:
            age = f["seconds_since_frame"]
            rows.append((
                f["camera_id"], wid, f["state"], f["source_fps"], f["resolution"],
                f["frames_read"], f["frames_submitted"], f["frames_skipped_static"],
                f["reconnects"], now - timedelta(seconds=age) if age is not None else None,
                f["last_error"]))
        try:
            with cursor() as cur:
                execute_values(cur, """
                    insert into net_camera_health
                      (camera_id, worker_id, state, source_fps, resolution, frames_read,
                       frames_submitted, frames_skipped_static, reconnects,
                       last_frame_at, last_error)
                    values %s
                    on conflict (camera_id) do update set
                      worker_id=excluded.worker_id, state=excluded.state,
                      source_fps=excluded.source_fps, resolution=excluded.resolution,
                      frames_read=excluded.frames_read,
                      frames_submitted=excluded.frames_submitted,
                      frames_skipped_static=excluded.frames_skipped_static,
                      reconnects=excluded.reconnects,
                      last_frame_at=excluded.last_frame_at,
                      last_error=excluded.last_error, updated_at=now()
                """, rows)
        except Exception as e:                          # noqa: BLE001
            self.health_failures += 1
            self.last_error = f"health publish: {type(e).__name__}: {e}"
            logger.warning(self.last_error)

    # --------------------------------------------------------------- status

    def _stats(self) -> dict:
        return {
            "cameras_owned": len(self.service.assigned or ()),
            "capture": self.capture.summary(),
            "rate": self.rate.stats(),
            "throughput_fps": self.service.pool.stats().get("throughput_fps", 0.0),
        }

    def status(self) -> dict:
        s = self.service.stats()
        return {
            "worker_id": self.membership.worker_id,
            "workers": self.membership.workers,
            "cameras_owned": sorted(self.service.assigned or ()),
            "capture": self.capture.summary(),
            "feeds": self.capture.feeds(),
            "rate": self.rate.stats(),
            "service": s,
            "membership": {"failures": self.membership.failures,
                           "last_error": self.membership.last_error},
            "loop": {"ticks": self.loop_ticks, "last_tick_at": self.last_tick_at,
                     "last_error": self.last_error},
        }

    def healthy(self) -> bool:
        if self.loop_ticks == 0:
            return time.time() - (self.service.started_at or time.time()) < 60
        return time.time() - self.last_tick_at < SCALE.worker_heartbeat_s * 6

    def ready(self) -> bool:
        return self.service.pool.running and self.service.models._vehicle is not None

    def metrics(self) -> str:
        st = self.service.stats()
        cap = self.capture.summary()
        pipe, bus = st["pipeline"], st["pipeline"]["bus"]
        lines: list[str] = []

        def m(name: str, value, help_: str, kind: str = "gauge", labels: str = "") -> None:
            lines.append(f"# HELP urbantrace_{name} {help_}")
            lines.append(f"# TYPE urbantrace_{name} {kind}")
            lines.append(f"urbantrace_{name}{labels} {value}")

        m("cameras_owned", len(self.service.assigned or ()), "Cameras this worker owns")
        m("workers_live", len(self.membership.workers), "Live workers in the cluster")
        for state, n in cap["by_state"].items():
            lines.append(f'urbantrace_feeds{{state="{state}"}} {n}')
        m("frames_read_total", cap["frames_read"], "Frames read from feeds", "counter")
        m("frames_submitted_total", cap["frames_submitted"], "Frames queued for analysis", "counter")
        m("frames_static_total", cap["frames_skipped_static"], "Frames skipped by the motion gate", "counter")
        m("feed_reconnects_total", cap["reconnects"], "Feed reconnects", "counter")
        m("frames_dropped_total", bus["dropped"], "Frames dropped from full queues", "counter")
        m("frames_pending", bus["frames_pending"], "Frames waiting for the detectors")
        m("analysis_fps_per_camera", round(self.rate.fps(), 3), "Current per-camera analysis rate")
        m("throughput_fps", pipe["throughput_fps"], "Frames analysed per second")
        m("sightings_total", st["sightings_seen"], "Confirmed sightings", "counter")
        m("sink_pending", st["sink"]["pending"], "Sightings awaiting a database write")
        m("sink_failures_total", st["sink"]["failures"], "Failed database writes", "counter")
        m("sink_dropped_total", st["sink"]["dropped"], "Sightings dropped by a full buffer", "counter")
        m("alerts_total", st["alerts"].get("raised", 0), "Alerts raised", "counter")
        return "\n".join(lines) + "\n"


def _serve_probes(worker: Worker, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: str, ctype: str = "text/plain; charset=utf-8"):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):                               # noqa: N802
            if self.path == "/healthz":
                ok = worker.healthy()
                self._send(200 if ok else 503, "ok\n" if ok else "stalled\n")
            elif self.path == "/readyz":
                ok = worker.ready()
                self._send(200 if ok else 503, "ready\n" if ok else "not ready\n")
            elif self.path == "/metrics":
                self._send(200, worker.metrics(), "text/plain; version=0.0.4")
            elif self.path == "/status":
                self._send(200, json.dumps(worker.status(), default=str),
                           "application/json")
            else:
                self._send(404, "not found\n")

        def log_message(self, *args):                   # probes are noisy
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, name="probes", daemon=True).start()
    return srv


def apply_schemas() -> None:
    from ..db.client import cursor
    here = Path(__file__).resolve().parent
    for sql in (here.parent / "db" / "schema.sql", here / "sql" / "scale_schema.sql"):
        with cursor() as cur:
            cur.execute(sql.read_text(encoding="utf-8"))
        print(f"applied {sql.name}")


def import_cameras(path: str) -> int:
    n = REGISTRY.load_file(path)
    save_cameras_to_db()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-db", action="store_true", help="apply schemas and exit")
    ap.add_argument("--import-cameras", metavar="FILE",
                    help="load cameras from JSON into the database and exit")
    ap.add_argument("--no-persist", action="store_true",
                    help="do not write sightings to the database")
    ap.add_argument("--port", type=int, default=SCALE.worker_port)
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.init_db:
        apply_schemas()
        return 0
    if a.import_cameras:
        print(f"imported {import_cameras(a.import_cameras)} camera(s)")
        return 0

    worker = Worker(persist=not a.no_persist)
    _serve_probes(worker, a.port)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: worker.request_stop())
    logger.info("worker %s starting; probes on :%d", worker.membership.worker_id, a.port)
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
