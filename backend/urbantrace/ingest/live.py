"""Live ingest: real-time detection and monitoring from a running source.

Until now the platform was batch-only - point it at a video file, wait, read
the results. This module runs the same vision stack continuously against a live
source and publishes what it finds as it finds it, so the dashboard shows the
present rather than a completed run.

SOURCES

  A source is anything OpenCV can open, which is what makes this real rather
  than a simulation:

    rtsp://…            a real ANPR camera
    0, 1, 2 …           a locally attached camera
    data/videos/x.mp4   a file, played at its own frame rate (`realtime=True`)
                        so a recorded feed behaves like a live one

WHAT IS PUBLISHED

  Every subscriber gets an event stream, not a snapshot:

    started / stopped   session lifecycle
    stats               rolling counts, fps, uptime - a few times a second
    vehicle             a vehicle became countable (crossed the gates)
    plate               a plate reached temporal consensus, with evidence
    alert               that plate is on the watchlist

  Each session also keeps a bounded ring of recent events, so a dashboard that
  connects late gets immediate context instead of an empty screen.

DESIGN NOTES

  Detection runs in a worker thread per source. Publishing is non-blocking:
  a subscriber that cannot keep up drops frames-worth of stats rather than
  stalling the detector, because a stalled detector would fall behind the feed
  and stop being live. Lifecycle events are never dropped.
"""
from __future__ import annotations

import base64
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2

from ..core.cameras import CAMERAS
from ..vision.aggregate import Reading, aggregate
from ..vision.counting import CountConfig, VehicleCounter
from ..vision.detectors import PlateDetector, VehicleTracker, bind_plates_to_tracks
from ..vision.ocr_engine import get_ocr
from ..vision.preprocess import crop_with_margin

# Match the batch pipeline so live and recorded runs agree.
MAX_READINGS_PER_TRACK = 14
MIN_PLATE_PIXELS = 480

# Stats cadence. More often than this is noise on the wire; less feels laggy.
STATS_EVERY_S = 0.5
# Recent-event ring handed to a late subscriber.
BACKLOG = 120
# A slow subscriber's queue. Beyond this, droppable events are discarded.
SUBSCRIBER_QUEUE = 256


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LiveEvent:
    type: str
    camera_id: str
    at: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, "camera_id": self.camera_id,
                "at": self.at, **self.data}


class LiveSession:
    """One live source, processed continuously in its own thread."""

    def __init__(self, camera_id: str, source: str, realtime: bool = True,
                 detect_every: int = 1, loop: bool = True,
                 watchlist: set | None = None,
                 evidence_dir: Path | None = None,
                 vehicle_tracker=None, plate_detector=None):
        self.camera_id = camera_id
        self.source = source
        self.realtime = realtime
        # Run detection on every Nth frame when the source outruns the GPU.
        # Frames in between are still displayed; they just carry no new boxes.
        self.detect_every = max(1, detect_every)
        # A file source can be replayed so a demo feed stays live. Counting
        # state is reset each cycle: without that the same vehicles are counted
        # again on every pass and the running total climbs without meaning.
        self.loop = loop
        self.cycles = 0
        self.watchlist = {p.upper().replace(" ", "") for p in (watchlist or set())}

        self.evidence_dir = Path(evidence_dir or Path("data/live_evidence")) / camera_id
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self._tracker = vehicle_tracker
        self._plates = plate_detector
        self._ocr = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._subs: list[queue.Queue] = []
        self._backlog: deque = deque(maxlen=BACKLOG)

        self.counter: VehicleCounter | None = None
        self.readings: dict = {}
        self.published: set = set()       # track ids whose plate was published
        self.counted: set = set()         # vehicle ids already announced
        self.latest_jpeg: bytes | None = None

        self.state = "idle"               # idle | starting | running | stopped | error
        self.error: str | None = None
        self.started_at: float | None = None
        self.frames = 0
        self.detections = 0
        self.plates_read = 0
        self.alerts = 0
        self.fps = 0.0
        self.source_fps = 0.0

    # ------------------------------------------------------------ pub/sub

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE)
        with self._lock:
            for ev in self._backlog:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    break
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _publish(self, kind: str, data: dict, droppable: bool = False) -> None:
        ev = LiveEvent(kind, self.camera_id, _now(), data).to_dict()
        with self._lock:
            if not droppable:
                self._backlog.append(ev)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                # Never block the detector for a slow client. Stats can be
                # missed without losing meaning; lifecycle events cannot, so
                # for those we make room by discarding the oldest item.
                if not droppable:
                    try:
                        q.get_nowait()
                        q.put_nowait(ev)
                    except (queue.Empty, queue.Full):
                        pass

    # ------------------------------------------------------------ control

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.state = "starting"
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"live-{self.camera_id}")
        self._thread.start()

    def stop(self, timeout: float = 6.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.state = "stopped"
        self._publish("stopped", self.status())

    def status(self) -> dict:
        up = (time.time() - self.started_at) if self.started_at else 0.0
        counts = self.counter.counts("traffic") if self.counter else {}
        return {
            "camera_id": self.camera_id,
            "name": CAMERAS[self.camera_id].name if self.camera_id in CAMERAS
                    else self.camera_id,
            "source": str(self.source),
            "state": self.state,
            "error": self.error,
            "uptime_s": round(up, 1),
            "frames": self.frames,
            "fps": round(self.fps, 1),
            "source_fps": round(self.source_fps, 1),
            # < 1.0 means detection is slower than the feed. On a real camera
            # that means dropped frames, so it is reported rather than hidden;
            # raising detect_every is the lever.
            "realtime_factor": round(self.fps / self.source_fps, 2)
                               if self.source_fps else None,
            "keeping_up": (self.fps >= self.source_fps * 0.9)
                          if self.source_fps else None,
            "detect_every": self.detect_every,
            "cycles": self.cycles,
            "detections": self.detections,
            "plates_read": self.plates_read,
            "alerts": self.alerts,
            "counts": counts,
            "watchlist_size": len(self.watchlist),
        }

    # ------------------------------------------------------------- worker

    def _open(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open live source: {self.source}")
        return cap

    def _run(self) -> None:
        try:
            self._tracker = self._tracker or VehicleTracker()
            self._plates = self._plates or PlateDetector()
            self._ocr = get_ocr()

            cap = self._open()
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.source_fps = fps_in
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            frame_period = 1.0 / fps_in if (self.realtime and fps_in > 0) else 0.0

            cfg = CountConfig()
            cam = CAMERAS.get(self.camera_id)
            if cam is not None and getattr(cam, "count_line", None):
                cl = cam.count_line
                cfg.line = ((cl[0], cl[1]), (cl[2], cl[3]))
            self.counter = VehicleCounter(w, h, cfg)
            self._tracker.reset()

            self.started_at = time.time()
            self.state = "running"
            self._publish("started", {**self.status(), "width": w, "height": h,
                                      "source_fps": round(fps_in, 2)})

            last_stats = 0.0
            t_prev = time.time()
            frame_no = 0
            loops = 0

            while not self._stop.is_set():
                t_frame = time.time()
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    if not self.loop:
                        self._publish("ended", self.status())
                        break
                    # Replay the source, and start counting afresh: the same
                    # vehicles reappearing on a second pass are not new ones.
                    loops += 1
                    if loops > 10_000:
                        raise RuntimeError("source kept ending; giving up")
                    try:
                        cap = self._open()
                    except Exception:
                        self.state = "error"
                        self.error = "source ended and could not be reopened"
                        break
                    self.cycles += 1
                    self.counter = VehicleCounter(w, h, cfg)
                    self._tracker.reset()
                    self.readings.clear()
                    self.published.clear()
                    self.counted.clear()
                    frame_no = 0
                    self._publish("cycle", self.status())
                    continue

                if frame_no % self.detect_every == 0:
                    self._process(frame, frame_no, fps_in)

                # Keep a JPEG for the dashboard preview. Small and cheap.
                if frame_no % 3 == 0:
                    small = cv2.resize(frame, (640, int(640 * h / max(1, w))))
                    ok_jpg, buf = cv2.imencode(".jpg", small,
                                               [cv2.IMWRITE_JPEG_QUALITY, 68])
                    if ok_jpg:
                        self.latest_jpeg = buf.tobytes()

                frame_no += 1
                self.frames += 1

                if now_stats := (time.time() - last_stats >= STATS_EVERY_S):
                    last_stats = time.time()

                # Pace a file source to its own frame rate BEFORE measuring, so
                # the reported fps is the rate frames are actually consumed at,
                # not raw GPU throughput. Reporting throughput here claimed
                # 111 fps on a 24 fps feed.
                if frame_period:
                    slack = frame_period - (time.time() - t_frame)
                    if slack > 0:
                        time.sleep(slack)

                now = time.time()
                dt = now - t_prev
                if dt > 0:
                    inst = 1.0 / dt
                    self.fps = 0.85 * self.fps + 0.15 * inst if self.fps else inst
                t_prev = now

                if now_stats:
                    self._publish("stats", self.status(), droppable=True)

            cap.release()
            if self.state != "error":
                self.state = "stopped"

        except Exception as e:                      # never kill the server
            self.state = "error"
            self.error = f"{type(e).__name__}: {e}"
            self._publish("error", {"error": self.error})
        finally:
            self._publish("stopped", self.status())

    # ------------------------------------------------------ per-frame work

    def _process(self, frame, frame_no: int, fps: float) -> None:
        tracks = self._tracker.update(frame)
        self.counter.update(tracks, frame_no)
        self.detections += len(tracks)

        # Announce vehicles as they become countable, so the running total on
        # screen is the same number the batch pipeline would report at the end.
        for v in self.counter.vehicles():
            if v.counted_traffic and v.vehicle_id not in self.counted:
                self.counted.add(v.vehicle_id)
                self._publish("vehicle", {
                    "vehicle_id": v.vehicle_id,
                    "vehicle_type": v.vehicle_type,
                    "frames_seen": v.frames_seen,
                    "counts": self.counter.counts("traffic"),
                })

        plates = self._plates.detect(frame)
        if not plates:
            return
        for track, plate in bind_plates_to_tracks(plates, tracks):
            self._handle_plate(frame, track, plate, frame_no, frame_no / max(1.0, fps))

    def _handle_plate(self, frame, track, plate, frame_no: int, t_s: float) -> None:
        tid = track.track_id if track else -1
        if tid < 0 or tid in self.published:
            return
        buf = self.readings.setdefault(tid, [])
        if len(buf) >= MAX_READINGS_PER_TRACK:
            return
        x1, y1, x2, y2 = plate.box
        if (x2 - x1) * (y2 - y1) < MIN_PLATE_PIXELS:
            return
        crop = crop_with_margin(frame, plate.box)
        if crop is None:
            return

        res = self._ocr.read(crop)
        if res is None or not res.text_final:
            return

        buf.append(Reading(
            frame_number=frame_no, timestamp_s=round(t_s, 3),
            text_raw=res.text_raw, text_final=res.text_final,
            ocr_confidence=res.confidence, plate_confidence=plate.confidence,
            is_valid_format=res.is_valid_format, repaired=res.repaired,
            variant=res.variant, char_confidences=res.char_confidences))

        # Publish once the track's readings agree, not on the first glimpse:
        # a single-frame OCR result is exactly what temporal aggregation exists
        # to distrust.
        agg = aggregate(buf, min_readings=3)
        if agg is None:
            return
        self.published.add(tid)
        self.plates_read += 1

        path = self.evidence_dir / f"live_{frame_no:07d}_{agg.plate}.jpg"
        try:
            cv2.imwrite(str(path), crop)
        except Exception:
            path = None

        cam = CAMERAS.get(self.camera_id)
        payload = {
            "plate_number": agg.plate,
            "ocr_confidence": round(agg.confidence, 4),
            "support": agg.support,
            "total_readings": agg.total_readings,
            "agreement": round(agg.agreement, 3),
            "is_valid_format": agg.is_valid_format,
            "track_id": tid,
            "frame_number": frame_no,
            "video_time_s": round(t_s, 2),
            "vehicle_type": track.cls_name if track else "unknown",
            "latitude": cam.latitude if cam else None,
            "longitude": cam.longitude if cam else None,
            "location": cam.name if cam else self.camera_id,
            "evidence": str(path) if path else None,
            "evidence_b64": base64.b64encode(
                cv2.imencode(".jpg", crop)[1].tobytes()).decode() if crop is not None else None,
        }
        self._publish("plate", payload)

        if agg.plate.upper().replace(" ", "") in self.watchlist:
            self.alerts += 1
            sev = ("HIGH" if agg.confidence >= 0.85
                   else "MEDIUM" if agg.confidence >= 0.60 else "LOW")
            self._publish("alert", {
                **payload, "severity": sev, "alert_type": "WATCHLIST_MATCH",
                "reason": (f"Watchlisted plate {agg.plate} seen live at "
                           f"{self.camera_id} ({payload['location']}), "
                           f"OCR confidence {agg.confidence:.3f} over "
                           f"{agg.support}/{agg.total_readings} agreeing frames."),
            })


class LiveManager:
    """All live sessions in this process. One per camera."""

    def __init__(self):
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()
        self._tracker = None
        self._plates = None

    def _models(self):
        # Models are heavy; share one copy across sessions. Each session keeps
        # its own tracker STATE via reset(), so ids do not leak between feeds.
        if self._tracker is None:
            self._tracker = VehicleTracker()
        if self._plates is None:
            self._plates = PlateDetector()
        return self._tracker, self._plates

    def start(self, camera_id: str, source: str, realtime: bool = True,
              detect_every: int = 1, loop: bool = True,
              watchlist: set | None = None) -> LiveSession:
        with self._lock:
            existing = self._sessions.get(camera_id)
            if existing and existing.state in ("running", "starting"):
                return existing
            tracker, plates = self._models()
            s = LiveSession(camera_id, source, realtime=realtime,
                            detect_every=detect_every, loop=loop,
                            watchlist=watchlist,
                            vehicle_tracker=tracker, plate_detector=plates)
            self._sessions[camera_id] = s
        s.start()
        return s

    def stop(self, camera_id: str) -> bool:
        s = self._sessions.get(camera_id)
        if not s:
            return False
        s.stop()
        return True

    def stop_all(self) -> int:
        n = 0
        for s in list(self._sessions.values()):
            if s.state in ("running", "starting"):
                s.stop()
                n += 1
        return n

    def get(self, camera_id: str) -> LiveSession | None:
        return self._sessions.get(camera_id)

    def status(self) -> dict:
        sessions = [s.status() for s in self._sessions.values()]
        return {
            "sessions": sessions,
            "running": [s["camera_id"] for s in sessions if s["state"] == "running"],
            "any_running": any(s["state"] == "running" for s in sessions),
        }


MANAGER = LiveManager()
