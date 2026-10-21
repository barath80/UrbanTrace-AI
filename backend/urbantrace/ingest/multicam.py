"""High-Speed Multi-Camera Ingest & Stream Manager (Continuous Auto-Loop Demo Mode).

Architecture:
  1. Continuous Auto-Looping: Video streams from phones or local files loop seamlessly
     from start to finish repeatedly until the user explicitly clicks STOP.
  2. Full ANPR & YOLO Parity: Every frame is annotated identically to the standard Cameras view:
     - Yellow bounding box around every vehicle with filled banner: `#{id} {class} | {plate}`
     - Green bounding box around detected license plates
     - Top-left translucent vehicle count statistics HUD: `VEHICLES (TRAFFIC)`
  3. Real-time MJPEG streaming back to the dashboard.
  4. Independent Stop/Resume controls per camera and on mobile phones.
  5. A central registry that links plates seen by more than one camera into a
     single journey, which is what makes this a network rather than three
     unrelated viewers.

Every number this module reports is measured. Nothing is estimated or filled in.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse

from ..vision.aggregate import Reading, aggregate
from ..vision.detectors import PlateDetector, VehicleTracker, bind_plates_to_tracks
from ..vision.ocr_engine import get_ocr
from ..vision.preprocess import crop_with_margin
from .live_run import LiveRunWriter

logger = logging.getLogger("urbantrace.multicam")

ROOT = Path(__file__).resolve().parents[3]

def _load_slot_configs() -> list[dict]:
    """Monitor slots, one per camera in the camera file.

    Read from UT_CAMERAS_FILE, else data/cameras.json. Nothing is invented:
    a camera needs an id and real coordinates to get a slot, and with no file
    the monitor simply has no cameras until one is added.
    """
    path = Path(os.environ.get("UT_CAMERAS_FILE") or ROOT / "data" / "cameras.json")
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.error("camera file %s unreadable: %s", path, e)
        return []
    entries = raw.get("cameras", raw) if isinstance(raw, dict) else raw
    out = []
    for e in entries or []:
        try:
            out.append({
                "camera_id": str(e["camera_id"]),
                "name": str(e.get("name") or e["camera_id"]),
                "road": str(e.get("road") or ""),
                "latitude": float(e["latitude"]),
                "longitude": float(e["longitude"]),
                "default_video": str(e.get("video_file") or ""),
            })
        except (KeyError, TypeError, ValueError):
            logger.warning("skipping camera entry without id/coordinates: %r", e)
    return out


CAMERA_CONFIGS = _load_slot_configs()

# A plate is only published once the temporal vote has enough independent
# frames behind it AND clears this confidence floor. Publishing the first
# single-frame read (the previous behaviour) put demonstrably wrong strings on
# screen before consensus had a chance to correct them.
MIN_READINGS_TO_PUBLISH = 3
MIN_PUBLISH_CONFIDENCE = 0.60

COUNT_COLORS = {
    "car": (229, 135, 57),
    "motorcycle": (38, 89, 217),
    "bus": (112, 158, 25),
    "truck": (25, 178, 250),
    "bicycle": (200, 200, 200),
    "total": (255, 255, 255),
}


def _lan_rank(ip: str) -> tuple:
    """Rank an address by how likely a phone on the Wi-Fi can actually reach it.

    Windows machines routinely carry WSL, Hyper-V and Docker adapters. Those
    sit on 172.16-31.x and answer a UDP route lookup first, so the "primary"
    address ended up being one no phone can reach. Real Wi-Fi and Ethernet
    networks are almost always 192.168.x or 10.x, so those rank first and the
    virtual ranges rank last.
    """
    parts = ip.split(".")
    try:
        a, b = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (9, ip)
    if a == 192 and b == 168:
        return (0, ip)
    if a == 10:
        return (1, ip)
    if a == 172 and 16 <= b <= 31:
        return (3, ip)          # usually WSL / Hyper-V / Docker on Windows
    if a == 169 and b == 254:
        return (8, ip)          # link-local, no DHCP
    return (2, ip)


def get_local_ips() -> list[str]:
    """LAN addresses of this laptop, most reachable first."""
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass

    if not ips:
        return ["127.0.0.1"]
    return sorted(ips, key=_lan_rank)


class CameraSlot:
    """Independent state, real-time visual overlay annotator, and ANPR engine."""

    def __init__(self, camera_id: str, name: str, road: str, lat: float, lon: float, default_video: str):
        self.camera_id = camera_id
        self.name = name
        self.road = road
        self.latitude = lat
        self.longitude = lon
        self.default_video = default_video

        # Stream state
        self.status = "waiting"  # "waiting" | "playing" | "disconnected" | "reconnecting" | "stopped" | "error"
        self.source_type = "none"  # "phone_web" | "network_url" | "local_file" | "none"
        self.source_label = ""
        self.video_name = ""
        self.fps = 0.0
        self.frame_count = 0
        self.resolution = "1280x720"
        self.video_time_s = 0.0
        self.loops = 0
        self.last_received_at = 0.0
        self.error_message: Optional[str] = None
        self.active_ws: Optional[WebSocket] = None

        # Frame buffers & thread safety
        self.latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Condition(self._lock)

        # Worker threads
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Vision Models
        self._tracker: Optional[VehicleTracker] = None
        self._plates: Optional[PlateDetector] = None
        self._ocr = None
        self.anpr_enabled = True
        self.detect_every = 2  # Run detection on keyframes for smooth 30 FPS playback
        # One frame in flight per camera. Inference costs tens of milliseconds;
        # a phone streaming at 30 fps offers frames far faster than that, and
        # queueing them would grow an unbounded backlog of stale frames. ANPR
        # needs a few good frames per vehicle, not every frame, so the honest
        # response to a busy slot is to drop.
        self._processing = threading.Lock()
        self.detections_skipped = 0     # frames shown without fresh detection
        self.detections_run = 0         # frames the detector actually ran on
        self.detect_failures = 0        # detection exceptions, reported in status
        self.last_detect_error: str | None = None
        # Latched by stop_stream(); until it clears, inbound phone frames are
        # dropped instead of restarting the feed.
        self.stopped_by_user = False

        # Detection cache & stats
        self.active_tracks: list = []
        self.active_plates: list = []
        self.readings: dict = {}
        self.best_label: dict = {}
        self.published_tracks: dict = {}      # track_id -> published confidence
        self.recent_plates: deque = deque(maxlen=30)
        self.vehicle_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "total": 0}

    def _ensure_models(self):
        """Thread-safe lazy initialization of vision models."""
        if self._tracker is None:
            try:
                self._tracker = VehicleTracker(conf=0.25, imgsz=640)
            except Exception as e:
                logger.warning(f"Tracker load error: {e}")
        if self._plates is None:
            try:
                self._plates = PlateDetector(conf=0.25, imgsz=960)
            except Exception as e:
                logger.warning(f"PlateDetector load error: {e}")
        if self._ocr is None:
            try:
                self._ocr = get_ocr()
            except Exception as e:
                logger.warning(f"OCR load error: {e}")

    def _draw_counter(self, frame: np.ndarray) -> np.ndarray:
        """Draw translucent top-left vehicle count statistics box."""
        counts = {
            "car": sum(1 for t in self.active_tracks if t.cls_name == "car"),
            "motorcycle": sum(1 for t in self.active_tracks if t.cls_name == "motorcycle"),
            "bus": sum(1 for t in self.active_tracks if t.cls_name == "bus"),
            "truck": sum(1 for t in self.active_tracks if t.cls_name == "truck"),
        }
        total = len(self.active_tracks)
        rows = [(c, counts[c]) for c in ("car", "motorcycle", "bus", "truck") if counts.get(c, 0) > 0]
        rows.append(("total", total))

        pad, lh = 8, 20
        w = 175
        h = pad * 2 + lh * (len(rows) + 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + w, 10 + h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + w, 10 + h), (70, 70, 70), 1)

        y = 10 + pad + 14
        cv2.putText(frame, "VEHICLES (TRAFFIC)", (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
        y += lh - 2
        for name, val in rows:
            col = COUNT_COLORS.get(name, (220, 220, 220))
            if name == "total":
                cv2.line(frame, (18, y - 12), (10 + w - 10, y - 12), (70, 70, 70), 1)
            cv2.putText(frame, f"{name}:", (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
            cv2.putText(frame, str(val), (10 + w - 18 - 10 * len(str(val)), y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)
            y += lh
        return frame

    def _draw_overlays(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Render active YOLO tracks, plate boxes, plate text, and traffic counter."""
        if not self.anpr_enabled:
            return frame_bgr

        # Draw vehicle bounding boxes & plate tags
        for t in self.active_tracks:
            x1, y1, x2, y2 = [int(v) for v in t.box]
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"#{t.track_id} {t.cls_name}"
            plate_txt = self.best_label.get(t.track_id)
            if plate_txt:
                label += f" | {plate_txt}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 200, 255), -1)
            cv2.putText(frame_bgr, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

        # Draw green plate boxes
        for box in self.active_plates:
            px1, py1, px2, py2 = [int(v) for v in box]
            cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), (60, 255, 60), 2)

        # Draw top-left traffic counter
        if len(self.active_tracks) > 0:
            frame_bgr = self._draw_counter(frame_bgr)

        return frame_bgr

    def _publish(self, agg, track, tid: int) -> None:
        """Record a confirmed sighting and hand it to the central registry.

        Re-publishing the same track replaces its earlier entry rather than
        appending a second one, so the feed shows one row per vehicle whose
        plate improves as more frames arrive.
        """
        sighting = {
            "plate": agg.plate,
            "confidence": round(agg.confidence, 3),
            "support": agg.support,
            "total_readings": agg.total_readings,
            "agreement": agg.agreement,
            "is_valid_format": agg.is_valid_format,
            "vehicle_type": track.cls_name if track else "vehicle",
            "track_id": tid,
            "camera_id": self.camera_id,
            "location": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "video_time_s": round(self.video_time_s, 2),
        }
        existing = [d for d in self.recent_plates if d.get("track_id") == tid]
        for d in existing:
            self.recent_plates.remove(d)
        self.recent_plates.appendleft(sighting)
        REGISTRY.record(sighting)

    def resume(self) -> None:
        """Clear a stopped state so an already-connected phone can carry on.

        Without this, `stop` latched `status == "stopped"` and `update_frame`
        dropped every subsequent frame, so the phone's own Resume button could
        never bring the feed back -- the WebSocket was still open, so nothing
        reconnected and the slot stayed dark.
        """
        self.stopped_by_user = False
        self._stop_event.clear()
        if self.status == "stopped":
            self.status = "waiting"
            self.error_message = None

    def update_frame(self, frame_bgr: np.ndarray, video_time_s: float = 0.0, video_name: str = "",
                     detect: bool = True):
        """Process frame: run YOLO + Plate + OCR detection and render live overlays.

        Frames are refused on `stopped_by_user`, not on `status`: this method
        sets `status = "playing"` a few lines down, so a frame that was already
        being processed when Stop arrived would set the feed running again. The
        latch only clears on an explicit Start or Resume.
        """
        if self.stopped_by_user or self.status == "stopped":
            return

        h, w = frame_bgr.shape[:2]
        self.resolution = f"{w}x{h}"
        self.video_time_s = video_time_s
        if video_name:
            self.video_name = video_name

        now = time.time()
        if self.last_received_at > 0:
            dt = now - self.last_received_at
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = 0.85 * self.fps + 0.15 * inst_fps if self.fps > 0 else inst_fps
        self.last_received_at = now
        self.frame_count += 1
        if self.stopped_by_user:
            return
        self.status = "playing"
        self.error_message = None

        # Run AI detection on keyframes
        if detect and self.anpr_enabled and (self.frame_count % self.detect_every == 0
                                             or len(self.active_tracks) == 0):
            self._ensure_models()
            if self._tracker and self._plates:
                try:
                    self.detections_run += 1
                    tracks = self._tracker.update(frame_bgr)
                    plates = self._plates.detect(frame_bgr)
                    pairs = bind_plates_to_tracks(plates, tracks)

                    self.active_tracks = tracks
                    self.active_plates = [p.box for _t, p in pairs]
                    self.vehicle_counts["total"] = len(tracks)

                    # Process plate OCR
                    for track, plate in pairs:
                        tid = track.track_id if track else -1
                        if tid < 0:
                            continue
                        buf = self.readings.setdefault(tid, [])
                        if len(buf) < 6:
                            # No margin override. Measured on the 80 labelled
                            # test plates (testing_accuracy.ipynb): reading the
                            # detector's box with a 0.15 margin scores 0.613
                            # exact against 0.750 with none. The detector box
                            # already carries context; padding it adds
                            # background the recogniser has to discount.
                            crop = crop_with_margin(frame_bgr, plate.box)
                            if crop is not None and self._ocr is not None:
                                res = self._ocr.read(crop)
                                if res and res.text_final:
                                    buf.append(
                                        Reading(
                                            frame_number=self.frame_count,
                                            timestamp_s=round(video_time_s, 2),
                                            text_raw=res.text_raw,
                                            text_final=res.text_final,
                                            ocr_confidence=res.confidence,
                                            plate_confidence=plate.confidence,
                                            is_valid_format=res.is_valid_format,
                                            repaired=res.repaired,
                                            variant=res.variant,
                                            char_confidences=res.char_confidences,
                                        )
                                    )
                                    self.best_label[tid] = res.text_final

                                    # Publish only once the temporal vote has
                                    # enough frames behind it. A single frame is
                                    # not evidence -- it is the reading that
                                    # consensus exists to correct.
                                    agg = aggregate(buf, min_readings=MIN_READINGS_TO_PUBLISH)
                                    if agg and agg.confidence >= MIN_PUBLISH_CONFIDENCE:
                                        prev = self.published_tracks.get(tid)
                                        if prev is None or agg.confidence > prev + 1e-9:
                                            self.published_tracks[tid] = agg.confidence
                                            self.best_label[tid] = agg.plate
                                            self._publish(agg, track, tid)
                except Exception as e:
                    # Counted and surfaced, never swallowed: a feed reading zero
                    # plates must be able to say whether the road was empty or
                    # the detector was failing on every frame.
                    self.detect_failures += 1
                    self.last_detect_error = f"{type(e).__name__}: {e}"
                    if self.detect_failures in (1, 10, 100) or self.detect_failures % 500 == 0:
                        logger.warning("%s detection failed %dx (latest: %s)",
                                       self.camera_id, self.detect_failures, self.last_detect_error)

        # Render overlays onto frame
        annotated = self._draw_overlays(frame_bgr)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 76])
        if ok:
            with self._new_frame_event:
                self.latest_jpeg = buf.tobytes()
                self._new_frame_event.notify_all()

    def submit_jpeg(self, jpeg_bytes: bytes, video_time_s: float = 0.0, video_name: str = "") -> bool:
        """Take a phone frame: always display it, detect on it only when free.

        Display and inference are deliberately separated. Showing a frame costs
        a JPEG decode; detecting on one costs vehicle detection, plate detection
        and OCR, which is orders of magnitude slower. Gating both on the same
        lock made the video freeze whenever the GPU was busy. So every frame
        reaches the operator's screen, and detection runs on whichever frames
        the GPU can actually take - the last known boxes stay drawn in between.

        MUST be called off the event loop. Returns True if this frame was
        detected on, False if it was shown without fresh detection.
        """
        if self.stopped_by_user or self.status == "stopped":
            return False
        if self._processing.acquire(blocking=False):
            try:
                self.update_jpeg(jpeg_bytes, video_time_s=video_time_s, video_name=video_name)
                return True
            finally:
                self._processing.release()

        self.detections_skipped += 1
        self.update_jpeg(jpeg_bytes, video_time_s=video_time_s,
                         video_name=video_name, detect=False)
        return False

    def update_jpeg(self, jpeg_bytes: bytes, video_time_s: float = 0.0, video_name: str = "",
                    detect: bool = True):
        """Decode incoming phone JPEG, run detection, render overlays, and stream."""
        if self.stopped_by_user or self.status == "stopped":
            return

        try:
            arr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.update_frame(frame, video_time_s=video_time_s,
                                  video_name=video_name, detect=detect)
                return
        except Exception:
            pass

        now = time.time()
        if self.last_received_at > 0:
            dt = now - self.last_received_at
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = 0.85 * self.fps + 0.15 * inst_fps if self.fps > 0 else inst_fps
        self.last_received_at = now
        self.frame_count += 1
        self.status = "playing"
        self.video_time_s = video_time_s
        if video_name:
            self.video_name = video_name
        self.error_message = None

        with self._new_frame_event:
            self.latest_jpeg = jpeg_bytes
            self._new_frame_event.notify_all()

    def mark_disconnected(self, reason: str = "Connection lost"):
        """Mark as disconnected while retaining slot order and metadata."""
        if self.status != "stopped":
            self.status = "disconnected"
            self.error_message = reason
            self.fps = 0.0

    def _begin_session(self) -> None:
        """A feed is starting, so the live session is now worth recording."""
        try:
            LIVE_RUN.start()
        except Exception as e:              # noqa: BLE001 - recording is optional
            logger.warning("live run writer failed to start: %s", e)

    def start_url_stream(self, url: str, video_name: str = ""):
        """Start background worker to ingest from network stream URL with continuous auto-loop."""
        self.stop_stream()
        self.stopped_by_user = False
        self._stop_event.clear()
        self._begin_session()
        self.source_type = "network_url"
        self.source_label = url
        self.video_name = video_name or "Network Stream"
        self.status = "reconnecting"
        self.error_message = None

        self._worker_thread = threading.Thread(
            target=self._run_url_capture, args=(url,), daemon=True, name=f"multicam-{self.camera_id}"
        )
        self._worker_thread.start()

    def start_local_file_stream(self, file_path: Path):
        """Start background worker to stream local video file with seamless continuous auto-looping."""
        self.stop_stream()
        self.stopped_by_user = False
        self._stop_event.clear()
        self._begin_session()
        self.source_type = "local_file"
        self.source_label = file_path.name
        self.video_name = file_path.name
        self.status = "playing"
        self.error_message = None

        self._worker_thread = threading.Thread(
            target=self._run_file_capture, args=(str(file_path),), daemon=True, name=f"multicam-{self.camera_id}"
        )
        self._worker_thread.start()

    def stop_stream(self):
        """Stop running worker thread or phone connection cleanly.

        Sets `stopped_by_user`, which the phone-ingest handlers check before
        accepting a frame. Without that latch a connected phone simply kept
        pushing frames and the feed restarted itself.
        """
        self.stopped_by_user = True
        self._stop_event.set()
        worker, self._worker_thread = self._worker_thread, None
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.0)

        # The socket is closed by whichever coroutine owns it; calling
        # asyncio.run() here raised RuntimeError inside the running loop and
        # the close never happened. Dropping the reference is enough - the
        # ingest handler sees the stop flag and returns.
        self.active_ws = None

        self.status = "stopped"
        self.fps = 0.0
        self.active_tracks = []
        self.active_plates = []
        self.error_message = "Stream stopped by user"

        img = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(img, f"{self.camera_id} - STREAM STOPPED", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 100, 255), 2)
        cv2.putText(img, "Click 'Resume' or connect phone to restart", (50, 285),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1)
        _, buf = cv2.imencode(".jpg", img)
        with self._new_frame_event:
            self.latest_jpeg = buf.tobytes()
            self._new_frame_event.notify_all()

    def _run_url_capture(self, url: str):
        """Worker loop for network URL streams with auto-reconnect/loop."""
        logger.info(f"Starting URL stream for {self.camera_id}: {url}")
        while not self._stop_event.is_set() and not self.stopped_by_user:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                self.status = "disconnected"
                self.error_message = f"Cannot open stream: {url}"
                time.sleep(2.0)
                continue

            self.status = "playing"
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_period = 1.0 / fps_src if fps_src > 0 else 0.033

            while not self._stop_event.is_set() and not self.stopped_by_user:
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    self.loops += 1
                    break

                self.update_frame(frame, video_time_s=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

                slack = frame_period - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)

            cap.release()
            if not self._stop_event.is_set():
                time.sleep(0.5)

    def _run_file_capture(self, filepath: str):
        """Worker loop for video file with seamless continuous looping."""
        logger.info(f"Starting file stream for {self.camera_id}: {filepath}")
        while not self._stop_event.is_set() and not self.stopped_by_user:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                self.status = "error"
                self.error_message = f"Cannot open file: {filepath}"
                break

            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_period = 1.0 / fps_src if fps_src > 0 else 0.033

            while not self._stop_event.is_set() and not self.stopped_by_user:
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    # Seamless continuous loop back to start
                    self.loops += 1
                    break

                self.update_frame(frame, video_time_s=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

                slack = frame_period - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)

            cap.release()

    def get_status(self) -> dict:
        """Return full slot status and ANPR stats."""
        if self.source_type == "phone_web" and self.status == "playing":
            if time.time() - self.last_received_at > 5.5:
                self.status = "disconnected"
                self.error_message = "Phone disconnected or paused"
                self.fps = 0.0

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "road": self.road,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "status": self.status,
            "source_type": self.source_type,
            "source_label": self.source_label,
            "video_name": self.video_name or f"Prerecorded Video ({self.camera_id})",
            "fps": round(self.fps, 1),
            "frame_count": self.frame_count,
            "resolution": self.resolution or "1280x720",
            "video_time_s": round(self.video_time_s, 1),
            "loops": self.loops,
            "timestamp": now_str,
            "has_feed": self.latest_jpeg is not None,
            "error_message": self.error_message,
            "anpr_enabled": self.anpr_enabled,
            "plates_detected_count": len(self.published_tracks),
            "recent_plates": list(self.recent_plates)[:5],
            "active_plate_boxes": len(self.active_plates),
            "detections_skipped": self.detections_skipped,
            "detections_run": self.detections_run,
            "detect_every": self.detect_every,
            "seconds_since_frame": (round(time.time() - self.last_received_at, 1)
                                    if self.last_received_at > 0 else None),
            "detect_failures": self.detect_failures,
            "last_detect_error": self.last_detect_error,
        }


class CentralRegistry:
    """Cross-camera identity and journey reconstruction for the live feeds.

    Each slot reports plates independently. This is the piece that makes the
    three feeds one network: it merges OCR variants of the same plate using the
    confusion-aware similarity already used by the batch pipeline, then orders
    the sightings into a journey and measures each leg.

    Distances are great-circle between the two camera positions scaled by an
    urban road factor, and every leg says so via `distance_source`. Speeds are
    derived from MEASURED wall-clock arrival times, not from any configured
    schedule.
    """

    ROAD_FACTOR = 1.25          # urban road length vs straight line
    MAX_PLAUSIBLE_KMH = 140.0
    WARN_KMH = 100.0

    def __init__(self):
        self._lock = threading.Lock()
        self._sightings: list = []          # every confirmed sighting, in order
        self._positions: dict = {}          # camera_id -> (lat, lon, name)

    def register_camera(self, camera_id: str, lat: float, lon: float, name: str) -> None:
        self._positions[camera_id] = (lat, lon, name)

    def reset(self) -> None:
        with self._lock:
            self._sightings.clear()

    def record(self, sighting: dict) -> None:
        """Store a confirmed sighting, replacing an earlier one for the same track."""
        with self._lock:
            key = (sighting["camera_id"], sighting["track_id"])
            self._sightings = [s for s in self._sightings
                               if (s["camera_id"], s["track_id"]) != key]
            self._sightings.append(sighting)
        # Also persist it, so the GIS map, Vehicle Tracking, Analytics and
        # Observations tabs show this session instead of the last batch run.
        # Queued and written by a background thread; a database problem can
        # never slow down or stop a camera.
        LIVE_RUN.record(sighting)

    @staticmethod
    def _haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
        import math
        r = 6371.0088
        p1, p2 = math.radians(a_lat), math.radians(b_lat)
        dp, dl = math.radians(b_lat - a_lat), math.radians(b_lon - a_lon)
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(h))

    def _canonical(self, plates: list) -> dict:
        """Union OCR variants of one plate. Highest-confidence spelling wins."""
        from ..identity.association import plate_similarity, MIN_SIMILARITY
        parent = {p: p for p in plates}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Weight by TOTAL evidence (how often a spelling was read, and how
        # strongly), not by the single best frame. A one-off misread that
        # happened to score high should not rename a vehicle that a dozen other
        # sightings agree on.
        best = {}
        for p in plates:
            reads = [s["confidence"] for s in self._sightings if s["plate"] == p]
            best[p] = sum(reads)
        for i, p1 in enumerate(plates):
            for p2 in plates[i + 1:]:
                if plate_similarity(p1, p2) < MIN_SIMILARITY:
                    continue
                r1, r2 = find(p1), find(p2)
                if r1 == r2:
                    continue
                keep, drop = (r1, r2) if best[r1] >= best[r2] else (r2, r1)
                parent[drop] = keep
        return {p: find(p) for p in plates}

    def journeys(self) -> list:
        """Every vehicle the network has seen, multi-camera ones first."""
        with self._lock:
            sightings = list(self._sightings)
        if not sightings:
            return []

        canonical = self._canonical(sorted({s["plate"] for s in sightings}))
        grouped: dict = {}
        for s in sightings:
            grouped.setdefault(canonical[s["plate"]], []).append(s)

        out = []
        for plate, group in grouped.items():
            group.sort(key=lambda s: s["observed_at"])

            # The journey is the order in which the vehicle FIRST reached each
            # camera. A camera that keeps re-detecting the same vehicle (a
            # looping demo clip, or a vehicle dwelling in view) would otherwise
            # produce a zig-zag route of dozens of meaningless hops.
            first_visit: dict = {}
            for s in group:
                first_visit.setdefault(s["camera_id"], s)
            visits = sorted(first_visit.values(), key=lambda s: s["observed_at"])

            route = [s["camera_id"] for s in visits]
            legs = [self._leg(a, b) for a, b in zip(visits, visits[1:])]
            cams = set(first_visit)
            out.append({
                "plate": plate,
                "variants": sorted({s["plate"] for s in group} - {plate}),
                "camera_count": len(cams),
                "sighting_count": len(group),
                "route": route,
                "route_display": " -> ".join(route),
                "first_seen": group[0]["observed_at"],
                "last_seen": group[-1]["observed_at"],
                "confidence": round(sum(s["confidence"] for s in group) / len(group), 3),
                "vehicle_type": group[-1]["vehicle_type"],
                "sightings": group,
                "legs": legs,
                "status": "MULTI_CAMERA" if len(cams) > 1 else "SINGLE_CAMERA",
            })

        out.sort(key=lambda v: (-v["camera_count"], -v["confidence"]))
        return out

    def _leg(self, a: dict, b: dict) -> dict:
        t1 = datetime.fromisoformat(a["observed_at"])
        t2 = datetime.fromisoformat(b["observed_at"])
        minutes = (t2 - t1).total_seconds() / 60.0
        km = round(self._haversine_km(a["latitude"], a["longitude"],
                                      b["latitude"], b["longitude"]) * self.ROAD_FACTOR, 2)
        speed = (km / (minutes / 60.0)) if minutes > 0 else float("inf")

        if minutes <= 0:
            verdict = "IMPLAUSIBLE"
            reason = f"Arrival at {b['camera_id']} is not after departure from {a['camera_id']}."
        elif speed > self.MAX_PLAUSIBLE_KMH:
            verdict = "IMPLAUSIBLE"
            reason = (f"{km:.2f} km in {minutes:.2f} min implies {speed:.0f} km/h, "
                      f"above the {self.MAX_PLAUSIBLE_KMH:.0f} km/h feasibility limit.")
        elif speed > self.WARN_KMH:
            verdict = "FAST"
            reason = f"{km:.2f} km in {minutes:.2f} min implies {speed:.0f} km/h."
        else:
            verdict = "PLAUSIBLE"
            reason = f"{km:.2f} km in {minutes:.2f} min ({speed:.0f} km/h)."

        return {
            "from_camera": a["camera_id"], "to_camera": b["camera_id"],
            "departed": a["observed_at"], "arrived": b["observed_at"],
            "travel_minutes": round(minutes, 2),
            "road_distance_km": km,
            "implied_speed_kmh": round(speed, 1) if minutes > 0 else None,
            "distance_source": "great-circle x1.25 estimate",
            "verdict": verdict, "reason": reason,
        }

    def summary(self) -> dict:
        j = self.journeys()
        multi = [v for v in j if v["camera_count"] > 1]
        return {
            "vehicles": len(j),
            "multi_camera_vehicles": len(multi),
            "sightings": sum(v["sighting_count"] for v in j),
            "cameras_with_sightings": len({s["camera_id"] for v in j for s in v["sightings"]}),
        }


REGISTRY = CentralRegistry()

# Mirrors the live session into a database run so the GIS map, Vehicle Tracking,
# Analytics and Observations tabs show what the cameras are seeing now rather
# than the newest batch run. Writes happen on its own thread and can fail
# without affecting any feed.
LIVE_RUN = LiveRunWriter(journeys_fn=REGISTRY.journeys)


class MultiCamManager:
    """Central manager for the monitor slots (one per configured camera)."""

    def __init__(self):
        self.slots: Dict[str, CameraSlot] = {}
        for cfg in CAMERA_CONFIGS:
            self.slots[cfg["camera_id"]] = CameraSlot(
                camera_id=cfg["camera_id"],
                name=cfg["name"],
                road=cfg["road"],
                lat=cfg["latitude"],
                lon=cfg["longitude"],
                default_video=cfg["default_video"],
            )
            REGISTRY.register_camera(cfg["camera_id"], cfg["latitude"],
                                     cfg["longitude"], cfg["name"])
            LIVE_RUN.register_camera(cfg["camera_id"], cfg["name"], cfg["road"],
                                     cfg["latitude"], cfg["longitude"])

    def get_slot(self, camera_id: str) -> Optional[CameraSlot]:
        return self.slots.get(camera_id)

    def get_all_status(self) -> list[dict]:
        return [self.slots[cfg["camera_id"]].get_status() for cfg in CAMERA_CONFIGS]

    def stop_all(self):
        LIVE_RUN.stop()
        for slot in self.slots.values():
            slot.stop_stream()

    def get_all_detections(self) -> list[dict]:
        """Aggregate recent plate detections across all 3 cameras sorted chronologically."""
        all_dets = []
        for slot in self.slots.values():
            all_dets.extend(list(slot.recent_plates))
        # Sort on the ISO instant. The previous "%H:%M:%S" key reordered the
        # feed wrongly across midnight and could not be compared to anything.
        return sorted(all_dets, key=lambda d: d.get("observed_at", ""), reverse=True)[:30]


# Singleton instance
MULTICAM = MultiCamManager()

# Router
router = APIRouter(tags=["Multi-Camera"])


# ------------------------------------------------------------- Ingest Endpoints


@router.websocket("/ws/multi-camera/ingest/{camera_id}")
async def ws_camera_ingest(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for Phone Web Streamer to stream frames continuously."""
    slot = MULTICAM.get_slot(camera_id)
    if not slot:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    slot.active_ws = websocket
    slot.source_type = "phone_web"
    slot.source_label = "Android Phone (Web Streamer)"
    slot.status = "playing"
    slot._begin_session()
    slot.error_message = None
    logger.info(f"Phone connected to {camera_id} via WebSocket")

    try:
        while True:
            msg = await websocket.receive()

            # A frame must never revive a feed the operator stopped. `resume()`
            # used to run on every inbound frame, so pressing Stop in the
            # dashboard was undone by the very next frame the phone sent and the
            # feed never actually stopped. Only an explicit resume restarts it;
            # while stopped, frames are dropped and the phone is told to stand
            # down so it is not encoding and uploading into a void.
            if slot.stopped_by_user:
                if "bytes" in msg or "text" in msg:
                    try:
                        await websocket.send_text(json.dumps({"type": "stopped"}))
                    except Exception:
                        pass
                    if "text" in msg and msg["text"]:
                        try:
                            if json.loads(msg["text"]).get("type") == "resume":
                                slot.resume()
                        except Exception:
                            pass
                continue

            if "bytes" in msg and msg["bytes"]:
                await run_in_threadpool(slot.submit_jpeg, msg["bytes"])
            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                msg_type = data.get("type")
                if msg_type == "resume":
                    slot.resume()
                elif msg_type == "frame":
                    frame_b64 = data.get("data", "")
                    if "," in frame_b64:
                        frame_b64 = frame_b64.split(",", 1)[1]
                    raw = base64.b64decode(frame_b64)
                    await run_in_threadpool(
                        slot.submit_jpeg, raw,
                        float(data.get("time", 0.0)),
                        data.get("video_name", ""),
                    )
                elif msg_type == "meta":
                    if data.get("video_name"):
                        slot.video_name = data["video_name"]
                    if "loops" in data:
                        slot.loops = int(data["loops"])
                    # Binary frames carry no metadata, so the phone reports its
                    # measured playback position here instead of the slot
                    # sitting at 0.0 for the whole session.
                    if "time" in data:
                        slot.video_time_s = float(data["time"])
                elif msg_type == "stop":
                    slot.stop_stream()
                    break
    except WebSocketDisconnect:
        logger.info(f"Phone disconnected from {camera_id}")
        slot.mark_disconnected("Phone connection closed")
    except Exception as e:
        logger.warning(f"Error in ws_camera_ingest {camera_id}: {e}")
        slot.mark_disconnected(str(e))
    finally:
        slot.active_ws = None


@router.post("/api/multi-camera/ingest/{camera_id}")
async def http_camera_ingest(camera_id: str, request: Request):
    """HTTP POST frame ingest fallback for phones."""
    slot = MULTICAM.get_slot(camera_id)
    if not slot:
        raise HTTPException(404, f"Unknown camera {camera_id}")

    if slot.stopped_by_user:
        # Same rule as the WebSocket path: a stopped feed stays stopped, and the
        # phone is told so it can stop uploading.
        return {"status": "stopped", "camera_id": camera_id}

    content_type = request.headers.get("content-type", "")
    slot.source_type = "phone_web"
    slot.source_label = "Android Phone (HTTP Ingest)"

    if "image/jpeg" in content_type or "application/octet-stream" in content_type:
        body = await request.body()
        accepted = await run_in_threadpool(slot.submit_jpeg, body)
        return {"status": "ok" if accepted else "busy", "camera_id": camera_id}
    else:
        payload = await request.json()
        frame_b64 = payload.get("data", "")
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        raw = base64.b64decode(frame_b64)
        accepted = await run_in_threadpool(
            slot.submit_jpeg, raw,
            float(payload.get("time", 0.0)),
            payload.get("video_name", ""),
        )
        return {"status": "ok" if accepted else "busy", "camera_id": camera_id}


# ------------------------------------------------------------- Stream & Status


def generate_mjpeg_stream(slot: CameraSlot):
    """Generator yielding multipart MJPEG frames for browser <img> tags at 25-30 FPS."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    placeholder = None

    while True:
        with slot._new_frame_event:
            if slot._new_frame_event.wait(timeout=0.1) and slot.latest_jpeg:
                frame_data = slot.latest_jpeg
            else:
                frame_data = slot.latest_jpeg

        if frame_data:
            yield boundary + frame_data + b"\r\n"
        else:
            if placeholder is None:
                img = np.zeros((480, 854, 3), dtype=np.uint8)
                cv2.putText(img, f"{slot.camera_id} - WAITING FOR PHONE STREAM", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(img, "Open Phone Streamer on Android Phone", (40, 290),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                _, buf = cv2.imencode(".jpg", img)
                placeholder = buf.tobytes()
            yield boundary + placeholder + b"\r\n"
            time.sleep(0.1)


@router.get("/api/multi-camera/stream/{camera_id}")
def get_camera_mjpeg_stream(camera_id: str):
    """Browser-compatible MJPEG stream for one camera."""
    slot = MULTICAM.get_slot(camera_id)
    if not slot:
        raise HTTPException(404, f"Unknown camera {camera_id}")

    return StreamingResponse(
        generate_mjpeg_stream(slot),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Connection": "close"},
    )


@router.get("/api/multi-camera/frame/{camera_id}")
def get_camera_single_frame(camera_id: str):
    """Single latest JPEG frame."""
    slot = MULTICAM.get_slot(camera_id)
    if not slot or not slot.latest_jpeg:
        return Response(status_code=204)
    return Response(
        content=slot.latest_jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/api/multi-camera/status")
def get_multi_camera_status():
    """Current status of all 3 camera feeds in fixed order."""
    cams = MULTICAM.get_all_status()
    return {
        "cameras": cams,
        "detections": MULTICAM.get_all_detections(),
        "network": REGISTRY.summary(),
        "live_run": LIVE_RUN.status.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "all_connected": all(c["status"] == "playing" for c in cams),
    }


@router.get("/api/multi-camera/vehicles")
def get_multi_camera_vehicles():
    """Cross-camera journeys reconstructed from the three live feeds."""
    return REGISTRY.journeys()


@router.post("/api/multi-camera/reset-registry")
def reset_multi_camera_registry():
    """Clear the centralised sighting history (does not touch the streams)."""
    REGISTRY.reset()
    return REGISTRY.summary()


@router.get("/api/multi-camera/detections")
def get_multi_camera_detections():
    """Live plate detections stream across all 3 cameras."""
    return MULTICAM.get_all_detections()


@router.get("/api/multi-camera/info")
def get_multi_camera_info(request: Request):
    """Network connection info and Phone URLs for Phone 1, 2, 3."""
    host = request.headers.get("host", "localhost:8000").split(":")[0]
    port = request.url.port or 8000
    local_ips = get_local_ips()
    primary_ip = local_ips[0] if local_ips else host

    phone_urls = {
        "CAM_001": f"http://{primary_ip}:{port}/phone-streamer?cam=CAM_001",
        "CAM_002": f"http://{primary_ip}:{port}/phone-streamer?cam=CAM_002",
        "CAM_003": f"http://{primary_ip}:{port}/phone-streamer?cam=CAM_003",
    }

    return {
        "primary_ip": primary_ip,
        "local_ips": local_ips,
        "port": port,
        "phone_urls": phone_urls,
        "cameras": MULTICAM.get_all_status(),
    }


@router.post("/api/multi-camera/connect-url")
def connect_url_stream(camera_id: str, url: str, video_name: str = ""):
    """Connect a network stream URL (VLC RTSP, ScreenStream, HTTP video) to a camera slot."""
    slot = MULTICAM.get_slot(camera_id)
    if not slot:
        raise HTTPException(404, f"Unknown camera {camera_id}")

    slot.start_url_stream(url, video_name=video_name)
    return slot.get_status()


@router.post("/api/multi-camera/connect-local")
def connect_local_stream(camera_id: str, video_file: str = ""):
    """Connect local video file (plays in seamless continuous loop)."""
    slot = MULTICAM.get_slot(camera_id)
    if not slot:
        raise HTTPException(404, f"Unknown camera {camera_id}")

    file_name = video_file or slot.default_video
    video_path = ROOT / "data" / "videos" / file_name
    if not file_name or not video_path.exists():
        # Fall back to a DIFFERENT clip per camera rather than handing every
        # slot the same file, which made three feeds look identical and made
        # cross-camera matching meaningless.
        pool = sorted((ROOT / "data" / "videos").glob("*.mp4"))
        if not pool:
            raise HTTPException(404, f"No videos in data/videos/ (wanted {file_name})")
        idx = [c["camera_id"] for c in CAMERA_CONFIGS].index(camera_id)
        video_path = pool[idx % len(pool)]

    slot.start_local_file_stream(video_path)
    return slot.get_status()


@router.post("/api/multi-camera/stop/{camera_id}")
def stop_camera_stream(camera_id: str):
    """Stop stream for an individual camera."""
    slot = MULTICAM.get_slot(camera_id)
    if not slot:
        raise HTTPException(404, f"Unknown camera {camera_id}")
    slot.stop_stream()
    return slot.get_status()


@router.post("/api/multi-camera/stop-all")
def stop_all_camera_streams():
    """Stop every camera stream at once."""
    MULTICAM.stop_all()
    return {"status": "all_stopped", "cameras": MULTICAM.get_all_status()}


# ------------------------------------------------------------- Mobile Phone Streamer HTML


PHONE_STREAMER_HTML_PATH = Path(__file__).resolve().parent / "phone_streamer.html"


@router.get("/phone-streamer", response_class=HTMLResponse)
def phone_streamer_page(cam: str = "CAM_001"):
    """Serve the Phone Web Streamer page for Android phones."""
    if PHONE_STREAMER_HTML_PATH.exists():
        return HTMLResponse(PHONE_STREAMER_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Phone Streamer file missing</h1>", status_code=500)
