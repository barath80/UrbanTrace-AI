"""Live stream capture for the camera network.

One reader thread per feed. Each reader owns a single `cv2.VideoCapture`, keeps
it open, and hands frames to the ingest service at the analysis rate rather
than the camera's rate.

What keeps the cost of a hundred feeds down:

* `grab()` runs for every frame so the stream stays in sync, but the costly
  part (colour conversion and copy, `retrieve()`) runs only for frames that are
  due. At 25 fps source and 6 fps analysis that is roughly three frames in four
  never converted.
* A motion gate compares a 64x36 greyscale thumbnail with the previous one. An
  unchanged scene is not sent to the detectors, except for a keyframe every
  `idle_keyframe_s`.
* Frames are downscaled to `max_frame_width` before they are queued.
* Opens are gated so 100+ feeds do not all connect in the same instant.
* A feed that fails or stalls is reopened with exponential backoff and jitter.

`grab()` still decodes. To cut decode cost itself, point each camera at its
low-resolution sub-stream.
"""
from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from .config import SCALE

logger = logging.getLogger("urbantrace.scale.capture")

STATES = ("connecting", "live", "reconnecting", "stopped")


def redact_url(url: str) -> str:
    """Hide credentials in a stream URL before it is logged or returned."""
    if not url or "@" not in url:
        return url
    try:
        p = urlsplit(url)
        if p.username or p.password:
            host = p.hostname or ""
            if p.port:
                host += f":{p.port}"
            return urlunsplit((p.scheme, f"***@{host}", p.path, p.query, p.fragment))
    except ValueError:
        pass
    return re.sub(r"//[^/@]*@", "//***@", url)


def configure_ffmpeg() -> None:
    """Set transport and timeouts for FFmpeg-backed captures.

    Must run before the first capture is opened. An existing value is kept so a
    deployment can override it.
    """
    if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
        return
    opts = [f"rtsp_transport;{SCALE.rtsp_transport}",
            f"stimeout;{SCALE.open_timeout_ms * 1000}"]
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)


def _open(source: str):
    """Open a feed. Numeric strings are device indexes, anything else a URL/path."""
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, SCALE.open_timeout_ms,
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, SCALE.read_timeout_ms]
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def is_file_source(source: str) -> bool:
    return not source.isdigit() and "://" not in source


class MotionGate:
    """Cheap scene-change test on a tiny greyscale thumbnail."""

    SIZE = (64, 36)

    def __init__(self, pixel_delta: int, min_fraction: float):
        self.pixel_delta = pixel_delta
        self.min_fraction = min_fraction
        self._prev: np.ndarray | None = None

    def moved(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, self.SIZE, interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev, self._prev = self._prev, grey
        if prev is None:
            return True
        changed = np.count_nonzero(cv2.absdiff(grey, prev) > self.pixel_delta)
        return changed / grey.size >= self.min_fraction


@dataclass
class FeedHealth:
    camera_id: str
    source: str
    state: str = "connecting"
    frames_read: int = 0
    frames_submitted: int = 0
    frames_static: int = 0
    reconnects: int = 0
    consecutive_failures: int = 0
    last_frame_at: float = 0.0
    last_error: str | None = None
    source_fps: float = 0.0
    width: int = 0
    height: int = 0
    opened_at: float = 0.0
    _rate_marks: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        now = time.time()
        return {
            "camera_id": self.camera_id,
            "source": redact_url(self.source),
            "state": self.state,
            "frames_read": self.frames_read,
            "frames_submitted": self.frames_submitted,
            "frames_skipped_static": self.frames_static,
            "reconnects": self.reconnects,
            "consecutive_failures": self.consecutive_failures,
            "seconds_since_frame": round(now - self.last_frame_at, 1)
            if self.last_frame_at else None,
            "source_fps": round(self.source_fps, 1),
            "resolution": f"{self.width}x{self.height}" if self.width else None,
            "last_error": self.last_error,
        }


class StreamReader(threading.Thread):
    """Reads one feed until stopped, reconnecting as needed."""

    def __init__(self, camera_id: str, source: str,
                 submit: Callable[..., bool],
                 fps_provider: Callable[[], float],
                 open_gate: threading.Semaphore):
        super().__init__(name=f"cap-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = source
        self._submit = submit
        self._fps = fps_provider
        self._gate = open_gate
        self._stop_evt = threading.Event()
        self.health = FeedHealth(camera_id, source)
        self._gatekeeper = MotionGate(SCALE.motion_pixel_delta,
                                      SCALE.motion_min_fraction) if SCALE.motion_gate else None

    def stop(self) -> None:
        self._stop_evt.set()

    # ------------------------------------------------------------ run

    def run(self) -> None:
        backoff = SCALE.reconnect_min_s
        while not self._stop_evt.is_set():
            cap = self._connect()
            if cap is None:
                self.health.state = "reconnecting"
                self.health.reconnects += 1
                # Full jitter so feeds that dropped together do not return together.
                if self._stop_evt.wait(random.uniform(0, backoff)):
                    break
                backoff = min(SCALE.reconnect_max_s, backoff * 2.0)
                continue
            backoff = SCALE.reconnect_min_s
            ended_cleanly = False
            try:
                ended_cleanly = self._pump(cap)
            except Exception as e:                      # noqa: BLE001 - keep the feed alive
                self.health.last_error = f"{type(e).__name__}: {e}"
                logger.warning("%s read loop error: %s", self.camera_id, self.health.last_error)
            finally:
                cap.release()
            if self._stop_evt.is_set():
                break
            if ended_cleanly and is_file_source(self.source) and not SCALE.loop_files:
                break
            self.health.state = "reconnecting"
            self.health.reconnects += 1
            self._stop_evt.wait(SCALE.reconnect_min_s)
        self.health.state = "stopped"

    def _connect(self):
        self.health.state = "connecting"
        with self._gate:
            if self._stop_evt.is_set():
                return None
            cap = _open(self.source)
        if not cap.isOpened():
            cap.release()
            self.health.consecutive_failures += 1
            self.health.last_error = "could not open stream"
            return None
        self.health.opened_at = time.time()
        self.health.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.health.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.health.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return cap

    def _pump(self, cap) -> bool:
        """Read until the feed ends or stalls. True when a file reached its end."""
        h = self.health
        next_due = 0.0
        last_sent = 0.0
        frame_number = 0
        file_source = is_file_source(self.source)
        while not self._stop_evt.is_set():
            if not cap.grab():
                if file_source:
                    if SCALE.loop_files:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    return True
                h.consecutive_failures += 1
                h.last_error = "read failed"
                return False
            now = time.time()
            h.frames_read += 1
            frame_number += 1
            h.last_frame_at = now
            h.consecutive_failures = 0
            h.state = "live"

            fps = max(0.01, self._fps())
            mono = time.monotonic()
            if mono < next_due:
                if file_source:
                    # A file would otherwise be consumed as fast as it decodes.
                    time.sleep(min(0.05, next_due - mono))
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            next_due = mono + 1.0 / fps

            if self._gatekeeper is not None and not self._gatekeeper.moved(frame):
                if now - last_sent < SCALE.idle_keyframe_s:
                    h.frames_static += 1
                    continue

            frame = self._shrink(frame)
            if self._submit(self.camera_id, frame,
                            captured_at=datetime.now(timezone.utc),
                            video_time_s=frame_number / (h.source_fps or 25.0),
                            frame_number=frame_number):
                h.frames_submitted += 1
            last_sent = now
        return False

    @staticmethod
    def _shrink(frame: np.ndarray) -> np.ndarray:
        limit = SCALE.max_frame_width
        w = frame.shape[1]
        if limit <= 0 or w <= limit:
            return frame
        scale = limit / w
        return cv2.resize(frame, (limit, int(frame.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)


class CaptureManager:
    """Owns the readers for the cameras this process is responsible for."""

    def __init__(self, submit: Callable[..., bool], fps_provider: Callable[[], float]):
        self._submit = submit
        self._fps = fps_provider
        self._readers: dict[str, StreamReader] = {}
        self._gate = threading.Semaphore(max(1, SCALE.max_concurrent_opens))
        self._lock = threading.Lock()
        configure_ffmpeg()

    def reconcile(self, wanted: dict[str, str]) -> dict:
        """Make the running readers match `wanted` ({camera_id: source}).

        Only the difference is touched, so a rebalance that moves a few cameras
        does not interrupt the rest.
        """
        started = stopped = 0
        with self._lock:
            for cid in list(self._readers):
                r = self._readers[cid]
                if cid not in wanted or wanted[cid] != r.source or not r.is_alive():
                    r.stop()
                    del self._readers[cid]
                    stopped += 1
            for cid, src in wanted.items():
                if cid in self._readers or not src:
                    continue
                r = StreamReader(cid, src, self._submit, self._fps, self._gate)
                self._readers[cid] = r
                r.start()
                started += 1
        return {"started": started, "stopped": stopped, "running": len(self._readers)}

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            readers = list(self._readers.values())
            self._readers.clear()
        for r in readers:
            r.stop()
        deadline = time.time() + timeout
        for r in readers:
            r.join(max(0.0, deadline - time.time()))

    def feeds(self) -> list[dict]:
        with self._lock:
            return [r.health.to_dict() for r in self._readers.values()]

    def summary(self) -> dict:
        with self._lock:
            hs = [r.health for r in self._readers.values()]
        by_state = {s: 0 for s in STATES}
        for h in hs:
            by_state[h.state] = by_state.get(h.state, 0) + 1
        return {
            "feeds": len(hs),
            "by_state": by_state,
            "frames_read": sum(h.frames_read for h in hs),
            "frames_submitted": sum(h.frames_submitted for h in hs),
            "frames_skipped_static": sum(h.frames_static for h in hs),
            "reconnects": sum(h.reconnects for h in hs),
        }
