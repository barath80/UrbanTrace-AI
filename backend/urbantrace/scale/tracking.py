"""Per-camera ByteTrack over a shared detector.

The single most important structural decision for a large network is here.

Ultralytics' convenience API, `model.track(persist=True)`, keeps tracker state
*inside the model object*. That is fine for one video and wrong for a city:
either every camera needs its own copy of the network (1000 x ~40 MB of
weights, which no GPU holds), or cameras share one model and their track
identities bleed into each other.

So the two concerns are separated:

    detection  -- ONE shared model, batched across cameras (see inference.py)
    tracking   -- one lightweight BYTETracker per camera, held here

A BYTETracker instance is a few kilobytes of Kalman state, so ten thousand of
them cost less memory than a single extra copy of the detector. Idle cameras
are evicted so a network that cycles through cameras does not leak.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import SCALE

# Ultralytics' own tracker defaults. Kept explicit rather than read from the
# packaged YAML so that a library upgrade cannot silently change tracking
# behaviour across an entire deployment.
BYTETRACK_ARGS = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.10,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.80,
    "fuse_score": True,
}

# COCO classes treated as vehicles.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}


class Detections:
    """Minimal Results-like view that BYTETracker accepts.

    BYTETracker needs `xywh`, `conf`, `cls`, `len()` and boolean indexing. It
    does NOT need image tensors or a full Results object, so passing this keeps
    detection payloads small enough to hand between threads cheaply.
    """

    __slots__ = ("xywh", "conf", "cls", "xyxy")

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.xyxy = xyxy
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        # xywh is centre-x, centre-y, width, height.
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = np.stack([xyxy[:, 0] + w / 2, xyxy[:, 1] + h / 2, w, h], axis=1)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, mask) -> "Detections":
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask])

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))


@dataclass
class Track:
    """One tracked vehicle in one camera, for one frame."""
    track_id: int
    cls_name: str
    confidence: float
    box: tuple            # (x1, y1, x2, y2)
    detection_index: int  # row in the Detections passed in, for binding plates


class _TrackerEntry:
    __slots__ = ("tracker", "last_used", "frames")

    def __init__(self, tracker):
        self.tracker = tracker
        self.last_used = time.time()
        self.frames = 0


class MultiCameraTracker:
    """Holds one independent ByteTrack per camera.

    Thread-safe: the inference workers call `update()` concurrently for
    different cameras. A per-camera lock keeps one camera's Kalman state
    consistent without serialising the whole network behind a global lock.
    """

    def __init__(self, idle_evict_seconds: float = 900.0, max_cameras: int = 20_000):
        self._entries: dict[str, _TrackerEntry] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self.idle_evict_seconds = idle_evict_seconds
        self.max_cameras = max_cameras

    def _args(self):
        from types import SimpleNamespace
        return SimpleNamespace(**BYTETRACK_ARGS)

    @staticmethod
    def _tracker_class():
        """BYTETracker with the global id counter neutralised.

        Ultralytics allocates track ids from `BaseTrack._count`, a CLASS
        attribute shared by every tracker in the process, and `BYTETracker
        .__init__` calls `reset_id()` on it. In a single-video script that is
        harmless. Across a camera network it is a correctness bug: bringing one
        camera online resets the counter, so cameras that are already running
        reissue ids they are still using and two different vehicles end up
        sharing a track id.

        Measured on this build: after one camera had allocated 3 ids, creating
        a second camera's tracker set the counter back to 1.

        Overriding `reset_id` to do nothing makes ids monotonic and unique
        across the whole deployment, which is what a centralised system wants
        anyway. `next_id` is also serialised, because several inference workers
        allocate ids concurrently and `_count += 1` is not atomic.
        """
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.basetrack import BaseTrack

        if not getattr(BaseTrack, "_ut_id_patched", False):
            id_lock = threading.Lock()

            def next_id() -> int:
                with id_lock:
                    BaseTrack._count += 1
                    return BaseTrack._count

            BaseTrack.next_id = staticmethod(next_id)
            BaseTrack.reset_id = staticmethod(lambda: None)
            BaseTrack._ut_id_patched = True

        class _NetworkByteTracker(BYTETracker):
            def reset_id(self):
                return None

        return _NetworkByteTracker

    def _entry(self, camera_id: str) -> tuple[_TrackerEntry, threading.Lock]:
        with self._guard:
            e = self._entries.get(camera_id)
            if e is None:
                e = _TrackerEntry(self._tracker_class()(self._args()))
                self._entries[camera_id] = e
                self._locks[camera_id] = threading.Lock()
                if len(self._entries) > self.max_cameras:
                    self._evict_locked(force=True)
            return e, self._locks[camera_id]

    def update(self, camera_id: str, dets: Detections) -> list[Track]:
        """Advance one camera's tracker by one frame."""
        entry, lock = self._entry(camera_id)
        with lock:
            entry.last_used = time.time()
            entry.frames += 1
            try:
                rows = entry.tracker.update(dets)
            except Exception:
                # A tracker that has become inconsistent is reset rather than
                # taking the camera offline; identities restart, which is
                # visible in the data, instead of the feed silently dying.
                entry.tracker = self._tracker_class()(self._args())
                return []

        out: list[Track] = []
        if rows is None or len(rows) == 0:
            return out
        for r in np.asarray(rows, dtype=np.float32):
            x1, y1, x2, y2, tid, score, cls_id, idx = (
                float(r[0]), float(r[1]), float(r[2]), float(r[3]),
                int(r[4]), float(r[5]), int(r[6]), int(r[7]),
            )
            out.append(Track(
                track_id=tid,
                cls_name=VEHICLE_CLASSES.get(cls_id, str(cls_id)),
                confidence=round(score, 4),
                box=(x1, y1, x2, y2),
                detection_index=idx,
            ))
        return out

    def reset(self, camera_id: str) -> None:
        with self._guard:
            self._entries.pop(camera_id, None)
            self._locks.pop(camera_id, None)

    def evict_idle(self) -> int:
        with self._guard:
            return self._evict_locked()

    def _evict_locked(self, force: bool = False) -> int:
        cutoff = time.time() - self.idle_evict_seconds
        stale = [c for c, e in self._entries.items() if e.last_used < cutoff]
        if force and not stale:
            # Nothing is idle but the cap was hit: drop the least recently used.
            stale = sorted(self._entries, key=lambda c: self._entries[c].last_used)[:1]
        for c in stale:
            self._entries.pop(c, None)
            self._locks.pop(c, None)
        return len(stale)

    def stats(self) -> dict:
        with self._guard:
            return {
                "cameras_tracked": len(self._entries),
                "frames_total": sum(e.frames for e in self._entries.values()),
            }


def containment(inner: tuple, outer: tuple) -> float:
    """Fraction of `inner`'s area lying inside `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (iw * ih) / area if area > 0 else 0.0


def bind_plates_to_tracks(plate_boxes: list, tracks: list, min_containment: float = 0.60) -> list:
    """Assign each plate box to the smallest track box that contains it.

    Smallest-containing-box wins, which resolves the common case of a plate
    falling inside both a car box and an overlapping bus behind it. Plates that
    match no track are returned with `None`, so the caller can decide what to
    do with them rather than having them silently dropped.
    """
    pairs = []
    for pb in plate_boxes:
        best, best_area = None, float("inf")
        for t in tracks:
            if containment(pb, t.box) >= min_containment:
                area = (t.box[2] - t.box[0]) * (t.box[3] - t.box[1])
                if area < best_area:
                    best, best_area = t, area
        pairs.append((best, pb))
    return pairs


TRACKERS = MultiCameraTracker()
