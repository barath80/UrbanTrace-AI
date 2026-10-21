"""Shared, batched inference for the whole camera network.

Throughput at city scale is decided here. Three properties matter:

1. **One set of models, not one per camera.** The detector weights are loaded
   once and every camera's frames pass through them. Per-camera state lives in
   `tracking.MultiCameraTracker` and is kilobytes, so camera count is bounded
   by bandwidth and CPU, not by GPU memory.

2. **Batching.** A GPU running one 960px frame at a time is mostly idle. Frames
   from *different cameras* are gathered into one batch, which is what turns a
   per-camera cost into a per-network cost. A partially filled batch runs after
   `batch_timeout_s` so a quiet network does not stall.

3. **Bounded queues that drop, never block.** Each camera holds at most
   `queue_depth_per_camera` frames. When a camera outruns the GPU the OLDEST
   frame is discarded. In live monitoring a stale frame has no value, and an
   unbounded queue converts a temporary slowdown into an out-of-memory crash.
   Dropped frames are counted and reported, never hidden.

The pool is deliberately synchronous inside a worker: pull a batch, detect,
track, OCR, emit. Ordering within a camera is preserved because a camera's
frames are only ever handled by the worker that drained them, and each camera's
tracker takes its own lock.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from .config import SCALE
from .identity import Sighting
from .tracking import TRACKERS, Detections, VEHICLE_CLASSES, bind_plates_to_tracks

logger = logging.getLogger("urbantrace.scale.inference")


@dataclass
class FrameItem:
    camera_id: str
    frame: np.ndarray
    captured_at: datetime
    video_time_s: float = 0.0
    frame_number: int = 0


class FrameBus:
    """Per-camera bounded frame slots with fair round-robin draining.

    Fairness matters: a naive shared queue lets a handful of high-frame-rate
    cameras crowd out the rest, so parts of the city go unwatched while the GPU
    looks fully busy. Round-robin gives every camera with pending work an equal
    share of each batch.
    """

    def __init__(self, depth: int | None = None, target_fps: float | None = None):
        self.depth = depth or SCALE.queue_depth_per_camera
        self.target_fps = SCALE.target_fps_per_camera if target_fps is None else target_fps
        self._queues: "OrderedDict[str, deque]" = OrderedDict()
        self._last_admit: dict[str, float] = {}
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self.submitted = 0
        self.dropped = 0
        self.rejected = 0

    def admit(self, camera_id: str, now: float | None = None) -> bool:
        """Rate-limit a camera to the configured analysis frame rate.

        Cameras deliver 25-30 fps, but ANPR does not need every frame: a
        vehicle crossing a lane is present for dozens of them. Sampling at the
        source is the cheapest possible optimisation because a rejected frame
        costs no decode, no transfer and no GPU time. At 1000 cameras this is
        the difference between 25,000 fps of demand and 6,000.
        """
        if self.target_fps <= 0:
            return True
        now = now or time.monotonic()
        # 10% tolerance: capture already paces frames at this rate, and the few
        # milliseconds of decode/resize jitter between its clock and this one
        # would otherwise reject frames that were on schedule and halve the
        # effective rate.
        period = 0.9 / self.target_fps
        with self._lock:
            last = self._last_admit.get(camera_id, 0.0)
            if now - last < period:
                self.rejected += 1
                return False
            self._last_admit[camera_id] = now
            return True

    def submit(self, item: FrameItem) -> bool:
        """Queue a frame. Returns False if an older frame had to be dropped."""
        with self._not_empty:
            q = self._queues.get(item.camera_id)
            if q is None:
                q = deque()
                self._queues[item.camera_id] = q
            dropped = False
            while len(q) >= self.depth:
                q.popleft()
                self.dropped += 1
                dropped = True
            q.append(item)
            self.submitted += 1
            self._not_empty.notify()
            return not dropped

    def drain_batch(self, max_items: int, timeout: float) -> list[FrameItem]:
        """Take at most one frame per camera, round-robin, up to `max_items`."""
        deadline = time.monotonic() + timeout
        with self._not_empty:
            while True:
                batch: list[FrameItem] = []
                empty: list[str] = []
                # Rotate so the camera served first changes between calls.
                for cam in list(self._queues.keys()):
                    q = self._queues[cam]
                    if q:
                        batch.append(q.popleft())
                        self._queues.move_to_end(cam)
                        if len(batch) >= max_items:
                            break
                    else:
                        empty.append(cam)
                for cam in empty:
                    if not self._queues[cam]:
                        del self._queues[cam]
                if batch:
                    return batch
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._not_empty.wait(remaining)

    def stats(self) -> dict:
        with self._lock:
            pending = sum(len(q) for q in self._queues.values())
            offered = self.submitted + self.rejected
            return {
                "cameras_pending": len(self._queues),
                "frames_pending": pending,
                "offered": offered,
                "submitted": self.submitted,
                "rejected_by_rate_limit": self.rejected,
                "dropped": self.dropped,
                "drop_rate": round(self.dropped / self.submitted, 4) if self.submitted else 0.0,
                "admit_rate": round(self.submitted / offered, 4) if offered else 0.0,
            }


class SharedModels:
    """The one copy of each network, loaded lazily and shared by all workers.

    Ultralytics models are not documented as thread-safe for concurrent
    `predict`, so calls are serialised on a lock. That is not the bottleneck it
    appears to be: the whole point of batching is that one locked call does the
    work of many frames.
    """

    def __init__(self):
        # Separate locks per model. With one lock a worker running the plate
        # pass blocks another worker's vehicle pass, so the two never overlap
        # and throughput collapses to the SUM of their times. Split, the GPU
        # can be kept busy by one while the other preprocesses on the CPU.
        self._lock = threading.Lock()
        self._veh_lock = threading.Lock()
        self._plate_lock = threading.Lock()
        self._vehicle = None
        self._plate = None
        self._ocr = None
        self.vehicle_calls = 0
        self.plate_calls = 0
        self.frames_processed = 0
        self.plate_frames_skipped = 0
        self.gpu_seconds = 0.0

    def _device(self):
        from ..core.runtime import device
        return 0 if device() == "cuda" else "cpu"

    def load(self) -> None:
        with self._lock:
            if self._vehicle is None:
                from ultralytics import YOLO
                from ..vision.detectors import PLATE_WEIGHTS, VEHICLE_WEIGHTS
                if not VEHICLE_WEIGHTS.exists():
                    raise FileNotFoundError(f"Vehicle weights missing: {VEHICLE_WEIGHTS}")
                if not PLATE_WEIGHTS.exists():
                    raise FileNotFoundError(f"Plate weights missing: {PLATE_WEIGHTS}")
                self._vehicle = YOLO(str(VEHICLE_WEIGHTS))
                self._plate = YOLO(str(PLATE_WEIGHTS))
            if self._ocr is None:
                from ..vision.ocr_engine import get_ocr
                self._ocr = get_ocr()

    @property
    def ocr(self):
        if self._ocr is None:
            self.load()
        return self._ocr

    def warmup(self, max_batch: int, size: tuple[int, int] = (1280, 720)) -> float:
        """Run both detectors once at every batch size 1..max_batch.

        The first call at each new input shape is slow (measured 4-100 s on an
        RTX 5060 while kernels are selected and compiled), and batches vary in
        size with load. Without this a freshly started worker analyses nothing
        for minutes while every frame it is offered is dropped. Doing it before
        the workers start moves that cost into startup, where it is expected.
        Returns the seconds spent.
        """
        self.load()
        dev = self._device()
        quant = 16 if dev != "cpu" else None
        blank = np.zeros((size[1], size[0], 3), np.uint8)
        t0 = time.perf_counter()
        for n in range(1, max(1, max_batch) + 1):
            frames = [blank] * n
            with self._veh_lock:
                self._vehicle.predict(frames, conf=SCALE.vehicle_conf, imgsz=SCALE.vehicle_imgsz,
                                      classes=list(VEHICLE_CLASSES), device=dev,
                                      quantize=quant, verbose=False)
            with self._plate_lock:
                self._plate.predict(frames, conf=SCALE.plate_conf, imgsz=SCALE.plate_imgsz,
                                    device=dev, quantize=quant, verbose=False)
        return time.perf_counter() - t0

    def detect_batch(self, frames: list, need_plates: list | None = None) -> tuple[list, list]:
        """Run both detectors over a batch. Returns (vehicle_dets, plate_boxes).

        `need_plates` marks, per frame, whether any track on that camera still
        wants a plate reading. Frames nobody is waiting on skip the plate pass
        entirely -- in steady state most vehicles are already identified, so
        this removes the majority of plate inferences without losing a reading.
        """
        if self._vehicle is None:
            self.load()
        dev = self._device()
        # FP16 on GPU: same boxes, roughly half the time. Ultralytics replaced
        # the old `half=` flag with `quantize=16`.
        quant = 16 if dev != "cpu" else None

        t0 = time.perf_counter()
        with self._veh_lock:
            vres = self._vehicle.predict(
                frames, conf=SCALE.vehicle_conf, imgsz=SCALE.vehicle_imgsz,
                classes=list(VEHICLE_CLASSES), device=dev, quantize=quant, verbose=False)
            self.vehicle_calls += 1
            self.frames_processed += len(frames)

        # The plate detector only runs on frames that contain a vehicle AND
        # whose camera still has an unresolved track. A plate cannot be bound
        # to a track that does not exist, and re-reading a settled plate cannot
        # change the answer.
        has_vehicle = [r.boxes is not None and len(r.boxes) > 0 for r in vres]
        if need_plates is None:
            need_plates = [True] * len(frames)
        keep = [hv and np_ for hv, np_ in zip(has_vehicle, need_plates)]
        live = [f for f, k in zip(frames, keep) if k]

        pres_live = []
        if live:
            with self._plate_lock:
                pres_live = self._plate.predict(
                    live, conf=SCALE.plate_conf, imgsz=SCALE.plate_imgsz,
                    device=dev, quantize=quant, verbose=False)
                self.plate_calls += 1
        self.plate_frames_skipped += len(frames) - len(live)
        self.gpu_seconds += time.perf_counter() - t0

        # ONE device-to-host transfer per frame. Reading `.xyxy`, `.conf` and
        # `.cls` separately issues three synchronising copies, and at batch
        # sizes that matter those dominate the postprocessing cost. `boxes.data`
        # is a single tensor laid out [x1, y1, x2, y2, (track), conf, cls].
        vehicles = []
        for r in vres:
            b = r.boxes
            if b is None or len(b) == 0:
                vehicles.append(Detections.empty())
                continue
            d = b.data.cpu().numpy()
            vehicles.append(Detections(d[:, :4], d[:, -2], d[:, -1]))
        plates = []
        it = iter(pres_live)
        for k in keep:
            if not k:
                plates.append([])
                continue
            r = next(it)
            b = r.boxes
            if b is None or len(b) == 0:
                plates.append([])
                continue
            d = b.data.cpu().numpy()
            plates.append([((float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                            float(row[-2])) for row in d])
        return vehicles, plates

    def stats(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "detector_calls": self.vehicle_calls,
            "mean_batch": round(self.frames_processed / self.vehicle_calls, 2)
            if self.vehicle_calls else 0.0,
            "plate_frames_skipped": self.plate_frames_skipped,
            "gpu_seconds": round(self.gpu_seconds, 2),
            "frames_per_gpu_second": round(self.frames_processed / self.gpu_seconds, 1)
            if self.gpu_seconds > 0 else 0.0,
        }


@dataclass
class _TrackBuffer:
    """Per-track OCR readings awaiting a confident consensus."""
    readings: list = field(default_factory=list)
    published_confidence: float = -1.0
    vehicle_type: str = "vehicle"
    last_seen: float = 0.0
    settled: bool = False


class InferencePool:
    """Worker threads that turn queued frames into confirmed sightings."""

    def __init__(self, models: SharedModels | None = None,
                 bus: FrameBus | None = None,
                 on_sighting: Callable[[Sighting], None] | None = None,
                 on_annotated=None):
        self.models = models or SharedModels()
        self.bus = bus or FrameBus()
        self.on_sighting = on_sighting
        self.on_annotated = on_annotated
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._buffers: dict[tuple, _TrackBuffer] = {}
        # Per-camera counters kept in step with _buffers so the plate-skip
        # decision is O(1) instead of a scan of every open track.
        self._live_tracks: dict[str, int] = {}
        self._unsettled: dict[str, int] = {}
        self._buf_lock = threading.Lock()
        self.sightings_emitted = 0
        self.ocr_calls = 0
        # A track read this confidently is considered settled and stops
        # consuming OCR. Set to 1.01 to disable the early exit.
        self.settle_confidence = 0.93
        # How long a track buffer counts as "live" when deciding whether a
        # camera still needs plate inference.
        self.plate_idle_seconds = 3.0
        self.started_at: float | None = None

    # ------------------------------------------------------- lifecycle

    def start(self, workers: int | None = None) -> None:
        if self._threads:
            return
        self.models.load()
        if SCALE.warmup and hasattr(self.models, "warmup"):
            logger.info("detector warm-up done in %.1fs",
                        self.models.warmup(SCALE.batch_size))
        self._stop.clear()
        self.started_at = time.time()
        n = workers or SCALE.inference_workers
        for i in range(n):
            t = threading.Thread(target=self._run, name=f"infer-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        logger.info("inference pool started with %d worker(s)", n)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    @property
    def running(self) -> bool:
        return bool(self._threads) and not self._stop.is_set()

    # ---------------------------------------------------------- worker

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self.bus.drain_batch(SCALE.batch_size, SCALE.batch_timeout_s)
            if not batch:
                continue
            try:
                self._process(batch)
            except Exception:
                logger.exception("inference batch failed")

    def _process(self, batch: list) -> None:
        frames = [b.frame for b in batch]
        need = [self._camera_wants_plates(b.camera_id) for b in batch]
        vehicle_dets, plate_dets = self.models.detect_batch(frames, need_plates=need)

        # Phase 1 -- detect, track, and COLLECT plate crops across the whole
        # batch. Nothing is OCR'd yet.
        pending: list = []          # (FrameItem, Track, plate_box, plate_conf)
        crops: list = []
        for item, dets, plates in zip(batch, vehicle_dets, plate_dets):
            tracks = TRACKERS.update(item.camera_id, dets)
            plate_boxes = [p[0] for p in plates]
            plate_confs = [p[1] for p in plates]
            pairs = bind_plates_to_tracks(plate_boxes, tracks)

            for (track, pbox), pconf in zip(pairs, plate_confs):
                if track is None:
                    continue
                crop = self._crop_for(item, track, pbox)
                if crop is None:
                    continue
                pending.append((item, track, pbox, pconf))
                crops.append(crop)

            if self.on_annotated is not None:
                try:
                    self.on_annotated(item, tracks, plate_boxes)
                except Exception:
                    logger.exception("annotation callback failed")

        if not crops:
            return

        # Phase 2 -- ONE OCR call for every plate in the batch. Reading plates
        # one at a time pays the engine's fixed dispatch cost per plate; batched
        # it is paid once. Measured at 3.2x on this hardware with byte-identical
        # readings.
        self.ocr_calls += len(crops)
        results = self.models.ocr.read_many(crops)

        # Phase 3 -- fold each reading into its track's temporal vote.
        for (item, track, _pbox, pconf), res in zip(pending, results):
            if res is None or not res.text_final:
                continue
            self._accumulate(item, track, pconf, res)

    def _camera_wants_plates(self, camera_id: str) -> bool:
        """Whether this camera still has a vehicle whose plate is unresolved.

        Maintained as a per-camera counter rather than by scanning every track
        buffer: the scan version was O(open tracks) on the hot path for every
        frame in every batch, under the buffer lock, and cost more than the
        inference it was trying to avoid.

        Defaults to True for a camera with no live tracks, because a vehicle
        that has just entered frame must get a plate pass.
        """
        with self._buf_lock:
            live = self._live_tracks.get(camera_id, 0)
            unsettled = self._unsettled.get(camera_id, 0)
        return unsettled > 0 or live == 0

    def _crop_for(self, item: FrameItem, track, pbox) -> "np.ndarray | None":
        """Crop a plate, unless this track is already resolved or saturated."""
        x1, y1, x2, y2 = pbox
        if (x2 - x1) * (y2 - y1) < SCALE.min_plate_pixels:
            return None

        key = (item.camera_id, track.track_id)
        with self._buf_lock:
            buf = self._buffers.get(key)
            if buf is None:
                buf = _TrackBuffer()
                self._buffers[key] = buf
                self._live_tracks[item.camera_id] = self._live_tracks.get(item.camera_id, 0) + 1
                self._unsettled[item.camera_id] = self._unsettled.get(item.camera_id, 0) + 1
            # Stop reading a track once it is saturated or already resolved
            # confidently. Continuing to OCR a plate that has been read
            # consistently at high confidence cannot change the answer, and on
            # a busy network that work is a large share of total OCR cost.
            if (len(buf.readings) >= SCALE.max_readings_per_track
                    or buf.published_confidence >= self.settle_confidence):
                if not buf.settled:
                    buf.settled = True
                    self._unsettled[item.camera_id] = max(
                        0, self._unsettled.get(item.camera_id, 1) - 1)
                return None
            buf.last_seen = time.time()
            buf.vehicle_type = track.cls_name

        from ..vision.preprocess import crop_with_margin
        return crop_with_margin(item.frame, pbox)

    def _accumulate(self, item: FrameItem, track, pconf: float, res) -> None:
        from ..vision.aggregate import Reading, aggregate

        reading = Reading(
            frame_number=item.frame_number, timestamp_s=round(item.video_time_s, 3),
            text_raw=res.text_raw, text_final=res.text_final,
            ocr_confidence=res.confidence, plate_confidence=pconf,
            is_valid_format=res.is_valid_format, repaired=res.repaired,
            variant=res.variant, char_confidences=res.char_confidences)

        key = (item.camera_id, track.track_id)
        with self._buf_lock:
            buf = self._buffers.get(key)
            if buf is None:
                return
            buf.readings.append(reading)
            readings = list(buf.readings)
            already = buf.published_confidence

        agg = aggregate(readings, min_readings=SCALE.min_readings_to_publish)
        if agg is None or agg.confidence < SCALE.min_publish_confidence:
            return
        # Only publish an improvement, so a track emits one sighting that gets
        # better rather than a stream of near-duplicates.
        if agg.confidence <= already:
            return
        with self._buf_lock:
            buf.published_confidence = agg.confidence
            vehicle_type = buf.vehicle_type

        from .registry import REGISTRY
        node = REGISTRY.get(item.camera_id)
        sighting = Sighting(
            plate=agg.plate,
            camera_id=item.camera_id,
            observed_at=item.captured_at,
            confidence=round(agg.confidence, 4),
            track_id=track.track_id,
            latitude=node.latitude if node else 0.0,
            longitude=node.longitude if node else 0.0,
            vehicle_type=vehicle_type,
            support=agg.support,
            total_readings=agg.total_readings,
            is_valid_format=agg.is_valid_format,
            site_id=SCALE.site_id,
        )
        self.sightings_emitted += 1
        if self.on_sighting is not None:
            try:
                self.on_sighting(sighting)
            except Exception:
                logger.exception("sighting callback failed")

    # ---------------------------------------------------------- upkeep

    def evict_stale_tracks(self, older_than_s: float = 300.0) -> int:
        cutoff = time.time() - older_than_s
        with self._buf_lock:
            dead = [k for k, b in self._buffers.items() if b.last_seen < cutoff]
            for k in dead:
                cam = k[0]
                buf = self._buffers.pop(k)
                self._live_tracks[cam] = max(0, self._live_tracks.get(cam, 1) - 1)
                if not buf.settled:
                    self._unsettled[cam] = max(0, self._unsettled.get(cam, 1) - 1)
        return len(dead)

    def stats(self) -> dict:
        uptime = time.time() - self.started_at if self.started_at else 0.0
        with self._buf_lock:
            open_tracks = len(self._buffers)
        return {
            "workers": len(self._threads),
            "running": self.running,
            "uptime_s": round(uptime, 1),
            "sightings_emitted": self.sightings_emitted,
            "ocr_calls": self.ocr_calls,
            "open_tracks": open_tracks,
            "throughput_fps": round(self.models.frames_processed / uptime, 1) if uptime > 0 else 0.0,
            "models": self.models.stats(),
            "bus": self.bus.stats(),
            "trackers": TRACKERS.stats(),
        }
