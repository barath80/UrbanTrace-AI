"""Deployment tunables for the city-scale ingest layer.

Everything that changes between a 3-camera bench and a 1000-camera city is a
number in this file or an environment variable, not a code change.

The defaults are sized for a single GPU workstation. The values that actually
bound throughput are `inference_workers`, `batch_size` and `target_fps_per_camera`:
a camera network is almost never GPU-bound on decode, it is bound on how many
frames per second the detector can consume in total.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _i(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _f(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _s(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _b(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ScaleSettings:
    # ---- inference -------------------------------------------------
    # One shared set of models serves every camera. Cameras contribute
    # per-camera TRACKER state (a few kilobytes), never their own copy of the
    # network, which is what makes 1000 cameras fit in one GPU.
    inference_workers: int = field(default_factory=lambda: _i("UT_INFER_WORKERS", 2))
    # Frames per detector call. Larger batches raise GPU utilisation and
    # throughput but add latency, because a frame waits for the batch to fill.
    batch_size: int = field(default_factory=lambda: _i("UT_BATCH_SIZE", 8))
    # How long a partially-filled batch waits before running anyway.
    batch_timeout_s: float = field(default_factory=lambda: _f("UT_BATCH_TIMEOUT", 0.05))
    # Vehicles are large in frame and are found reliably at 640; plates are
    # small and lose glyph detail below 960. Running the two detectors at
    # different resolutions is measurably faster than running both at 960 and
    # costs nothing in vehicle recall.  Measured on an RTX 5060 Laptop, batch 8:
    #   both @960 -> 36.5 fps combined
    #   veh @640 + plate @960 -> 53.5 fps combined
    vehicle_imgsz: int = field(default_factory=lambda: _i("UT_VEHICLE_IMGSZ", 640))
    plate_imgsz: int = field(default_factory=lambda: _i("UT_PLATE_IMGSZ", 960))
    vehicle_conf: float = field(default_factory=lambda: _f("UT_VEHICLE_CONF", 0.30))
    plate_conf: float = field(default_factory=lambda: _f("UT_PLATE_CONF", 0.25))

    # ---- per-camera admission -------------------------------------
    # Analysed frames per second per camera. Cameras deliver 25-30 fps, but
    # ANPR does not need every frame: a vehicle crossing a lane is present for
    # many frames. Sampling is what makes the arithmetic work at city scale.
    target_fps_per_camera: float = field(default_factory=lambda: _f("UT_TARGET_FPS", 6.0))
    # Frames held per camera before the oldest is dropped. A backlog is always
    # dropped rather than queued: in live monitoring a stale frame has no value
    # and an unbounded queue turns a slow GPU into an out-of-memory crash.
    queue_depth_per_camera: int = field(default_factory=lambda: _i("UT_QUEUE_DEPTH", 2))

    # ---- OCR -------------------------------------------------------
    ocr_batch_size: int = field(default_factory=lambda: _i("UT_OCR_BATCH", 16))
    max_readings_per_track: int = field(default_factory=lambda: _i("UT_MAX_READINGS", 14))
    min_plate_pixels: int = field(default_factory=lambda: _i("UT_MIN_PLATE_PX", 480))
    # A track is published as a sighting only when the temporal vote has this
    # many frames behind it and clears this confidence.
    min_readings_to_publish: int = field(default_factory=lambda: _i("UT_MIN_READINGS_PUB", 3))
    min_publish_confidence: float = field(default_factory=lambda: _f("UT_MIN_PUB_CONF", 0.60))

    # ---- identity resolution ---------------------------------------
    min_plate_similarity: float = field(default_factory=lambda: _f("UT_MIN_PLATE_SIM", 0.82))
    # A cluster stops accepting new sightings after this long, so the identity
    # index does not grow without bound over a multi-day deployment.
    identity_ttl_minutes: float = field(default_factory=lambda: _f("UT_IDENTITY_TTL_MIN", 720.0))

    # ---- spatio-temporal validation --------------------------------
    # Two sightings separated by more than this are NOT one continuous journey.
    # Without this bound a vehicle seen on Monday and again on Friday produces a
    # single four-day "leg" at 0.1 km/h, which is then averaged into the
    # origin-destination matrix and the congestion baseline and destroys both.
    # The vehicle did not crawl for four days; it made two separate trips.
    max_leg_gap_minutes: float = field(default_factory=lambda: _f("UT_MAX_LEG_GAP_MIN", 90.0))
    max_plausible_speed_kmh: float = field(default_factory=lambda: _f("UT_MAX_SPEED", 140.0))
    warn_speed_kmh: float = field(default_factory=lambda: _f("UT_WARN_SPEED", 100.0))
    road_factor: float = field(default_factory=lambda: _f("UT_ROAD_FACTOR", 1.25))
    freeflow_kmh: float = field(default_factory=lambda: _f("UT_FREEFLOW_KMH", 55.0))
    congested_kmh: float = field(default_factory=lambda: _f("UT_CONGESTED_KMH", 22.0))

    # ---- camera graph ----------------------------------------------
    # Edges are built to the k nearest cameras within a radius, NOT between all
    # pairs. All-pairs is 499,500 edges at 1000 cameras and none of the distant
    # ones describe a road a vehicle could plausibly take between sightings.
    graph_neighbours: int = field(default_factory=lambda: _i("UT_GRAPH_K", 6))
    graph_radius_km: float = field(default_factory=lambda: _f("UT_GRAPH_RADIUS_KM", 12.0))

    # ---- persistence -----------------------------------------------
    # Sightings are written in batches. One INSERT per sighting is the single
    # most common reason an ANPR backend falls over at scale.
    sink_batch_size: int = field(default_factory=lambda: _i("UT_SINK_BATCH", 200))
    sink_flush_seconds: float = field(default_factory=lambda: _f("UT_SINK_FLUSH_S", 2.0))
    sink_enabled: bool = field(default_factory=lambda: _b("UT_SINK_ENABLED", True))
    # Bound on the in-memory spill buffer used when the database is unreachable.
    sink_max_pending: int = field(default_factory=lambda: _i("UT_SINK_MAX_PENDING", 50_000))

    site_id: str = field(default_factory=lambda: _s("UT_SITE_ID", "default"))

    # ---- stream capture --------------------------------------------
    # Longest edge of a frame handed to the detectors. The plate detector runs
    # at 960, so anything wider than this is decoded, queued and copied for
    # nothing. 0 disables the downscale.
    max_frame_width: int = field(default_factory=lambda: _i("UT_MAX_FRAME_WIDTH", 1280))
    rtsp_transport: str = field(default_factory=lambda: _s("UT_RTSP_TRANSPORT", "tcp"))
    open_timeout_ms: int = field(default_factory=lambda: _i("UT_OPEN_TIMEOUT_MS", 8000))
    read_timeout_ms: int = field(default_factory=lambda: _i("UT_READ_TIMEOUT_MS", 8000))
    reconnect_min_s: float = field(default_factory=lambda: _f("UT_RECONNECT_MIN_S", 1.0))
    reconnect_max_s: float = field(default_factory=lambda: _f("UT_RECONNECT_MAX_S", 60.0))
    # Streams being opened at once. Starting 100+ feeds together stalls the
    # network and the decoder; a small gate staggers the start.
    max_concurrent_opens: int = field(default_factory=lambda: _i("UT_MAX_OPENS", 6))
    # Motion gate: an unchanged scene is not worth a detector pass. A frame is
    # still sent every `idle_keyframe_s`, so a stopped vehicle is not missed.
    motion_gate: bool = field(default_factory=lambda: _b("UT_MOTION_GATE", True))
    # Run the detectors once at every batch size before taking traffic, so the
    # slow first call per input shape is paid at startup, not on live frames.
    warmup: bool = field(default_factory=lambda: _b("UT_WARMUP", True))
    motion_pixel_delta: int = field(default_factory=lambda: _i("UT_MOTION_PIXEL_DELTA", 14))
    motion_min_fraction: float = field(default_factory=lambda: _f("UT_MOTION_MIN_FRACTION", 0.004))
    idle_keyframe_s: float = field(default_factory=lambda: _f("UT_IDLE_KEYFRAME_S", 2.0))
    # Files (not live feeds) restart at the end. Meant for bench testing.
    loop_files: bool = field(default_factory=lambda: _b("UT_LOOP_FILES", False))

    # ---- adaptive load shedding -------------------------------------
    # The analysis rate per camera moves between these bounds so the pipeline
    # sheds load smoothly when the GPU saturates instead of dropping frames at
    # random, and recovers when it frees up.
    adaptive_fps: bool = field(default_factory=lambda: _b("UT_ADAPTIVE_FPS", True))
    min_fps_per_camera: float = field(default_factory=lambda: _f("UT_MIN_FPS", 0.5))
    adapt_interval_s: float = field(default_factory=lambda: _f("UT_ADAPT_INTERVAL_S", 5.0))
    # Fraction of submitted frames dropped (oldest-first) that triggers a cut.
    adapt_drop_high: float = field(default_factory=lambda: _f("UT_ADAPT_DROP_HIGH", 0.10))
    adapt_drop_low: float = field(default_factory=lambda: _f("UT_ADAPT_DROP_LOW", 0.01))

    # ---- cluster membership ----------------------------------------
    worker_id: str = field(default_factory=lambda: _s("UT_WORKER_ID", ""))
    worker_heartbeat_s: float = field(default_factory=lambda: _f("UT_WORKER_HEARTBEAT_S", 5.0))
    # A worker silent for this long is presumed dead and its cameras move.
    worker_ttl_s: float = field(default_factory=lambda: _f("UT_WORKER_TTL_S", 20.0))
    # Cameras in one `sector` (else one grid cell of this size) are always
    # owned by the same worker, so neighbouring cameras share an identity index
    # and the legs between them can be measured.
    shard_cell_km: float = field(default_factory=lambda: _f("UT_SHARD_CELL_KM", 3.0))
    worker_port: int = field(default_factory=lambda: _i("UT_WORKER_PORT", 9100))


SCALE = ScaleSettings()
