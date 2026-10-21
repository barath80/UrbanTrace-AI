# 2. Technology choices — why this, and why not the others

**Read this first.** The project measured *its own* models against each other (the
OCR model comparison in §2.3 is a real measured table). It did **not** benchmark
YOLO against Faster R-CNN, ByteTrack against DeepSORT, or PostgreSQL against
MongoDB. Where a comparison below is an argument rather than a measurement, it is
marked **[rationale]**. These are the reasons the choice is sound for *this* problem,
not claims that the alternatives are bad in general.

The deciding constraints for the whole project:

1. Many cameras, a **limited GPU** (reference: 8 GB laptop RTX 5060).
2. **Real-time** — a stale frame is worthless.
3. **Auditability** — an alert may lead to a vehicle being stopped.
4. Indian plates: small, often blurred, a fixed grammar, frequent OCR confusions.
5. Must be runnable and verifiable by one developer without special infrastructure.

## 2.1 Object detection — Ultralytics YOLO11 (vehicles: yolo11m; plates: fine-tuned YOLO11m)

| Option | Verdict |
|---|---|
| **YOLO11 (chosen)** | One-stage, single pass, batchable, FP16 on GPU, ONNX/TensorRT export path, mature tooling for fine-tuning and validation (`ultralytics val` gives standard COCO-protocol mAP). |
| Faster R-CNN / Mask R-CNN | **[rationale]** Two-stage detectors are typically more accurate on small objects but are slower per frame; the budget here is a fixed number of analysed frames per second per GPU, so throughput dominates. |
| SSD / RetinaNet | **[rationale]** Older one-stage designs; YOLO's recent versions are generally the better accuracy/speed trade and have the better training tooling. |
| RT-DETR / transformer detectors | **[rationale]** Strong accuracy, but heavier and less convenient to batch on a modest GPU; the tracker and plate pipeline are built around the Ultralytics `Results` API. |
| Roboflow hosted inference | Supported as an *optional* backend (`plate_backends.py`). Costs one API call per frame and needs credits, so local GPU weights are the default. If hosted fails, the system reports the exact reason and falls back to local. |

**Two detectors, not one.** Vehicles are large and found reliably at 640 px; plates
are small and lose glyph detail below 960 px. Running vehicles at 640 and plates at
960 is measurably faster than 960/960: **53.5 vs 36.5 frames/s combined** (batch 8,
RTX 5060 Laptop, from `scale/config.py`), with no loss of vehicle recall observed.

**Why `yolo11m` for vehicles?** It is the COCO-pretrained medium model: recorded COCO
mAP50 0.684 / mAP50-95 0.514, 20.0 M parameters, 67.6 GFLOPs (fused). Larger models
would reduce the cameras-per-GPU figure; smaller ones lose far-field vehicles.
*Limit:* COCO has no autorickshaw class, so autos are forced into the least-bad class
and the **per-class breakdown is not trustworthy on Indian traffic** (totals are
unaffected). Fixing that needs a fine-tuned detector, not a threshold.

**Why `license-plate-finetune-v1m` over `-v1l` for plates?** Measured on the 83-image
held-out set: v1l has slightly higher mAP50 (0.829 vs 0.804) but **lower recall
(0.754 vs 0.808)** and costs more (15.4 vs 12.1 ms/image, 25.3 M vs 20.0 M
parameters). For ANPR a missed plate loses the whole sighting, so recall and speed
win. Full table in document 3.

## 2.2 Multi-object tracking — ByteTrack

See document 4 for the full argument. In one paragraph: the plate, not the
appearance, identifies a vehicle across cameras, so the tracker only needs reliable
*short-term continuity inside one camera*. ByteTrack does that with motion
(Kalman + IoU) alone, needs **no appearance/re-ID network** (so a per-camera tracker
is a few kilobytes and adds no GPU cost), and recovers occluded vehicles by also
associating *low-confidence* detections in a second pass.

| Tracker | Why not here **[rationale]** |
|---|---|
| SORT | ByteTrack's predecessor in spirit; discards low-confidence boxes, so it loses tracks through occlusion more often. |
| DeepSORT / StrongSORT | Add an appearance (re-ID) CNN per detection — extra GPU work for every camera, and cross-camera identity is already supplied by the plate. |
| BoT-SORT | Adds camera-motion compensation and optional re-ID; useful for moving cameras, unnecessary for fixed CCTV. |
| OC-SORT | Good for non-linear motion; a reasonable alternative, but ByteTrack ships in Ultralytics and its behaviour is pinned explicitly here. |

The tracker parameters are written out in `scale/tracking.py` (`BYTETRACK_ARGS`) rather
than read from the library YAML, so a library upgrade cannot silently change tracking
across a whole deployment.

## 2.3 Plate OCR — `fast-plate-ocr` `cct-xs-v2-global-model` (ONNX Runtime, CUDA)

This one **is** measured. All six published `fast-plate-ocr` recognisers were run on
the same 80 hand-transcribed plates (document 3, §3.4):

| Model | Exact-match | Character error rate | ms/plate |
|---|---|---|---|
| **cct-xs-v2-global-model (chosen)** | **0.7125** | **0.0777** | 6.2 |
| cct-s-v2-global-model | 0.2750 | 0.1949 | 7.0 |
| cct-xs-v1-global-model | 0.1250 | 0.2446 | 4.5 |
| cct-s-v1-global-model | 0.0500 | 0.2344 | 6.1 |
| cct-s-relu-v1-global-model | 0.0250 | 0.2611 | 6.3 |
| cct-xs-relu-v1-global-model | 0.0000 | 1.0000 | 1.5 |

The *smaller* model wins by a wide margin: that is a measurement, not an assumption,
and it is why the docs no longer say `cct-s`. The mobile-ViT models take 1-channel
input and could not be run with the pipeline's colour crops (the notebook logged the
input-shape errors rather than hiding them).

| OCR family | Why not **[rationale — not benchmarked here]** |
|---|---|
| Tesseract | General document OCR; plates are a different domain (single line, fixed length, perspective, glare). Needs heavy per-plate tuning. |
| EasyOCR / PaddleOCR | Strong general scene-text engines but heavier; a plate-specialised recogniser with a fixed 128×64 input is far cheaper per plate and trivially batchable. |
| CRNN + CTC trained in-house | Would need a large labelled Indian-plate text dataset. The available dataset has plate **boxes only**, no text. |

**Why a plate-specialised, fixed-slot model also has a ceiling:** `max_plate_slots=10`
and an input of 128×64. Section 3.4 shows the dominant failure — 17 of 23 wrong
readings are *shorter* than the truth — i.e. the model drops a trailing character.
No preprocessing recovers a character the model never emits.

## 2.4 Database — PostgreSQL + PostGIS (Supabase in dev, PostGIS container in production)

Detail in document 5. Summary:

| Option | Verdict |
|---|---|
| **PostgreSQL + PostGIS (chosen)** | ACID transactions, rich indexing (B-tree, GiST), native **range partitioning by day**, `jsonb`, a generated `geography` column for spatial queries, `INSERT … ON CONFLICT` for idempotent upserts. One engine for relational, spatial and semi-structured data. |
| MySQL / MariaDB | **[rationale]** Weaker spatial and partitioning ergonomics for this schema; no equivalent of PostGIS's `geography` type with a generated column. |
| MongoDB | **[rationale]** The core data is relational (cameras ↔ sightings ↔ vehicles ↔ legs) and needs joins, constraints and transactions. Document storage would push integrity into application code. |
| TimescaleDB | **[rationale]** A strong fit for time series and a natural future upgrade (it *is* PostgreSQL). Native partitioning was enough to reach the design target without another extension. |
| Redis | Wrong role as the system of record; the hot, bounded state (identity index, tracks) is already in process memory by design. |
| SQLite | No concurrent writers across worker processes; it was used only in the earliest local prototype (`data/urbantrace.db*` is git-ignored). |

`DATABASE_URL` selects any PostGIS server; if it is unset the Supabase settings are
used, so development and production run identical code.

## 2.5 API — FastAPI + Uvicorn

Typed request validation through Pydantic (`CameraIn` enforces id pattern, lat/lon
ranges, lengths), automatic OpenAPI, easy dependency-free middleware (the API-key
gate), `StreamingResponse` for server-sent events. **[rationale]** Flask has no
built-in validation; Django is a heavier framework than a thin data API needs.
Sync handlers run on Starlette's thread pool, which is why the database pool needed
a wait-queue (document 9, defect 3).

## 2.6 Live updates — Server-Sent Events (SSE), not WebSocket or polling

Traffic is one-directional (server → browser), SSE rides ordinary HTTP so it passes
proxies, and the browser reconnects automatically. A 15-second keep-alive comment
stops intermediaries closing idle streams, and nginx is configured with
`proxy_buffering off` and a 3600 s read timeout for those routes. **[rationale]**
WebSocket would add a bidirectional channel nothing needs; polling would add latency
and load.

## 2.7 Front end — React + TypeScript + Vite, Leaflet

Typed components for 13 views (overview, live, multi-camera, city network, GIS map,
fleet, camera health, vehicle tracking, analytics, alerts, observations, cameras).
The original vanilla-JS dashboard is retained at `/legacy` so nothing that worked was
removed. The React bundle is built in a Node stage of the Dockerfile and served as
static files by the API.

## 2.8 Load balancer — nginx

Chosen for resolver-based re-resolution of the `api` service name (`valid=5s`) so
`docker compose up --scale api=N` changes the upstream set **without reloading nginx**,
plus `limit_req`, gzip, `proxy_next_upstream` retry, and trivial streaming
configuration. **[rationale]** HAProxy and Traefik are equally capable; nginx is the
most widely known and needs the least configuration here.

## 2.9 Packaging — Docker, multi-stage, CUDA runtime image

One image runs both the API and the workers. Node builds the front end in a throw-away
stage; the runtime stage is `nvidia/cuda:12.8.1-cudnn-runtime`, runs as **non-root
(uid 10001)** under `tini` for correct signal handling (graceful shutdown flushes
buffered sightings). Notebook tooling is stripped from the service image.
Detector weights are *mounted*, never baked in.

## 2.10 Why Python

The whole ML stack (PyTorch, Ultralytics, ONNX Runtime, OpenCV) is Python. The GIL
is not a problem on the hot path because the heavy work (CUDA kernels, OpenCV, NumPy)
releases it; the lock design in document 7 keeps the Python-level critical sections
tiny.
