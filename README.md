# UrbanTrace AI

**City-Wide Multi-Camera ANPR, Spatio-Temporal Vehicle Trajectory Tracking, GIS Traffic Intelligence and Real-Time Urban Mobility Analytics Platform**

> From License Plate Recognition to City-Wide Mobility Intelligence

A centralised AI platform that turns isolated ANPR camera feeds into connected
intelligence: it recognises license plates, tracks every vehicle, links the same
vehicle across cameras, reconstructs its journey on a GIS map, aggregates all
movement into city-wide traffic analytics, and raises explainable alerts.

**Everything in this repository runs on real video.** There is no sample data, no
placeholder plate, no pre-filled result. If the input videos are missing, the
pipeline stops rather than fabricating output.

---

## Table of contents

- [The problem](#the-problem)
- [Architecture](#architecture)
- [Features](#features)
- [Measured results](#measured-results)
- [Quick start](#quick-start)
- [Running the notebook](#running-the-notebook)
- [Running the dashboard](#running-the-dashboard)
- [Camera configuration](#camera-configuration)
- [Database](#database)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Verification](#verification)
- [Honest limitations](#honest-limitations)
- [Security](#security)

---

## The problem

Modern cities deploy large CCTV/ANPR networks, but most process each feed in
isolation — detecting a plate, storing it, and stopping there. Authorities end up
knowing that a vehicle was seen at three places without any system that joins those
sightings into one journey, and without any way to derive macro traffic trends from
the infrastructure they already own.

UrbanTrace AI adds the missing intelligence layer:

| Question | Answered by |
|---|---|
| **Who is this vehicle?** | Vehicle detection → plate detection → preprocessing → OCR → format validation |
| **Where and when did it travel?** | Cross-camera association → spatio-temporal validation → trajectory reconstruction |
| **How is traffic moving?** | Window aggregation → volume, density, speed, travel time, OD flows, congestion |
| **What needs attention?** | Watchlist matches + explainable spatio-temporal anomalies |

---

## Architecture

```
                        ┌──────────────────────────────┐
                        │      CAMERA NETWORK          │
                        │  N video sources (1 … many)  │
                        │  one camera per video file   │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │      INGESTION LAYER         │
                        │  auto-discovery, properties  │
                        │  read from the file itself   │
                        └──────────────┬───────────────┘
                                       │
        ┌──────────────────────────────▼──────────────────────────────┐
        │                     AI VISION LAYER  (GPU)                  │
        │                                                             │
        │   ┌─────────────────┐   ┌──────────────────┐                │
        │   │ VEHICLE DETECT  │   │  PLATE DETECT    │                │
        │   │ YOLO11m (COCO)  │   │  YOLOv11 local   │                │
        │   │  car/moto/bus/  │   │  ── or ──        │                │
        │   │  truck/bicycle  │   │  Roboflow hosted │                │
        │   └────────┬────────┘   └────────┬─────────┘                │
        │            │                     │                          │
        │   ┌────────▼────────┐            │                          │
        │   │ MULTI-OBJECT    │            │                          │
        │   │ TRACKING        │◄───────────┘  bind plate → track      │
        │   │ ByteTrack       │               (smallest container)    │
        │   └────────┬────────┘                                       │
        │            │                                                │
        │   ┌────────▼──────────────────────────────────┐             │
        │   │ PLATE CROP → PREPROCESSING VARIANTS       │             │
        │   │ deskew · CLAHE · sharpen · denoise · Otsu │             │
        │   └────────┬──────────────────────────────────┘             │
        │            │                                                │
        │   ┌────────▼────────┐   ┌──────────────────────┐            │
        │   │  OCR ENGINE     │──►│ CROSS-VARIANT        │            │
        │   │  CCT transformer│   │ CONSENSUS            │            │
        │   └─────────────────┘   └──────────┬───────────┘            │
        │                                    │                        │
        │   ┌────────────────────────────────▼───────────┐            │
        │   │ INDIAN PLATE GRAMMAR VALIDATION            │            │
        │   │ digit/letter position repair · state codes │            │
        │   │ raw OCR always retained                    │            │
        │   └────────────────┬───────────────────────────┘            │
        │                    │                                        │
        │   ┌────────────────▼───────────────────────────┐            │
        │   │ TEMPORAL AGGREGATION (per track)           │            │
        │   │ string vote + character vote + peak        │            │
        │   └────────────────┬───────────────────────────┘            │
        └────────────────────┼────────────────────────────────────────┘
                             │
                ┌────────────▼─────────────┐
                │  STRUCTURED OBSERVATION  │
                │  PLATE + TIMESTAMP +     │
                │  CAMERA + TRACK ID       │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │   CENTRAL DATA STORE     │
                │   Supabase / PostgreSQL  │
                │   + PostGIS geometry     │
                └────────────┬─────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│  TRAJECTORY    │  │   ANALYTICS     │  │     ALERT       │
│  ENGINE        │  │   ENGINE        │  │     ENGINE      │
│                │  │                 │  │                 │
│ plate          │  │ unique vehicle  │  │ watchlist match │
│  similarity    │  │  counts         │  │                 │
│ (OCR-confusion │  │ volume/density  │  │ spatio-temporal │
│  aware)        │  │ speed estimate  │  │  anomaly        │
│ camera graph   │  │ travel time     │  │                 │
│ travel-time    │  │ OD flow matrix  │  │ every alert     │
│  feasibility   │  │ route density   │  │  carries its    │
│ route          │  │ heatmap         │  │  reasoning      │
│  reconstruction│  │ trends          │  │                 │
└───────┬────────┘  └────────┬────────┘  └────────┬────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                ┌────────────▼─────────────┐
                │       API LAYER          │
                │   FastAPI · 17 endpoints │
                └────────────┬─────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
┌───────▼─────────────────┐          ┌────────────▼────────────┐
│  GIS WEB DASHBOARD      │          │   JUPYTER NOTEBOOK      │
│  Leaflet · 7 views      │          │   31 cells, runnable    │
│                         │          │   one at a time         │
│  overview · map ·       │          │                         │
│  tracking · analytics · │          │  inline annotated video │
│  video · alerts · data  │          │  tables · maps · audit  │
└─────────────────────────┘          └─────────────────────────┘
```

### Data flow in one line

```
video → vehicle detect → MOT → plate detect → crop → preprocess → OCR
      → consensus → grammar validation → temporal aggregation
      → observation (plate+time+camera) → cross-camera association
      → spatio-temporal validation → trajectory → analytics → alerts
      → PostgreSQL → API → dashboard + notebook
```

---

## Features

### 1 · High-accuracy ANPR / OCR engine

| Capability | Implementation |
|---|---|
| Vehicle detection | YOLO11m (COCO) — car, motorcycle, bus, truck, bicycle |
| Multi-object tracking | ByteTrack, persistent IDs, state reset per camera |
| Plate detection | Local YOLOv11 on GPU **or** your Roboflow model (auto-selected) |
| Preprocessing | Deskew (Hough), CLAHE, unsharp mask, bilateral denoise, Otsu |
| OCR | `cct-s-v2-global-model` (CCT transformer) via ONNX Runtime on GPU |
| Cross-variant consensus | Variants vote; agreement tempers confidence |
| Temporal aggregation | String vote + per-character vote + peak-evidence blend |
| Format validation | Indian plate grammar, state-code check, position-forced repair |
| Audit trail | Every raw per-frame OCR reading retained and stored |

**Grammar-forced repair never invents characters.** It only corrects a character in
a position the plate grammar *requires* to be a digit or letter (`TNO1AB1234` →
`TN01AB1234`), caps repairs at 2, reduces confidence per repair, and always keeps
the raw string.

### 2 · Single-plate trajectory tracking

- Cross-camera association on OCR-confusion-aware plate similarity (`8↔B`, `0↔O`, `1↔I` cost less than arbitrary substitutions)
- Camera graph with derived road distances and expected travel-time windows
- Spatio-temporal validation: every transition gets a verdict — `PLAUSIBLE` / `SLOW` / `FAST` / `IMPLAUSIBLE` — **with a written reason**
- Route classification: `COMPLETE_CORRIDOR`, `PARTIAL_ROUTE`, `DROPPED_AT_ORIGIN`, `SINGLE_CAMERA_NETWORK`
- Query-based tracking interface: plate → chronological history with timestamps, locations, travel times, evidence

### 3 · Macro traffic flow analytics

- **Unique vehicle counts by class**, rendered live on the output video and stored per camera
- Per-camera volume and plate-read counts
- Origin–destination flow matrix with mean travel time and estimated speed
- Traffic density heatmap on real geography
- Detections-over-time trends, vehicle-class mix
- Processing performance metrics per camera

### 4 · Alert system

- Watchlist matches (database-driven — never auto-seeded)
- Spatio-temporal anomalies (physically impossible transitions)
- Every alert carries what happened, where, when, and **why it was flagged**

### 5 · Platform

- **Any number of cameras** — 1, 2, 5, 20; one per video file, auto-discovered
- GIS dashboard with 7 views
- Jupyter notebook, runnable cell by cell
- Supabase/PostgreSQL with PostGIS
- Run-scoped history: repeated runs accumulate, never overwrite
- Component failure counters — an empty result is never ambiguous

---

## Measured results

All figures below were produced by running this code. Nothing is estimated.

### On the supplied test footage (`camera1.mp4`, 1280×720, 24 fps, 240 frames)

```
vehicle detections      4130
plate detections         217
unique vehicles           83   (car 37 · motorcycle 40 · bus 1 · truck 5)
plates recognised          1   TN01AB1234  @ 0.943 confidence
frames agreeing        12/14   (agreement 0.857), valid Indian format
processing speed        18.2   fps on RTX 5060 Laptop GPU
component failures         0
```

**The OCR audit trail for that plate**, showing why multi-frame consensus matters:

```
frame 25   raw=135119       → garbage
frame 26   raw=TI01AA1334   → wrong
frame 27   raw=TNO1AB1234   → TN01AB1234   (O→0 grammar repair)
frame 34   raw=TN01AB1234   → TN01AB1234   (clean)
                            ⇒ consensus: TN01AB1234
```

A single-frame system would have emitted a wrong plate at high confidence.

### OCR robustness under controlled degradation

Real plate crops, degraded at rising severity, measuring recovery of the clean reading:

| Condition | Multi-variant preprocessing vs raw |
|---|---|
| Low light (darkest) | **+12.5%** |
| Defocus blur (mild–moderate) | **+10.0% / +7.5% / +7.5%** |
| Sensor noise | **+7.5%** |
| Motion blur k≥7, downscale ≥3× | **0% either way** — a real hard limit |

Reproduce with `python scripts/benchmark_ocr_robustness.py`.

### System verification

```
54/54 checks passed   (single camera)
57/57 checks passed   (multi-camera, with overlapping footage)
```

Covers runtime, dynamic camera network, models, pipeline, observation integrity,
association, spatio-temporal validation, plate grammar, alerts, visualisation and
database persistence. Run with `python scripts/verify_system.py`.

---

## Quick start

### Requirements

- Python 3.12
- NVIDIA GPU with CUDA (CPU works, slower)
- A Supabase project (or any PostgreSQL with PostGIS)

### Install

```bash
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env      # then edit with your credentials
```

```ini
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_PROJECT_REF=<project-ref>
SUPABASE_DB_PASSWORD=<database-password>
SUPABASE_REGION=ap-south-1
SUPABASE_POOLER_PORT=5432
SUPABASE_PREFER=pooler

ROBOFLOW_API_KEY=<optional>
ROBOFLOW_MODEL_ID=<optional>
```

> **Connection note.** Supabase's *direct* host (`db.<ref>.supabase.co`) is
> **IPv6-only**. On networks without IPv6 egress it is unreachable, so the code
> defaults to the IPv4 **pooler** and falls back automatically. Passwords
> containing `@`, `:` or `/` are percent-encoded when the DSN is assembled — a
> hand-written connection string with a raw `@` in the password will not parse.

### Create the database schema

Paste [`supabase_schema.sql`](supabase_schema.sql) into the Supabase SQL Editor,
or run:

```python
from urbantrace.db.client import apply_schema
apply_schema()          # idempotent, safe to re-run
```

### Add videos

```
data/videos/
├── camera1.mp4        →  CAM_01
├── camera2.mp4        →  CAM_02
└── camera3.mp4        →  CAM_03
```

Any count, any common format (`.mp4 .avi .mov .mkv .m4v .webm`). Files are sorted
naturally, so `camera2` precedes `camera10`.

---

## Running the notebook

```bash
jupyter notebook UrbanTrace_AI_Realtime.ipynb
```

**Cells run one at a time, top to bottom** — verified: 31/31 execute individually
with zero forward dependencies.

| Cell | What it does |
|---|---|
| 1–5 | Environment, video discovery, models, camera network |
| **6** | **Processes every video** — detection, tracking, OCR. The slow one |
| 7–8 | Processing statistics, health check |
| **9** | **PLATE + TIMESTAMP + CAMERA table** (exported to CSV) |
| 10–12 | Per-camera timeline, OCR audit trail, grammar corrections |
| **13** | **Annotated videos play inline** with boxes, IDs, plates, live counter |
| 14–17 | Cross-camera association, routes, dropped vehicles |
| 18–20 | Route sketch map, GIS map, traffic heatmap |
| 21–22 | Per-camera vehicle counts, traffic analytics |
| 23–24 | Alerts, evidence crops |
| 25 | OCR robustness benchmark on your own footage |
| **26–27** | **Connect to Supabase and persist everything** |
| 28–30 | Read it back via SQL (these work standalone) |
| 31 | Export full JSON |

`MAX_FRAMES = None` processes entire videos; set a number for a fast first pass.

`UrbanTrace_AI_Realtime_executed.ipynb` is a saved copy **with outputs** for
demoing without re-running.

---

## Running the dashboard

```bash
python scripts/serve.py           # http://127.0.0.1:8000
```

| View | Contents |
|---|---|
| **Overview** | KPI tiles, vehicles per camera, route outcomes, route table |
| **GIS Map** | Camera nodes, flow lines weighted by volume, traffic heat layer |
| **Vehicle Tracking** | Plate search → timeline, route map, evidence, OCR audit trail |
| **Traffic Analytics** | Unique vehicle counts, OD flows, trends, vehicle mix, performance |
| **Cameras & Video** | Annotated videos inline, per-camera counts, network table |
| **Alerts** | Watchlist matches, anomalies, watchlist management |
| **Observations** | Every plate observation, evidence gallery |

> **If you edit backend code, restart the server.** On Windows a stale process
> keeps serving old code — `pkill` does not kill Python there. Use:
> ```powershell
> Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
>   Where-Object { $_.CommandLine -match 'serve.py' } |
>   ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
> ```

---

## Camera configuration

Videos contain no GPS and no wall clock, so two things are **configuration**:

1. **Camera coordinates**
2. **Capture start times**

Everything else is measured from the video. Each observation stores
`video_time_s` (measured: frame index ÷ fps) alongside `observed_at`
(derived: configured start + offset), and `clock_source` records which is which —
the system never presents a derived clock as measured.

Generate a config template, then edit it:

```python
from urbantrace.core.cameras import write_config_template
write_config_template()          # → data/cameras.json
```

```json
{
  "cameras": [
    {
      "camera_id": "CAM_01",
      "video_file": "camera1.mp4",
      "name": "Koyambedu Junction",
      "road": "100 Feet Road / Jawaharlal Nehru Salai",
      "latitude": 13.0694,
      "longitude": 80.1948,
      "direction": "WEST",
      "start_time": "2026-08-24T10:00:00"
    }
  ]
}
```

Cameras without an entry fall back to a built-in Chennai-corridor pool (Koyambedu →
Ambattur → Avadi → Thiruninravur → Tiruvallur → Poonamallee → Porur → Guindy →
Tambaram → Chengalpattu → Sriperumbudur → Kanchipuram) and are flagged
`location_source: default` in the UI.

Road distances and expected travel-time windows are **derived** from these
coordinates and drive the plausibility verdicts, so wrong coordinates produce
wrong verdicts.

---

## Database

12 tables + 4 SQL views, PostGIS-enabled. Full DDL in
[`supabase_schema.sql`](supabase_schema.sql).

| Table | Holds |
|---|---|
| `runs` | One row per execution — device, GPU, models, counts |
| `cameras` | Location, direction, capture start + **PostGIS point** |
| `camera_edges` | Derived road links, distance, expected travel-time window |
| `videos` | Source file properties |
| `camera_stats` | Frames, detections, fps, failures, **unique vehicle counts by class** |
| `observations` | **Plate + timestamp + camera + track ID** — the core record |
| `plate_readings` | Every raw per-frame OCR read — the audit trail |
| `global_vehicles` | Cross-camera identity, route array, status, duration |
| `vehicle_observations` | Links global vehicles to their sightings |
| `route_legs` | Per-hop travel time, distance, speed, verdict, reason |
| `alerts` | Fired alerts with reasoning |
| `watchlist` | Plates to alert on |

| View | Purpose |
|---|---|
| `v_camera_counts` | Vehicles and observations per camera |
| `v_camera_flows` | Origin–destination flows with mean travel time and speed |
| `v_plate_timeline` | Chronological sightings for any plate |
| `v_vehicle_counts` | Unique vehicles per camera by class |

Every row is scoped to a `run_id`, so repeated runs accumulate history.

---

## API reference

FastAPI, interactive docs at `/docs`.

| Endpoint | Returns |
|---|---|
| `GET /api/health` | Runtime, GPU, database status |
| `GET /api/runs` | All pipeline runs |
| `GET /api/summary` | KPI summary for a run |
| `GET /api/cameras` | Cameras with locations, stats, counts |
| `GET /api/edges` | Derived camera road links |
| `GET /api/camera-counts` | Vehicles/observations per camera |
| `GET /api/vehicle-counts` | Unique vehicles by class per camera |
| `GET /api/observations` | Plate observations (filter by camera/plate) |
| `GET /api/vehicles` | Global vehicles with routes |
| `GET /api/vehicle/{plate}` | Full history: timeline, legs, raw OCR readings |
| `GET /api/search?q=` | Plate search |
| `GET /api/flows` | Origin–destination flows |
| `GET /api/heatmap` | Observation density per camera |
| `GET /api/timeline` | Detections over time |
| `GET /api/vehicle-types` | Vehicle class distribution |
| `GET /api/stats` | Per-camera processing statistics |
| `GET /api/alerts` | Alerts for a run |
| `GET /api/watchlist` · `POST /api/watchlist/{plate}` | Watchlist |
| `GET /api/video/{camera_id}` | Annotated video (transcodes to H.264 on demand) |
| `GET /api/evidence?path=` | Plate crop image |

---

## Project structure

```
UrbanTrace-AI/
├── UrbanTrace_AI_Realtime.ipynb           main notebook (run this)
├── UrbanTrace_AI_Realtime_executed.ipynb  saved copy with outputs
├── supabase_schema.sql                    paste into Supabase SQL Editor
│
├── backend/urbantrace/
│   ├── core/
│   │   ├── cameras.py          dynamic N-camera network, graph, config
│   │   ├── runtime.py          GPU/cuDNN setup for onnxruntime
│   │   └── config.py           tunables
│   ├── vision/
│   │   ├── detectors.py        YOLO vehicle + plate detection, ByteTrack
│   │   ├── plate_backends.py   Roboflow ↔ local backend selection
│   │   ├── preprocess.py       crop + preprocessing variants
│   │   ├── ocr_engine.py       OCR with cross-variant consensus
│   │   ├── plate_format.py     Indian plate grammar validation
│   │   ├── aggregate.py        temporal aggregation
│   │   └── pipeline.py         per-camera pipeline + counter overlay
│   ├── identity/
│   │   └── association.py      cross-camera association, routes, validation
│   ├── viz/
│   │   ├── route_map.py        sketch map, GIS map, heatmap
│   │   └── video_utils.py      H.264 transcode for browser playback
│   ├── db/
│   │   ├── client.py           Supabase connection
│   │   ├── repository.py       persistence + queries
│   │   └── schema.sql          DDL
│   └── api/main.py             FastAPI app
│
├── frontend/
│   ├── index.html              dashboard, 7 views
│   ├── css/style.css           validated colour palette
│   └── js/app.js               data binding, Leaflet, charts
│
├── scripts/
│   ├── serve.py                     launch dashboard
│   ├── verify_system.py             full self-test
│   ├── build_realtime_notebook.py   regenerate the notebook
│   ├── benchmark_ocr_robustness.py  OCR degradation benchmark
│   ├── benchmark_temporal_blur.py   temporal aggregation benchmark
│   └── run_all_cameras.py           CLI pipeline runner
│
├── models/weights/             YOLO weights (downloaded on first use)
└── data/
    ├── videos/                 ← your input videos
    ├── output/                 annotated videos, CSV, JSON, maps
    ├── evidence/               plate crops
    ├── cameras.json            your camera → location mapping
    └── watchlist.txt           plates to alert on
```

---

## Verification

```bash
python scripts/verify_system.py                 # test on data/videos
python scripts/verify_system.py --cameras 3     # generate overlapping test clips
python scripts/benchmark_ocr_robustness.py      # OCR degradation curve
python scripts/benchmark_temporal_blur.py       # temporal aggregation benefit
```

`verify_system.py` runs 54 checks across every subsystem and reports PASS/FAIL per
check, then purges its own test artefacts.

---

## Honest limitations

These are real and stated deliberately — each would otherwise surface as a question
you could not answer.

**OCR accuracy is not yet measured.** The plate in the test footage was recognised
correctly at 0.943 confidence, but that is a sample size of one. A defensible
accuracy figure needs a few hundred human-verified plate crops. The available
detection dataset contains plate *boxes* only, no plate *text*. Robustness under
degradation **is** measured; absolute accuracy is not. **Do not quote a >90%
accuracy figure until it has been measured on held-out labelled data.**

**Unique vehicle counts are track-based, not ground truth.** The count is sensitive
to the persistence threshold:

| Threshold | Total |
|---|---|
| ≥1 frame | 110 |
| ≥3 frames (default) | 83 |
| ≥8 frames | 63 |
| ≥20 frames | 48 |

A vehicle occluded and re-acquired gets a new track ID and is counted twice; parked
background vehicles are counted as "seen". The honest phrasing is *"83 unique
tracked vehicles"*, not *"83 vehicles passed"*. A virtual counting line would give a
stricter, better-defined metric.

**Severe motion blur is unrecoverable in a single frame.** Measured: 0% exact
recovery at blur kernel ≥7 or 3× downscale. The system's answer to blur is
multi-frame consensus — it needs *some* frame in the track to be sharp.

**Cross-camera routing requires shared vehicles.** With one video, or with videos
of unrelated scenes, every vehicle is correctly reported as single-camera and no
routes are drawn. This is the data speaking, not a failure.

**Timestamps are part measured, part configured.** `video_time_s` is measured;
wall-clock is `configured camera start + offset`. Set real capture times in
`cameras.json` if you know them.

**Roboflow hosted inference is optional.** If the workspace has no credits the
system reports the exact reason and uses local GPU weights. Local inference is
faster and free; hosted inference costs one API call per frame.

---

## Security

- Credentials live in `.env`, which is **git-ignored**. `.env.example` holds placeholders only.
- The connection status endpoint redacts the password.
- The evidence endpoint is path-constrained to the evidence directory (no traversal).
- Watchlist data is database-driven and never auto-populated from detections.

> **Rotate any credential that has been shared in plain text** — chat, email, or a
> screenshot. In Supabase: Settings → Database → Reset password, then update `.env`.
> No code changes are needed.

---

## Tech stack

| Layer | Technology |
|---|---|
| Vehicle detection | Ultralytics YOLO11m (COCO) |
| Tracking | ByteTrack |
| Plate detection | YOLOv11 (local) / Roboflow (hosted) |
| OCR | CCT transformer via ONNX Runtime (CUDA) |
| CV | OpenCV, NumPy |
| Backend | FastAPI, Uvicorn |
| Database | Supabase / PostgreSQL 17 + PostGIS |
| Frontend | Vanilla JS, Leaflet, leaflet.heat |
| Notebook | Jupyter, pandas, matplotlib, folium |
| Video | imageio-ffmpeg (H.264 transcode) |
