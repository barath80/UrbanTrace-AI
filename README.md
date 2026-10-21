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
- [Live monitoring](#live-monitoring)
- [Vehicle counting](#vehicle-counting)
- [Real-road routing](#real-road-routing)
- [React dashboard](#react-dashboard)
- [Camera configuration](#camera-configuration)
- [Database](#database)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Verification](#verification)
- [Production deployment](#production-deployment-100-live-cameras)
- [Honest limitations](#honest-limitations)
- [Security](#security)
- [Technical documentation](#technical-documentation)

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
| OCR | `cct-xs-v2-global-model` (CCT transformer) via ONNX Runtime on GPU |
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

- **Unique vehicle counts by class**, deduped and auditable — see [Vehicle counting](#vehicle-counting)
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
raw tracker ids          110   ← NOT a vehicle count; see below
unique vehicles (traffic) 28   (car 10 · motorcycle 16 · truck 2)   ← converged
unique vehicles (presence) 45   (car 18 · motorcycle 22 · bus 1 · truck 4)  ← approximate
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
- Node.js 18+ (for the React dashboard)
- NVIDIA GPU with CUDA (CPU works, slower)
- A Supabase project (or any PostgreSQL with PostGIS)

### Install

```bash
pip install -r requirements.txt
cd web && npm install && cd ..
```

### Run everything

`run.py` checks prerequisites, builds the frontend if needed, processes every
video in `data/videos/`, persists the run, and serves the dashboard:

```bash
python run.py
```

Then open **http://127.0.0.1:8000**.

| Command | Does |
|---|---|
| `python run.py` | check → build → process → serve |
| `python run.py check` | prerequisites only (models, videos, database, routing) |
| `python run.py process` | process + persist, no server |
| `python run.py serve` | serve only, using the last saved run |
| `python run.py build-web` | rebuild the React frontend |
| `python run.py test` | 33 unit tests + the 55-check self-test |

Add `--no-video` to skip writing annotated output videos (much faster), and
`--port 8080` to move the server.

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

## Live monitoring

Batch processing answers "what was in that video". Live ingest answers "what is
happening now". [`ingest/live.py`](backend/urbantrace/ingest/live.py) runs the
same vision stack continuously against a source and streams what it finds.

**A source is anything OpenCV can open**, which is what makes this real rather
than a replay of a stored result:

| Source | Example |
|---|---|
| A real ANPR camera | `rtsp://user:pass@10.0.0.7/stream1` |
| An attached camera | `0`, `1`, `2` … |
| A video file | played at its own frame rate, so a recording behaves like a feed |

```bash
python scripts/serve.py        # then open Live Monitor and press Start
```

```bash
# or drive it directly
curl -X POST "localhost:8000/api/live/start?camera_id=CAM_01&detect_every=3"
curl -X POST "localhost:8000/api/live/start?camera_id=CAM_02&source=rtsp://…"
curl  -N     "localhost:8000/api/live/stream?camera_id=CAM_01"   # SSE
curl -X POST "localhost:8000/api/live/stop"
```

Events arrive over **server-sent events** — one-directional traffic, survives
proxies, and the browser reconnects on its own:

| Event | Fired when |
|---|---|
| `started` / `stopped` / `ended` | session lifecycle |
| `stats` | twice a second: counts, fps, uptime |
| `vehicle` | a vehicle passed the counting gates |
| `plate` | a plate reached **temporal consensus**, with its evidence crop |
| `alert` | that plate is on the watchlist |
| `cycle` | a file source replayed, and counting state reset |

### Three things it is honest about

**A plate is published on consensus, not on a glimpse.** A single-frame OCR
result is exactly what temporal aggregation exists to distrust, so nothing is
emitted until enough frames agree.

**Throughput is reported against the source.** On an RTX 5060 the full stack
runs at roughly **17 fps against a 24 fps feed**, so the UI shows a `BEHIND`
badge and names the lever (`detect_every`). A real camera cannot be slowed
down, so pretending to keep up would mean silently dropping frames.

**A replayed file resets its counters.** Without that the same vehicles are
counted again on every pass and the running total climbs without meaning.

Alerts fire live against the same watchlist the batch pipeline uses, and the
same identity logic — a plate read under a merged OCR variant still matches.

---

## Vehicle counting

**A tracker id is not a vehicle.** Counting distinct ByteTrack ids overcounts
badly, in three independent ways. On the supplied 10-second clip the tracker
emits **110 ids** for a scene containing nothing like 110 vehicles:

| Inflation source | What it looks like here |
|---|---|
| **Fragmentation** | The tracker loses a vehicle behind the signal pole and re-acquires it under a new id. 110 raw ids collapse to **53** vehicles once fragments are rejoined — **57 stitches**. |
| **Clutter** | 40×40px far-field blobs and vehicles parked at the kerb. Neither is traffic passing the camera. |
| **Flicker** | One-to-three frame spurious detections, which get an id like anything else. |

[`counting.py`](backend/urbantrace/vision/counting.py) addresses each:

- **Fragment stitching** — when a new tracker id appears where a recently lost
  track was predicted to be (constant-velocity, size-compatible, and not alive
  in the same frame), it is treated as the same vehicle.
- **Evidence gates** — a track counts if it is *either* big enough to be
  unambiguous *or* persistent enough that noise cannot explain it. Requiring
  both discarded a distant motorcycle held for 174 consecutive frames and a
  truck entering two seconds before the clip ended.
- **Depth-invariant motion** — pixel displacement scales with 1/depth, so a flat
  pixel threshold silently drops all far-field traffic. Motion is measured in
  **body lengths** (displacement ÷ √median box area) instead.
- **Area-weighted class vote** — the frames where the vehicle was close decide
  its class, so a distant blob's guess cannot outvote them.

### Two counts, both reported

"How many vehicles" has two honest answers at an intersection where half the
vehicles in shot are parked, so both are always computed:

| Mode | Means | `camera1.mp4` | Duplicate-free? |
|---|---|---|---|
| `traffic` *(default)* | Vehicles that **travelled through** the scene, or crossed the counting line when one is configured | **28** | Yes — converged (see below) |
| `presence` | Every distinct real vehicle that **appeared**, parked included | **45** | **No — treat as approximate** |

**`traffic` is the trustworthy number; `presence` is not.** A moving vehicle has
a velocity, so predicting where a lost track should reappear is unambiguous and
fragments rejoin reliably. Parked vehicles have no velocity, and this scene has
an autorickshaw stand where several sit clustered, partly behind a pole, at
~40px. Whether that is two autos or three is genuinely ambiguous from the
footage. Use `presence` as a rough upper bound, not a figure to quote.

Set a `count_line` for a camera in `cameras.json` and `traffic` becomes
"vehicles that crossed this line" — the strictest definition, and the one you
can verify by watching the video.

### Checking the count

A bare number is unverifiable, so the count ships with its evidence:

```bash
python scripts/cache_tracks.py --all          # detect + track once, cache it
python scripts/count_vehicles.py --all --sheet --video
```

| Output | What it is for |
|---|---|
| `<video>.sheet.jpg` | **Contact sheet** — one crop per counted vehicle, numbered. Duplicates and false positives are obvious on sight; the count is auditable in about a minute. |
| `<video>.counts.json` | Every vehicle, its class and frame span, its raw tracker ids — and for each vehicle **not** counted, the exact gate that rejected it. |
| `<video>.counted.mp4` | Boxes coloured by class, the counting line, and a running total that increments at the frame each vehicle qualifies. Rejected clutter is drawn grey. |

### The duplicate test

Counting is per **vehicle**, never per frame — that part is structural. On this
clip 28 counted vehicles account for 2991 frame-appearances out of 4130 raw
detections; the longest-lived vehicle appears in 238 frames and contributes 1.

Whether *fragments* are fully rejoined is the empirical question, and
`--sensitivity` answers it by re-counting across a 100× range of stitching
settings. If one vehicle were still counted twice, stitching harder would find
the join and the total would fall:

```bash
python scripts/count_vehicles.py --all --sensitivity
```

```
  gap   dist | stitched  traffic  presence
    0  0.055 |        0       25        60
   30  0.055 |       57       28        45  <- default
  120   0.15 |       75       27        31
  TRAFFIC   25-29  CONVERGED - no duplicates left for stitching to merge
  PRESENCE  31-60  NOT converged - duplicates (or over-merges) remain
```

`traffic` holds at 25–29 however hard you stitch: it has converged, which is
what "no duplicate counting" actually means in measurable terms. `presence`
halves, which is why it is flagged approximate above.

Tracking is cached, so re-counting with different thresholds does not re-run the
models. Thresholds are all overridable:

```bash
python scripts/count_vehicles.py --all --mode presence
python scripts/count_vehicles.py --all --min-frames 12 --min-disp-frac 0.06
python scripts/count_vehicles.py --all --line 0.0,0.62,1.0,0.55   # count crossings
```

The counting logic is covered by [`tests/test_counting.py`](tests/test_counting.py)
(15 tests, no model or video needed) — each test names the real failure it guards
against, including the two false negatives found while tuning against this clip.

---

## Real-road routing

Camera-to-camera paths are **real driven roads**, not straight lines. Geometry,
distance and typical driving time come from the
[GraphHopper Routing API](https://graphhopper.com/), and everything downstream
uses them: the map, the per-vehicle route, and the travel-time verdicts.

This is a correctness fix, not decoration. Distance used to be great-circle
× 1.25, and the PLAUSIBLE / FAST / SLOW verdict is computed from it:

| Koyambedu → Ambattur | Distance | Implied speed for a 14.0 min run |
|---|---|---|
| Great-circle × 1.25 (old) | 8.26 km | 35 km/h |
| **Real road (GraphHopper)** | **9.18 km** | **39 km/h** |

### Setup

```bash
# .env  (git-ignored)
GRAPHHOPPER_API_KEY=your_graphhopper_key
```

**The key never reaches the browser.** The frontend asks `/api/road-route` and
`/api/vehicle-paths`; the server calls GraphHopper. Every response is cached on
disk under `data/cache/routes/`, keyed by coordinates — cameras do not move, so
a corridor is fetched once and redraws are free.

### It always returns a route

With no key, no network, or a quota error, routing falls back to the
great-circle estimate rather than failing. The fallback is **labelled**, end to
end — `source`, `is_real_road`, a dashed line on the map and an `ESTIMATE`
badge in the UI — so a guess is never presented as a measured road.

| Endpoint | Returns |
|---|---|
| `GET /api/routing` | Whether a key is configured, and how many routes are cached |
| `GET /api/road-route?from_camera=&to_camera=` | Road geometry between two cameras |
| `GET /api/vehicle-paths` | Road path for every vehicle with a multi-camera route |
| `GET /api/vehicle-path/{plate}` | One vehicle's path, ready to draw |

### What is measured and what is not

A route is a sequence of **sightings**, not a continuous track. What happened
between two cameras was never observed. The path drawn is the road a driver
would take, and every response carries that caveat:

> Camera sightings are measured. The road between two sightings is the route a
> driver would take, from GraphHopper — inferred, not observed.

Run it, and persist so the dashboard picks it up:

```bash
python scripts/run_all_cameras.py --save
python scripts/run_all_cameras.py --no-road-routing   # offline, estimates only
python scripts/serve.py
```

---

## React dashboard

The dashboard is a Vite + React + TypeScript app in [`web/`](web/), built to
`web-dist/` and served by FastAPI at the same origin as the API. It is
**responsive**: one build works on desktop and in a phone browser, where the
tab rail becomes a thumb-reachable bottom bar.

```bash
cd web && npm install && npm run build   # -> web-dist/, served at /
cd .. && python scripts/serve.py         # http://127.0.0.1:8000
```

For frontend work, `npm run dev` in `web/` proxies `/api` to port 8000.
The original vanilla-JS dashboard is untouched and still served at `/legacy`.

| Piece | What it does |
|---|---|
| **MapLibre GL** | Vector basemap with real tilt and 3D building extrusions, a native GPU heatmap layer, and road geometry drawn as line layers. |
| **Framer Motion** | View transitions, the sliding tab underline, and count-up statistics. Everything respects `prefers-reduced-motion`. |
| **Recharts** | Class mix, per-camera counts and the detections-over-time series. |

### Search a plate, get everything

The Vehicle Tracking view is one request to `/api/vehicle-dossier/{plate}`,
which assembles server-side:

- every **sighting** with timestamp, camera and location
- the **road path** it travelled, on real roads
- that vehicle's **own heat surface** — sightings weighted heavily (measured),
  the road between them weighted low (inferred)
- **per-leg verdicts** with the real road distance behind each one
- its **watchlist alerts** and the **raw per-frame OCR** behind every identification

Assembling it server-side is deliberate. When the client stitched these from
separate endpoints the halves disagreed — the view once reported "2 cameras"
and "1 sighting" for the same car.

---

## Camera configuration

Videos contain no GPS and no wall clock, so two things are **configuration**:

1. **Camera coordinates**
2. **Capture start times**
3. **The counting line**, if you want counts defined as line crossings

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
      "start_time": "2026-08-24T10:00:00",
      "count_line": [0.0, 0.62, 1.0, 0.55]
    }
  ]
}
```

`count_line` is `[x1, y1, x2, y2]` in **normalised 0–1 frame coordinates**, so it
survives a change of resolution. When set, the camera's `traffic` count becomes
"vehicles whose centre crossed this line" instead of "vehicles that moved".
Leave it out (or `null`) to count by motion. Draw it across the carriageway you
want to count and check the result in `<video>.counted.mp4`, which renders the
line.

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
│   │   ├── counting.py         unique-vehicle counting (dedupe + gates)
│   │   └── pipeline.py         per-camera pipeline + counter overlay
│   ├── identity/
│   │   └── association.py      cross-camera association, routes, validation
│   ├── alerts/
│   │   └── engine.py           watchlist matching over vehicle identities
│   ├── ingest/
│   │   └── live.py             real-time sessions + SSE event stream
│   ├── analytics/
│   │   └── vehicle.py          per-vehicle dossier stats + heat surface
│   ├── routing/
│   │   └── graphhopper.py      real road geometry + distance, disk-cached
│   ├── trajectory/
│   │   └── paths.py            per-vehicle road path for the map
│   ├── viz/
│   │   ├── route_map.py        sketch map, GIS map, heatmap
│   │   └── video_utils.py      H.264 transcode for browser playback
│   ├── db/
│   │   ├── client.py           Supabase connection
│   │   ├── repository.py       persistence + queries
│   │   ├── migrations/         additive DDL for existing databases
│   │   └── schema.sql          DDL
│   └── api/main.py             FastAPI app
│
├── web/                        React dashboard (Vite + TS) -> web-dist/
│   └── src/
│       ├── api.ts              typed client for every endpoint
│       ├── components/CityMap.tsx   MapLibre GL 3D map surface
│       └── views/              one file per dashboard view
│
├── frontend/                   legacy vanilla-JS dashboard, served at /legacy
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
│   ├── cache_tracks.py              cache detection+tracking per video
│   ├── count_vehicles.py            count + contact sheet + counted video
│   └── run_all_cameras.py           CLI pipeline runner
│
├── tests/
│   ├── test_counting.py        unique-vehicle counting (no model needed)
│   ├── test_live.py            live sessions (models stubbed, no GPU)
│   └── test_routing.py         real-road routing + fallback (no network)
│
├── models/weights/             YOLO weights (downloaded on first use)
└── data/
    ├── videos/                 ← your input videos
    ├── cache/tracks/           cached per-frame tracks, for fast re-counting
    ├── cache/routes/           cached GraphHopper road geometry
    ├── output/                 annotated videos, CSV, JSON, maps
    ├── output/counts/          per-video counts, contact sheets, counted video
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

## Production deployment (100+ live cameras)

```
 browsers ──► nginx (lb) ──► api ×N        stateless, any replica serves any request
                                │
                                ▼
                      PostgreSQL / PostGIS ◄──── worker ×M   ingest, GPU, one per slice of the city
                                                   │
                                      RTSP / HTTP / file feeds (100s of cameras)
```

```bash
cp .env.example .env                       # set POSTGRES_PASSWORD, UT_API_KEY
# put detector weights in models/weights/
docker compose up -d --scale api=2 --scale worker=4
```

Cameras are data, not code. Add them in the **Fleet** tab (one at a time or from a
CSV), through `POST /api/network/cameras[/bulk]`, or with
`python -m urbantrace.scale.worker --import-cameras data/cameras.json`. Each has a
`source` (an `rtsp://` URL, an HTTP stream, a file, or empty for push-only) and
real coordinates. Workers re-read the estate every 15 s; no restart is needed.

**How the load is spread**

* Workers heartbeat into `net_workers`. Each computes, from the same list, which
  cameras it owns, by rendezvous hashing on the camera's sector (else a 3 km grid cell), bounded
  to 1.25x the average load so clumped cameras still spread evenly. A lost worker's
  cameras move after a 20 s TTL. Nothing coordinates, so nothing is a single point of failure.
* Neighbouring cameras land on the same worker so their shared identity index can
  measure the legs between them. A leg between two cameras on either side of a cell
  edge is not written to `net_legs`; trajectory queries, which join in SQL, still show it.
* API replicas are stateless behind nginx; scale them independently with `--scale api=N`.
  `/api/ready` takes a replica out of rotation when its database is unreachable.

**How the compute is kept low** (`urbantrace/scale/capture.py`, `adaptive.py`)

* Each feed is sampled at `UT_TARGET_FPS` (default 6), not at the camera's 25-30. Only
  due frames are converted; the rest are `grab()`-ed and discarded.
* A motion gate skips detector passes on an unchanged scene, with a keyframe every 2 s.
* Frames are downscaled to 1280 px wide before queueing.
* All feeds share one detector and one OCR engine, batched across cameras; the plate
  pass is skipped when no vehicle on that camera is still unread.
* If the GPU saturates, the analysis rate for every camera falls together (down to
  `UT_MIN_FPS`) and recovers when load eases, instead of random frames being lost.
* Point cameras at their low-resolution sub-stream: `grab()` still decodes.

**Operations**

| | |
|---|---|
| Worker probes | `:9100/healthz`, `/readyz`, `/metrics` (Prometheus), `/status` |
| Fleet view | `GET /api/network/fleet`, `/workers`, `/feeds` |
| Auth | `UT_API_KEY` protects writes; `UT_API_AUTH=all` protects reads too |
| Database | `DATABASE_URL` for any PostGIS server; Supabase still works |
| Shutdown | workers flush buffered sightings and deregister on SIGTERM |

Measured on an RTX 5060 Laptop (8 GB), 16 looping 1080p clips through the whole path
(`python scripts/bench_fleet.py --cameras 16`): **about 50 analysed frames/s**, peak
VRAM under 1.5 GiB. VRAM is not the limit. Larger batches, a third worker, and smaller
detector input sizes did not raise it (they measured 33-49 fps), so treat ~50 fps as
this machine's ceiling: roughly 8 cameras at 6 fps, 25 at 2 fps, or 50 at 1 fps. The
clips are continuously busy; real feeds have quiet periods the motion gate skips, so
re-run the benchmark against your own footage before sizing. More cameras than that
need more workers on more GPUs.

Capacity is set by the GPU, not by the camera count: see
[backend/urbantrace/scale/README.md](backend/urbantrace/scale/README.md) for measured
figures. Plan workers from `frames/s per worker` in the Fleet tab, not from a guess.

## Honest limitations

These are real and stated deliberately — each would otherwise surface as a question
you could not answer.

**OCR accuracy is measured, and it is modest.** On 80 hand-transcribed held-out
plates (9 illegible ones excluded) single-frame OCR is **71.3 % exact-match** and
**92.2 % character accuracy**; end to end with the plate detector it is **72.5 %**.
The dominant failure is a dropped trailing character. Do not quote a >90 % plate
accuracy figure. Multi-frame voting demonstrably changes answers but its accuracy has
not been scored (no video with plate ground truth). Full numbers and caveats:
[docs/03-models-and-accuracy.md](docs/03-models-and-accuracy.md).

**Vehicle class labels are wrong for autorickshaws.** The vehicle detector is
`yolo11m` on COCO, and COCO has no autorickshaw class. On this Chennai footage
autorickshaws are a large share of traffic, and each one is forced into whichever
COCO class fits worst-least — they appear scattered across `car`, `truck`, `bus`
and `motorcycle` in the contact sheet. **The totals are unaffected** (each auto is
still counted exactly once) but the per-class breakdown is not trustworthy for
Indian traffic. Fixing it needs a detector fine-tuned with an `autorickshaw`
class; it is not a threshold change.

**Counts are auditable, not ground truth.** No human-labelled count exists for
this footage, so "correct" here means every counted vehicle is a real distinct
vehicle and no real vehicle passing the camera is missed — checked against the
contact sheet, not asserted. Thresholds are defaults that suit this scene; a
camera looking down a motorway would want its own, and `count_line` in
`cameras.json` is the way to pin the definition down hard.

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
| Database | PostgreSQL + PostGIS (Supabase in development, `postgis/postgis:16` container in production) |
| Frontend | React + TypeScript + Vite, Leaflet (legacy vanilla-JS dashboard kept at `/legacy`) |
| Notebook | Jupyter, pandas, matplotlib, folium |
| Video | imageio-ffmpeg (H.264 transcode) |

---

## Technical documentation

In-depth design documents live in [docs/](docs/README.md):

1. [Solution, innovation and novelty](docs/01-solution-and-novelty.md)
2. [Technology choices — why this, why not the others](docs/02-technology-choices.md)
3. [Models and measured accuracy](docs/03-models-and-accuracy.md)
4. [Tracking and identity resolution](docs/04-tracking-and-identity.md)
5. [Database design (DBMS)](docs/05-database-dbms.md)
6. [Networking and load balancing](docs/06-networking-and-load-balancing.md)
7. [Concurrency and thread safety](docs/07-concurrency-and-thread-safety.md)
8. [Security](docs/08-security.md)
9. [Verification, defects found, and limitations](docs/09-verification-and-limitations.md)
