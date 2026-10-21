# 1. Solution, innovation and novelty

## 1.1 The problem

Cities already own large CCTV/ANPR (automatic number-plate recognition) networks,
but most deployments treat each camera as an island: detect a plate, store a row,
stop. The operator learns *that* a plate was seen, never *the journey*, and gets no
traffic intelligence from the infrastructure they already pay for.

Four questions are left unanswered by a per-camera system:

| Question | What is needed |
|---|---|
| Who is this vehicle? | Detection → plate localisation → OCR → format validation, robust to blur and noise |
| Where did it go, and when? | Linking sightings across cameras into one journey, and judging whether the journey was physically possible |
| How is traffic moving? | Counts, density, origin–destination flows, speed and congestion relative to each road's own normal |
| What needs attention? | Watchlist hits and route anomalies, each with a written reason |

## 1.2 The solution

UrbanTrace AI is a central platform that turns isolated camera feeds into one
connected picture. It has two cooperating halves that share the same models:

* **Batch pipeline** (`urbantrace.vision`, notebook, `scripts/`): process a folder
  of videos into a *run* — fully auditable, reproducible, with an annotated video,
  evidence crops and a raw-reading audit trail.
* **City-scale live layer** (`urbantrace.scale`): a continuous network of cameras,
  one shared set of models, a streaming sighting feed, horizontally scalable
  workers, and analytics served from PostgreSQL.

End to end:

```
video/RTSP → vehicle detect → per-camera ByteTrack → plate detect → crop
  → preprocessing variants → batched OCR → grammar validation
  → per-track temporal vote → Sighting (plate, camera, time, confidence)
  → identity resolution (same vehicle across cameras?)
  → feasibility check (could it physically get there?) → leg + verdict
  → alerts → batched write → PostgreSQL/PostGIS
  → REST API → React dashboard / notebook
```

## 1.3 What is innovative

These are design contributions, each backed by code and, where possible, a
measurement. They are listed by how much they matter, not by how impressive they
sound.

### (a) Detection is shared; tracking is per-camera

The convenient Ultralytics call `model.track(persist=True)` stores tracker state
inside the model object. For one video that is fine. For a network it forces a bad
choice: one model per camera (1000 × tens of MB of weights — no GPU holds that) or
one shared model whose track identities bleed between cameras.

UrbanTrace separates the two: **one** detector serves every camera, batched across
cameras, and `MultiCameraTracker` holds a lightweight `BYTETracker` per camera
(kilobytes of Kalman state). Camera count is then bounded by bandwidth and CPU, not
GPU memory. Measured peak VRAM for 16 cameras through the full path: **0.75–0.78 GiB**.

### (b) A real defect in the tracking library was found and neutralised

Ultralytics allocates track ids from a *class-level* counter and `BYTETracker.__init__`
resets it. Measured on this build: after camera A had allocated 3 ids, creating
camera B's tracker reset the counter to 1, so camera A began reissuing ids it was
still using and two vehicles shared one identity. `tracking._tracker_class()` makes
the reset a no-op and serialises allocation, so ids are unique network-wide. This is
pinned by `test_new_camera_does_not_reissue_ids_of_a_running_camera`.

### (c) Identity resolution without all-pairs comparison

The batch pipeline compares every plate with every other (O(n²) — 5 billion
comparisons at 100,000 plates). The live layer uses **blocking**: each plate yields
keys (its OCR-confusion-folded form plus single-deletion variants) and two plates
can only match if they share a key. Measured: **0.83 candidate comparisons per
insert, constant** from 2,000 to 40,000 vehicles, with exact cluster counts
(2,000 → 2,000; 10,000 → 10,000; 40,000 → 40,000).

### (d) Precision over recall in identity, stated and enforced

Merging two vehicles into one identity is the worst error the system can make — it
puts one owner's journey under another's plate. A similarity threshold alone is too
permissive (`TN01AB1234` vs `TN01AB1235` scores 0.90). `same_vehicle()` therefore
requires every difference to be explainable as an OCR confusion or one dropped
glyph. The cost of this design is measured honestly in document 3 (§3.7): plates
that differ only by a confusable character *are* merged by plate logic, and it is the
physical-feasibility check — a car cannot be in two places — that separates them.

### (e) Every verdict carries a written reason

Each camera-to-camera transition is `PLAUSIBLE`, `SLOW`, `FAST` or `IMPLAUSIBLE`
with a human-readable sentence ("9.18 km in 4.2 min implies 131 km/h…"). Each alert
likewise explains itself. An alert that cannot explain itself cannot be acted on,
and in this domain acting on one means stopping a vehicle.

### (f) Honest uncertainty as a feature

The system refuses to fabricate. No sample data, no pre-seeded watchlist, no
invented plate; empty results are empty. Raw OCR is always retained next to the
repaired string. Grammar repair never invents characters — it only corrects a
character in a position the plate grammar *requires* to be a digit or a letter, caps
repairs at 2 and lowers confidence per repair. Illegible test plates are excluded
from accuracy **and their count is printed beside every result**.

### (g) Load is shed smoothly and fairly

* `FrameBus` drains round-robin so a few high-frame-rate cameras cannot starve the
  rest, and drops the **oldest** frame when a queue is full (a stale frame has no
  value in live monitoring; an unbounded queue turns a slow GPU into an OOM crash).
* `RateController` lowers the analysis rate for *every* camera together when the
  drop ratio rises (multiplicative decrease ×0.8, gentle increase ×1.1 + 0.05) so
  coverage degrades evenly instead of random cameras going dark.
* The plate detector is skipped for frames whose camera has no unresolved track
  (measured: ~40 % of plate inferences skipped), and OCR is batched across plates
  (measured 3.2× faster, byte-identical readings).

### (h) Coordination-free worker assignment

Workers heartbeat into a table, read the same list, and each independently computes
which cameras it owns using **rendezvous hashing with a load bound**. There is no
coordinator to fail. Verified live (document 9): three workers took 15 cameras, a
hard-killed worker's cameras were re-owned by the survivors after the TTL, and a
camera added through the API was picked up within one refresh cycle.

## 1.4 What is novel — and what is not

Honest framing matters in a project report, so:

* **Not novel:** YOLO detection, ByteTrack, CCT-based plate OCR, PostgreSQL, nginx,
  rendezvous hashing, adaptive rate control. These are deliberately *standard,
  proven components*.
* **Novel is the integration and the engineering around it:** a camera-network
  architecture that shares models while isolating per-camera state; blocking-based
  identity resolution tuned for OCR-confusion noise with an explicit
  precision-first merge rule; spatio-temporal feasibility verdicts with reasons;
  and a measurement-first methodology where every claim has a source and every
  weakness is published.

## 1.5 Feature inventory

| Area | Capabilities |
|---|---|
| ANPR | Vehicle + plate detection, multi-variant preprocessing (deskew, CLAHE, sharpen, denoise, Otsu), batched OCR, Indian plate grammar validation, per-track temporal voting, full raw-reading audit |
| Counting | Unique vehicles per class with fragmentation stitching, clutter/flicker gates, two modes (traffic vs presence), optional count line |
| Trajectory | Cross-camera association, camera graph with road distances, per-leg verdicts, route classification, plate → chronological history |
| Analytics | Density, heatmap, origin–destination matrix, congestion vs per-corridor baseline, flow trends, camera health |
| Alerts | Watchlist (OCR-confusion tolerant), route anomalies, explainable reasons |
| Live | RTSP/HTTP/file/device feeds, server-sent events, phone-as-camera streamer, motion gate, reconnect with backoff |
| Platform | FastAPI, React dashboard, legacy dashboard, notebook, Docker Compose, nginx, Prometheus metrics, worker health probes |
