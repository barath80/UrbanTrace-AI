# City-scale layer

The `urbantrace.vision` pipeline processes a folder of videos into a *run*.
This package runs a **live camera network** instead: thousands of cameras, one
shared set of models, a continuous sighting stream, and analytics served from
Postgres rather than from process memory.

Both coexist. Nothing here changes the batch pipeline or its endpoints.

---

## Data flow

```
  camera 1 ─┐
  camera 2 ─┤  admit()          drain_batch()          detect_batch()
     ...    ├─► FrameBus ──────► InferencePool ───────► SharedModels
  camera N ─┘  per-camera        N worker threads       ONE vehicle model
               bounded queue,                           ONE plate model
               drop-oldest                              ONE OCR engine
                                        │
                                        ▼
                              MultiCameraTracker
                              one ByteTrack per camera
                                        │
                                        ▼
                              temporal OCR vote  ──► Sighting
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
             IdentityIndex        AlertEngine         SightingSink
             (which vehicle?)     (who cares?)        (batched INSERT)
                    │                   │                   │
                    └───────────────────┴───────────────────┘
                                        ▼
                                   PostgreSQL
                                        │
                            trajectory · density · O-D
                            congestion · heatmap · alerts
```

---

## The five decisions that make it scale

### 1. Models are shared; tracker state is per-camera

`model.track(persist=True)` keeps tracker state **inside the model object**.
For one video that is convenient. For a city it forces a choice between one
model per camera (1000 × ~40 MB of weights — no GPU holds that) and one shared
model whose track identities bleed between cameras.

So detection and tracking are separated. One detector serves everyone;
`MultiCameraTracker` holds a `BYTETracker` per camera. A tracker instance is a
few kilobytes of Kalman state, so ten thousand of them cost less than one extra
copy of the detector.

**A real defect this exposed.** Ultralytics allocates track ids from
`BaseTrack._count`, a *class* attribute, and `BYTETracker.__init__` calls
`reset_id()` on it. Measured on this build: after one camera had allocated 3
ids, creating a second camera's tracker set the counter back to 1 — so a camera
already running began reissuing ids it was still using, and two vehicles shared
one identity. `tracking.MultiCameraTracker._tracker_class()` neutralises the
reset and serialises allocation, making ids monotonic and unique network-wide.
Covered by `test_new_camera_does_not_reissue_ids_of_a_running_camera`.

### 2. Work is batched, and skipped when it cannot change the answer

- Frames from *different cameras* are batched into one detector call.
- The vehicle detector runs at 640, the plate detector at 960. Vehicles are
  large and are found reliably at the lower size; plates lose glyph detail
  below 960.
- The plate pass is skipped entirely for a frame whose camera has no
  unresolved track. Measured: **40% of plate inferences skipped** in steady
  state.
- OCR runs **once per batch**, not once per plate: every preprocessing variant
  of every plate is flattened into a single engine call. Measured **3.2×
  faster, byte-identical readings** (`ocr_engine.read_many`).
- A track stops consuming OCR once it is read confidently (`settle_confidence`).

Net effect on the reference machine: **35.9 → 52.3 analysed frames/sec**.

### 3. Queues are bounded and drop the oldest

Each camera holds at most `queue_depth_per_camera` frames. When a camera
outruns the GPU the **oldest** frame is discarded — in live monitoring a stale
frame has no value, and an unbounded queue turns a temporary slowdown into an
out-of-memory crash. Drops are counted and reported, never hidden.

`FrameBus.admit()` additionally rate-limits each camera to
`target_fps_per_camera`. A rejected frame costs no decode, no transfer and no
GPU time, which is the cheapest optimisation available.

### 4. Identity resolution is blocked, not all-pairs

The batch pipeline compares every plate with every other plate. At 100,000
plates that is 5 billion comparisons.

Here each plate is reduced to **blocking keys** chosen so two plates can only
be similar if they share one: the OCR-confusion-folded canonical form, plus
single-deletion variants. Lookup touches a handful of candidates.

Measured, 45% OCR noise, 1000 cameras:

| vehicles | sightings | ingest | rate | clusters (ideal) | comparisons/insert |
|---|---|---|---|---|---|
| 2,000 | 12,000 | 0.37 s | 32,169/s | 2,000 (2,000) | 0.83 |
| 10,000 | 60,000 | 2.00 s | 29,950/s | 10,000 (10,000) | 0.83 |
| 40,000 | 240,000 | 25.9 s | 9,276/s | 40,000 (40,000) | 0.83 |

Constant per-insert cost, and exact cluster counts — no splitting, no merging.

**Precision matters more than recall here.** Similarity alone is too permissive:
on a ten-character plate a 0.82 threshold tolerates nearly two full character
edits, so `TN01AB1234` and `TN01AB1235` score 0.90 and would merge — putting one
owner's journey under another's plate. `same_vehicle()` instead requires every
difference to be explainable as OCR confusion (canonical forms identical) or a
single dropped/spurious glyph. See `test_same_vehicle_rejects_real_differences`.

### 5. The camera graph is k-nearest, not all-pairs

All-pairs on 1000 cameras is 499,500 edges, almost all between cameras on
opposite sides of the city that describe no road a vehicle would take between
consecutive sightings.

| cameras | build | edges | all-pairs | `near()` |
|---|---|---|---|---|
| 1,000 | 49 ms | 7,106 | 499,500 | 29 µs |
| 5,000 | 252 ms | 35,408 | 12,497,500 | 134 µs |
| 10,000 | 697 ms | 70,652 | 49,995,000 | 294 µs |

Lookups use a uniform lat/lon grid, so cost is independent of estate size.
A non-adjacent pair still gets an edge (a vehicle *can* be seen at two cameras
when the ones between missed it) — it is labelled `estimate-nonadjacent`.

---

## Storage

`net_sightings` is **range-partitioned by day**. A 1000-camera network reading
one plate per camera per second produces ~86 million sightings a day; retention
then becomes a partition drop rather than a `DELETE` over tens of millions of
rows. `ensure_sighting_partition(day)` creates partitions on demand.

Writes are batched (`SightingSink`). One `INSERT` per sighting is the most
common reason an ANPR backend falls over. Measured to remote Supabase:
**4,000 sightings in 1.63 s (2,456/sec)**.

If the database is unreachable the buffer is retained up to `sink_max_pending`,
the failure is counted and surfaced, and past the bound the oldest are dropped
and reported — never silently discarded, never grown until the process dies.

---

## Measured capacity, and how to go beyond it

On the reference machine (RTX 5060 Laptop, 8-frame batches, real footage):

| stage | throughput |
|---|---|
| vehicle detector, 640, fp16 | 132 fps |
| plate detector, 960, fp16 | 199 fps @640 / 91 fps @960 |
| **end-to-end pipeline** | **52.3 analysed fps** |

At 1 analysed frame per camera per second that is roughly **50 cameras per
GPU**; at 6 fps/camera, roughly 9.

**One GPU does not serve 1000 cameras, and no amount of tuning changes that.**
The platform scales *horizontally* instead: `service.shard_of()` assigns each
camera to a shard by a stable CRC32 of its id. Run N ingest processes with
`shard_index` 0..N-1 against the same database and each owns a disjoint slice
of the estate. Identity resolution and analytics both happen in shared storage,
so nothing else changes. A 1000-camera deployment at 1 fps/camera is roughly 20
such processes.

Reproduce all of the above:

```bash
python scripts/benchmark_scale.py              # everything
python scripts/benchmark_scale.py --skip-gpu   # data structures only
```

---

## Module map

| module | responsibility |
|---|---|
| `config.py` | every scale tunable, env-overridable |
| `registry.py` | camera estate, spatial grid, k-nearest graph |
| `tracking.py` | per-camera ByteTrack over a shared detector |
| `inference.py` | frame bus, batched detection, batched OCR, sighting emission |
| `identity.py` | blocking-based cross-camera identity resolution |
| `sink.py` | batched sighting writes, vehicle/leg upserts |
| `analytics.py` | trajectory, density, O-D, congestion, heatmap, trend |
| `alerts.py` | watchlist matching and route-anomaly detection |
| `service.py` | wiring, sharding, lifecycle, housekeeping |
| `sql/scale_schema.sql` | partitioned continuous schema |

HTTP surface: `urbantrace.api.scale_routes`, mounted at `/api/network`.
