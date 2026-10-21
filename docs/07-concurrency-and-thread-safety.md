# 7. Concurrency and thread safety

## 7.1 The threading model

A worker process is a small set of cooperating threads plus per-camera readers:

| Thread(s) | Count | Job | Blocks on |
|---|---|---|---|
| `cap-<camera>` (`StreamReader`) | one per owned camera | `grab()` frames, motion-gate, shrink, submit | network / decoder |
| `infer-<n>` (`InferencePool`) | `UT_INFER_WORKERS` (2) | drain a batch → detect → track → OCR → emit sighting | the GPU, model locks |
| `sighting-sink` | 1 | flush buffered sightings to Postgres | database |
| `scale-maint` | 1 | flush vehicles/alerts, refresh the watchlist | database |
| worker control loop (main) | 1 | heartbeat, rebalance, adapt rate, publish health | database |
| `probes` | 1 + per-request | `/healthz`, `/readyz`, `/metrics`, `/status` | clients |

The API process is separate: Uvicorn's event loop plus Starlette's thread pool (40
threads) for synchronous handlers, a camera-estate refresh thread (30 s), and live-feed
sessions.

**Python's GIL is not the bottleneck here.** The expensive work — CUDA kernels, ONNX
Runtime, OpenCV, NumPy — releases the GIL, so the design goal is simply to keep every
Python-level critical section *tiny* and never hold a lock across I/O or inference unless
the library requires it.

## 7.2 Lock inventory — every shared structure and what protects it

| Structure | Lock | Granularity and reasoning |
|---|---|---|
| `FrameBus._queues`, counters | `threading.Condition` over one lock | Held only for `deque` push/pop and counter updates, never across inference. `notify` wakes a drainer; `wait(remaining)` is bounded by the batch timeout. |
| `FrameBus._last_admit` | same lock | Per-camera rate limiter state. |
| Detector `predict` | **`_veh_lock` and `_plate_lock`, separate** | Ultralytics models are not documented as thread-safe for concurrent `predict`, so calls are serialised *per model*. A single shared lock would force vehicle and plate passes never to overlap and collapse throughput to the **sum** of their times; split locks let one thread's plate pass overlap another's vehicle pass and CPU preprocessing. |
| `SharedModels.load` | `_lock` | Lazy one-time load, double-checked inside the lock. |
| `MultiCameraTracker._entries` | `_guard` | Held only to find/create/evict an entry. |
| One camera's `BYTETracker` | **per-camera lock** | Kalman state of one camera is consistent without serialising the whole network behind a global lock; different cameras update fully in parallel. |
| Track-id allocation | module-level `id_lock` around `BaseTrack._count += 1` | `_count += 1` is **not atomic**, and several inference threads allocate ids concurrently. |
| `InferencePool._buffers`, `_live_tracks`, `_unsettled` | `_buf_lock` | Per-track OCR buffers. Counters are maintained incrementally so the "does this camera still need plates?" check is O(1) rather than a scan under the lock (the scan version cost more than the inference it was avoiding). The expensive `aggregate()` runs on a **copy** taken under the lock, outside it. |
| `IdentityIndex` | `RLock` | `add()` is the only mutator and is O(candidates); reentrant because `find`/`get` may be called from within callbacks. |
| `SightingSink._buf` | `_lock` + `Event` | Buffer swap (`list(...)` then `clear()`) is atomic under the lock; the **database write happens outside the lock**, so submitters never wait on the network. On failure, items are re-inserted at the front. |
| `IngestService._dirty`, `_legs` | `_lock` | Same swap-then-work pattern in `flush()`. |
| `Watchlist._by_canonical` | `RLock` | The table is **built aside** and swapped in under the lock; `match()` takes the lock only for a dict lookup. |
| `AlertEngine._pending` | `_lock` | Pending alerts drained in `flush()`. |
| `CaptureManager._readers` | `_lock` | `reconcile` computes the diff and starts/stops readers under the lock. |
| `RateController` | `_lock` | `tick()` updates the rate from counters atomically. |
| `Membership._workers` | `_lock` | Returns copies; a failed heartbeat keeps the previous list. |
| `CameraRegistry.replace` | *no lock* — **build aside, then swap references** | New state is built in a fresh registry and four attributes are rebound. A concurrent reader sees either the old or the new object for each attribute (reference assignment is atomic under the GIL); it never sees an empty registry. A reader could momentarily pair new nodes with old edges, which is benign (a lookup misses and falls back). Pinned by a test that hammers `replace()` while reading and asserts the registry is never empty. |
| DB connections | `ThreadedConnectionPool` + `BoundedSemaphore(POOL_MAX)` + `_POOL_LOCK` | Pool created once (double-checked); the semaphore makes callers queue (document 5). |
| Live session pub/sub | `_lock` + bounded `queue.Queue` per subscriber | A slow browser cannot grow memory: the queue is bounded and `droppable` events are shed. |

## 7.3 Patterns used throughout

1. **Swap-then-work.** Take the lock, move the shared state into a local variable
   (`pending = list(self._buf); self._buf.clear()`), release, then do the slow work. Used
   by the sink, the service flush and the watchlist.
2. **Build aside, swap atomically.** Registry and watchlist refresh.
3. **Per-key locks, not a global lock.** One lock per camera tracker.
4. **Bounded everything.** Per-camera queues (drop-oldest), sink buffer, subscriber
   queues, pool, track buffers, identity index, tracker count. A fast producer meeting a
   slow consumer degrades gracefully; it does not grow until the process dies.
5. **Lock ordering is trivial by construction.** No code path holds two of these locks at
   once except `IdentityIndex` (re-entering its own `RLock`) and the tracker
   (`_guard` is released before the per-camera lock is taken), so deadlock cycles do not
   arise.
6. **No lock held across I/O or inference** — except the per-model `predict` lock, which
   exists precisely to serialise the GPU call.
7. **Graceful stop.** `threading.Event` for every loop; `stop()` sets the event, joins
   with a timeout, then flushes. SIGTERM → `request_stop()` → capture stops → service
   flushes buffered sightings → worker deregisters. Compose gives workers a 30 s grace
   period for this.
8. **Daemon threads + explicit shutdown** so a crashed main thread never leaves a hung
   process, while a clean stop still drains data.

## 7.4 Distributed concurrency (across processes)

* **No shared memory between workers.** All coordination is through Postgres rows
  (`net_workers`), which is why a worker can die without taking anything else with it.
* **Unique ids across processes.** Cluster ids carry a per-process nonce
  (document 9, defect 1) because the id is a primary key in a shared table.
* **Idempotent writes.** `ON CONFLICT … DO UPDATE` makes concurrent or repeated flushes
  converge.
* **Time from the database** for liveness, so host clock drift cannot cause false
  failover.
* **Accepted overlap during rebalance:** two workers may briefly read the same camera and
  duplicate a sighting for ≤ one heartbeat interval. This is a deliberate trade for not
  needing a distributed lock.

## 7.5 Races that are known and accepted

Honest accounting, because "thread-safe" claims should list their exceptions:

* **Statistics counters** such as `IngestService.sightings_seen`, `InferencePool.ocr_calls`,
  `SharedModels.frames_processed` and `SightingSink.written/failures` are incremented
  without a lock from more than one thread in some paths. A lost increment makes a
  *displayed statistic* slightly low; it cannot corrupt data or control flow.
  (`vehicle_calls` and `frames_processed` are updated inside the model lock.)
* **`FeedHealth` fields** are written by the reader thread and read by the control loop
  without a lock; each is a single attribute write, and readers only publish a snapshot.
* **Batch pipeline (`ingest/multicam.py`)** has its own locks (`_lock`, a `Condition` for
  new frames, a `_processing` lock); it predates the scale layer and was verified by the
  existing test-suite and live runs rather than redesigned.

## 7.6 How it is tested

`tests/test_scale.py` and `tests/test_fleet.py` (97 tests total in the suite, all
passing) include concurrency-specific tests:

* track ids never reissued across cameras (`test_new_camera_does_not_reissue_ids_of_a_running_camera`);
* `FrameBus` fairness, drop-oldest and rate limiting (`test_bus_is_fair_across_cameras`, `test_bus_drops_oldest_and_never_blocks`, `test_admission_rate_limits_per_camera`);
* registry `replace()` under concurrent reads never exposes an empty registry (`test_registry_replace_never_exposes_an_empty_estate`);
* cluster ids unique across independent indexes (added in this pass);
* DB pool queues rather than raising when more threads than connections ask (added in this pass);
* worker assignment: every camera has exactly one owner, load is roughly even, clumped cameras still spread, losing a worker moves only its cameras, membership keeps its last view when the database is down;
* the sink's retain-and-bound behaviour on a database outage is implemented (`SightingSink.flush`) but **has no dedicated automated test**.

What is **not** covered by an automated test: a long soak under real load with thread
sanitisation (not available for CPython), so the lock inventory above is reasoned and
reviewed, backed by the live multi-worker runs in document 9.
