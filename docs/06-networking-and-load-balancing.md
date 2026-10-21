# 6. Networking and load balancing

Load is balanced at **four independent layers**. Each was checked against the code, and
the first three that can run on a single machine were exercised live (document 9).

```
 browsers ──► nginx (L7 LB) ──► api ×N  (stateless)
                                   │
                                   ▼
                     PostgreSQL / PostGIS  ◄──── worker ×M  (GPU ingest, coordination-free)
                                                    │
                          RTSP / HTTP / file / device feeds
```

## 6.1 Layer 1 — HTTP load balancing (nginx → API replicas)

`deploy/nginx.conf`:

| Directive | Effect |
|---|---|
| `resolver 127.0.0.11 valid=5s` + `set $api http://api:8000` | Docker's DNS is queried per request (cached 5 s) so `docker compose up --scale api=N` takes effect **without reloading nginx**; Docker's resolver round-robins the replicas. |
| `proxy_next_upstream error timeout http_502 http_503` (3 tries) | A replica that is down or restarting is skipped, not shown to the user. |
| `proxy_connect_timeout 3s`, `proxy_read_timeout 60s` | Fail fast to the next replica; bounded waits. |
| `limit_req zone=api rate=30r/s burst=100 nodelay` (per client IP) | Back-pressure and abuse limit on ordinary routes. |
| `gzip` for JSON/CSS/JS/SVG ≥ 1 KB | Smaller dashboard payloads. |
| Streaming locations (`/api/live/stream`, `/api/multi-camera/stream`, `/ws/`) | `proxy_buffering off`, HTTP/1.1, `Upgrade`/`Connection` mapping, 3600 s read timeout — server-sent events and WebSocket survive the proxy. |
| `X-Forwarded-For/Proto`, `X-Real-IP`, `Host` | Original client preserved; the API runs `--proxy-headers`. |
| `/lb-health` | LB's own liveness, no logging. |

**Statelessness is what makes this safe.** All state is in Postgres; any request can go
to any replica. The API replica list is also refreshed in-process: each replica re-reads
the camera estate every 30 s so a camera added via *any* replica appears in *all*.

**Readiness vs liveness.** `/api/ready` pings the database and returns 503 when it is
unreachable; the compose healthcheck uses it, and `lb` `depends_on` the API being
healthy. `/api/health` always answers and reports detail.

**Detector weights are not loaded in API replicas** (`UT_API_PREWARM=0` in compose) — a
deployment with separate workers would otherwise waste GPU/CPU memory per replica.

**Honest scope:** nginx itself could **not** be run in this documentation pass (no
Docker in the environment), so its configuration was reviewed, not exercised.
Equivalent behaviour at the application layer — two independent API replicas on separate
ports hitting one database — was exercised and is reported below.

### Measured: two API replicas against one database
(`GET /api/network/summary`, 6 sub-queries, local PostgreSQL 14, client on the same
machine, via `127.0.0.1`)

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 100 | 57 req/s | 17 ms | 34 ms | 37 ms | 0 |
| 30 | 1,000 | **306 req/s** | 98 ms | 232 ms | 277 ms | 0 |
| 100 | 2,000 | 217 req/s | 391 ms | 810 ms | 1,086 ms | 0 |

The query itself takes **1.2 ms** in-process; the rest is HTTP, JSON and Python.
Past ~30 concurrent clients latency rises while throughput falls: the replicas, the
Python test client and the database share one laptop, so treat this as "no errors,
sane latency", not as a capacity ceiling. Before the pool fix, the same 30-concurrent
test returned HTTP 500s.

*Measurement trap worth recording:* Python's `urllib` against `localhost` on Windows
adds a fixed ~2 s (IPv6 `::1` tried first; Uvicorn listens on IPv4). That looked like a
2-second server latency until `curl` and `127.0.0.1` showed 17 ms. It is a client
artifact, not a server defect.

## 6.2 Layer 2 — assigning cameras to workers (coordination-free)

Ingest scales **horizontally**: N worker processes against one database, each owning a
disjoint set of cameras. `scale/cluster.py`:

1. Each worker heartbeats into `net_workers` every `UT_WORKER_HEARTBEAT_S` (5 s) and
   reads the list of workers whose heartbeat is newer than `UT_WORKER_TTL_S` (20 s).
2. Every worker runs the **same deterministic function** on the **same inputs** and so
   independently reaches the **same assignment** — there is no coordinator, so there is no
   single point of failure.
3. The unit assigned is the camera's *affinity key*: its `sector`, else a grid cell of
   `UT_SHARD_CELL_KM` (3 km). Neighbouring cameras land on the same worker so one identity
   index sees a vehicle at both and can measure the leg between them.
4. **Algorithm — rendezvous (highest-random-weight) hashing with a load bound.** For a key,
   each worker's weight is `blake2b(worker_id ‖ key)`; groups are placed **largest first**
   on their highest-weighted worker that is still under `1.25 ×` the average load, falling
   back to the least-loaded worker.

Why this and not the alternatives **[rationale]**:

| Approach | Problem for this use |
|---|---|
| `hash(camera) % N` | Changing N reshuffles almost every camera; every feed reconnects. |
| Consistent-hash ring | Minimal movement, but balance is poor with few, uneven groups (a busy junction can hold a dozen cameras). |
| Central scheduler / leader election | Adds a component that can fail; heartbeats in the database already give a shared view. |
| **Rendezvous + bound (chosen)** | Minimal movement on membership change (a group stays on its top-ranked worker unless that worker is full) **and** bounded imbalance. |

Only the *difference* is applied on rebalance (`CaptureManager.reconcile`), so cameras
that did not move are not interrupted.

### Verified live (three real workers, 15 cameras, real video sources, local PostgreSQL)

| Event | Observed |
|---|---|
| 3 workers start (4 sector groups of 3 + 3 isolated cameras) | **w1 5, w2 3, w3 7 cameras**; all 15 feeds `live`, each reported by exactly one worker |
| w3 hard-killed (`kill -9` equivalent) | after the TTL, **w1 10 + w2 5 = 15**; all feeds still `live`; w3 shown `alive: false` |
| a replacement worker joins | it took 3 cameras; w1 7, w2 5 |
| camera `T99` added through `POST /api/network/cameras` | **16 feeds live** within one refresh cycle, no restart |

**How even is it?** With groups of 3 cameras the *best possible* maximum on 3 workers is 6
(4 groups cannot be split evenly); the bound allowed 7 (`ceil(1.25 × 15/3)`), and the
observed layouts were 5/3/7 and 7/5/3 — max/average of 1.4. The bound is a *ceiling on
imbalance*, not an equaliser. Tighten it (`slack` in `cluster.assign`) if you prefer more
even load over fewer moves. Large estates with many small groups balance much better than
this deliberately clumpy test.

**Overlap during rebalance.** For up to one heartbeat interval two workers may read the
same camera; this can duplicate a sighting for a few seconds and does nothing else.

**DB outage.** If the heartbeat fails, the previous membership is kept, so a brief outage
does not make every worker abandon its cameras.

**Known limit.** A leg between cameras on opposite sides of an affinity-cell edge, owned by
different workers, is not written to `net_legs` (separate in-process identity indexes).
Trajectory queries — which join sightings in SQL — still show it.

## 6.3 Layer 3 — inside one worker: fairness, shedding, batching

| Mechanism | Where | Behaviour |
|---|---|---|
| Per-camera rate limit | `FrameBus.admit` | At most `target_fps` per camera; 10 % tolerance so frames already paced by capture are not rejected by clock jitter (fixed in this pass). A rejected frame costs no decode/transfer/GPU. |
| Bounded per-camera queue | `FrameBus.submit` | `UT_QUEUE_DEPTH` (2); **drop-oldest**, counted and reported. |
| Fair draining | `FrameBus.drain_batch` | At most one frame per camera per pass, round-robin (`move_to_end`), so high-fps cameras cannot starve others. |
| Cross-camera batching | `InferencePool` | Up to `UT_BATCH_SIZE` (8) frames from different cameras per detector call; a partial batch runs after `UT_BATCH_TIMEOUT` (50 ms). |
| Adaptive load shedding | `RateController` | Every 5 s: if dropped/submitted > 0.10 → fps × 0.8 (floor `UT_MIN_FPS` 0.5); if < 0.01 → fps × 1.1 + 0.05 (cap = target). All cameras sample less together; none go dark. |
| Motion gate | `capture.MotionGate` | 64×36 grey thumbnail diff; an unchanged scene is not sent to the detectors except a keyframe every 2 s. |
| Plate-pass skipping | `SharedModels.detect_batch` | Frames whose camera has no unresolved track skip the plate detector (~40 % of plate passes). |
| Downscale | `capture._shrink` | Frames > 1280 px wide are reduced before queueing. |
| `grab()` vs `retrieve()` | `StreamReader` | Every frame is `grab()`-bed to keep the stream in sync; the costly colour conversion runs only for due frames. |
| Warm-up | `SharedModels.warmup` | Both detectors run at every batch size 1..8 before taking traffic (first call per shape measured 4–100 s). |

## 6.4 Layer 4 — the network edge: camera streams

* **Protocols.** `cv2.VideoCapture` over FFmpeg: RTSP, HTTP(S) streams, files, local device
  indexes. RTSP defaults to **TCP** transport (`UT_RTSP_TRANSPORT`) — robust through NAT and
  loss, at some latency cost versus UDP.
* **Timeouts.** Open and read timeouts of 8 s (`UT_OPEN_TIMEOUT_MS`, `UT_READ_TIMEOUT_MS`)
  so a dead camera cannot hang a reader thread.
* **Reconnect with exponential backoff and full jitter** (1 s → 60 s): feeds that dropped
  together do not return together (avoids a reconnect stampede on the camera or network).
* **Staggered opens.** A semaphore (`UT_MAX_OPENS`, 6) limits simultaneous stream opens so
  100+ feeds do not connect in the same instant.
* **Credential hygiene.** `redact_url` masks `user:pass@` in anything logged or returned by
  the API.
* **Browser live updates.** Server-sent events with 15 s keep-alive; the browser reconnects
  by itself.

## 6.5 Observability

Worker probes on `:9100`: `/healthz` (control loop alive), `/readyz` (models loaded, pool
running), `/metrics` (Prometheus text: cameras owned, live workers, feeds by state, frames
read/submitted/static/dropped, analysis fps, throughput, sightings, sink pending/failures/
dropped, alerts) and `/status` (full JSON). Fleet views: `/api/network/fleet`, `/workers`,
`/feeds`. A feed whose worker has stopped reporting is shown as `unmonitored`, never
trusted at its last known state.

## 6.6 Capacity planning

Capacity is set by GPU throughput, not camera count. Cameras per worker ≈ analysed
frames/s ÷ frames/s sampled per camera. The measured throughput is machine- and
footage-specific (document 3, §3.8: 7–52 frames/s observed across runs and sessions), so
**benchmark on the target hardware** with `scripts/bench_fleet.py`, then add workers
(`docker compose up -d --scale worker=M`). API replicas scale independently
(`--scale api=N`); neither needs the other restarted.
