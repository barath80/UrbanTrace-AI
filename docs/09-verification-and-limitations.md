# 9. Verification, defects found, and limitations

This is the record of the verification pass done while writing this documentation:
what was run, what it found, what was fixed, and what is still unproven.

## 9.1 Environment of the pass

| | |
|---|---|
| Date | 21 Oct 2026 |
| Machine | Windows 11 laptop, NVIDIA RTX 5060 Laptop (8 GB), Python 3.12 |
| Database | **Supabase project unreachable** (pooler: "tenant/user not found"; direct host does not resolve — typically a paused or deleted project). Verified instead against a real **PostgreSQL 14 + PostGIS** started in WSL (port 5433), schemas applied from scratch |
| Docker / nginx | **Not available**; nginx and compose were reviewed, not run |
| Test suite | `python -m pytest tests` → **97 passed** (93 existing + 4 added) |

## 9.2 What was exercised live

| Area | What was run | Result |
|---|---|---|
| Schema | `python -m urbantrace.scale.worker --init-db` on an empty database | both schemas apply cleanly |
| Multi-worker ingest | 3 real workers, 15 cameras (4 sectors × 3 + 3 isolated), real video sources, GPU | 5/3/7 split; all 15 feeds `live` |
| Failover | hard-killed one worker | survivors took over all its cameras after the TTL; all 15 feeds stayed `live` |
| Rejoin | started a replacement worker | rebalanced to 7/5/3 |
| Dynamic estate | `POST /api/network/cameras` while running | 16th camera live within one refresh cycle, no restart |
| Sightings | workers persisted real sightings | 10 sightings → 8 vehicles across 6 cameras; **no cluster-id collisions** |
| API replicas | 2 Uvicorn replicas on one DB | both report identical fleet state |
| API endpoints | summary, density, origin-destination, congestion, trend, heatmap, alerts, fleet, ready | all HTTP 200 on both replicas |
| API load | 100 / 1,000 / 2,000 requests at concurrency 1 / 30 / 100 | 0 errors; 306 req/s at 30 concurrent (document 6) |
| Security | auth, validation, injection, traversal, CORS | all as designed (document 8) |
| Fleet GPU benchmark | `bench_fleet.py --cameras 16`, 5 runs | 7.2–14.2 frames/s (document 3 §3.8) |

## 9.3 Defects found and fixed

| # | Defect | How it showed | Fix | Test |
|---|---|---|---|---|
| 1 | **Cluster ids collided across workers and restarts.** `GV_<counter>` restarted at 1 in every process, but `cluster_id` is the primary key of `net_vehicles` and `net_legs` | code review of a multi-worker deployment: two workers would overwrite each other's vehicles, putting one car's journey under another's identity | `GV_<8-hex process nonce>_<seq>` | `test_cluster_ids_do_not_collide_across_workers` |
| 2 | **Rate limiter applied twice.** Capture paces frames at `1/fps`; `FrameBus.admit` re-limited at exactly `1/fps` on a different clock, so milliseconds of decode/resize jitter could reject on-schedule frames | analysis of the two clocks | 10 % tolerance in `admit` | `test_admission_tolerates_paced_frames_with_clock_jitter` |
| 3 | **HTTP 500 under concurrency.** psycopg2's pool raises when empty; Starlette runs 40 sync handlers over 8 connections | 30 concurrent requests → `PoolError: connection pool exhausted` → 500 | `BoundedSemaphore` queue with timeout | `test_db_pool_queues_instead_of_failing_when_exhausted`; re-run of 3,000+ requests, 0 errors |
| 4 | **Watchlist refresh on the inference hot path.** `match()` could call the database from an inference thread; a slow or unreachable DB would stall inference | benchmark log filled with refresh failures | `match()` never touches the DB; refreshed by the maintenance thread / worker housekeeping | existing suite |
| 5 | **`persist=False` still wrote to the database** (vehicle upserts, alerts) and spammed errors | `bench_fleet.py` (documented as "nothing is written") logged DB failures | `flush()` and watchlist loading are skipped when the sink is disabled | existing suite |
| 6 | **Sink and vehicle upserts opened a fresh connection per flush** (TCP+TLS each time) | code review against the pooling rationale in `db/client.py` | use `pooled()` | existing suite |
| 7 | **Partition cache updated before commit** — a rolled-back `ensure_sighting_partition` stayed "created" in memory, so later rows went to the default partition | code review | cache updated after commit | — |
| 8 | **`/api/video/{camera_id}` built a file path from the raw id** | code review | allow-pattern → HTTP 400 | `test_video_endpoint_rejects_unsafe_camera_id` |

Corrected in the docs (not code defects): the README named the OCR default `cct-s-v2`
(it is `cct-xs-v2`, the measured winner) and said OCR accuracy was "not yet measured"
(it now is: 71.3 % exact, 92.2 % character accuracy).

Defects 1, 3 and the throughput finding below are the ones that matter most for
"does it work correctly under load with several workers".

## 9.4 Findings that were *not* fixed

1. **Fleet throughput is not reproducible today.** The documented ~50 analysed frames/s
   (16 cameras) was not reproduced: 7.2–14.2 frames/s on the current code and 11.5–12.0 on
   a clean checkout of the previous commit, same machine. The fixes above do not explain the
   gap. The cause was not isolated (thermal or power state, or other load, are the leading
   suspects). **Action:** re-run `scripts/bench_fleet.py` on the target hardware before
   sizing workers.
2. **Dead worker rows accumulate** in `net_workers` after a hard kill, and `/fleet`'s
   `workers` counts them. Harmless; prune periodically.
3. **Cross-worker legs are not written** to `net_legs` (separate in-process identity
   indexes); sightings and trajectory queries are unaffected.
4. **Worker balance is bounded, not equal.** Clumped groups gave max/avg ≈ 1.4 on 3
   workers (best possible for those groups: 6; observed: 7).
5. **FastAPI `@app.on_event` is deprecated** (4 warnings in the test run). Behaviour is
   unchanged; migrate to `lifespan` handlers when convenient.
6. **A sink-outage unit test is missing** (the behaviour is implemented, not tested).
7. **The summary endpoints are not cached.** Fine at the measured load; under many
   dashboards a few seconds of TTL cache per replica would cut repeated window scans.

## 9.5 Limitations of the models and data (consolidated)

* **OCR is the weak link:** 71.3 % exact plate match single-frame; the dominant failure is a
  dropped trailing character (17 of 23 errors). Confidence is informative but over-confident
  at the top (0.99 → 86.5 % right). Details: document 3.
* **Plate detector:** recall 0.788 at the production threshold; its training score (0.98
  mAP50) overstates held-out performance (0.80) by 0.15–0.18.
* **No vehicle-detection accuracy on Indian traffic** — no vehicle boxes exist; and no
  autorickshaw class, so per-class counts are unreliable (totals are not).
* **Accuracy after temporal voting is not measured** — no video with plate ground truth.
* **Counts are auditable, not ground truth.**
* **Severe motion blur, extreme distance and near-darkness are not recoverable** from one
  frame (measured 0 % recovery at blur ≥ 7 px or 3× downscale).
* **Plate-only identity merges confusable twins** unless physical feasibility separates them.
* **Timestamps are part measured, part configured:** `video_time_s` is measured; wall-clock
  is configured camera start + offset unless a live feed supplies capture time.
* **Cross-camera routes need shared vehicles:** unrelated scenes correctly yield
  single-camera vehicles and no routes.
* **Roboflow hosted inference is optional** and needs credits; local weights are the default.
* **Scale evidence:** the "city scale" figures come from synthetic identity/registry tests
  and a small real-video fleet (≤ 16 cameras on one GPU). A real 1,000-camera deployment has
  not been run.

## 9.6 How to reproduce

```bash
python -m pytest tests                                  # 97 tests, no GPU/video/network needed
python scripts/verify_system.py                         # end-to-end checks against real video
jupyter nbconvert --to notebook --execute testing_accuracy.ipynb   # every accuracy number in doc 3
python scripts/benchmark_ocr_robustness.py              # OCR under degradation
python scripts/benchmark_scale.py --skip-gpu            # identity / registry scaling
python scripts/bench_fleet.py --cameras 16 --seconds 40 # GPU capacity — run on YOUR hardware

# multi-worker check against any Postgres+PostGIS
export DATABASE_URL=postgresql://user@host:5432/urbantrace UT_DB_SSLMODE=disable
python -m urbantrace.scale.worker --init-db
python -m urbantrace.scale.worker --import-cameras data/cameras.json
UT_WORKER_ID=w1 python -m urbantrace.scale.worker --port 9101 &
UT_WORKER_ID=w2 python -m urbantrace.scale.worker --port 9102 &
curl localhost:8000/api/network/fleet                   # who owns what; kill a worker and watch it move
```
