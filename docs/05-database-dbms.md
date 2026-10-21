# 5. Database design (DBMS)

**DBMS: PostgreSQL with the PostGIS and pgcrypto extensions.**
Development target: a managed Supabase Postgres. Production target: the
`postgis/postgis:16-3.4` container in `docker-compose.yml`. The same code runs against
both; `DATABASE_URL` (if set) wins over the `SUPABASE_*` settings.

Verification note: in this documentation pass the configured Supabase project was
**unreachable** (the pooler answered "tenant/user not found" and the direct host did
not resolve, which usually means a paused or deleted project). All database-backed
behaviour below was therefore verified against a **real PostgreSQL 14 + PostGIS**
instance started locally in WSL, with both schemas applied from scratch. PostgreSQL 14
is older than the 16 the compose file pins, so partition and `ON CONFLICT` features used
are the conservative common subset.

## 5.1 Two schemas, one database

| Schema file | Models | Scoped by |
|---|---|---|
| `db/schema.sql` | **Batch runs** — reprocessing a folder of videos | `run_id` (uuid) — runs accumulate, never overwrite |
| `scale/sql/scale_schema.sql` | **Continuous city network** — live cameras, no "runs" | time (partitioned by day) |

They coexist; applying both is idempotent (`create … if not exists`,
`create or replace`). `python -m urbantrace.scale.worker --init-db` applies both and is
run once per deploy by the compose `init` service.

## 5.2 Why a relational, ACID store

The data is inherently relational: cameras ↔ sightings ↔ vehicles ↔ legs, with
referential integrity (`net_edges` references `net_cameras … on delete cascade`), check
constraints (`latitude between -90 and 90`), uniqueness (`unique (cluster_id, seq)`) and
multi-statement atomicity (partition creation and the batch insert happen in one
transaction). An alert may lead to a vehicle being stopped; integrity belongs in the
database, not only in application code.

## 5.3 Continuous schema (`net_*`)

| Table | Purpose | Key | Notable design |
|---|---|---|---|
| `net_cameras` | camera estate | `camera_id` | generated column `geom geography(Point,4326)` from lat/lon, GiST index |
| `net_edges` | plausible road links (k-nearest) | `(source, destination)` | `on delete cascade` |
| `net_sightings` | **the fundamental fact**: one confirmed plate at one camera at one instant | `(id, observed_at)` | **range-partitioned by day** |
| `net_vehicles` | resolved identity (a cluster of sightings) | `cluster_id` | `spellings jsonb` |
| `net_legs` | one measured camera-to-camera hop | `id`, `unique(cluster_id, seq)` | verdict + reason |
| `net_watchlist`, `net_alerts` | alerting | plate / id | `evidence jsonb` |
| `net_camera_minute` | per-camera per-minute rollup | `(camera_id, bucket)` | for density/heatmap |
| `net_flow_baseline` | normal speed per corridor and hour of day | `(from, to, hour_of_day)` | `check (hour_of_day between 0 and 23)` |
| `net_workers` | ingest worker heartbeats | `worker_id` | `stats jsonb`, `heartbeat_at` |
| `net_camera_health` | latest state of each feed | `camera_id` | written by the owning worker |

### Partitioning — retention is a `DROP`, not a `DELETE`

A 1,000-camera network reading one plate per camera per second produces ~86 million
sightings a day. `net_sightings` is `partition by range (observed_at)`:

* retention = dropping a daily partition (instant) instead of a `DELETE` over tens of
  millions of rows (long locks, bloat, vacuum storm);
* queries bounded by a time window touch only the relevant partitions (partition pruning);
* `ensure_sighting_partition(day)` creates `net_sightings_YYYYMMDD` on demand, and a
  **default partition** means an insert can never fail because tomorrow's partition does
  not exist yet.

The primary key includes the partition key (`(id, observed_at)`) because PostgreSQL
requires that of a partitioned table.

### Indexing (every dashboard query is index-served)

| Index | Serves |
|---|---|
| `(plate, observed_at desc)` | trajectory of one plate |
| `(plate_canonical, observed_at desc)` | OCR-confusion-tolerant search |
| `(camera_id, observed_at desc)` | per-camera density and flow |
| `(cluster_id, observed_at)` | assembling a vehicle's journey |
| `(observed_at desc)` | time-window scans |
| GiST on `geom` | spatial queries |
| `net_legs (from_camera, to_camera, arrived desc)` | origin–destination and congestion |
| `net_workers (heartbeat_at desc)` | live-worker lookup |

Indexes are declared on the partitioned parent so every partition inherits them. A
sequential scan of a table this size is "not a query, it is an outage".

### Rollups instead of re-aggregation

`net_camera_minute` is written once per camera per minute and every density/heatmap
query reads it, instead of scanning the whole day's sightings on every dashboard refresh.
`net_flow_baseline` stores each corridor's normal speed per hour of day so congestion is
judged **relative to that road's own normal**, not a fixed number that means nothing on a
given road.

## 5.4 Write path: batching and idempotence

* **Batched inserts.** `SightingSink` buffers sightings and writes with
  `psycopg2.extras.execute_values` on a size trigger (`UT_SINK_BATCH` = 200) or time
  trigger (`UT_SINK_FLUSH_S` = 2 s). One INSERT per sighting is the commonest cause of ANPR
  backends falling over (a network round trip and commit each). Measured to remote
  Supabase: **4,000 sightings in 1.63 s (2,456/s)**.
* **Idempotent upserts.** `net_vehicles` and `net_legs` use `INSERT … ON CONFLICT … DO
  UPDATE`, so repeated flushes of a growing cluster converge instead of duplicating;
  `first_seen = least(…)` and `last_seen = greatest(…)` keep the bounds correct.
* **Failure behaviour.** If the database is unreachable the buffer is retained at the
  front (order preserved) up to `UT_SINK_MAX_PENDING` (50,000); past that the *oldest* are
  dropped and **counted and surfaced** — never silently, never until the process dies.
* **Partition cache after commit.** (Fixed in this pass.) The "partition exists" cache is
  updated only after the transaction commits; previously a rolled-back partition creation
  stayed cached and later rows silently fell into the default partition.

## 5.5 Connection pooling and transactions

`db/client.py` keeps a per-process `ThreadedConnectionPool` (`UT_DB_POOL_MAX`, default 8).
Every connection costs a TCP+TLS handshake, which once made a seven-query dashboard call
take six seconds against a remote region; pooling removed that.

**Pool exhaustion defect (found and fixed in this pass).** psycopg2's pool *raises*
`PoolError` when empty instead of waiting, and FastAPI runs synchronous handlers on a
40-thread pool. 30 concurrent requests against 8 connections produced HTTP 500s. A
`BoundedSemaphore` now makes extra callers queue (up to `UT_DB_POOL_WAIT_S`, default 30 s)
and then raise a clear `TimeoutError`. Re-measured: 1,000 and 2,000 requests, zero
errors (document 9).

`pooled()` commits on success, rolls back on error and **discards** a connection that
errored rather than returning it to the pool. The sink and vehicle upserts now use the
pool (they previously opened and leaked a fresh connection per flush).

SSL: `UT_DB_SSLMODE` (default `prefer` for `DATABASE_URL`; `require` for Supabase).

## 5.6 Query safety

All SQL is **parameterised** (`%s` placeholders); the only f-string SQL is the sink's
fixed column list and `reset_schema`'s constant table names. Verified live: a search for
`' or 1=1 --` returned `[]`, not rows. Time-window endpoints take `hours` bounded by
`Query(gt=0, le=720)`, so a request cannot ask for an unbounded scan.

## 5.7 Time and clocks

Worker liveness uses `now()` **from the database**, not the host clock, so clock drift
between worker machines cannot make a healthy worker look dead. Sightings use
`timestamptz` throughout; naive datetimes are treated as UTC.

## 5.8 Operational notes and known gaps

* **Dead worker rows persist.** `net_workers` rows for crashed workers are never deleted
  (graceful shutdown deregisters; a hard kill does not). `/api/network/workers` reports
  them with `alive: false` and `/fleet` counts them in `workers` but not `workers_alive`.
  Harmless, but a periodic prune (`delete … where heartbeat_at < now() - interval '1
  day'`) is recommended.
* **No automatic partition dropping.** Retention is possible by design; a scheduled job to
  drop old `net_sightings_YYYYMMDD` partitions is not included.
* **Per-process pool.** With N API replicas × `UT_DB_POOL_MAX` plus M workers, size
  Postgres `max_connections` (or put PgBouncer in front) accordingly.
