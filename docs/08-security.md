# 8. Security

The platform stores *location history of vehicles identified by plate*. That is
sensitive personal data in most jurisdictions, and an alert may lead to a person being
stopped. Security here therefore covers both the usual web concerns and data-protection
responsibilities. This document separates what is **implemented and verified**, what is
**implemented but not verified**, and what is **a gap you must close for production**.

## 8.1 Verified live (API run with `UT_API_KEY=secret123`, local PostgreSQL)

| Check | Request | Result |
|---|---|---|
| Read without key (default mode `writes`) | `GET /api/network/fleet` | 200 |
| Write without key | `POST /api/network/watchlist/ABC123` | **401** |
| Write with wrong key | `X-API-Key: nope` | **401** |
| Write with correct key | `X-API-Key: secret123` | 200 |
| Health stays open for probes | `GET /api/health` | 200 |
| Camera id with SQL/shell characters | `camera_id: "a;drop table x"` | **422** (Pydantic pattern) |
| Latitude out of range | `latitude: 99` | **422** |
| SQL-injection-style search | `q=' or 1=1 --` | `[]` — parameterised, no rows |
| Path traversal, video route | `/api/video/..%2f..%2fetc` | 404 |
| Path traversal, evidence route | `/api/evidence?path=../../.env` | **403** |
| Cross-origin request from an unlisted origin | `Origin: https://evil.example` | no `Access-Control-Allow-Origin` header |

## 8.2 Controls implemented

### Authentication and authorisation
* **Optional API-key gate** (`require_api_key` middleware). With `UT_API_KEY` unset the API
  is open (development). With it set, every non-`GET/HEAD/OPTIONS` request to `/api/*`
  needs the key; `UT_API_AUTH=all` extends this to reads. `/api/health` and `/api/ready`
  are always exempt so load balancers can probe.
* Key accepted as `X-API-Key`, `Authorization: Bearer`, or `?api_key=` (the last only
  because `<img>` and `EventSource` cannot set headers).
* **Constant-time comparison** (`hmac.compare_digest`) — no timing oracle.

### Input validation
* `CameraIn` (Pydantic): `camera_id` matches `^[A-Za-z0-9_.\-]+$`, 1–64 chars; names
  ≤ 200; latitude −90…90, longitude −180…180. Bulk upload capped at 5,000 cameras.
* Every time-window parameter is bounded (`hours` in `(0, 720]`, `limit` ≤ 500/2000/5000),
  so one request cannot ask for an unbounded scan.
* **Camera ids that become file names** (`/api/video/{camera_id}`) are validated against
  `[A-Za-z0-9_.\-]{1,64}` (added in this pass; the route previously relied on the
  filename suffix and routing to stay inside the output directory).

### Injection and traversal
* All SQL is parameterised; the only dynamic SQL is a fixed, code-defined column list and
  constant table names.
* `/api/evidence` resolves the path and requires it to be inside `data/evidence`
  (`relative_to`), defeating `..` and absolute-path tricks.

### Secrets
* Credentials live in `.env` (git-ignored); `.env.example` holds placeholders only.
* The compose file **refuses to start** without `POSTGRES_PASSWORD`
  (`${POSTGRES_PASSWORD:?…}`).
* DSNs are assembled with proper percent-encoding; the connection-status endpoint and the
  camera API return a **redacted** target / stream URL (`redact_url` masks `user:pass@`).
* The GraphHopper key is used **server-side only**; the browser asks `/api/road-route` for
  geometry and never sees the key.
* Stream credentials in a camera's `source` are returned by the API only after redaction.

### Network
* **CORS is off by default**: the dashboard is served from the API's own origin. Listing
  origins in `UT_CORS_ORIGINS` enables exactly those.
* nginx rate-limits ordinary routes (30 req/s per client, burst 100) and restricts body
  size (20 MB).
* Streaming routes use SSE over plain HTTP(S), so they inherit the same auth middleware.
* RTSP defaults to TCP transport.

### Containers
* Runs as a **non-root user** (`uid 10001`); only `data/` is writable.
* `tini` as PID 1 for correct signal handling.
* Model weights mounted **read-only**; notebook/dev tooling excluded from the service
  image; multi-stage build so Node is not in the runtime image.
* Detector weights and `.env` are never baked into the image.

### Robustness (denial-of-service surface)
* Every queue, buffer and pool is bounded (document 7); the DB pool queues rather than
  failing and times out with a clear error.
* Per-camera rate limit, drop-oldest queues and adaptive load shedding keep a flood of
  frames from exhausting memory.
* Feed reconnects use jittered exponential backoff.

## 8.3 Known gaps — close these before a real deployment

| Gap | Why it matters | Recommended fix |
|---|---|---|
| **No TLS in the compose stack** | API keys, plates and locations travel in clear on the LAN | Terminate TLS at nginx (or an ingress/cloud LB); set HSTS |
| **Default is an open API** (`UT_API_KEY` unset; reads open even when set) | Anyone reachable can read vehicle histories | Always set `UT_API_KEY`; set `UT_API_AUTH=all` for any non-demo use |
| **One shared API key**, no users/roles, no audit log of who queried which plate | No accountability for searches on individuals | Put SSO/OIDC in front; add per-user roles and a query audit trail |
| **`?api_key=` in the URL** | Keys appear in access logs and browser history | Prefer headers; rotate keys; scrub logs |
| **`POST /api/live/start` accepts an arbitrary `source`** (file path, URL, device) and camera `source` fields are opened by workers | A holder of the write key can make the server open local files or internal URLs (SSRF-style) | Treat the write key as administrator-level; add an allow-list of schemes/hosts/directories |
| **No database row-level security / least privilege** | The app role can do everything | Use a role with only needed grants; Supabase RLS if exposing the REST layer |
| **Evidence images and plate text are unencrypted at rest** | Disk theft exposes personal data | Encrypt volumes; set retention (partition drop) policy |
| **Rate limiting only per IP at nginx** | Shared NAT/proxy IPs | Add per-key limits in the app or gateway |
| **Dead worker / stale rows, no automated retention job** | Data kept longer than intended | Schedule partition drops and a worker prune |
| **Secrets in `.env` on disk** | Local compromise exposes them | Use Docker/K8s secrets or a vault in production |
| Credentials shared in plain text (chat, screenshot, email) | Anything shared that way must be treated as compromised | **Rotate** (Supabase → Settings → Database → Reset password); no code change is needed |

## 8.4 Privacy and lawful use

Automatic plate recognition and movement tracking are regulated activities. Before
operating this on real people's data: establish a lawful basis and purpose limitation,
publish a retention period and enforce it (the partitioned schema makes deletion by day
cheap), restrict and audit access, and be aware that **watchlists must never be
auto-populated from detections** — in this system the watchlist is database-driven and
manual. The system also states its own uncertainty (documents 3 and 9): single-frame
exact-match OCR accuracy is 71.3 %, and confusable plates can merge, so an alert is a
*lead for a human to verify*, never proof.
