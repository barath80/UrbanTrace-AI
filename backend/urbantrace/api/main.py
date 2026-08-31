"""UrbanTrace AI - REST API and dashboard host.

Serves the GIS command-centre frontend and exposes every artefact the pipeline
produced. All data is read from Supabase; nothing is generated here. When the
database is empty the endpoints return empty collections and the dashboard says
so rather than inventing content.

    python -m uvicorn urbantrace.api.main:app --reload --port 8000
    (or)  python scripts/serve.py
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import json

from ..core.cameras import CAMERAS
from ..core.runtime import describe
from ..db import repository as repo
from ..db.client import ping
from ..alerts import load_watchlist_file
from ..ingest import MANAGER as LIVE
from ..ingest.multicam import router as multicam_router
from .scale_routes import router as network_router
from ..analytics import journey_stats, path_heat_points
from ..routing import describe as routing_describe, road_route
from ..trajectory.paths import build_vehicle_paths

ROOT = Path(__file__).resolve().parents[3]
FRONTEND = ROOT / "frontend"        # legacy vanilla-JS dashboard
WEB_DIST = ROOT / "web-dist"        # built React app (npm run build in web/)
OUTPUT = ROOT / "data" / "output"
EVIDENCE = ROOT / "data" / "evidence"

app = FastAPI(title="UrbanTrace AI",
              description="City-wide multi-camera ANPR, trajectory and traffic intelligence",
              version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.include_router(multicam_router)
app.include_router(network_router)


@app.on_event("startup")
def startup_event():
    import threading
    from ..ingest.multicam import MULTICAM
    def _load_network():
        # The city registry is read from the database once at boot so spatial
        # lookups and graph edges are ready before the first sighting arrives.
        try:
            from ..scale.service import load_cameras_from_db
            n = load_cameras_from_db()
            print(f"[api] camera network loaded: {n} camera(s)")
        except Exception as e:
            print(f"[api] camera network not loaded: {type(e).__name__}: {e}")

    threading.Thread(target=_load_network, daemon=True).start()

    def _prewarm():
        for slot in MULTICAM.slots.values():
            slot._ensure_models()
    threading.Thread(target=_prewarm, daemon=True).start()


def _run(run_id: str | None) -> str:
    rid = run_id or repo.latest_run()
    if not rid:
        raise HTTPException(404, "No pipeline run found. Process videos first.")
    return rid


# ------------------------------------------------------------ status

@app.get("/api/health")
def health():
    out = {"runtime": describe(), "database": {"connected": False}}
    try:
        out["database"] = ping()
    except Exception as e:
        out["database"] = {"connected": False, "error": f"{type(e).__name__}: {e}"}
    try:
        out["latest_run"] = repo.latest_run()
    except Exception:
        out["latest_run"] = None
    return out


@app.get("/api/runs")
def runs():
    return repo._fetch(
        """select run_id, started_at, finished_at, device, gpu_name, plate_backend,
                  ocr_model, camera_count, video_count, observation_count, notes
           from runs order by started_at desc""")


@app.get("/api/summary")
def summary(run_id: str | None = None):
    rid = _run(run_id)
    s = repo.run_summary(rid)
    extra = repo._fetch(
        """select
             (select count(*) from observations where run_id=%s) as observations,
             (select count(distinct plate_number) from observations where run_id=%s) as unique_plates,
             (select count(*) from global_vehicles where run_id=%s and camera_count>1) as multi_camera,
             (select count(*) from alerts where run_id=%s) as alerts,
             (select round(avg(implied_speed_kmh)::numeric,1) from route_legs
                where run_id=%s and verdict<>'IMPLAUSIBLE') as avg_speed_kmh,
             (select round(avg(travel_minutes)::numeric,1) from route_legs
                where run_id=%s and verdict<>'IMPLAUSIBLE') as avg_travel_min,
             (select count(*) from cameras where run_id=%s) as cameras,
             (select round(avg(ocr_confidence)::numeric,3) from observations where run_id=%s) as mean_ocr_conf,
             (select count(*) from plate_readings where run_id=%s) as raw_readings
        """, (rid,) * 9)[0]
    return {"run_id": rid, **{k: v for k, v in s.items() if k != "run_id"}, **extra}


# ------------------------------------------------------------ network

@app.get("/api/cameras")
def cameras(run_id: str | None = None):
    rid = _run(run_id)
    cams = repo._fetch(
        """select c.camera_id, c.name, c.road, c.latitude, c.longitude, c.direction,
                  c.video_file, c.capture_start, c.location_source,
                  coalesce(s.frames,0) as frames, coalesce(s.vehicle_dets,0) as vehicle_dets,
                  coalesce(s.plate_dets,0) as plate_dets,
                  coalesce(s.processing_fps,0) as processing_fps,
                  coalesce(s.unique_vehicles,0) as unique_vehicles,
                  s.count_mode,
                  coalesce(s.count_traffic,0) as count_traffic,
                  coalesce(s.count_presence,0) as count_presence,
                  coalesce(s.count_car,0) as count_car,
                  coalesce(s.count_motorcycle,0) as count_motorcycle,
                  coalesce(s.count_bus,0) as count_bus,
                  coalesce(s.count_truck,0) as count_truck,
                  (select count(distinct o.plate_number) from observations o
                     where o.run_id=c.run_id and o.camera_id=c.camera_id) as vehicles_with_plate
           from cameras c
           left join camera_stats s on s.run_id=c.run_id and s.camera_id=c.camera_id
           where c.run_id=%s order by c.camera_id""", (rid,))
    for c in cams:
        c["has_video"] = (OUTPUT / f"{c['camera_id']}_annotated_h264.mp4").exists() or \
                         (OUTPUT / f"{c['camera_id']}_annotated.mp4").exists()
    return cams


@app.get("/api/edges")
def edges(run_id: str | None = None):
    return repo._fetch("""select * from camera_edges where run_id=%s
                          order by source_camera, destination_camera""", (_run(run_id),))


@app.get("/api/camera-counts")
def camera_counts(run_id: str | None = None):
    return repo.camera_counts(_run(run_id))


# ------------------------------------------------------- observations

@app.get("/api/observations")
def observations(run_id: str | None = None, camera_id: str | None = None,
                 plate: str | None = None, limit: int = Query(500, le=5000)):
    rid = _run(run_id)
    sql = """select observation_id, camera_id, plate_number, observed_at, video_time_s,
                    frame_number, local_track_id, vehicle_type, ocr_confidence,
                    plate_detection_confidence, support, total_readings, agreement,
                    is_valid_format, aggregation_method, latitude, longitude, evidence_crop
             from observations where run_id=%s"""
    params = [rid]
    if camera_id:
        sql += " and camera_id=%s"; params.append(camera_id)
    if plate:
        sql += " and plate_number ilike %s"; params.append(f"%{plate}%")
    sql += " order by observed_at limit %s"; params.append(limit)
    return repo._fetch(sql, tuple(params))


@app.get("/api/vehicles")
def vehicles(run_id: str | None = None):
    return repo.vehicle_routes(_run(run_id))


@app.get("/api/vehicle/{plate}")
def vehicle_detail(plate: str, run_id: str | None = None):
    """Query-based tracking interface: full history for one plate."""
    rid = _run(run_id)
    plate = plate.upper()
    timeline = repo.plate_timeline(plate, rid)
    if not timeline:
        raise HTTPException(404, f"No sightings of {plate} in this run.")
    veh = repo._fetch("""select * from global_vehicles
                         where run_id=%s and plate_number=%s""", (rid, plate))
    legs = repo._fetch("""select l.* from route_legs l
                          join global_vehicles g on g.run_id=l.run_id
                           and g.global_vehicle_id=l.global_vehicle_id
                          where l.run_id=%s and g.plate_number=%s
                          order by l.leg_index""", (rid, plate))
    readings = repo._fetch("""select observation_id, frame_number, text_raw, text_final,
                                     ocr_confidence, plate_confidence, variant, repaired
                              from plate_readings
                              where run_id=%s and observation_id in
                                (select observation_id from observations
                                 where run_id=%s and plate_number=%s)
                              order by observation_id, frame_number""", (rid, rid, plate))
    return {"plate": plate, "vehicle": veh[0] if veh else None,
            "timeline": timeline, "legs": legs, "raw_readings": readings}


# ------------------------------------------------------- road routing
# The GraphHopper key lives in .env and is used here, server-side. The browser
# never sees it: the frontend asks these endpoints for geometry instead. Every
# response says whether the path is a real road or a straight-line estimate.

@app.get("/api/routing")
def routing_status():
    return routing_describe()


@app.get("/api/road-route")
def road_route_between(from_camera: str, to_camera: str,
                       run_id: str | None = None):
    """Real road geometry between two cameras of the current run."""
    rid = _run(run_id)
    rows = {c["camera_id"]: c for c in repo._fetch(
        "select camera_id, name, latitude, longitude from cameras where run_id=%s",
        (rid,))}
    for cam in (from_camera, to_camera):
        if cam not in rows:
            raise HTTPException(404, f"Unknown camera {cam} in this run.")
    a, b = rows[from_camera], rows[to_camera]
    r = road_route(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
    return {"from_camera": from_camera, "to_camera": to_camera, **r.to_dict()}


@app.get("/api/vehicle-paths")
def vehicle_paths(run_id: str | None = None):
    """Road path for every vehicle that has a multi-camera route."""
    rid = _run(run_id)
    return build_vehicle_paths(_routes_for_paths(rid))


@app.get("/api/vehicle-path/{plate}")
def vehicle_path(plate: str, run_id: str | None = None):
    """The road path one plate took, ready to draw on the map."""
    rid = _run(run_id)
    plate = plate.upper()
    paths = build_vehicle_paths(
        [v for v in _routes_for_paths(rid) if v["plate_number"] == plate])
    if not paths:
        raise HTTPException(404, f"No route for {plate} in this run.")
    return next(iter(paths.values()))


def _routes_for_paths(rid: str) -> list:
    """Global vehicles shaped the way build_vehicle_paths expects."""
    rows = repo._fetch(
        """select global_vehicle_id, plate_number, route_display,
                  total_duration_minutes
           from global_vehicles where run_id=%s
           order by camera_count desc, plate_number""", (rid,))
    for r in rows:
        r["route"] = [c.strip() for c in (r.get("route_display") or "").split("->")
                      if c.strip()]
    return rows


@app.get("/api/vehicle-dossier/{plate}")
def vehicle_dossier(plate: str, run_id: str | None = None):
    """Everything about one vehicle in one payload.

    Sightings, per-leg verdicts, the real road path, this vehicle's own heat
    surface, its alerts and its journey statistics - assembled server-side so
    the parts cannot disagree with each other.
    """
    rid = _run(run_id)
    plate = plate.upper()

    timeline = repo.plate_timeline(plate, rid)
    if not timeline:
        raise HTTPException(404, f"No sightings of {plate} in this run.")

    veh = repo._fetch("""select * from global_vehicles
                         where run_id=%s and plate_number=%s""", (rid, plate))
    legs = repo._fetch("""select l.* from route_legs l
                          join global_vehicles g on g.run_id=l.run_id
                           and g.global_vehicle_id=l.global_vehicle_id
                          where l.run_id=%s and g.plate_number=%s
                          order by l.leg_index""", (rid, plate))
    readings = repo._fetch(
        """select observation_id, frame_number, text_raw, text_final,
                  ocr_confidence, plate_confidence, variant, repaired
           from plate_readings
           where run_id=%s and observation_id in
             (select observation_id from observations where run_id=%s
              and plate_number = any(%s))
           order by observation_id, frame_number""",
        (rid, rid, list({t["plate_number"] for t in timeline})))
    alerts = repo._fetch(
        """select alert_type, severity, camera_id, location, occurred_at,
                  ocr_confidence, evidence, reason, status
           from alerts where run_id=%s and plate_number=%s
           order by occurred_at""", (rid, plate))

    paths = build_vehicle_paths(
        [v for v in _routes_for_paths(rid) if v["plate_number"] == plate])
    path = next(iter(paths.values()), None)

    return {
        "plate": plate,
        "vehicle": veh[0] if veh else None,
        "timeline": timeline,
        "legs": legs,
        "path": path,
        "heat": path_heat_points(path, timeline),
        "stats": journey_stats(timeline, legs, path or {}),
        "alerts": alerts,
        "raw_readings": readings,
    }


@app.get("/api/search")
def search(q: str, run_id: str | None = None, limit: int = 25):
    rid = _run(run_id)
    return repo._fetch(
        """select plate_number, route_display, route_status, camera_count,
                  first_seen, last_seen, confidence
           from global_vehicles
           where run_id=%s and plate_number ilike %s
           order by camera_count desc limit %s""", (rid, f"%{q.upper()}%", limit))


# ---------------------------------------------------------- analytics

@app.get("/api/flows")
def flows(run_id: str | None = None):
    return repo.camera_flows(_run(run_id))


@app.get("/api/heatmap")
def heatmap(run_id: str | None = None):
    """Observation density per camera, for the GIS heat layer."""
    rid = _run(run_id)
    return repo._fetch(
        """select c.camera_id, c.name, c.latitude, c.longitude,
                  count(o.id) as observations,
                  count(distinct o.plate_number) as vehicles
           from cameras c
           left join observations o on o.run_id=c.run_id and o.camera_id=c.camera_id
           where c.run_id=%s
           group by c.camera_id, c.name, c.latitude, c.longitude
           order by c.camera_id""", (rid,))


@app.get("/api/timeline")
def timeline(run_id: str | None = None, bucket_seconds: int = 30):
    """Detections over time per camera - the traffic-trend series."""
    rid = _run(run_id)
    return repo._fetch(
        """select camera_id,
                  floor(video_time_s / %s) * %s as bucket_s,
                  count(*) as detections
           from observations where run_id=%s
           group by camera_id, bucket_s order by bucket_s""",
        (bucket_seconds, bucket_seconds, rid))


@app.get("/api/vehicle-types")
def vehicle_types(run_id: str | None = None):
    return repo._fetch("""select vehicle_type, count(*) as count
                          from observations where run_id=%s
                          group by vehicle_type order by count desc""", (_run(run_id),))


@app.get("/api/vehicle-counts")
def vehicle_counts(run_id: str | None = None):
    """Unique vehicles counted per camera, by class (each track counted once)."""
    return repo._fetch("select * from v_vehicle_counts where run_id=%s order by camera_id",
                       (_run(run_id),))


@app.get("/api/stats")
def stats(run_id: str | None = None):
    return repo._fetch("select * from camera_stats where run_id=%s order by camera_id",
                       (_run(run_id),))


# ------------------------------------------------------------- alerts

@app.get("/api/alerts")
def alerts(run_id: str | None = None):
    return repo._fetch(
        """select id, alert_type, severity, plate_number, camera_id, location,
                  occurred_at, ocr_confidence, evidence, reason, status
           from alerts where run_id=%s order by occurred_at desc""", (_run(run_id),))


@app.get("/api/watchlist")
def watchlist():
    """Both watchlist stores, so the dashboard cannot contradict the alerts.

    Alerts are generated against the union of data/watchlist.txt and the
    database table; showing only the table made the UI report an empty
    watchlist beside live matches from the file.
    """
    rows = repo._fetch("select plate_number, note, active, created_at from watchlist "
                       "order by plate_number")
    have = {r["plate_number"].upper() for r in rows}
    for plate in sorted(load_watchlist_file(ROOT / "data" / "watchlist.txt")):
        if plate not in have:
            rows.append({"plate_number": plate, "note": "from data/watchlist.txt",
                         "active": True, "created_at": None})
    return sorted(rows, key=lambda r: r["plate_number"])


@app.get("/api/watchlist/db")
def watchlist_db():
    return repo._fetch("select plate_number, note, active, created_at from watchlist "
                       "order by created_at desc")


@app.post("/api/watchlist/{plate}")
def watchlist_add(plate: str, note: str = ""):
    repo.add_to_watchlist(plate, note)
    return {"added": plate.upper()}


# ------------------------------------------------------------- media

@app.get("/api/video/{camera_id}")
def video(camera_id: str):
    """Serve the annotated video, transcoding to H.264 on first request.

    OpenCV writes mp4v, which browsers refuse to decode. We re-encode once with
    the bundled ffmpeg and cache the result beside the original.
    """
    h264 = OUTPUT / f"{camera_id}_annotated_h264.mp4"
    if h264.exists():
        return FileResponse(h264, media_type="video/mp4")

    raw = OUTPUT / f"{camera_id}_annotated.mp4"
    if not raw.exists():
        raise HTTPException(404, f"No annotated video for {camera_id}")
    try:
        from ..viz.video_utils import to_h264
        out = to_h264(raw, h264)
        return FileResponse(out, media_type="video/mp4")
    except Exception as e:
        # Serve the original rather than nothing; some players cope with mp4v.
        print(f"[api] transcode failed for {camera_id}: {e}")
        return FileResponse(raw, media_type="video/mp4")


@app.get("/api/evidence")
def evidence(path: str):
    """Serve a plate crop. Path is constrained to the evidence directory."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    try:
        p = p.resolve()
        p.relative_to(EVIDENCE.resolve())     # prevents path traversal
    except (ValueError, OSError):
        raise HTTPException(403, "Outside the evidence directory")
    if not p.exists():
        raise HTTPException(404, "Evidence not found")
    return FileResponse(p, media_type="image/jpeg")


# ---------------------------------------------------------- live feeds
# Real-time detection. A session runs the same vision stack continuously
# against a source OpenCV can open - an RTSP camera, a local device index, or
# a video file played at its own frame rate - and streams what it finds.

def _live_watchlist() -> set:
    """Union of both watchlist stores, same as the batch pipeline uses."""
    wl = load_watchlist_file(ROOT / "data" / "watchlist.txt")
    try:
        wl |= repo.get_watchlist()
    except Exception:
        pass
    return wl


@app.get("/api/live/status")
def live_status():
    return LIVE.status()


@app.post("/api/live/start")
def live_start(camera_id: str, source: str | None = None,
               realtime: bool = True, detect_every: int = 1,
               loop: bool = True):
    """Begin live detection for one camera.

    `source` defaults to that camera's configured video file, so the dashboard
    can start a real feed with one click on a machine that has no RTSP camera
    attached. Pass an rtsp:// URL or a device index for actual live hardware.
    """
    src = source
    if not src:
        cam = CAMERAS.get(camera_id)
        if cam is None:
            raise HTTPException(404, f"Unknown camera {camera_id}")
        p = ROOT / "data" / "videos" / cam.video_file
        if not p.exists():
            raise HTTPException(
                404, f"No source given and {p.name} is not in data/videos/")
        src = str(p)
    s = LIVE.start(camera_id, src, realtime=realtime, loop=loop,
                   detect_every=max(1, detect_every), watchlist=_live_watchlist())
    return s.status()


@app.post("/api/live/stop")
def live_stop(camera_id: str | None = None):
    if camera_id:
        if not LIVE.stop(camera_id):
            raise HTTPException(404, f"No live session for {camera_id}")
        return LIVE.status()
    LIVE.stop_all()
    return LIVE.status()


@app.get("/api/live/frame/{camera_id}")
def live_frame(camera_id: str):
    """Latest JPEG from a live session, for the dashboard preview."""
    s = LIVE.get(camera_id)
    if not s or not s.latest_jpeg:
        # 204 rather than 404: a preview poll that arrives before the first
        # frame is decoded is normal, not an error, and a 404 fills the
        # browser console with noise that hides real failures.
        return Response(status_code=204)
    return Response(content=s.latest_jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/live/stream")
def live_stream(camera_id: str):
    """Server-sent events for one live session.

    SSE rather than WebSocket: the traffic is one-directional, it survives
    proxies, and the browser reconnects on its own.
    """
    s = LIVE.get(camera_id)
    if not s:
        raise HTTPException(404, f"No live session for {camera_id}. Start one first.")

    def sse(kind: str, payload: dict) -> str:
        return f"event: {kind}\ndata: {json.dumps(payload)}\n\n"

    def gen():
        q = s.subscribe()
        try:
            yield sse("hello", s.status())
            while True:
                try:
                    ev = q.get(timeout=15)
                except Exception:
                    # Keep-alive comment so proxies do not close an idle stream.
                    yield ": keep-alive\n\n"
                    continue
                yield sse(ev["type"], ev)
                if ev["type"] == "stopped":
                    break
        finally:
            s.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        "Connection": "keep-alive"})


@app.on_event("shutdown")
def _stop_live():
    LIVE.stop_all()


# ----------------------------------------------------------- frontend
# The React app (web/) builds to web-dist/ and is served at / when present.
# The original vanilla-JS dashboard stays reachable at /legacy so nothing that
# already works is taken away by adding the new one.

if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST / "assets")), name="assets")


@app.get("/legacy")
def legacy_index():
    f = FRONTEND / "index.html"
    if not f.exists():
        return JSONResponse({"error": "legacy frontend missing"}, status_code=404)
    return FileResponse(f)


@app.get("/")
@app.get("/multi-camera")
def index():
    """The React dashboard, falling back to the legacy one if it is not built."""
    react = WEB_DIST / "index.html"
    if react.exists():
        return FileResponse(react)
    f = FRONTEND / "index.html"
    if not f.exists():
        return JSONResponse(
            {"error": "No frontend built. Run: cd web && npm install && npm run build"},
            status_code=404)
    return FileResponse(f)
