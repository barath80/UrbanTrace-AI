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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..core.runtime import describe
from ..db import repository as repo
from ..db.client import ping

ROOT = Path(__file__).resolve().parents[3]
FRONTEND = ROOT / "frontend"
OUTPUT = ROOT / "data" / "output"
EVIDENCE = ROOT / "data" / "evidence"

app = FastAPI(title="UrbanTrace AI",
              description="City-wide multi-camera ANPR, trajectory and traffic intelligence",
              version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


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


# ----------------------------------------------------------- frontend

if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
def index():
    f = FRONTEND / "index.html"
    if not f.exists():
        return JSONResponse({"error": "frontend not built"}, status_code=404)
    return FileResponse(f)
