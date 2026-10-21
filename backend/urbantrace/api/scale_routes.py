"""HTTP surface for the city-scale network.

These endpoints read the continuous `net_*` tables, so they describe the live
camera estate rather than a single batch run. The batch endpoints in `main.py`
are unchanged and still serve the run-scoped views.

Every list endpoint takes a time window in hours and is backed by an index, so
response time depends on how much data the window covers, not on how much
history the database holds.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..scale import analytics
from ..scale.capture import redact_url
from ..scale.alerts import recent_alerts
from ..scale.config import SCALE
from ..scale.registry import REGISTRY
from ..scale.service import SERVICE, load_cameras_from_db

router = APIRouter(prefix="/api/network", tags=["City network"])


@router.get("/summary")
def network_summary(hours: float = Query(1.0, gt=0, le=720)):
    """Headline figures for the whole network over the window."""
    return analytics.network_summary(hours)


class CameraIn(BaseModel):
    camera_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    name: str = Field(min_length=1, max_length=200)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    road: str = ""
    sector: str = ""
    direction: str = "UNKNOWN"
    source: str = ""          # rtsp://, http(s)://, file path, or empty for push-only
    enabled: bool = True


def _public(row: dict) -> dict:
    """A camera row safe to return: stream credentials never leave the server."""
    out = dict(row)
    if out.get("source"):
        out["source"] = redact_url(out["source"])
    return out


def _upsert_cameras(cams: list[CameraIn]) -> int:
    from psycopg2.extras import execute_values

    from ..db.client import cursor
    rows = [(c.camera_id, SCALE.site_id, c.name, c.road, c.sector, c.latitude,
             c.longitude, c.direction, c.source, c.enabled) for c in cams]
    with cursor() as cur:
        execute_values(cur, """
            insert into net_cameras
              (camera_id, site_id, name, road, sector, latitude, longitude,
               direction, source, enabled)
            values %s
            on conflict (camera_id) do update set
              name=excluded.name, road=excluded.road, sector=excluded.sector,
              latitude=excluded.latitude, longitude=excluded.longitude,
              direction=excluded.direction, source=excluded.source,
              enabled=excluded.enabled, updated_at=now()
        """, rows)
    return len(rows)


@router.get("/cameras")
def cameras():
    """The camera estate as stored in the database, credentials redacted."""
    from ..db import repository as repo
    rows = repo._fetch("""select camera_id, name, road, sector, latitude, longitude,
                                 direction, source, enabled, updated_at
                          from net_cameras order by camera_id""")
    return {"registry": REGISTRY.summary(), "cameras": [_public(r) for r in rows]}


@router.post("/cameras", status_code=201)
def camera_upsert(cam: CameraIn):
    """Add a camera, or update it if the id exists. Workers pick it up within
    one refresh interval and a rebalance assigns it to a worker."""
    return {"saved": _upsert_cameras([cam]), "camera_id": cam.camera_id}


@router.post("/cameras/bulk", status_code=201)
def camera_bulk(cams: list[CameraIn]):
    """Add or update many cameras in one transaction."""
    if not cams:
        raise HTTPException(422, "No cameras supplied.")
    if len(cams) > 5000:
        raise HTTPException(413, "At most 5000 cameras per request.")
    return {"saved": _upsert_cameras(cams)}


@router.delete("/cameras/{camera_id}")
def camera_delete(camera_id: str):
    from ..db.client import cursor
    with cursor() as cur:
        cur.execute("delete from net_cameras where camera_id = %s", (camera_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, f"Unknown camera {camera_id}")
        cur.execute("delete from net_camera_health where camera_id = %s", (camera_id,))
    return {"deleted": camera_id}


@router.patch("/cameras/{camera_id}/enabled")
def camera_enable(camera_id: str, enabled: bool):
    from ..db.client import cursor
    with cursor() as cur:
        cur.execute("update net_cameras set enabled = %s, updated_at = now() "
                    "where camera_id = %s", (enabled, camera_id))
        if cur.rowcount == 0:
            raise HTTPException(404, f"Unknown camera {camera_id}")
    return {"camera_id": camera_id, "enabled": enabled}


@router.get("/workers")
def workers():
    """Live ingest workers and how the estate is spread across them."""
    from ..db import repository as repo
    rows = repo._fetch("""
        select worker_id, host, started_at, heartbeat_at, stats,
               extract(epoch from (now() - heartbeat_at)) as age_s
        from net_workers order by worker_id""")
    for r in rows:
        r["alive"] = r["age_s"] is not None and r["age_s"] <= SCALE.worker_ttl_s
        r["age_s"] = round(float(r["age_s"]), 1) if r["age_s"] is not None else None
    return rows


@router.get("/feeds")
def feeds(state: Optional[str] = None):
    """Per-camera feed health. A feed whose worker has stopped reporting is
    shown as `unmonitored` rather than trusted at its last known state."""
    from ..db import repository as repo
    rows = repo._fetch("""
        select c.camera_id, c.name, c.sector, c.enabled, c.source,
               h.worker_id, h.state, h.source_fps, h.resolution, h.frames_read,
               h.frames_submitted, h.frames_skipped_static, h.reconnects,
               h.last_frame_at, h.last_error,
               extract(epoch from (now() - h.updated_at)) as report_age_s
        from net_cameras c left join net_camera_health h using (camera_id)
        order by c.camera_id""")
    out = []
    stale = SCALE.worker_ttl_s * 2
    for r in rows:
        r = _public(r)
        age = r.pop("report_age_s")
        if not r["enabled"]:
            r["state"] = "disabled"
        elif not r["source"]:
            r["state"] = "push-only"
        elif r["state"] is None or age is None or age > stale:
            r["state"] = "unmonitored"
        if state and r["state"] != state:
            continue
        out.append(r)
    return out


@router.get("/fleet")
def fleet():
    """One-call summary for the operations screen."""
    ws = workers()
    fs = feeds()
    by_state: dict = {}
    for f in fs:
        by_state[f["state"]] = by_state.get(f["state"], 0) + 1
    return {"workers_alive": sum(1 for w in ws if w["alive"]), "workers": len(ws),
            "cameras": len(fs), "feeds_by_state": by_state}


@router.get("/cameras/near")
def cameras_near(latitude: float, longitude: float,
                 radius_km: float = Query(5.0, gt=0, le=200),
                 limit: int = Query(25, ge=1, le=500)):
    """Spatial lookup: which cameras cover a point."""
    found = REGISTRY.near(latitude, longitude, radius_km, limit=limit)
    return [{"camera_id": cid, "distance_km": km,
             **(REGISTRY.get(cid).to_dict() if REGISTRY.get(cid) else {})}
            for cid, km in found]


# ------------------------------------------------------------ trajectory

@router.get("/trajectory/{plate}")
def trajectory(plate: str, hours: float = Query(24.0, gt=0, le=720)):
    """Complete movement history for one plate, chronologically.

    This is the query-based tracking interface: sightings in order, each with
    its camera, location and timestamp, plus the measured leg between
    consecutive cameras and a verdict on whether that journey was possible.
    """
    result = analytics.plate_trajectory(plate, hours=hours)
    if not result["sightings"]:
        raise HTTPException(404, f"No sightings of {plate.upper()} in the last {hours:g}h.")
    return result


@router.get("/search")
def search(q: str, hours: float = Query(24.0, gt=0, le=720),
           limit: int = Query(50, ge=1, le=500)):
    """Plate search, tolerant of OCR confusion."""
    return analytics.search_plates(q, hours=hours, limit=limit)


# -------------------------------------------------------------- analytics

@router.get("/density")
def density(hours: float = Query(1.0, gt=0, le=720)):
    """Traffic density per camera."""
    return analytics.camera_density(hours)


@router.get("/heatmap")
def heatmap(hours: float = Query(1.0, gt=0, le=720)):
    """Measured presence per camera, for the GIS heat layer."""
    return analytics.heatmap(hours)


@router.get("/origin-destination")
def origin_destination(hours: float = Query(6.0, gt=0, le=720),
                       min_vehicles: int = Query(1, ge=1),
                       limit: int = Query(200, ge=1, le=2000)):
    """Camera-to-camera movement matrix."""
    return analytics.origin_destination(hours, min_vehicles, limit)


@router.get("/congestion")
def congestion(hours: float = Query(1.0, gt=0, le=720),
               min_samples: int = Query(5, ge=1),
               speed_ratio: float = Query(0.60, gt=0, le=1.0)):
    """Corridors running slower than their own baseline for this hour."""
    return analytics.congestion(hours, min_samples, speed_ratio)


@router.post("/congestion/baseline")
def refresh_baseline(days: int = Query(28, ge=1, le=365)):
    """Recompute per-corridor normal speeds. Run on a schedule."""
    return {"rows": analytics.refresh_baseline(days), "days": days}


@router.get("/trend")
def trend(hours: float = Query(6.0, gt=0, le=720),
          bucket_minutes: int = Query(15, ge=1, le=1440)):
    """Network-wide volume over time."""
    return analytics.flow_trend(hours, bucket_minutes)


# ----------------------------------------------------------------- alerts

@router.get("/alerts")
def alerts(hours: float = Query(24.0, gt=0, le=720),
           limit: int = Query(200, ge=1, le=2000)):
    return recent_alerts(hours, limit)


@router.post("/watchlist/{plate}")
def watchlist_add(plate: str, reason: str = "", severity: str = "HIGH"):
    SERVICE.alerts.watchlist.add(plate, reason, severity)
    return {"added": plate.upper(), "entries": SERVICE.alerts.watchlist.entries}


@router.get("/watchlist")
def watchlist():
    from ..db import repository as repo
    return repo._fetch("""select plate, reason, severity, active, added_at
                          from net_watchlist order by added_at desc""")


# ------------------------------------------------------------- operations

@router.get("/status")
def status():
    """Fleet health: live workers and the state of every feed."""
    return fleet()


@router.get("/config")
def config():
    """The tunables this process is running with."""
    from dataclasses import asdict
    return asdict(SCALE)
