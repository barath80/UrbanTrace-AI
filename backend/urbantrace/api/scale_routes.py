"""HTTP surface for the city-scale network.

These endpoints read the continuous `net_*` tables, so they describe the live
camera estate rather than a single batch run. The batch endpoints in `main.py`
are unchanged and still serve the run-scoped views.

Every list endpoint takes a time window in hours and is backed by an index, so
response time depends on how much data the window covers, not on how much
history the database holds.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..scale import analytics
from ..scale.alerts import recent_alerts
from ..scale.config import SCALE
from ..scale.registry import REGISTRY
from ..scale.service import SERVICE, load_cameras_from_db, save_cameras_to_db

router = APIRouter(prefix="/api/network", tags=["City network"])


@router.get("/summary")
def network_summary(hours: float = Query(1.0, gt=0, le=720)):
    """Headline figures for the whole network over the window."""
    return analytics.network_summary(hours)


@router.get("/cameras")
def cameras():
    """The camera estate, with its graph statistics."""
    return {
        "registry": REGISTRY.summary(),
        "cameras": [c.to_dict() for c in REGISTRY],
    }


@router.post("/cameras/reload")
def cameras_reload():
    """Re-read the camera estate from the database into the live registry."""
    n = load_cameras_from_db()
    return {"loaded": n, "registry": REGISTRY.summary()}


@router.post("/cameras/persist")
def cameras_persist():
    """Write the in-memory registry and its edges back to the database."""
    return {"saved": save_cameras_to_db(), "registry": REGISTRY.summary()}


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
    """Live ingest health: throughput, queue depth, drops, sink lag."""
    return SERVICE.stats()


@router.get("/config")
def config():
    """The tunables this process is running with."""
    from dataclasses import asdict
    return asdict(SCALE)


@router.post("/housekeeping")
def housekeeping():
    """Evict expired identities and idle tracker state."""
    return SERVICE.housekeeping()


@router.post("/flush")
def flush():
    """Force a persistence flush. Normally happens on a timer."""
    return SERVICE.flush()
