"""City-wide traffic analytics and trajectory reconstruction.

Every function here answers its question in SQL, against indexed and
partitioned tables, and returns plain dictionaries. That is a deliberate scale
decision: pulling tens of millions of sightings into Python to aggregate them
would be slower than the database by orders of magnitude and would bound the
platform by the API server's memory instead of the database's disk.

Three families of question, matching the three things the platform is for:

    trajectory  -- where did ONE vehicle go?
    flow        -- where is traffic going in general?
    congestion  -- where is it going more slowly than it should?

Congestion is measured against each corridor's OWN historical baseline for the
same hour of day, not against a fixed speed. A 25 km/h arterial in dense city
traffic is normal; the same 25 km/h on a bypass that normally runs at 70 is a
bottleneck. A single global threshold cannot tell those apart.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..db import repository as repo
from .config import SCALE
from .identity import canonical_key, same_vehicle


def _window(hours: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


# ------------------------------------------------------------ trajectory

def plate_trajectory(plate: str, hours: float = 24.0, limit: int = 500) -> dict:
    """Chronological movement history for one plate.

    Lookup is by canonical (OCR-confusion-folded) key, so a query for
    TN01AB1234 also finds the sighting that was read as TNO1AB1234. Candidates
    are then confirmed with the strict `same_vehicle` rule, so folding widens
    recall without merging genuinely different plates.
    """
    plate = (plate or "").upper().strip()
    if not plate:
        return {"plate": plate, "sightings": [], "legs": [], "cameras": 0}

    rows = repo._fetch("""
        select s.sighting_id, s.camera_id, s.plate, s.cluster_id, s.observed_at,
               s.confidence, s.vehicle_type, s.track_id, s.support, s.total_readings,
               s.is_valid_format, s.latitude, s.longitude,
               c.name as camera_name, c.road, c.sector, c.direction
        from net_sightings s
        left join net_cameras c on c.camera_id = s.camera_id
        where s.plate_canonical = %s and s.observed_at >= %s
        order by s.observed_at
        limit %s
    """, (canonical_key(plate), _window(hours), limit))

    sightings = [r for r in rows if same_vehicle(plate, r["plate"])]
    legs = _legs_from(sightings)
    cameras = {s["camera_id"] for s in sightings}
    return {
        "plate": plate,
        "sightings": sightings,
        "legs": legs,
        "cameras": len(cameras),
        "first_seen": sightings[0]["observed_at"] if sightings else None,
        "last_seen": sightings[-1]["observed_at"] if sightings else None,
        "spellings": sorted({s["plate"] for s in sightings}),
        "window_hours": hours,
    }


def _legs_from(sightings: list) -> list:
    """Measured hop between consecutive sightings at DIFFERENT cameras."""
    from .registry import REGISTRY

    legs = []
    prev = None
    for s in sightings:
        if prev is not None and s["camera_id"] != prev["camera_id"]:
            legs.append(_leg(prev, s, REGISTRY))
        prev = s
    return legs


def _leg(a: dict, b: dict, registry) -> dict:
    minutes = (b["observed_at"] - a["observed_at"]).total_seconds() / 60.0
    edge = registry.edge(a["camera_id"], b["camera_id"])
    if edge is not None:
        km, source = edge.road_distance_km, edge.distance_source
        lo, hi = edge.expected_min_minutes, edge.expected_max_minutes
    else:
        from .registry import haversine_km
        km = round(haversine_km(a["latitude"] or 0, a["longitude"] or 0,
                                b["latitude"] or 0, b["longitude"] or 0) * SCALE.road_factor, 3)
        source, lo, hi = "estimate-unregistered", None, None

    speed = (km / (minutes / 60.0)) if minutes > 0 else None
    if minutes <= 0:
        verdict = "IMPLAUSIBLE"
        reason = f"Arrival at {b['camera_id']} is not after departure from {a['camera_id']}."
    elif speed is not None and speed > SCALE.max_plausible_speed_kmh:
        verdict = "IMPLAUSIBLE"
        reason = (f"{km:.2f} km in {minutes:.2f} min implies {speed:.0f} km/h, above the "
                  f"{SCALE.max_plausible_speed_kmh:.0f} km/h feasibility limit.")
    elif speed is not None and speed > SCALE.warn_speed_kmh:
        verdict = "FAST"
        reason = f"{km:.2f} km in {minutes:.2f} min implies {speed:.0f} km/h."
    elif hi is not None and minutes > hi * 1.5:
        verdict = "SLOW"
        reason = (f"Took {minutes:.1f} min against an expected {lo:.1f}-{hi:.1f} min; "
                  f"congestion or an intermediate stop.")
    else:
        verdict = "PLAUSIBLE"
        reason = f"{km:.2f} km in {minutes:.1f} min ({speed:.0f} km/h)." if speed else "Same instant."
    return {
        "from_camera": a["camera_id"], "to_camera": b["camera_id"],
        "departed": a["observed_at"], "arrived": b["observed_at"],
        "travel_minutes": round(minutes, 2), "road_distance_km": km,
        "implied_speed_kmh": round(speed, 1) if speed is not None else None,
        "distance_source": source, "verdict": verdict, "reason": reason,
    }


def search_plates(fragment: str, hours: float = 24.0, limit: int = 50) -> list:
    """Plate lookup for the tracking interface."""
    frag = (fragment or "").upper().strip()
    if not frag:
        return []
    return repo._fetch("""
        select plate, count(*) as sightings, count(distinct camera_id) as cameras,
               min(observed_at) as first_seen, max(observed_at) as last_seen,
               round(avg(confidence)::numeric, 3) as mean_confidence
        from net_sightings
        where observed_at >= %s and (plate like %s or plate_canonical like %s)
        group by plate
        order by cameras desc, sightings desc
        limit %s
    """, (_window(hours), f"%{frag}%", f"%{canonical_key(frag)}%", limit))


# ------------------------------------------------------------- density

def camera_density(hours: float = 1.0) -> list:
    """Per-camera traffic density, vehicles per minute, over the window."""
    return repo._fetch("""
        select c.camera_id, c.name, c.road, c.sector, c.latitude, c.longitude,
               count(s.id) as sightings,
               count(distinct s.plate_canonical) as unique_vehicles,
               round(avg(s.confidence)::numeric, 3) as mean_confidence,
               round((count(s.id) / (%s * 60.0))::numeric, 2) as vehicles_per_minute,
               max(s.observed_at) as last_sighting
        from net_cameras c
        left join net_sightings s
               on s.camera_id = c.camera_id and s.observed_at >= %s
        where c.enabled
        group by c.camera_id, c.name, c.road, c.sector, c.latitude, c.longitude
        order by sightings desc
    """, (hours, _window(hours)))


def heatmap(hours: float = 1.0) -> list:
    """Points for the GIS heat layer: measured presence per camera."""
    return repo._fetch("""
        select c.camera_id, c.name, c.latitude, c.longitude,
               count(s.id) as observations,
               count(distinct s.plate_canonical) as vehicles
        from net_cameras c
        left join net_sightings s
               on s.camera_id = c.camera_id and s.observed_at >= %s
        where c.enabled
        group by c.camera_id, c.name, c.latitude, c.longitude
        having count(s.id) > 0
        order by observations desc
    """, (_window(hours),))


def network_summary(hours: float = 1.0) -> dict:
    since = _window(hours)
    row = repo._fetch("""
        select
          (select count(*) from net_cameras where enabled) as cameras,
          (select count(*) from net_sightings where observed_at >= %s) as sightings,
          (select count(distinct plate_canonical) from net_sightings
             where observed_at >= %s) as unique_vehicles,
          (select count(distinct camera_id) from net_sightings
             where observed_at >= %s) as active_cameras,
          (select count(*) from net_vehicles where camera_count > 1
             and last_seen >= %s) as multi_camera_vehicles,
          (select count(*) from net_alerts where occurred_at >= %s) as alerts,
          (select round(avg(confidence)::numeric, 3) from net_sightings
             where observed_at >= %s) as mean_confidence,
          (select round(avg(implied_speed_kmh)::numeric, 1) from net_legs
             where arrived >= %s and verdict <> 'IMPLAUSIBLE') as mean_speed_kmh
    """, (since,) * 7)[0]
    row["window_hours"] = hours
    return row


# ---------------------------------------------------- origin-destination

def origin_destination(hours: float = 6.0, min_vehicles: int = 1, limit: int = 200) -> list:
    """Camera-to-camera movement matrix, measured from reconstructed legs."""
    return repo._fetch("""
        select l.from_camera, l.to_camera,
               a.name as from_name, b.name as to_name,
               a.latitude as from_lat, a.longitude as from_lon,
               b.latitude as to_lat,   b.longitude as to_lon,
               count(*) as vehicles,
               round(avg(l.travel_minutes)::numeric, 2) as mean_travel_minutes,
               round(avg(l.implied_speed_kmh)::numeric, 1) as mean_speed_kmh,
               max(l.road_distance_km) as road_distance_km
        from net_legs l
        left join net_cameras a on a.camera_id = l.from_camera
        left join net_cameras b on b.camera_id = l.to_camera
        where l.arrived >= %s and l.verdict <> 'IMPLAUSIBLE'
        group by l.from_camera, l.to_camera, a.name, b.name,
                 a.latitude, a.longitude, b.latitude, b.longitude
        having count(*) >= %s
        order by vehicles desc
        limit %s
    """, (_window(hours), min_vehicles, limit))


# ---------------------------------------------------------- congestion

def refresh_baseline(days: int = 28) -> int:
    """Recompute each corridor's normal speed per hour of day.

    Congestion is meaningless without this: it is defined as a corridor running
    slower than IT normally does at THIS hour, not slower than a fixed number.
    """
    from ..db.client import connect
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                insert into net_flow_baseline
                  (from_camera, to_camera, hour_of_day, samples,
                   mean_speed_kmh, mean_travel_minutes, updated_at)
                select from_camera, to_camera,
                       extract(hour from arrived)::int as hour_of_day,
                       count(*), avg(implied_speed_kmh), avg(travel_minutes), now()
                from net_legs
                where arrived >= %s and verdict <> 'IMPLAUSIBLE'
                  and implied_speed_kmh is not null
                group by from_camera, to_camera, hour_of_day
                on conflict (from_camera, to_camera, hour_of_day) do update set
                  samples = excluded.samples,
                  mean_speed_kmh = excluded.mean_speed_kmh,
                  mean_travel_minutes = excluded.mean_travel_minutes,
                  updated_at = now()
            """, (since,))
            n = cur.rowcount
        conn.commit()
    return n


def congestion(hours: float = 1.0, min_samples: int | None = None,
               speed_ratio: float = 0.60) -> list:
    """Corridors currently running slower than their own baseline.

    A corridor is reported only when it has enough historical samples to have a
    meaningful normal. Without that guard, a corridor seen twice in its life
    would be declared a bottleneck on its second observation.
    """
    min_samples = 5 if min_samples is None else min_samples
    return repo._fetch("""
        with live as (
            select from_camera, to_camera,
                   extract(hour from arrived)::int as hour_of_day,
                   count(*) as vehicles,
                   avg(implied_speed_kmh) as speed_now,
                   avg(travel_minutes) as travel_now
            from net_legs
            where arrived >= %s and verdict <> 'IMPLAUSIBLE'
              and implied_speed_kmh is not null
            group by from_camera, to_camera, hour_of_day
        )
        select l.from_camera, l.to_camera,
               a.name as from_name, b.name as to_name,
               a.latitude as from_lat, a.longitude as from_lon,
               b.latitude as to_lat,   b.longitude as to_lon,
               l.vehicles,
               round(l.speed_now::numeric, 1)  as speed_kmh,
               round(f.mean_speed_kmh::numeric, 1) as baseline_kmh,
               round((l.speed_now / nullif(f.mean_speed_kmh, 0))::numeric, 3) as speed_ratio,
               round(l.travel_now::numeric, 2) as travel_minutes,
               round(f.mean_travel_minutes::numeric, 2) as baseline_travel_minutes,
               f.samples as baseline_samples,
               case
                 when l.speed_now / nullif(f.mean_speed_kmh, 0) < 0.40 then 'SEVERE'
                 when l.speed_now / nullif(f.mean_speed_kmh, 0) < 0.55 then 'HEAVY'
                 else 'MODERATE'
               end as severity
        from live l
        join net_flow_baseline f
          on f.from_camera = l.from_camera and f.to_camera = l.to_camera
         and f.hour_of_day = l.hour_of_day
        left join net_cameras a on a.camera_id = l.from_camera
        left join net_cameras b on b.camera_id = l.to_camera
        where f.samples >= %s
          and f.mean_speed_kmh > 0
          and l.speed_now / f.mean_speed_kmh < %s
        order by speed_ratio asc
    """, (_window(hours), min_samples, speed_ratio))


def flow_trend(hours: float = 6.0, bucket_minutes: int = 15) -> list:
    """Network-wide sighting volume over time -- the traffic trend series."""
    return repo._fetch("""
        select to_timestamp(floor(extract(epoch from observed_at) / (%s * 60))
                            * (%s * 60)) at time zone 'UTC' as bucket,
               count(*) as sightings,
               count(distinct plate_canonical) as unique_vehicles,
               count(distinct camera_id) as active_cameras
        from net_sightings
        where observed_at >= %s
        group by bucket
        order by bucket
    """, (bucket_minutes, bucket_minutes, _window(hours)))
