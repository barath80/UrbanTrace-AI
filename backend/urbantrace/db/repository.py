"""Persistence layer: pipeline output -> Supabase.

Every execution opens a `run`, and all rows are scoped to that run_id, so
repeated runs accumulate history instead of overwriting. Bulk inserts use
execute_values (one round trip per table, not per row) because a full video
can yield thousands of raw OCR readings.
"""
from __future__ import annotations

import json
from datetime import datetime

from psycopg2.extras import execute_values, RealDictCursor

from .client import cursor, pooled

# ------------------------------------------------------------------ runs


def start_run(runtime: dict, models: dict, camera_count: int,
              video_count: int, notes: str = "") -> str:
    with cursor() as cur:
        cur.execute(
            """insert into runs (device, gpu_name, vehicle_model, plate_backend,
                                 plate_model, ocr_model, camera_count, video_count, notes)
               values (%s,%s,%s,%s,%s,%s,%s,%s,%s) returning run_id""",
            (runtime.get("device"), runtime.get("gpu_name"),
             models.get("vehicle_model"), models.get("plate_backend"),
             models.get("plate_model"), models.get("ocr_model"),
             camera_count, video_count, notes))
        return str(cur.fetchone()[0])


def finish_run(run_id: str, observation_count: int) -> None:
    with cursor() as cur:
        cur.execute("update runs set finished_at = now(), observation_count = %s "
                    "where run_id = %s", (observation_count, run_id))


# --------------------------------------------------------------- network


def save_cameras(run_id: str, cameras: dict) -> int:
    rows = [(run_id, c.camera_id, c.name, c.road, c.latitude, c.longitude,
             c.longitude, c.latitude,           # geom: ST_MakePoint(lon, lat)
             c.direction, c.video_file,
             c.start_time or None, c.location_source)
            for c in cameras.values()]
    if not rows:
        return 0
    with cursor() as cur:
        execute_values(cur,
            """insert into cameras (run_id, camera_id, name, road, latitude, longitude,
                                    geom, direction, video_file, capture_start, location_source)
               values %s
               on conflict (run_id, camera_id) do update set
                 name = excluded.name, latitude = excluded.latitude,
                 longitude = excluded.longitude, geom = excluded.geom""",
            rows,
            template="(%s,%s,%s,%s,%s,%s,ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography,"
                     "%s,%s,%s,%s)")
    return len(rows)


def save_edges(run_id: str, edges: list) -> int:
    rows = [(run_id, e.source, e.destination, e.road_distance_km,
             e.expected_min_minutes, e.expected_max_minutes, e.road_segment,
             getattr(e, "distance_source", "estimate"),
             getattr(e, "typical_duration_min", None))
            for e in edges]
    if not rows:
        return 0
    with cursor() as cur:
        execute_values(cur,
            """insert into camera_edges (run_id, source_camera, destination_camera,
                    road_distance_km, expected_min_minutes, expected_max_minutes,
                    road_segment, distance_source, typical_duration_min)
               values %s on conflict (run_id, source_camera, destination_camera) do nothing""",
            rows)
    return len(rows)


def save_videos(run_id: str, video_rows: list) -> int:
    rows = []
    for v in video_rows:
        res = str(v.get("resolution", "x"))
        w, _, h = res.partition("x")
        rows.append((run_id, v.get("camera"), v.get("file"),
                     int(w) if w.isdigit() else None,
                     int(h) if h.isdigit() else None,
                     v.get("fps"), v.get("frames"), v.get("duration_s"),
                     v.get("size_MB")))
    if not rows:
        return 0
    with cursor() as cur:
        execute_values(cur,
            """insert into videos (run_id, camera_id, file_name, width, height,
                                   fps, frame_count, duration_s, size_mb)
               values %s""", rows)
    return len(rows)


def save_camera_stats(run_id: str, stats: dict) -> int:
    rows = []
    for cam, s in stats.items():
        c = s.get("counts_by_class", {}) or {}
        rep = s.get("count_report", {}) or {}
        cfg = rep.get("config", {}) or {}
        rows.append((run_id, cam, s.get("frames"), s.get("vehicle_dets"),
                     s.get("plate_dets"), s.get("ocr_calls"), s.get("ocr_success"),
                     s.get("wall_seconds"), s.get("processing_fps"),
                     s.get("ocr_engine_failures", 0), s.get("plate_detector_failures", 0),
                     s.get("unique_vehicles"), s.get("count_mode"),
                     (rep.get("traffic") or {}).get("total"),
                     (rep.get("presence") or {}).get("total"),
                     c.get("car"), c.get("motorcycle"),
                     c.get("bus"), c.get("truck"), c.get("bicycle"),
                     cfg.get("min_frames"), rep.get("fragments_stitched"),
                     rep.get("rejected")))
    if not rows:
        return 0
    with cursor() as cur:
        execute_values(cur,
            """insert into camera_stats (run_id, camera_id, frames, vehicle_dets,
                    plate_dets, ocr_calls, ocr_success, wall_seconds, processing_fps,
                    ocr_engine_failures, plate_detector_failures,
                    unique_vehicles, count_mode, count_traffic, count_presence,
                    count_car, count_motorcycle, count_bus,
                    count_truck, count_bicycle, count_min_frames,
                    count_fragments_stitched, count_rejected)
               values %s on conflict (run_id, camera_id) do nothing""", rows)
    return len(rows)


# ---------------------------------------------------------- observations


def save_observations(run_id: str, observations: list) -> tuple:
    """Insert observations plus their raw per-frame OCR audit trail."""
    obs = [o if isinstance(o, dict) else o.to_dict() for o in observations]
    if not obs:
        return 0, 0

    obs_rows, read_rows = [], []
    for o in obs:
        obs_rows.append((
            run_id, o["observation_id"], o["camera_id"], o["plate_number"],
            o.get("ocr_confidence"), o.get("plate_detection_confidence"),
            o.get("local_track_id"), o.get("vehicle_type"), o.get("frame_number"),
            o.get("video_time_s"), o.get("timestamp"), o.get("clock_source"),
            o.get("latitude"), o.get("longitude"),
            o.get("longitude"), o.get("latitude"),
            o.get("direction"), o.get("aggregation_method"), o.get("support"),
            o.get("total_readings"), o.get("agreement"), o.get("is_valid_format"),
            o.get("evidence_crop")))
        for r in o.get("raw_readings", []):
            read_rows.append((
                run_id, o["observation_id"], o["camera_id"], o.get("local_track_id"),
                r.get("frame_number"), r.get("timestamp_s"), r.get("text_raw"),
                r.get("text_final"), r.get("ocr_confidence"), r.get("plate_confidence"),
                r.get("is_valid_format"), r.get("repaired"), r.get("variant")))

    with cursor() as cur:
        execute_values(cur,
            """insert into observations (run_id, observation_id, camera_id, plate_number,
                  ocr_confidence, plate_detection_confidence, local_track_id, vehicle_type,
                  frame_number, video_time_s, observed_at, clock_source, latitude, longitude,
                  geom, direction, aggregation_method, support, total_readings, agreement,
                  is_valid_format, evidence_crop)
               values %s on conflict (run_id, observation_id) do nothing""",
            obs_rows,
            template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                     "ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography,%s,%s,%s,%s,%s,%s,%s)")
        if read_rows:
            execute_values(cur,
                """insert into plate_readings (run_id, observation_id, camera_id,
                      local_track_id, frame_number, timestamp_s, text_raw, text_final,
                      ocr_confidence, plate_confidence, is_valid_format, repaired, variant)
                   values %s""", read_rows, page_size=500)
    return len(obs_rows), len(read_rows)


# ------------------------------------------------------- global vehicles


def save_global_vehicles(run_id: str, vehicles: list) -> tuple:
    vs = [v if isinstance(v, dict) else v.to_dict() for v in vehicles]
    if not vs:
        return 0, 0

    veh_rows, leg_rows, link_rows = [], [], []
    for v in vs:
        veh_rows.append((
            run_id, v["global_vehicle_id"], v["plate_number"], v.get("route") or [],
            " -> ".join(v.get("route") or []), v.get("route_status"), v.get("dropped_at"),
            v.get("status_reason"), v.get("first_seen") or None,
            v.get("last_seen") or None, v.get("total_duration_minutes"),
            v.get("camera_count"), v.get("confidence"), v.get("merged_variants") or []))
        for i, l in enumerate(v.get("legs", [])):
            leg_rows.append((
                run_id, v["global_vehicle_id"], i, l.get("from_camera"), l.get("to_camera"),
                l.get("departed") or None, l.get("arrived") or None, l.get("travel_minutes"),
                l.get("road_distance_km"), l.get("implied_speed_kmh"),
                l.get("expected_min_minutes"), l.get("expected_max_minutes"),
                l.get("distance_source", "estimate"), l.get("typical_duration_min"),
                l.get("verdict"), l.get("reason")))
        for i, o in enumerate(v.get("observations", [])):
            link_rows.append((run_id, v["global_vehicle_id"], o["observation_id"], i))

    with cursor() as cur:
        execute_values(cur,
            """insert into global_vehicles (run_id, global_vehicle_id, plate_number, route,
                  route_display, route_status, dropped_at, status_reason, first_seen,
                  last_seen, total_duration_minutes, camera_count, confidence, merged_variants)
               values %s on conflict (run_id, global_vehicle_id) do nothing""", veh_rows)
        if leg_rows:
            execute_values(cur,
                """insert into route_legs (run_id, global_vehicle_id, leg_index, from_camera,
                      to_camera, departed, arrived, travel_minutes, road_distance_km,
                      implied_speed_kmh, expected_min_minutes, expected_max_minutes,
                      distance_source, typical_duration_min,
                      verdict, reason) values %s""", leg_rows)
        if link_rows:
            execute_values(cur,
                """insert into vehicle_observations (run_id, global_vehicle_id,
                      observation_id, seq) values %s""", link_rows, page_size=500)
    return len(veh_rows), len(leg_rows)


# -------------------------------------------------------------- alerts


def save_alerts(run_id: str, alerts: list) -> int:
    rows = [(run_id, a.get("alert_type"), a.get("severity"), a.get("plate"),
             a.get("camera"), a.get("location"), a.get("timestamp") or None,
             a.get("ocr_confidence"), a.get("evidence"), a.get("reason"))
            for a in alerts]
    if not rows:
        return 0
    with cursor() as cur:
        execute_values(cur,
            """insert into alerts (run_id, alert_type, severity, plate_number, camera_id,
                  location, occurred_at, ocr_confidence, evidence, reason) values %s""", rows)
    return len(rows)


def get_watchlist() -> set:
    with cursor(commit=False) as cur:
        cur.execute("select plate_number from watchlist where active")
        return {r[0].upper() for r in cur.fetchall()}


def add_to_watchlist(plate: str, note: str = "") -> None:
    with cursor() as cur:
        cur.execute("""insert into watchlist (plate_number, note) values (%s,%s)
                       on conflict (plate_number) do update
                       set active = true, note = excluded.note""",
                    (plate.upper(), note))


# ------------------------------------------------------------- queries


def _fetch(sql: str, params=()) -> list:
    # Pooled: a read-only dashboard request runs several of these, and paying a
    # TLS handshake for each one dominated the response time.
    with pooled(commit=False) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def latest_run() -> str | None:
    r = _fetch("select run_id from runs order by started_at desc limit 1")
    return str(r[0]["run_id"]) if r else None


def plate_timeline(plate: str, run_id: str | None = None) -> list:
    """Every sighting of a vehicle, chronologically.

    Resolved through the GLOBAL VEHICLE, not the plate string. Association
    merges OCR variants - TN01AB1234 and TN01A8123 are one car - and the
    sighting at the second camera is stored under whichever text was read
    there. Matching observations on plate_number alone drops it, which is how
    a two-camera vehicle came to show "1 sighting" on a two-stop route.

    Falls back to the plate string for observations never linked to a vehicle.
    """
    plate = plate.upper()
    if run_id:
        rows = _fetch("""select * from v_vehicle_timeline
                         where vehicle_plate = %s and run_id = %s
                         order by observed_at""", (plate, run_id))
        if rows:
            return rows
        return _fetch("""select * from v_plate_timeline
                         where plate_number = %s and run_id = %s
                         order by observed_at""", (plate, run_id))
    rows = _fetch("""select * from v_vehicle_timeline where vehicle_plate = %s
                     order by observed_at""", (plate,))
    if rows:
        return rows
    return _fetch("select * from v_plate_timeline where plate_number = %s "
                  "order by observed_at", (plate,))


def camera_counts(run_id: str) -> list:
    return _fetch("select * from v_camera_counts where run_id = %s order by camera_id",
                  (run_id,))


def camera_flows(run_id: str) -> list:
    return _fetch("select * from v_camera_flows where run_id = %s "
                  "order by vehicles desc", (run_id,))


def vehicle_routes(run_id: str) -> list:
    return _fetch("""select global_vehicle_id, plate_number, route_display, route_status,
                            camera_count, first_seen, last_seen, total_duration_minutes,
                            confidence
                     from global_vehicles where run_id = %s
                     order by camera_count desc, confidence desc""", (run_id,))


def run_summary(run_id: str) -> dict:
    r = _fetch("""select r.*,
                    (select count(*) from observations o where o.run_id = r.run_id) as observations,
                    (select count(*) from global_vehicles g where g.run_id = r.run_id) as global_vehicles,
                    (select count(*) from alerts a where a.run_id = r.run_id) as alerts,
                    (select count(*) from plate_readings p where p.run_id = r.run_id) as raw_readings
                  from runs r where r.run_id = %s""", (run_id,))
    return r[0] if r else {}


def save_all(runtime: dict, models: dict, cameras: dict, edges: list,
             video_rows: list, camera_stats: dict, observations: list,
             vehicles: list, alerts: list, notes: str = "") -> dict:
    """One call to persist a complete pipeline run. Returns a write report."""
    run_id = start_run(runtime, models, len(cameras), len(video_rows), notes)
    report = {"run_id": run_id}
    report["cameras"] = save_cameras(run_id, cameras)
    report["camera_edges"] = save_edges(run_id, edges)
    report["videos"] = save_videos(run_id, video_rows)
    report["camera_stats"] = save_camera_stats(run_id, camera_stats)
    n_obs, n_read = save_observations(run_id, observations)
    report["observations"], report["plate_readings"] = n_obs, n_read
    n_veh, n_legs = save_global_vehicles(run_id, vehicles)
    report["global_vehicles"], report["route_legs"] = n_veh, n_legs
    report["alerts"] = save_alerts(run_id, alerts)
    finish_run(run_id, n_obs)
    return report
