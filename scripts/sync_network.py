"""Make the city-network layer reflect the videos that actually exist.

The camera estate is DERIVED FROM THE VIDEOS in data/videos/, one camera per
file, exactly as the batch pipeline does it. Two videos means two cameras and
two map markers; drop a third video in and the third camera appears on the next
sync. Nothing is invented and nothing is left behind.

This script:

  1. rebuilds `net_cameras` and `net_edges` from the discovered videos,
     DELETING any camera that no longer corresponds to a video;
  2. mirrors the latest batch run's observations into `net_sightings`, so the
     City Analytics and network tracking views show the same real detections
     the Cameras & Video tab shows;
  3. rebuilds `net_vehicles` and `net_legs` from the reconstructed routes;
  4. mirrors watchlist alerts.

    python scripts/sync_network.py
    python scripts/sync_network.py --run-id <uuid>    # a specific run
    python scripts/sync_network.py --prune-only       # just drop stale cameras
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from psycopg2.extras import execute_values  # noqa: E402

from urbantrace.core.cameras import CAMERAS, discover_videos, load_network  # noqa: E402
from urbantrace.db import repository as repo  # noqa: E402
from urbantrace.db.client import connect  # noqa: E402
from urbantrace.scale.config import SCALE  # noqa: E402
from urbantrace.scale.identity import canonical_key  # noqa: E402
from urbantrace.scale.registry import REGISTRY  # noqa: E402


def sync_cameras(prune: bool = True) -> dict:
    """Rebuild the camera estate from the videos on disk."""
    load_network()                       # re-discover videos -> CAMERAS
    if not CAMERAS:
        raise SystemExit("No videos in data/videos/, so there are no cameras to sync.")

    REGISTRY.clear()
    REGISTRY.load([{
        "camera_id": c.camera_id, "name": c.name, "latitude": c.latitude,
        "longitude": c.longitude, "road": c.road, "direction": c.direction,
        "sector": c.road or "", "source": c.video_file,
    } for c in CAMERAS.values()])
    info = REGISTRY.build_graph()

    keep = sorted(CAMERAS)
    rows = [(c.camera_id, SCALE.site_id, c.name, c.road, c.road,
             c.latitude, c.longitude, c.direction, c.video_file, True)
            for c in CAMERAS.values()]
    edges = [(e.source, e.destination, e.road_distance_km,
              e.expected_min_minutes, e.expected_max_minutes, e.distance_source)
             for e in REGISTRY.edges()]

    removed = []
    with connect() as conn:
        with conn.cursor() as cur:
            if prune:
                cur.execute("select camera_id from net_cameras where not (camera_id = any(%s))",
                            (keep,))
                removed = [r[0] for r in cur.fetchall()]
                if removed:
                    # Sightings carry no foreign key, so they are removed
                    # explicitly; edges cascade from the camera row.
                    cur.execute("delete from net_sightings where camera_id = any(%s)", (removed,))
                    cur.execute("delete from net_alerts where camera_id = any(%s)", (removed,))
                    cur.execute("delete from net_legs where from_camera = any(%s) "
                                "or to_camera = any(%s)", (removed, removed))
                    cur.execute("delete from net_cameras where camera_id = any(%s)", (removed,))

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
            cur.execute("delete from net_edges")
            if edges:
                execute_values(cur, """
                    insert into net_edges
                      (source_camera, destination_camera, road_distance_km,
                       expected_min_minutes, expected_max_minutes, distance_source)
                    values %s
                """, edges)
        conn.commit()

    return {"cameras": len(rows), "edges": len(edges), "removed": removed,
            "graph": info}


def mirror_run(run_id: str | None) -> dict:
    """Copy a batch run's detections into the continuous network tables."""
    rid = run_id or repo.latest_run()
    if not rid:
        return {"run_id": None, "sightings": 0, "vehicles": 0, "legs": 0, "alerts": 0}

    obs = repo._fetch("""
        select observation_id, camera_id, plate_number, observed_at, ocr_confidence,
               support, total_readings, is_valid_format, vehicle_type,
               local_track_id, latitude, longitude, direction
        from observations where run_id=%s order by observed_at""", (rid,))
    known = set(CAMERAS)
    obs = [o for o in obs if o["camera_id"] in known]

    vehicles = repo._fetch("""
        select global_vehicle_id, plate_number, first_seen, last_seen, camera_count,
               confidence, merged_variants
        from global_vehicles where run_id=%s""", (rid,))
    legs = repo._fetch("""
        select l.global_vehicle_id, l.leg_index, l.from_camera, l.to_camera,
               l.departed, l.arrived, l.travel_minutes, l.road_distance_km,
               l.implied_speed_kmh, l.verdict, l.reason
        from route_legs l where l.run_id=%s order by l.global_vehicle_id, l.leg_index""",
        (rid,))
    legs = [l for l in legs if l["from_camera"] in known and l["to_camera"] in known]

    alerts = repo._fetch("""
        select alert_type, severity, plate_number, camera_id, occurred_at,
               ocr_confidence, reason
        from alerts where run_id=%s""", (rid,))
    alerts = [a for a in alerts if a["camera_id"] in known]

    # Map each observation to the vehicle whose route contains it.
    plate_to_cluster = {v["plate_number"]: v["global_vehicle_id"] for v in vehicles}
    for v in vehicles:
        for variant in (v.get("merged_variants") or []):
            plate_to_cluster.setdefault(variant, v["global_vehicle_id"])

    sight_rows = []
    days = set()
    for o in obs:
        days.add(o["observed_at"].date())
        sight_rows.append((
            SCALE.site_id, o["camera_id"], o["plate_number"],
            canonical_key(o["plate_number"]),
            plate_to_cluster.get(o["plate_number"]),
            o["observed_at"], float(o["ocr_confidence"] or 0.0),
            int(o["support"] or 0), int(o["total_readings"] or 0),
            bool(o["is_valid_format"]), o["vehicle_type"],
            int(o["local_track_id"] or 0),
            float(o["latitude"] or 0.0), float(o["longitude"] or 0.0),
            o["direction"],
        ))

    veh_rows = [(
        v["global_vehicle_id"], SCALE.site_id, v["plate_number"],
        json.dumps({v["plate_number"]: float(v["confidence"] or 0.0)}),
        v["first_seen"], v["last_seen"], int(v["camera_count"] or 0),
        sum(1 for o in obs if plate_to_cluster.get(o["plate_number"]) == v["global_vehicle_id"]),
        float(v["confidence"] or 0.0), None,
    ) for v in vehicles]

    leg_rows = [(
        l["global_vehicle_id"], int(l["leg_index"] or 0), l["from_camera"], l["to_camera"],
        l["departed"], l["arrived"], float(l["travel_minutes"] or 0.0),
        float(l["road_distance_km"] or 0.0), l["implied_speed_kmh"],
        "batch-run", l["verdict"], l["reason"],
    ) for l in legs]

    alert_rows = [(
        SCALE.site_id, a["alert_type"], a["severity"], a["plate_number"],
        plate_to_cluster.get(a["plate_number"]), a["camera_id"], a["occurred_at"],
        a["ocr_confidence"], a["reason"], json.dumps({"source": "batch run", "run_id": rid}),
    ) for a in alerts]

    with connect() as conn:
        with conn.cursor() as cur:
            for d in sorted(days):
                cur.execute("select ensure_sighting_partition(%s)", (d,))
            # Replace this run's mirrored rows rather than accumulating copies.
            cur.execute("delete from net_sightings where site_id=%s", (SCALE.site_id,))
            cur.execute("delete from net_legs")
            cur.execute("delete from net_vehicles")
            cur.execute("delete from net_alerts where site_id=%s", (SCALE.site_id,))

            if sight_rows:
                execute_values(cur, """
                    insert into net_sightings
                      (site_id, camera_id, plate, plate_canonical, cluster_id, observed_at,
                       confidence, support, total_readings, is_valid_format, vehicle_type,
                       track_id, latitude, longitude, direction)
                    values %s""", sight_rows)
            if veh_rows:
                execute_values(cur, """
                    insert into net_vehicles
                      (cluster_id, site_id, plate, spellings, first_seen, last_seen,
                       camera_count, sighting_count, confidence, vehicle_type)
                    values %s
                    on conflict (cluster_id) do nothing""", veh_rows)
            if leg_rows:
                execute_values(cur, """
                    insert into net_legs
                      (cluster_id, seq, from_camera, to_camera, departed, arrived,
                       travel_minutes, road_distance_km, implied_speed_kmh,
                       distance_source, verdict, reason)
                    values %s
                    on conflict (cluster_id, seq) do nothing""", leg_rows)
            if alert_rows:
                execute_values(cur, """
                    insert into net_alerts
                      (site_id, alert_type, severity, plate, cluster_id, camera_id,
                       occurred_at, confidence, reason, evidence)
                    values %s""", alert_rows)
        conn.commit()

    return {"run_id": rid, "sightings": len(sight_rows), "vehicles": len(veh_rows),
            "legs": len(leg_rows), "alerts": len(alert_rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--prune-only", action="store_true")
    ap.add_argument("--no-prune", action="store_true",
                    help="keep cameras that no longer have a video")
    args = ap.parse_args()

    videos = discover_videos()
    print(f"videos in data/videos/ : {len(videos)}")
    for v in videos:
        print(f"    {v.name}")

    cam = sync_cameras(prune=not args.no_prune)
    print(f"\ncameras synced         : {cam['cameras']}  (one per video)")
    for c in CAMERAS.values():
        print(f"    {c.camera_id}  {c.name:24} {c.latitude:8.4f}, {c.longitude:8.4f}"
              f"   <- {c.video_file}")
    if cam["removed"]:
        print(f"  removed stale cameras : {cam['removed']}")
    print(f"  graph edges           : {cam['edges']}")

    if args.prune_only:
        return

    m = mirror_run(args.run_id)
    print(f"\nmirrored batch run     : {m['run_id']}")
    print(f"    sightings {m['sightings']}   vehicles {m['vehicles']}   "
          f"legs {m['legs']}   alerts {m['alerts']}")
    print("\nThe City Analytics and network tracking views now reflect exactly "
          "these cameras.")


if __name__ == "__main__":
    main()
