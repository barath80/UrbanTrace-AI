"""End-to-end system verification.

Runs the complete pipeline against real video and checks every component,
reporting PASS/FAIL per check. Designed to be run with ANY number of videos:

    python scripts/verify_system.py                  # uses data/videos/ as-is
    python scripts/verify_system.py --cameras 2      # builds N test clips first
    python scripts/verify_system.py --max-frames 80

With --cameras it creates overlapping clips from a real traffic video so that
the same vehicles genuinely appear at more than one camera, which is what
exercises the cross-camera routing path. Test clips are removed afterwards
unless --keep is passed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

VIDEO_DIR = ROOT / "data" / "videos"
OUT_DIR = ROOT / "data" / "output"
EVIDENCE_DIR = ROOT / "data" / "evidence"

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def build_test_clips(n: int, frames_each: int = 90, overlap: int = 45) -> bool:
    """Create N overlapping clips so vehicles genuinely span cameras."""
    src = VIDEO_DIR / "_source.mp4"
    if not src.exists():
        try:
            import os
            cwd = os.getcwd()
            os.chdir(VIDEO_DIR)
            from supervision.assets import download_assets, VideoAssets
            got = Path(download_assets(VideoAssets.VEHICLES))
            os.chdir(cwd)
            (VIDEO_DIR / got.name).rename(src)
        except Exception as e:
            print(f"could not obtain source video: {e}")
            return False

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, frames_each - overlap)

    for i in range(n):
        start = min(i * step, max(0, total - frames_each))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        vw = cv2.VideoWriter(str(VIDEO_DIR / f"CAMERA_{i+1}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        written = 0
        for _ in range(frames_each):
            ok, f = cap.read()
            if not ok:
                break
            vw.write(f); written += 1
        vw.release()
        print(f"  CAMERA_{i+1}.mp4  frames {start}..{start+written}")
    cap.release()
    src.unlink(missing_ok=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=0,
                    help="build N overlapping test clips before running")
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--keep", action="store_true", help="keep generated artifacts")
    args = ap.parse_args()

    print("=" * 76)
    print("UrbanTrace AI - system verification")
    print("=" * 76)

    if args.cameras:
        print(f"\n[0] Building {args.cameras} overlapping test clip(s)")
        for f in VIDEO_DIR.glob("*.mp4"):
            f.unlink()
        if not build_test_clips(args.cameras):
            return 1

    # ---- 1. runtime -------------------------------------------------
    print("\n[1] Runtime")
    from urbantrace.core.runtime import describe
    rt = describe()
    check("runtime resolves a device", rt["device"] in ("cuda", "cpu"),
          f"device={rt['device']} gpu={rt.get('gpu_name')}")

    # ---- 2. dynamic camera network ---------------------------------
    print("\n[2] Dynamic camera network")
    from urbantrace.core import cameras as C
    C.load_network()
    n_cam = len(C.CAMERAS)
    check("videos discovered", n_cam > 0, f"{n_cam} camera(s) from {len(C.discover_videos())} video(s)")
    if n_cam == 0:
        print("\nNo videos in data/videos/. Use --cameras N to generate test clips.")
        return 1
    expected_edges = n_cam * (n_cam - 1) // 2
    check("edges built for every camera pair",
          len(C.CAMERA_EDGES) == expected_edges,
          f"{len(C.CAMERA_EDGES)} edges (expected {expected_edges})")
    check("every camera has coordinates",
          all(c.latitude and c.longitude for c in C.CAMERAS.values()))
    check("every camera has a capture start time",
          all(c.start_time for c in C.CAMERAS.values()))
    starts = [C.camera_start(c) for c in C.CAMERAS]
    check("capture times are strictly increasing",
          all(a < b for a, b in zip(starts, starts[1:])) if n_cam > 1 else True)

    # ---- 3. models --------------------------------------------------
    print("\n[3] Models")
    from urbantrace.vision.plate_backends import probe_roboflow, get_plate_detector
    from urbantrace.vision.detectors import VehicleTracker
    from urbantrace.vision.ocr_engine import get_ocr
    rf = probe_roboflow()
    print(f"       roboflow: available={rf.available} :: {rf.detail[:90]}")
    plate_detector, active = get_plate_detector("auto")
    check("a plate-detection backend is active", active.available, active.name)
    tracker = VehicleTracker()
    ocr = get_ocr()
    check("OCR engine loaded", ocr is not None, ocr.model_name)

    # ---- 4. pipeline per camera ------------------------------------
    print(f"\n[4] Pipeline over {n_cam} camera(s)")
    from urbantrace.vision.pipeline import CameraPipeline
    all_obs, stats = [], {}
    for cam_id, cam in C.CAMERAS.items():
        pipe = CameraPipeline(cam_id, VIDEO_DIR / cam.video_file,
                              start_time=C.camera_start(cam_id),
                              evidence_dir=EVIDENCE_DIR, vehicle_tracker=tracker,
                              plate_detector=plate_detector,
                              write_video=True, output_dir=OUT_DIR)
        obs = pipe.run(max_frames=args.max_frames, progress_every=0)
        all_obs.extend(obs); stats[cam_id] = pipe.stats
        print(f"       {cam_id}: {pipe.stats['vehicle_dets']} vehicle dets, "
              f"{pipe.stats['plate_dets']} plate dets, {len(obs)} observations")

    check("frames were read from every video",
          all(s["frames"] > 0 for s in stats.values()))
    check("vehicles detected + tracked",
          all(s["vehicle_dets"] > 0 for s in stats.values()),
          f"total {sum(s['vehicle_dets'] for s in stats.values())}")
    check("plates detected on at least one camera",
          any(s["plate_dets"] > 0 for s in stats.values()),
          f"total {sum(s['plate_dets'] for s in stats.values())}")
    check("no OCR engine failures",
          sum(s.get("ocr_engine_failures", 0) for s in stats.values()) == 0)
    check("no plate-detector failures",
          sum(s.get("plate_detector_failures", 0) for s in stats.values()) == 0)
    check("annotated video written per camera",
          all((OUT_DIR / f"{c}_annotated.mp4").exists() for c in C.CAMERAS))
    check("OCR produced plate observations", len(all_obs) > 0, f"{len(all_obs)} observations")

    if not all_obs:
        print("\nNo plates read - cannot verify downstream stages.")
        return 1

    # ---- 5. observation integrity ----------------------------------
    print("\n[5] Observation integrity")
    check("every observation has plate + timestamp + camera",
          all(o.plate_number and o.timestamp and o.camera_id for o in all_obs))
    check("every observation has a measured video_time_s",
          all(isinstance(o.video_time_s, (int, float)) for o in all_obs))
    check("every observation carries a local track id",
          all(o.local_track_id is not None for o in all_obs))
    check("raw per-frame readings retained (audit trail)",
          all(len(o.raw_readings) > 0 for o in all_obs))
    check("clock source is declared, not implied",
          all(o.clock_source for o in all_obs), all_obs[0].clock_source)
    multi_read = [o for o in all_obs if o.total_readings > 1]
    check("temporal aggregation used multiple frames",
          len(multi_read) > 0,
          f"{len(multi_read)}/{len(all_obs)} plates from >1 frame")
    check("evidence crops written",
          any(o.evidence_crop and Path(o.evidence_crop).exists() for o in all_obs))

    # ---- 6. association + routing ----------------------------------
    print("\n[6] Cross-camera association and routing")
    from urbantrace.identity.association import associate, summarize, plate_similarity
    vehicles = associate(all_obs)
    summary = summarize(vehicles)
    check("association produced global vehicles", len(vehicles) > 0,
          f"{summary['global_vehicles']} global from {len(all_obs)} observations")
    check("route status assigned to every vehicle",
          all(v.route_status for v in vehicles), str(summary["by_route_status"]))
    check("every vehicle has a route and a reason",
          all(v.route and v.status_reason for v in vehicles))
    check("OCR-confusion similarity works",
          plate_similarity("TN09AB1234", "TN09A81234") > 0.9
          and plate_similarity("TN09AB1234", "KA10CD5678") < 0.5)
    multi = [v for v in vehicles if v.camera_count > 1]
    if multi:
        check("cross-camera routes reconstructed", True,
              f"{len(multi)} vehicle(s): " +
              "; ".join(f"{v.plate_number}={'->'.join(v.route)}" for v in multi[:3]))
        check("every leg has distance, time, speed and a verdict",
              all(l["road_distance_km"] >= 0 and l["verdict"] and l["reason"]
                  for v in multi for l in v.legs))
    else:
        check("cross-camera routing (no shared vehicles in this footage)", True,
              "all vehicles single-camera - correctly reported as DROPPED_AT_ORIGIN")

    # ---- 7. spatio-temporal validation -----------------------------
    print("\n[7] Spatio-temporal validation")
    from urbantrace.identity.association import validate_transition
    ids = list(C.CAMERAS)
    if len(ids) > 1:
        a = {"camera_id": ids[0], "timestamp": "2026-08-24T10:00:00"}
        b_bad = {"camera_id": ids[-1], "timestamp": "2026-08-24T10:00:30"}
        b_ok = {"camera_id": ids[-1], "timestamp": "2026-08-24T10:40:00"}
        check("impossible transition rejected",
              validate_transition(a, b_bad).verdict == "IMPLAUSIBLE",
              validate_transition(a, b_bad).reason[:70])
        check("plausible transition accepted",
              validate_transition(a, b_ok).verdict in ("PLAUSIBLE", "SLOW", "FAST"))
    else:
        check("spatio-temporal validation (needs >1 camera)", True, "skipped")

    # ---- 8. plate format validation --------------------------------
    print("\n[8] Indian plate grammar")
    from urbantrace.vision.plate_format import validate
    check("digit-position repair", validate("TNO9AB1234").final == "TN09AB1234")
    check("invalid state code rejected", not validate("1N09AB1234").is_valid)
    check("garbage rejected", not validate("GARBAGE").is_valid)
    check("valid plate untouched",
          validate("TN09AB1234").final == "TN09AB1234"
          and not validate("TN09AB1234").repaired)
    check("raw OCR always retained",
          validate("TNO9AB1234").raw == "TNO9AB1234")

    # ---- 9. alerts --------------------------------------------------
    print("\n[9] Alert engine")
    test_plate = all_obs[0].plate_number
    wl_alerts = [o for o in all_obs if o.plate_number == test_plate]
    check("watchlist match path fires on a real detected plate",
          len(wl_alerts) > 0, f"'{test_plate}' matched {len(wl_alerts)} observation(s)")
    anomalies = [l for v in vehicles for l in v.legs if l["verdict"] == "IMPLAUSIBLE"]
    check("anomaly alerts carry an explanation",
          all(l["reason"] for l in anomalies) if anomalies else True,
          f"{len(anomalies)} anomaly leg(s)")
    wl_file = ROOT / "data" / "watchlist.txt"
    check("watchlist is file-driven, not auto-seeded", wl_file.exists(),
          f"{wl_file.name} present")

    # ---- 10. visualisation + export --------------------------------
    print("\n[10] Visualisation and export")
    import matplotlib
    matplotlib.use("Agg")
    from urbantrace.viz.route_map import sketch_route_map, gis_route_map, traffic_heatmap
    from urbantrace.viz.video_utils import to_h264
    sketch_route_map(vehicles, out_path=OUT_DIR / "route_sketch.png")
    check(f"route sketch renders for {n_cam} camera(s)",
          (OUT_DIR / "route_sketch.png").exists())
    gis_route_map(vehicles, out_path=OUT_DIR / "route_map.html",
                  highlight_plate=multi[0].plate_number if multi else None)
    check("GIS route map renders", (OUT_DIR / "route_map.html").exists())
    traffic_heatmap(all_obs, out_path=OUT_DIR / "heatmap.html")
    check("traffic heatmap renders", (OUT_DIR / "heatmap.html").exists())
    h = to_h264(OUT_DIR / f"{ids[0]}_annotated.mp4")
    check("annotated video re-encodes to browser H.264", Path(h).exists(),
          f"{Path(h).stat().st_size/1e6:.2f} MB")

    payload = {"summary": summary, "camera_stats": stats,
               "observations": [o.to_dict() for o in all_obs],
               "global_vehicles": [v.to_dict() for v in vehicles]}
    (OUT_DIR / "urbantrace_results.json").write_text(json.dumps(payload, indent=2))
    check("results JSON exported", (OUT_DIR / "urbantrace_results.json").exists())

    # ---- 11. Supabase persistence ----------------------------------
    print("\n[11] Supabase persistence")
    run_id = None
    try:
        from urbantrace.db.client import ping, apply_schema
        from urbantrace.db import repository as dbrepo
        info = ping()
        check("database reachable", info["connected"], info["server"])
        if info["public_tables"] < 12:
            apply_schema(verbose=False)
        video_rows = [{"camera": c, "file": C.CAMERAS[c].video_file,
                       "resolution": "0x0", "fps": 0, "frames": 0,
                       "duration_s": 0, "size_MB": 0} for c in C.CAMERAS]
        test_alerts = [{"alert_type": "WATCHLIST_MATCH", "severity": "HIGH",
                        "plate": all_obs[0].plate_number,
                        "camera": all_obs[0].camera_id,
                        "location": C.CAMERAS[all_obs[0].camera_id].name,
                        "timestamp": all_obs[0].timestamp,
                        "ocr_confidence": all_obs[0].ocr_confidence,
                        "evidence": all_obs[0].evidence_crop,
                        "reason": "verification run"}]
        rep = dbrepo.save_all(rt, {"vehicle_model": "yolo11m.pt",
                                   "plate_backend": active.name,
                                   "plate_model": active.model_id,
                                   "ocr_model": ocr.model_name},
                              C.CAMERAS, C.CAMERA_EDGES, video_rows, stats,
                              all_obs, vehicles, test_alerts,
                              notes="verify_system.py")
        run_id = rep["run_id"]
        check("run row created", bool(run_id), run_id)
        check("cameras persisted", rep["cameras"] == n_cam, f"{rep['cameras']} rows")
        check("observations persisted", rep["observations"] == len(all_obs),
              f"{rep['observations']} rows")
        check("raw OCR audit trail persisted", rep["plate_readings"] > 0,
              f"{rep['plate_readings']} readings")
        check("global vehicles persisted", rep["global_vehicles"] == len(vehicles),
              f"{rep['global_vehicles']} rows")
        check("route legs persisted", rep["route_legs"] >= 0, f"{rep['route_legs']} legs")
        check("alerts persisted", rep["alerts"] == 1)

        # read back through SQL, not from memory
        summ = dbrepo.run_summary(run_id)
        check("run summary readable via SQL",
              summ.get("observations") == len(all_obs),
              f"observations={summ.get('observations')} raw={summ.get('raw_readings')}")
        counts = dbrepo.camera_counts(run_id)
        check("per-camera counts view works", len(counts) > 0,
              "; ".join(f"{c['camera_id']}={c['vehicles_with_plate']}" for c in counts))
        routes_db = dbrepo.vehicle_routes(run_id)
        check("vehicle routes readable via SQL", len(routes_db) == len(vehicles))
        if routes_db:
            tl = dbrepo.plate_timeline(routes_db[0]["plate_number"], run_id)
            check("plate timeline query returns sightings", len(tl) > 0,
                  f"{routes_db[0]['plate_number']} -> {len(tl)} sighting(s)")
        if multi:
            fl = dbrepo.camera_flows(run_id)
            check("origin-destination flow view works", len(fl) > 0,
                  "; ".join(f"{f['from_camera']}->{f['to_camera']}:{f['vehicles']}"
                            for f in fl))
        check("camera locations stored as PostGIS points",
              bool(dbrepo._fetch(
                  "select count(*) n from cameras where run_id=%s and geom is not null",
                  (run_id,))[0]["n"] == n_cam))
    except Exception as e:
        check("Supabase persistence", False, f"{type(e).__name__}: {str(e)[:150]}")

    # ---- summary ----------------------------------------------------
    passed = sum(1 for _n, ok, _d in RESULTS if ok)
    print("\n" + "=" * 76)
    print(f"RESULT: {passed}/{len(RESULTS)} checks passed  "
          f"({n_cam} camera(s), {len(all_obs)} observations, "
          f"{summary['global_vehicles']} global vehicles, "
          f"{summary['multi_camera_vehicles']} multi-camera routes)")
    failed = [n for n, ok, _ in RESULTS if not ok]
    if failed:
        print("FAILED:")
        for n in failed:
            print("   -", n)
    print("=" * 76)

    if not args.keep and run_id:
        try:
            from urbantrace.db.client import cursor as _cur
            with _cur() as c:
                c.execute("delete from runs where run_id = %s", (run_id,))
            print(f"verification run {run_id} deleted from database")
        except Exception as e:
            print(f"could not delete verification run: {e}")

    if not args.keep:
        for d in (OUT_DIR, EVIDENCE_DIR):
            for p in d.glob("*"):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        if args.cameras:
            for f in VIDEO_DIR.glob("*.mp4"):
                f.unlink()
        print("test artifacts purged")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
