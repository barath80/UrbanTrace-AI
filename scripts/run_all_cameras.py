"""Process CAMERA_1/2/3 end to end and reconstruct city-wide vehicle routes.

    python scripts/run_all_cameras.py                 # full run
    python scripts/run_all_cameras.py --max-frames 300
    python scripts/run_all_cameras.py --no-video      # skip annotated output

Camera start times are STAGGERED to reflect the real corridor travel times
(Koyambedu -> Ambattur -> Avadi). They are a declared assumption, not measured
wall-clock data: the videos carry no embedded real-world timestamp. Every
observation keeps `video_time_s`, which IS measured, alongside the derived clock.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from urbantrace.core.cameras import CAMERAS, network_summary  # noqa: E402
from urbantrace.core.runtime import describe                   # noqa: E402
from urbantrace.identity.association import associate, summarize  # noqa: E402
from urbantrace.vision.detectors import PlateDetector, VehicleTracker  # noqa: E402
from urbantrace.vision.pipeline import CameraPipeline  # noqa: E402

VIDEO_DIR = ROOT / "data" / "videos"
OUT_DIR = ROOT / "data" / "output"
EVIDENCE_DIR = ROOT / "data" / "evidence"

# Declared capture start for CAM_01; the others are offset by typical corridor
# travel time so that a vehicle appearing in all three yields plausible legs.
BASE_START = datetime(2026, 8, 24, 10, 0, 0)
CAMERA_START_OFFSETS = {"CAM_01": 0, "CAM_02": 14, "CAM_03": 25}  # minutes


def resolve_videos() -> dict:
    """Map camera_id -> video path, tolerating common filename variants."""
    found = {}
    for cam_id, cam in CAMERAS.items():
        candidates = [
            VIDEO_DIR / cam.video_file,
            VIDEO_DIR / cam.video_file.lower(),
            VIDEO_DIR / f"{cam_id}.mp4",
            VIDEO_DIR / f"camera_{cam_id[-1]}.mp4",
            VIDEO_DIR / f"camera{cam_id[-1]}.mp4",
        ]
        for c in candidates:
            if c.exists():
                found[cam_id] = c
                break
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("UrbanTrace AI - multi-camera ANPR + trajectory reconstruction")
    print("Runtime:", describe())
    print("=" * 78)

    videos = resolve_videos()
    missing = [c for c in CAMERAS if c not in videos]
    if missing:
        print(f"\nMISSING VIDEOS for {', '.join(missing)}")
        print(f"Place the files in {VIDEO_DIR} named:")
        for c in missing:
            print(f"   {CAMERAS[c].video_file}   ->  {CAMERAS[c].name}")
        if not videos:
            return 1
        print("\nContinuing with the cameras that ARE available.\n")

    # Models are loaded once and shared across cameras; tracker state is reset
    # per video so local track IDs restart at 1 for each camera.
    tracker = VehicleTracker()
    plates = PlateDetector()

    all_obs, stats = [], {}
    for cam_id, path in sorted(videos.items()):
        start = BASE_START + timedelta(minutes=CAMERA_START_OFFSETS.get(cam_id, 0))
        pipe = CameraPipeline(cam_id, path, start_time=start,
                              evidence_dir=EVIDENCE_DIR,
                              vehicle_tracker=tracker, plate_detector=plates,
                              write_video=not args.no_video, output_dir=OUT_DIR)
        obs = pipe.run(max_frames=args.max_frames, frame_stride=args.frame_stride)
        stats[cam_id] = pipe.stats
        all_obs.extend(obs)
        (OUT_DIR / f"{cam_id}_observations.json").write_text(
            json.dumps([o.to_dict() for o in obs], indent=2))
        print(f"[{cam_id}] {len(obs)} observations | {pipe.stats}\n")

    if not all_obs:
        print("No plate observations produced - nothing to associate.")
        return 1

    print("=" * 78)
    print("CROSS-CAMERA ASSOCIATION")
    print("=" * 78)
    vehicles = associate(all_obs, min_confidence=args.min_confidence)
    summary = summarize(vehicles)

    for v in vehicles:
        if v.camera_count < 2:
            continue
        print(f"\n{v.global_vehicle_id}  PLATE {v.plate_number}  "
              f"({v.camera_count} cameras, conf {v.confidence:.3f})")
        if v.merged_variants:
            print(f"   merged OCR variants: {v.merged_variants}")
        for o in v.observations:
            print(f"   {o['timestamp']}  {o['camera_id']}  "
                  f"track#{o['local_track_id']:<4} conf={o['ocr_confidence']:.3f}")
        print(f"   ROUTE: {' -> '.join(v.route)}")
        for l in v.legs:
            print(f"     {l['from_camera']}->{l['to_camera']}: "
                  f"{l['travel_minutes']:.1f} min, {l['implied_speed_kmh']:.0f} km/h "
                  f"[{l['verdict']}] {l['reason']}")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": describe(),
        "network": network_summary(),
        "camera_stats": stats,
        "timestamp_note": ("Wall-clock timestamps are derived as "
                           "configured_camera_start + measured video offset. "
                           "video_time_s is the measured quantity."),
        "summary": summary,
        "observations": [o.to_dict() for o in all_obs],
        "global_vehicles": [v.to_dict() for v in vehicles],
    }
    out = OUT_DIR / "urbantrace_results.json"
    out.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 78)
    print(f"observations       : {len(all_obs)}")
    print(f"global vehicles    : {summary['global_vehicles']}")
    print(f"multi-camera routes: {summary['multi_camera_vehicles']}")
    print(f"results            : {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
