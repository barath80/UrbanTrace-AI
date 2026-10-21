"""End-to-end capacity test: N simulated cameras through capture -> GPU -> identity.

    python scripts/bench_fleet.py --cameras 50 --seconds 60

Each simulated camera is a looping read of a real clip from data/videos, so the
whole path is exercised: decode, motion gate, downscale, batched detection,
tracking, OCR and identity. Nothing is written to the database. The result says
how many cameras this machine sustains, which is the number to size workers from.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--fps", type=float, help="analysis fps per camera (UT_TARGET_FPS)")
    a = ap.parse_args()
    os.environ["UT_LOOP_FILES"] = "1"
    os.environ["UT_ADAPTIVE_FPS"] = "0"           # measure raw capacity, not the governor
    if a.fps:
        os.environ["UT_TARGET_FPS"] = str(a.fps)

    import torch

    from urbantrace.scale.capture import CaptureManager
    from urbantrace.scale.config import SCALE
    from urbantrace.scale.registry import REGISTRY
    from urbantrace.scale.service import IngestService

    clips = sorted(p for p in (ROOT / "data" / "videos").iterdir()
                   if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
    if not clips:
        print("No clips in data/videos.")
        return 1

    # Cameras spread over a grid so the registry and graph are realistic in size.
    REGISTRY.replace([{"camera_id": f"B{i:03d}", "name": f"bench {i}",
                       "latitude": 13.0 + (i // 10) * 0.01,
                       "longitude": 80.0 + (i % 10) * 0.01} for i in range(a.cameras)])
    svc = IngestService(persist=False)
    svc.start()
    cap = CaptureManager(svc.submit_frame, lambda: SCALE.target_fps_per_camera)
    cap.reconcile({f"B{i:03d}": str(clips[i % len(clips)]) for i in range(a.cameras)})

    print(f"{a.cameras} cameras, {SCALE.target_fps_per_camera:g} fps each "
          f"= {a.cameras * SCALE.target_fps_per_camera:g} fps demanded")
    time.sleep(a.warmup)
    s0, t0 = svc.pool.models.frames_processed, time.time()
    sub0, drop0 = svc.bus.submitted, svc.bus.dropped
    torch.cuda.reset_peak_memory_stats()
    time.sleep(a.seconds)
    dt = time.time() - t0
    done = svc.pool.models.frames_processed - s0
    sub = svc.bus.submitted - sub0
    dropped = svc.bus.dropped - drop0
    cs = cap.summary()
    print(f"analysed          : {done / dt:6.1f} frames/s")
    print(f"submitted/dropped : {sub / dt:6.1f} /s  /  {dropped / dt:5.1f} /s  "
          f"({100 * dropped / max(1, sub):.1f}% dropped)")
    print(f"per camera        : {done / dt / a.cameras:5.2f} fps actually analysed")
    print(f"feeds live        : {cs['by_state'].get('live', 0)}/{a.cameras}  "
          f"static-skipped {cs['frames_skipped_static']}")
    print(f"sightings         : {svc.sightings_seen}")
    print(f"VRAM peak         : {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated")
    cap.stop()
    svc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
