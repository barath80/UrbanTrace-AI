"""Dump per-frame vehicle tracks for a video to a JSON cache.

Detection+tracking is the slow part of counting; the counting logic on top of
it is milliseconds. Caching the tracks lets the counter be tuned and audited
against a fixed input instead of re-running the models on every change.

    python scripts/cache_tracks.py data/videos/camera1.mp4
    python scripts/cache_tracks.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import cv2  # noqa: E402

from urbantrace.core.cameras import discover_videos  # noqa: E402
from urbantrace.vision.detectors import VehicleTracker  # noqa: E402

CACHE_DIR = ROOT / "data" / "cache" / "tracks"


def cache_video(video: Path, tracker: VehicleTracker, out_dir: Path) -> Path:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video}")
    meta = {
        "video": video.name,
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    tracker.reset()
    frames = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracks = tracker.update(frame)
        frames.append([[t.track_id, t.cls_name, t.confidence,
                        [round(v, 1) for v in t.box]] for t in tracks])
        n += 1
        if n % 100 == 0:
            print(f"  {video.name}: {n}/{meta['frame_count']}")
    cap.release()
    meta["frames_processed"] = n

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{video.stem}.tracks.json"
    out.write_text(json.dumps({"meta": meta, "frames": frames}))
    ids = {d[0] for f in frames for d in f}
    print(f"  {video.name}: {n} frames, {len(ids)} raw tracker ids -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="every video in data/videos")
    ap.add_argument("--out-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()

    videos = list(args.videos)
    if args.all or not videos:
        videos = discover_videos()
    if not videos:
        print("No videos found in data/videos/")
        return 1

    tracker = VehicleTracker()
    for v in videos:
        cache_video(Path(v), tracker, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
