"""Count the vehicles in a video - and produce the evidence to check the count.

    python scripts/count_vehicles.py --all
    python scripts/count_vehicles.py data/videos/camera1.mp4 --sheet --video
    python scripts/count_vehicles.py --all --mode presence

A bare number is unverifiable, so this writes three things per video:

  <stem>.counts.json   every deduped vehicle, its class, its frame span, and -
                       for the ones NOT counted - the gate that rejected it.
  <stem>.sheet.jpg     a contact sheet: ONE crop per counted vehicle, numbered.
                       Duplicates and false positives are obvious on sight, so
                       the count can be audited in about a minute.
  <stem>.counted.mp4   the video with boxes, the counting line, and the running
                       total, so each increment can be watched as it happens.

Tracks come from the cache written by scripts/cache_tracks.py when present, so
re-counting with different thresholds does not re-run the models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from urbantrace.core.cameras import discover_videos  # noqa: E402
from urbantrace.vision.counting import (  # noqa: E402
    COUNT_CLASSES, CountConfig, VehicleCounter)

CACHE_DIR = ROOT / "data" / "cache" / "tracks"
OUT_DIR = ROOT / "data" / "output" / "counts"

COLORS = {"car": (229, 135, 57), "motorcycle": (38, 89, 217),
          "bus": (112, 158, 25), "truck": (25, 178, 250),
          "bicycle": (200, 200, 200), "unknown": (150, 150, 150)}


class _T:
    """Minimal track record matching what VehicleCounter expects."""
    __slots__ = ("track_id", "cls_name", "confidence", "box")

    def __init__(self, tid, cls, conf, box):
        self.track_id, self.cls_name, self.confidence, self.box = tid, cls, conf, box


def load_tracks(video: Path, cache_dir: Path):
    """(meta, frames) from cache, or by running the models if not cached."""
    cached = cache_dir / f"{video.stem}.tracks.json"
    if cached.exists():
        d = json.loads(cached.read_text())
        return d["meta"], d["frames"]
    print(f"  no cache for {video.name}; running detector (slow)")
    from cache_tracks import cache_video
    from urbantrace.vision.detectors import VehicleTracker
    cache_video(video, VehicleTracker(), cache_dir)
    d = json.loads(cached.read_text())
    return d["meta"], d["frames"]


def count(video: Path, cfg: CountConfig, cache_dir: Path):
    meta, frames = load_tracks(video, cache_dir)
    counter = VehicleCounter(meta["width"], meta["height"], cfg)
    for i, dets in enumerate(frames):
        counter.update([_T(d[0], d[1], d[2], tuple(d[3])) for d in dets], i)
    return meta, frames, counter


# ---------------- evidence ----------------

def contact_sheet(video: Path, vehicles, out: Path, cols: int = 8,
                  cell: int = 168) -> Path | None:
    """One crop per counted vehicle, numbered - the count made checkable."""
    if not vehicles:
        return None
    wanted = {}
    for v in vehicles:
        wanted.setdefault(v.best_frame, []).append(v)

    crops = {}
    cap = cv2.VideoCapture(str(video))
    fno = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for v in wanted.get(fno, []):
            x1, y1, x2, y2 = [int(b) for b in v.best_box]
            pad = 6
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)
            if x2 > x1 and y2 > y1:
                crops[v.vehicle_id] = frame[y1:y2, x1:x2].copy()
        fno += 1
    cap.release()

    rows = (len(vehicles) + cols - 1) // cols
    label_h = 22
    sheet = np.full((rows * (cell + label_h) + 40, cols * cell, 3), 24, np.uint8)
    cv2.putText(sheet, f"{video.name}  -  {len(vehicles)} vehicles counted",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2,
                cv2.LINE_AA)

    for i, v in enumerate(vehicles):
        r, c = divmod(i, cols)
        y0 = 40 + r * (cell + label_h)
        x0 = c * cell
        crop = crops.get(v.vehicle_id)
        if crop is not None and crop.size:
            h, w = crop.shape[:2]
            s = min((cell - 8) / w, (cell - 8) / h)
            rs = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
            yh, xw = rs.shape[:2]
            oy, ox = y0 + (cell - yh) // 2, x0 + (cell - xw) // 2
            sheet[oy:oy + yh, ox:ox + xw] = rs
        col = COLORS.get(v.vehicle_type, (200, 200, 200))
        cv2.rectangle(sheet, (x0 + 2, y0 + 2), (x0 + cell - 2, y0 + cell - 2),
                      col, 1)
        cv2.putText(sheet, f"{i + 1}. {v.vehicle_type}", (x0 + 6, y0 + cell + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    return out


def annotate_video(video: Path, meta, frames, counter, vehicles, mode: str,
                   out: Path) -> Path:
    """Boxes + counting line + a running total that increments on screen."""
    counted = {v.vehicle_id: v for v in vehicles}
    # A vehicle joins the on-screen total at the frame it actually became
    # countable: for TRAFFIC that is where it crossed the line or had travelled
    # far enough; for PRESENCE it is simply where it was first seen.
    joins_at = {v.vehicle_id: (v.counted_at_frame if mode == "traffic"
                               and v.counted_at_frame is not None
                               else v.first_frame)
                for v in vehicles}
    cap = cv2.VideoCapture(str(video))
    w, h = meta["width"], meta["height"]
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             meta["fps"], (w, h))

    running = set()
    fno = 0
    while True:
        ok, frame = cap.read()
        if not ok or fno >= len(frames):
            break
        for d in frames[fno]:
            cid = counter.canonical_id(d[0])
            v = counted.get(cid)
            x1, y1, x2, y2 = [int(b) for b in d[3]]
            if v is None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                continue
            if fno >= joins_at[cid]:
                running.add(cid)
            col = COLORS.get(v.vehicle_type, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{v.vehicle_id} {v.vehicle_type}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

        if counter.line is not None:
            a, b = counter.line
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     (0, 255, 255), 2)

        seen = [counted[c] for c in running]
        rows = [(c, sum(1 for v in seen if v.vehicle_type == c))
                for c in COUNT_CLASSES]
        rows = [r for r in rows if r[1]] + [("total", len(seen))]
        pad, lh, bw = 10, 24, 210
        bh = pad * 2 + lh * (len(rows) + 1)
        ov = frame.copy()
        cv2.rectangle(ov, (10, 10), (10 + bw, 10 + bh), (18, 18, 18), -1)
        cv2.addWeighted(ov, 0.68, frame, 0.32, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + bw, 10 + bh), (70, 70, 70), 1)
        y = 10 + pad + 16
        cv2.putText(frame, f"VEHICLES ({mode.upper()})", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        y += lh - 4
        for name, val in rows:
            col = COLORS.get(name, (255, 255, 255))
            if name == "total":
                cv2.line(frame, (20, y - 15), (bw, y - 15), (70, 70, 70), 1)
            cv2.putText(frame, f"{name}:", (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.56, col, 2, cv2.LINE_AA)
            cv2.putText(frame, str(val), (bw - 22 - 12 * len(str(val)), y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, col, 2, cv2.LINE_AA)
            y += lh
        writer.write(frame)
        fno += 1
    cap.release()
    writer.release()
    return out


def sensitivity(video: Path, cfg: CountConfig, cache_dir: Path) -> dict:
    """Re-count while varying how aggressively fragments are stitched.

    This is the duplicate test. If one vehicle is still being counted twice,
    stitching harder will find the join and the total will fall. A total that
    holds steady across a wide range of stitching settings has converged: there
    is nothing left for more stitching to merge.
    """
    meta, frames = load_tracks(video, cache_dir)
    print(f"  {'gap':>5} {'dist':>6} | {'stitched':>8} {'traffic':>8} {'presence':>9}")
    traffic, presence = [], []
    for gap in (0, 10, 30, 60, 120):
        for dist in (0.055, 0.09, 0.15):
            c = VehicleCounter(meta["width"], meta["height"],
                               CountConfig(**{**cfg.to_dict(),
                                              "stitch_max_gap_frames": gap,
                                              "stitch_max_dist_frac": dist}))
            for i, dets in enumerate(frames):
                c.update([_T(d[0], d[1], d[2], tuple(d[3])) for d in dets], i)
            r = c.report()
            traffic.append(r["traffic"]["total"])
            presence.append(r["presence"]["total"])
            mark = "  <- default" if (gap, dist) == (
                cfg.stitch_max_gap_frames, cfg.stitch_max_dist_frac) else ""
            print(f"  {gap:>5} {dist:>6} | {r['fragments_stitched']:>8} "
                  f"{r['traffic']['total']:>8} {r['presence']['total']:>9}{mark}")

    def spread(xs):
        return (max(xs) - min(xs)) / max(1, max(xs))

    out = {"traffic_range": [min(traffic), max(traffic)],
           "presence_range": [min(presence), max(presence)],
           "traffic_spread": round(spread(traffic), 3),
           "presence_spread": round(spread(presence), 3)}
    for mode in ("traffic", "presence"):
        lo, hi = out[f"{mode}_range"]
        verdict = ("CONVERGED - no duplicates left for stitching to merge"
                   if out[f"{mode}_spread"] <= 0.15 else
                   "NOT converged - the total still depends on stitching, so "
                   "duplicates (or over-merges) remain")
        print(f"  {mode.upper():9} {lo}-{hi}  {verdict}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mode", choices=("traffic", "presence"), default="traffic")
    ap.add_argument("--sheet", action="store_true", help="write the contact sheet")
    ap.add_argument("--video", action="store_true", help="write annotated video")
    ap.add_argument("--sensitivity", action="store_true",
                    help="re-count across stitching settings to test for "
                         "leftover duplicate counting")
    ap.add_argument("--min-frames", type=int, default=CountConfig.min_frames)
    ap.add_argument("--min-area-frac", type=float,
                    default=CountConfig.min_median_area_frac)
    ap.add_argument("--min-disp-frac", type=float,
                    default=CountConfig.min_displacement_frac)
    ap.add_argument("--line", type=str, default=None,
                    help="counting line x1,y1,x2,y2 in 0..1 normalised coords")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    videos = [Path(v) for v in args.videos]
    if args.all or not videos:
        videos = discover_videos()
    if not videos:
        print("No videos found in data/videos/")
        return 1

    line = None
    if args.line:
        p = [float(x) for x in args.line.split(",")]
        line = ((p[0], p[1]), (p[2], p[3]))

    cfg = CountConfig(min_frames=args.min_frames,
                      min_median_area_frac=args.min_area_frac,
                      min_displacement_frac=args.min_disp_frac,
                      line=line)

    for video in videos:
        print(f"\n=== {video.name}")
        meta, frames, counter = count(video, cfg, args.cache_dir)
        rep = counter.report()
        vehicles = [v for v in counter.vehicles()
                    if getattr(v, f"counted_{args.mode}")]

        print(f"  raw tracker ids   : {rep['raw_tracker_ids']}")
        print(f"  fragments stitched: {rep['fragments_stitched']}")
        print(f"  rejected as clutter/flicker: {rep['rejected']}")
        print(f"  TRAFFIC  count: {rep['traffic']['total']}  "
              + "  ".join(f"{c}={rep['traffic'][c]}" for c in COUNT_CLASSES
                          if rep['traffic'][c]))
        print(f"  PRESENCE count: {rep['presence']['total']}  "
              + "  ".join(f"{c}={rep['presence'][c]}" for c in COUNT_CLASSES
                          if rep['presence'][c]))

        args.out_dir.mkdir(parents=True, exist_ok=True)
        jpath = args.out_dir / f"{video.stem}.counts.json"
        jpath.write_text(json.dumps({
            "video": video.name, "meta": meta, "report": rep, "mode": args.mode,
            "vehicles": [v.to_dict() for v in counter.vehicles()],
        }, indent=2))
        print(f"  -> {jpath}")

        if args.sensitivity:
            print("  -- duplicate test: does the total depend on stitching?")
            rep["sensitivity"] = sensitivity(video, cfg, args.cache_dir)
            jpath.write_text(json.dumps({
                "video": video.name, "meta": meta, "report": rep,
                "mode": args.mode,
                "vehicles": [v.to_dict() for v in counter.vehicles()],
            }, indent=2))

        if args.sheet:
            s = contact_sheet(video, vehicles,
                              args.out_dir / f"{video.stem}.sheet.jpg")
            print(f"  -> {s}")
        if args.video:
            p = annotate_video(video, meta, frames, counter, vehicles,
                               args.mode, args.out_dir / f"{video.stem}.counted.mp4")
            print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
