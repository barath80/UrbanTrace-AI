"""Generate UrbanTrace_AI_Realtime.ipynb - the strict real-data notebook.

Design rule: NOTHING in this notebook is pre-baked. Every plate, timestamp,
route, map and metric is computed from the three uploaded videos at run time.
If the videos are absent the notebook STOPS with a clear message rather than
producing placeholder output.

Regenerate with:  python scripts/build_realtime_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "UrbanTrace_AI_Realtime.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {},
                "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}

cells = []

cells.append(md("""
# UrbanTrace AI - Real-Data Pipeline

**Multi-camera ANPR, multi-object tracking, cross-camera vehicle routing and GIS analytics.**

Every value produced below is computed from the uploaded camera videos when the
cells are run. Any number of videos is supported - one camera per video file. There is no sample data, no placeholder plate, no pre-filled result.
If the videos are missing, the notebook stops.

### What is measured vs configured

| | |
|---|---|
| **Measured from video** | vehicle boxes, plate boxes, plate text, OCR confidence, track IDs, frame numbers, `video_time_s`, travel times, routes, flows |
| **Configured by you** | camera geographic coordinates, camera capture start times |

Camera coordinates are configuration - the videos contain no GPS. Wall-clock timestamps
are `configured camera start + measured video offset`; the measured quantity is
`video_time_s` and both are stored on every observation.

### Pipeline

```
video -> vehicle detection -> ByteTrack multi-object tracking -> plate detection
      -> plate crop -> preprocessing variants -> OCR -> cross-variant consensus
      -> temporal aggregation over the track -> ONE timestamped observation
      -> cross-camera association -> route reconstruction -> GIS + analytics + alerts
```
"""))

# ---------------------------------------------------------------- 1
cells.append(md("## 1. Environment"))
cells.append(code("""
import sys, json, subprocess, warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd
from urbantrace.core.runtime import describe

rt = describe()
for k, v in rt.items():
    print(f"{k:12}: {v}")
assert rt["device"] == "cuda" or True, "running on CPU - this will be slow but works"
"""))

# ---------------------------------------------------------------- 2
cells.append(md("""
## 2. Input videos - ANY number of cameras

Drop **as many videos as you like** into `data/videos/` (`.mp4`, `.avi`, `.mov`, `.mkv`, ...).
One camera is created per video, numbered in natural filename order:
`CAMERA_1.mp4 -> CAM_01`, `CAMERA_2.mp4 -> CAM_02`, and so on. Two videos, or nine -
the whole pipeline adapts.

Each file's real properties are read from the file itself. **The cell raises if the
folder is empty** - nothing downstream can be real without footage.
"""))
cells.append(code("""
import cv2
from urbantrace.core.cameras import (CAMERAS, load_network, discover_videos,
                                     camera_start, VIDEO_DIR)

# Rebuild the network from whatever is in data/videos/ right now.
load_network()

found = discover_videos()
print(f"videos discovered : {len(found)}")
print(f"cameras created   : {len(CAMERAS)}\\n")

if not CAMERAS:
    raise FileNotFoundError(
        f"No videos found in {VIDEO_DIR}. Add one or more video files "
        f"(.mp4/.avi/.mov/.mkv) and re-run. This notebook does not fabricate data.")

videos, video_rows = {}, []
for cam_id, cam in CAMERAS.items():
    p = VIDEO_DIR / cam.video_file
    videos[cam_id] = p
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_rows.append({"camera": cam_id, "file": cam.video_file,
                 "location": cam.name, "loc_source": cam.location_source,
                 "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                 "fps": round(fps, 2), "frames": n,
                 "duration_s": round(n / fps, 1) if fps else None,
                 "size_MB": round(p.stat().st_size / 1e6, 1)})
    cap.release()

display(pd.DataFrame(video_rows))
print(f"\\n{len(CAMERAS)} camera(s) ready.")
"""))

# ---------------------------------------------------------------- 3
cells.append(md("""
## 3. Models

Three models. The plate detector has two interchangeable backends and reports which one
is actually active - if the Roboflow workspace can serve inference it is used, otherwise
local GPU weights are used and the reason is printed.
"""))
cells.append(code("""
from urbantrace.vision.plate_backends import probe_roboflow, get_plate_detector
from urbantrace.vision.detectors import VehicleTracker, VEHICLE_WEIGHTS, PLATE_WEIGHTS
from urbantrace.vision.ocr_engine import get_ocr

status = probe_roboflow()
print(f"Roboflow model : {status.model_id}")
print(f"  available    : {status.available}")
print(f"  detail       : {status.detail}\\n")

plate_detector, active = get_plate_detector("auto")
tracker = VehicleTracker()
ocr = get_ocr()

print(f"VEHICLE DETECTOR : {VEHICLE_WEIGHTS.name} (COCO) + ByteTrack")
print(f"PLATE DETECTOR   : {active.name}")
print(f"                   {active.detail}")
print(f"OCR ENGINE       : {ocr.model_name}  variants={ocr.variants}  consensus={ocr.consensus}")
"""))

# ---------------------------------------------------------------- 4
cells.append(md("""
## 4. Camera network - locations and capture times

Each camera needs a real-world location. `loc_source` tells you where it came from:

- **`config`** - you mapped it in `data/cameras.json` (authoritative)
- **`default`** - auto-assigned from a Chennai-corridor fallback pool

**To set your real locations**, run `write_config_template()` below, edit the generated
`data/cameras.json`, and re-run this notebook from cell 2.

Road distances and expected travel-time windows are **derived** from these coordinates,
and they are what the system uses to decide whether a vehicle could plausibly have
travelled between two cameras. Wrong coordinates = wrong plausibility verdicts.
"""))
cells.append(code("""
from urbantrace.core.cameras import CAMERA_EDGES, network_summary, write_config_template

cam_df = pd.DataFrame([c.to_dict() for c in CAMERAS.values()])
display(cam_df[["camera_id", "name", "road", "latitude", "longitude",
                "direction", "location_source", "start_time", "video_file"]])

n_default = int((cam_df.location_source == "default").sum())
if n_default:
    print(f"NOTE: {n_default} camera(s) are using FALLBACK locations, not your real ones.")
    print(f"      Run write_config_template() and edit data/cameras.json to fix.\\n")

print(f"Derived road links ({len(CAMERA_EDGES)} edges between {len(CAMERAS)} cameras):")
display(pd.DataFrame([e.to_dict() for e in CAMERA_EDGES]))

print("\\nConfigured capture start times (videos carry no wall clock):")
for cam_id in CAMERAS:
    print(f"  {cam_id}: {camera_start(cam_id)}")
"""))
cells.append(code("""
# Uncomment to emit data/cameras.json, then edit it with your real locations.
# print("template written to:", write_config_template())
"""))

# ---------------------------------------------------------------- 5
cells.append(md("""
## 5. Process all three videos

Runs the full pipeline once per discovered camera, however many there are.
`MAX_FRAMES = None` processes the entire video.
Every detected vehicle gets a track ID; every plate found on a tracked vehicle is
cropped, preprocessed, read, and accumulated until the track ends.
"""))
cells.append(code("""
from urbantrace.vision.pipeline import CameraPipeline

MAX_FRAMES  = None     # None = the whole video. Set e.g. 400 for a fast first pass.
WRITE_VIDEO = True     # annotated mp4 with vehicle + plate boxes, track IDs, plate text

OUT_DIR      = ROOT / "data" / "output";   OUT_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR = ROOT / "data" / "evidence"; EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

all_observations, camera_stats = [], {}

for cam_id, path in videos.items():
    pipe = CameraPipeline(
        cam_id, path, start_time=camera_start(cam_id),
        evidence_dir=EVIDENCE_DIR, vehicle_tracker=tracker,
        plate_detector=plate_detector, write_video=WRITE_VIDEO, output_dir=OUT_DIR)
    obs = pipe.run(max_frames=MAX_FRAMES, progress_every=100)
    all_observations.extend(obs)
    camera_stats[cam_id] = pipe.stats
    print(f"[{cam_id}] {len(obs)} plate observations\\n")

print("=" * 70)
print("TOTAL PLATE OBSERVATIONS:", len(all_observations))
if not all_observations:
    print("\\nNo plates were read. Possible causes: plates too small/blurry to resolve,")
    print("no vehicles with visible plates, or the footage is not vehicle traffic.")
"""))

cells.append(md("### 5a. Processing statistics (measured)"))
cells.append(code("""
st = pd.DataFrame(camera_stats).T
st["plates_per_frame"] = (st["plate_dets"] / st["frames"]).round(3)
st["ocr_success_rate"] = (st["ocr_success"] / st["ocr_calls"].replace(0, 1)).round(3)
display(st)
"""))

# ---------------------------------------------------------------- 6
cells.append(md("""
### 5b. Pipeline health check

An empty result table can mean two very different things: *there were no plates in the
footage*, or *a component was failing on every call*. This cell distinguishes them.
"""))
cells.append(code("""
problems = []
for cam_id, s in camera_stats.items():
    if s.get("ocr_engine_failures"):
        problems.append(f"{cam_id}: OCR engine failed {s['ocr_engine_failures']}x "
                        f"-> {s.get('ocr_last_error')}")
    if s.get("plate_detector_failures"):
        problems.append(f"{cam_id}: plate detector failed {s['plate_detector_failures']}x "
                        f"-> {s.get('plate_detector_last_error')}")
    if s.get("preprocess_variant_failures"):
        problems.append(f"{cam_id}: preprocessing variants failed "
                        f"{s['preprocess_variant_failures']}")
    if s["frames"] == 0:
        problems.append(f"{cam_id}: no frames were read from the video file")
    elif s["vehicle_dets"] == 0:
        problems.append(f"{cam_id}: no vehicles detected in any frame "
                        f"(is this vehicle traffic footage?)")
    elif s["plate_dets"] == 0:
        problems.append(f"{cam_id}: vehicles detected but NO plates found "
                        f"(plates may be too small/angled - try a higher-resolution "
                        f"source or a closer camera angle)")
    elif s["ocr_calls"] and s["ocr_success"] / s["ocr_calls"] < 0.5:
        problems.append(f"{cam_id}: OCR succeeded on only "
                        f"{s['ocr_success']}/{s['ocr_calls']} plate crops")

if problems:
    print("ISSUES DETECTED:")
    for p in problems:
        print("  -", p)
else:
    print("All components healthy: frames read, vehicles detected, plates found, OCR ran.")

print(f"\\ncomponent failure counts (0 = no failures):")
print(f"  OCR engine      : {sum(s.get('ocr_engine_failures',0) for s in camera_stats.values())}")
print(f"  plate detector  : {sum(s.get('plate_detector_failures',0) for s in camera_stats.values())}")
"""))

cells.append(md("""
## 6. Timestamped plate observations

**The fundamental output: PLATE NUMBER + TIMESTAMP + CAMERA.**
One row per vehicle per camera, with the local track ID it came from.
"""))
cells.append(code("""
obs_df = pd.DataFrame([o.to_dict() for o in all_observations])
if len(obs_df):
    sheet = obs_df[["observation_id", "camera_id", "local_track_id", "plate_number",
                    "timestamp", "video_time_s", "frame_number", "vehicle_type",
                    "ocr_confidence", "plate_detection_confidence",
                    "support", "total_readings", "agreement", "is_valid_format"]]
    sheet = sheet.sort_values(["camera_id", "video_time_s"]).reset_index(drop=True)
    display(sheet)
    sheet.to_csv(OUT_DIR / "observations_sheet.csv", index=False)
    print("saved ->", OUT_DIR / "observations_sheet.csv")
else:
    print("No observations.")
"""))

cells.append(md("### 6a. Per-camera vehicle timeline"))
cells.append(code("""
for cam_id in CAMERAS:
    sub = obs_df[obs_df.camera_id == cam_id] if len(obs_df) else pd.DataFrame()
    print(f"\\n=== {cam_id} - {CAMERAS[cam_id].name} ({len(sub)} vehicles) ===")
    for _, r in sub.sort_values("video_time_s").iterrows():
        print(f"  track#{int(r.local_track_id):<4} {r.plate_number:12} "
              f"{r.timestamp}  (video t={r.video_time_s:7.2f}s)  "
              f"conf={r.ocr_confidence:.3f}  {r.vehicle_type}")
"""))

cells.append(md("""
### 6b. Audit trail - the raw per-frame OCR behind one plate

Proof that the final plate is a multi-frame consensus and not one lucky read.
Individual frames disagree; the aggregator resolves them.
"""))
cells.append(code("""
if len(obs_df):
    best = max(all_observations, key=lambda o: o.total_readings)
    print(f"FINAL PLATE : {best.plate_number}")
    print(f"camera      : {best.camera_id}   track #{best.local_track_id}")
    print(f"method      : {best.aggregation_method}")
    print(f"support     : {best.support}/{best.total_readings} frames agreed "
          f"({best.agreement:.0%})\\n")
    display(pd.DataFrame(best.raw_readings)[
        ["frame_number", "timestamp_s", "text_raw", "text_final",
         "ocr_confidence", "plate_confidence", "variant", "repaired"]])
"""))

cells.append(md("""
### 6c. Plate-format validation on the real readings

Corrections applied to actual OCR output from your videos. The validator only makes
changes forced by the plate grammar (a position that must be a digit, read as a
visually identical letter). Raw output is always retained.
"""))
cells.append(code("""
rows = []
for o in all_observations:
    for r in o.raw_readings:
        if r["text_raw"] != r["text_final"]:
            rows.append({"camera": o.camera_id, "track": o.local_track_id,
                         "frame": r["frame_number"], "ocr_raw": r["text_raw"],
                         "after_validation": r["text_final"],
                         "confidence": r["ocr_confidence"], "repaired": r["repaired"]})
if rows:
    display(pd.DataFrame(rows).head(30))
    print(f"{len(rows)} readings were adjusted by grammar-forced validation.")
else:
    print("No grammar-forced corrections were needed on this footage.")
"""))

# ---------------------------------------------------------------- 7
cells.append(md("""
## 7. Annotated output videos

Bounding boxes for every detected vehicle (orange, with ByteTrack ID and recognised
plate) and every detected plate (green). Re-encoded to H.264 so they play inline.
"""))
cells.append(code("""
from urbantrace.viz.video_utils import show_video
from IPython.display import display as _d, HTML as _H

# One player per camera, however many there are.
for cam_id, cam in CAMERAS.items():
    p = OUT_DIR / f"{cam_id}_annotated.mp4"
    if not p.exists():
        print(f"{cam_id}: no annotated video (set WRITE_VIDEO=True and re-run cell 9)")
        continue
    n_veh = len({o.plate_number for o in all_observations if o.camera_id == cam_id})
    _d(_H(f"<h3 style='margin:14px 0 4px'>{cam_id} &mdash; {cam.name} "
          f"<span style='font-weight:400;color:#888'>({n_veh} vehicles with plates read)"
          f"</span></h3>"))
    _d(show_video(p, width=920))
"""))

# ---------------------------------------------------------------- 8
cells.append(md("""
## 8. Cross-camera association

Track IDs are local to each video - track #17 at CAM_01 and track #6 at CAM_02 may be the
same car. Association scores candidate pairs on OCR-confusion-aware plate similarity,
OCR confidence, and travel-time plausibility against the camera graph. Two plates are
only merged if at least one supporting pair forms a physically possible transition.
"""))
cells.append(code("""
from urbantrace.identity.association import associate, summarize

vehicles = associate(all_observations)
summary = summarize(vehicles)

print("global vehicles      :", summary["global_vehicles"])
print("multi-camera vehicles:", summary["multi_camera_vehicles"])
print("single-camera only   :", summary["single_camera_vehicles"])
print("\\nroute status breakdown:")
for k, v in summary["by_route_status"].items():
    print(f"   {k:20} {v}")
print("\\ndropped by camera:", summary["dropped_by_camera"])
"""))
cells.append(code("""
veh_df = pd.DataFrame([{
    "global_id": v.global_vehicle_id, "plate": v.plate_number,
    "route": " -> ".join(v.route), "status": v.route_status,
    "cameras": v.camera_count, "first_seen": v.first_seen, "last_seen": v.last_seen,
    "duration_min": v.total_duration_minutes, "confidence": v.confidence,
    "merged_ocr_variants": v.merged_variants,
} for v in vehicles])
display(veh_df)
veh_df.to_csv(OUT_DIR / "global_vehicles.csv", index=False)
"""))

# ---------------------------------------------------------------- 9
cells.append(md("""
## 9. Route reconstruction per vehicle

For each vehicle: the camera sequence, timestamp at each camera, travel time and implied
speed between them, and an explicit verdict on whether that movement is physically
plausible. Vehicles seen at only one camera are reported as `DROPPED_AT_ORIGIN` -
they did not continue along the corridor.
"""))
cells.append(code("""
multi = [v for v in vehicles if v.camera_count > 1]
dropped = [v for v in vehicles if v.camera_count == 1]

print(f"{len(multi)} vehicle(s) crossed more than one camera")
print(f"{len(dropped)} vehicle(s) were seen at one camera only\\n")

for v in multi:
    print("=" * 78)
    print(f"{v.global_vehicle_id}   PLATE {v.plate_number}   [{v.route_status}]   "
          f"confidence {v.confidence:.3f}")
    if v.merged_variants:
        print(f"  merged OCR variants : {v.merged_variants}")
    print(f"  {v.status_reason}")
    print("  TIMELINE:")
    for o in v.observations:
        print(f"    {o['timestamp']}   {o['camera_id']}  "
              f"({CAMERAS[o['camera_id']].name})  track#{o['local_track_id']}  "
              f"conf={o['ocr_confidence']:.3f}")
    print(f"  ROUTE: {' -> '.join(v.route)}")
    for l in v.legs:
        print(f"    {l['from_camera']} -> {l['to_camera']}: {l['travel_minutes']:.1f} min, "
              f"{l['road_distance_km']:.2f} km, {l['implied_speed_kmh']:.0f} km/h  "
              f"[{l['verdict']}]")
        print(f"      {l['reason']}")

if not multi:
    print("No vehicle was observed at more than one camera.")
    print("Cross-camera routes require the SAME vehicle to be visible in multiple videos.")
"""))

cells.append(md("### 9a. Vehicles that did not continue past their first camera"))
cells.append(code("""
if dropped:
    display(pd.DataFrame([{
        "plate": v.plate_number, "seen_at": v.route[0] if v.route else None,
        "location": CAMERAS[v.route[0]].name if v.route else None,
        "timestamp": v.first_seen, "confidence": v.confidence,
    } for v in dropped]))
else:
    print("Every detected vehicle appeared at more than one camera.")
"""))

# ---------------------------------------------------------------- 10
cells.append(md("## 10. Route sketch map"))
cells.append(code("""
%matplotlib inline
from urbantrace.viz.route_map import sketch_route_map, gis_route_map, traffic_heatmap, flow_counts
sketch_route_map(vehicles, out_path=OUT_DIR / "route_sketch.png")
"""))

cells.append(md("## 11. GIS route map"))
cells.append(code("""
gis_route_map(vehicles, out_path=OUT_DIR / "route_map.html",
              highlight_plate=multi[0].plate_number if multi else None)
"""))

cells.append(md("## 12. Traffic heatmap"))
cells.append(code("""
traffic_heatmap(all_observations, out_path=OUT_DIR / "heatmap.html")
"""))

# ---------------------------------------------------------------- 13
cells.append(md("""
## 12b. Vehicles passing each camera

How many vehicles passed each camera, and how many of them continued onward.
"""))
cells.append(code("""
per_cam = []
for cam_id, cam in CAMERAS.items():
    cam_obs = [o for o in all_observations if o.camera_id == cam_id]
    plates_here = {o.plate_number for o in cam_obs}
    onward = sum(1 for v in vehicles
                 if cam_id in v.route and v.route.index(cam_id) < len(v.route) - 1)
    ended = sum(1 for v in vehicles if v.route and v.route[-1] == cam_id)
    st = camera_stats.get(cam_id, {})
    per_cam.append({
        "camera": cam_id, "location": cam.name,
        "vehicles_tracked": st.get("vehicle_dets", 0),
        "plates_detected": st.get("plate_dets", 0),
        "vehicles_with_plate": len(plates_here),
        "continued_onward": onward,
        "route_ended_here": ended,
        "frames": st.get("frames", 0),
    })
per_cam_df = pd.DataFrame(per_cam)
display(per_cam_df)
per_cam_df.to_csv(OUT_DIR / "per_camera_counts.csv", index=False)

print()
for r in per_cam:
    print(f"{r['camera']} ({r['location']}): {r['vehicles_with_plate']} vehicle(s) "
          f"identified, {r['continued_onward']} continued to another camera, "
          f"{r['route_ended_here']} ended here")
"""))

cells.append(md("## 13. Traffic analytics (computed from the detections above)"))
cells.append(code("""
from urbantrace.core.cameras import get_edge

print("PER-CAMERA VOLUME")
if len(obs_df):
    display(obs_df.groupby("camera_id").agg(
        unique_vehicles=("plate_number", "nunique"),
        observations=("observation_id", "count"),
        mean_ocr_conf=("ocr_confidence", "mean"),
        mean_frames_per_plate=("total_readings", "mean")).round(3))

print("\\nCAMERA-TO-CAMERA FLOW (origin-destination)")
flows = flow_counts(vehicles)
if flows:
    rows = []
    for (a, b), n in flows.items():
        e = get_edge(a, b)
        legs = [l for v in vehicles for l in v.legs
                if l["from_camera"] == a and l["to_camera"] == b
                and l["verdict"] != "IMPLAUSIBLE"]
        rows.append({"from": a, "to": b, "vehicles": n,
                     "distance_km": e.road_distance_km if e else None,
                     "mean_travel_min": round(sum(l["travel_minutes"] for l in legs)/len(legs), 2) if legs else None,
                     "mean_speed_kmh": round(sum(l["implied_speed_kmh"] for l in legs)/len(legs), 1) if legs else None})
    display(pd.DataFrame(rows))
else:
    print("  no multi-camera vehicles, so no flows")

print("\\nVEHICLE TYPE MIX")
if len(obs_df):
    display(obs_df.vehicle_type.value_counts().to_frame("count"))
"""))

# ---------------------------------------------------------------- 14
cells.append(md("""
## 14. Watchlist alerts

The watchlist is read from `data/watchlist.txt` (one plate per line). It is **not**
auto-populated from detections - an alert only fires if a plate you listed actually
appears in the footage. If the file is absent or empty, no watchlist alerts are produced.

Spatio-temporal anomalies are separate: they come from route validation and fire when a
transition is physically impossible.
"""))
cells.append(code("""
WATCHLIST_FILE = ROOT / "data" / "watchlist.txt"
WATCHLIST = set()
if WATCHLIST_FILE.exists():
    WATCHLIST = {l.strip().upper() for l in WATCHLIST_FILE.read_text().splitlines()
                 if l.strip() and not l.startswith("#")}
print(f"watchlist entries loaded: {len(WATCHLIST)}"
      + (f" -> {sorted(WATCHLIST)}" if WATCHLIST else
         f"  (create {WATCHLIST_FILE.name} with one plate per line to enable)"))

alerts = []
for o in all_observations:
    if o.plate_number in WATCHLIST:
        alerts.append({"alert_type": "WATCHLIST_MATCH", "severity": "HIGH",
                       "plate": o.plate_number, "camera": o.camera_id,
                       "location": CAMERAS[o.camera_id].name, "timestamp": o.timestamp,
                       "ocr_confidence": o.ocr_confidence, "evidence": o.evidence_crop,
                       "reason": f"Watchlisted plate {o.plate_number} observed at "
                                 f"{o.camera_id} ({CAMERAS[o.camera_id].name}) at {o.timestamp}."})
for v in vehicles:
    for l in v.legs:
        if l["verdict"] == "IMPLAUSIBLE":
            alerts.append({"alert_type": "SPATIO_TEMPORAL_ANOMALY", "severity": "MEDIUM",
                           "plate": v.plate_number, "camera": l["to_camera"],
                           "location": CAMERAS[l["to_camera"]].name,
                           "timestamp": l["arrived"], "ocr_confidence": v.confidence,
                           "evidence": None, "reason": l["reason"]})

if alerts:
    display(pd.DataFrame(alerts))
else:
    print("\\nNo alerts. (Empty watchlist and no physically impossible transitions.)")
"""))

# ---------------------------------------------------------------- 15
cells.append(md("## 15. Evidence crops (real plate images from your videos)"))
cells.append(code("""
import matplotlib.pyplot as plt

crops = [o for o in all_observations if o.evidence_crop and Path(o.evidence_crop).exists()]
print(f"{len(crops)} evidence crops stored")
show = crops[:12]
if show:
    # squeeze=False + ravel() keeps the axes array 1-D for ANY crop count,
    # including exactly one (where subplots would otherwise return a bare Axes
    # or a 1-D row, and naive flattening breaks).
    ncol = min(4, len(show))
    nrow = (len(show) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.7 * nrow), squeeze=False)
    flat = axes.ravel()
    for ax, o in zip(flat, show):
        ax.imshow(cv2.cvtColor(cv2.imread(o.evidence_crop), cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"{o.plate_number}\\n{o.camera_id}  {o.timestamp[-8:]}  "
                     f"({o.ocr_confidence:.2f})", fontsize=9)
    for ax in flat[len(show):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()
else:
    print("No evidence crops were written.")
"""))

# ---------------------------------------------------------------- 16
cells.append(md("""
## 16. OCR robustness on YOUR footage (measured)

Runs the degradation benchmark against the plate crops just extracted from your videos.
Real crops, controlled degradation, measured recovery.

This is a robustness curve, not an absolute accuracy figure: the reference strings are the
model's own clean-image readings, not human-verified ground truth. A true exact-plate
accuracy number requires a hand-labelled test set.
"""))
cells.append(code("""
print(subprocess.run([sys.executable, "scripts/benchmark_ocr_robustness.py"],
                     capture_output=True, text=True, cwd=str(ROOT)).stdout)
"""))

# ---------------------------------------------------------------- 17
cells.append(md("""
## 16b. Persist everything to Supabase

All detections, tracks, plates, timestamps, routes, camera locations (as PostGIS
points) and alerts are written to your Supabase Postgres database.

Each execution opens a **run** and every row is scoped to its `run_id`, so repeated
runs accumulate history rather than overwriting. Credentials come from the git-ignored
`.env` - never from this notebook.

> **Run the two cells below in order.** The first connects and creates the schema; the
> second writes this run's data and defines `RUN_ID`. The read-back sections that follow
> fall back to the most recent run if `RUN_ID` is not set, so they also work standalone.
"""))
cells.append(code("""
from urbantrace.db.client import ping, apply_schema
from urbantrace.db import repository as repo

info = ping()
print(f"connected : {info['connected']}")
print(f"server    : {info['server']}")
print(f"dsn       : {info['dsn']}")          # password redacted
print(f"tables    : {info['public_tables']}")

# Idempotent - creates tables/views only if missing.
if info["public_tables"] < 12:
    apply_schema()
"""))
cells.append(code("""
write_report = repo.save_all(
    runtime=rt,
    models={"vehicle_model": VEHICLE_WEIGHTS.name,
            "plate_backend": active.name,
            "plate_model": active.model_id,
            "ocr_model": ocr.model_name},
    cameras=CAMERAS,
    edges=CAMERA_EDGES,
    video_rows=video_rows,
    camera_stats=camera_stats,
    observations=all_observations,
    vehicles=vehicles,
    alerts=alerts,
    notes=f"{len(CAMERAS)} camera(s), MAX_FRAMES={MAX_FRAMES}",
)
RUN_ID = write_report["run_id"]
print("RUN ID:", RUN_ID, "\\n")
display(pd.DataFrame([{"table": k, "rows_written": v}
                      for k, v in write_report.items() if k != "run_id"]))
"""))

cells.append(md("""
### 16c. Read it back out of the database

Proof the data is stored and queryable - these tables are built by SQL running on
Supabase, not from the in-memory Python objects.
"""))
cells.append(code("""
# Self-sufficient: these read-back cells work even if run out of order.
from urbantrace.db import repository as repo
RUN_ID = globals().get("RUN_ID") or repo.latest_run()
if not RUN_ID:
    raise RuntimeError("No run found in the database. Run cell 27 (save_all) first.")
print("RUN ID:", RUN_ID)

print("\\nRUN SUMMARY")
display(pd.DataFrame([repo.run_summary(RUN_ID)]).T.rename(columns={0: "value"}))

print("\\nPER-CAMERA COUNTS  (SQL view v_camera_counts)")
display(pd.DataFrame(repo.camera_counts(RUN_ID)))

print("\\nORIGIN-DESTINATION FLOWS  (SQL view v_camera_flows)")
flows_db = pd.DataFrame(repo.camera_flows(RUN_ID))
display(flows_db if len(flows_db) else "no multi-camera flows")

print("\\nVEHICLE ROUTES  (table global_vehicles)")
display(pd.DataFrame(repo.vehicle_routes(RUN_ID)))
"""))

cells.append(md("""
### 16d. Query-based vehicle tracking

The trajectory interface from the problem statement: give it a plate, get the complete
chronological movement history with timestamps and camera locations - straight from SQL.
"""))
cells.append(code("""
from urbantrace.db import repository as repo
RUN_ID = globals().get("RUN_ID") or repo.latest_run()

routes_db = repo.vehicle_routes(RUN_ID)
SEARCH_PLATE = routes_db[0]["plate_number"] if routes_db else None
print(f"SEARCH PLATE: {SEARCH_PLATE}\\n")

if SEARCH_PLATE:
    tl = pd.DataFrame(repo.plate_timeline(SEARCH_PLATE, RUN_ID))
    display(tl)
    for _, r in tl.iterrows():
        print(f"  {r['observed_at']}  {r['camera_id']}  {r['location']}  "
              f"track#{r['local_track_id']}  conf={r['ocr_confidence']:.3f}  "
              f"({r['latitude']}, {r['longitude']})")
"""))

cells.append(md("""
### 16e. Watchlist stored in the database

The watchlist lives in the `watchlist` table. Add plates with
`repo.add_to_watchlist("TN09AB1234", "reason")`; `repo.get_watchlist()` is what the
alert engine should read in production.
"""))
cells.append(code("""
from urbantrace.db import repository as repo
RUN_ID = globals().get("RUN_ID") or repo.latest_run()

db_watchlist = repo.get_watchlist()
print(f"watchlist entries in DB: {len(db_watchlist)}"
      + (f" -> {sorted(db_watchlist)}" if db_watchlist else "  (empty)"))

# Alerts already written for this run:
alerts_db = repo._fetch(
    "select alert_type, severity, plate_number, camera_id, location, occurred_at, reason "
    "from alerts where run_id = %s order by occurred_at", (RUN_ID,))
display(pd.DataFrame(alerts_db) if alerts_db else "no alerts stored for this run")
"""))

cells.append(md("## 17. Export everything"))
cells.append(code("""
payload = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "runtime": rt,
    "models": {
        "vehicle_detector": VEHICLE_WEIGHTS.name,
        "plate_detector": {"backend": active.name, "detail": active.detail},
        "ocr": ocr.model_name,
    },
    "videos": video_rows,
    "network": network_summary(),
    "camera_stats": camera_stats,
    "capture_times": {c: str(camera_start(c)) for c in CAMERAS},
    "per_camera_counts": per_cam,
    "timestamp_note": ("Wall-clock timestamps = configured camera start + measured video "
                       "offset. video_time_s is the measured quantity."),
    "supabase_run_id": RUN_ID,
    "supabase_write_report": write_report,
    "summary": summary,
    "observations": [o.to_dict() for o in all_observations],
    "global_vehicles": [v.to_dict() for v in vehicles],
    "alerts": alerts,
}
out = OUT_DIR / "urbantrace_results.json"
out.write_text(json.dumps(payload, indent=2))
print("written:", out, f"({out.stat().st_size/1024:.1f} KB)\\n")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name:38} {f.stat().st_size/1024:10.1f} KB")
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
