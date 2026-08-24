"""Per-camera video processing pipeline.

    frame -> vehicle detect + ByteTrack -> plate detect -> bind to track
          -> crop + preprocess -> OCR -> per-track reading buffer
          -> temporal aggregation -> ONE timestamped observation per track

Timestamps: the video files carry no wall-clock metadata, so every observation
records BOTH `video_time_s` (authoritative, measured from frame index / fps) and
`timestamp` (= configured camera start time + video_time_s). The `clock_source`
field states which it is, so the UI never presents a derived clock as measured.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from ..core.cameras import CAMERAS
from .aggregate import Reading, aggregate
from .detectors import PlateDetector, VehicleTracker, bind_plates_to_tracks
from .ocr_engine import get_ocr
from .preprocess import crop_with_margin

# OCR is expensive relative to detection; cap work per track.
MAX_READINGS_PER_TRACK = 14
MIN_PLATE_PIXELS = 480          # w*h below this the crop cannot carry 10 glyphs

# A track must persist this many frames before it is counted. ByteTrack can emit
# a short-lived id for a spurious detection; requiring persistence stops those
# inflating the totals.
MIN_FRAMES_TO_COUNT = 3

# Classes shown in the on-video counter, in display order.
COUNT_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle")
COUNT_COLORS = {                       # BGR
    "car":        (229, 135, 57),
    "motorcycle": (38, 89, 217),
    "bus":        (112, 158, 25),
    "truck":      (25, 178, 250),
    "bicycle":    (200, 200, 200),
    "total":      (255, 255, 255),
}


@dataclass
class Observation:
    """One vehicle, one camera - the system's fundamental record."""
    observation_id: str
    camera_id: str
    plate_number: str
    ocr_confidence: float
    plate_detection_confidence: float
    local_track_id: int
    vehicle_type: str
    frame_number: int
    video_time_s: float
    timestamp: str                  # ISO, derived from camera start time
    clock_source: str               # how `timestamp` was produced
    latitude: float
    longitude: float
    direction: str
    aggregation_method: str
    support: int
    total_readings: int
    agreement: float
    is_valid_format: bool
    evidence_crop: str | None = None
    raw_readings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


class CameraPipeline:
    def __init__(self, camera_id: str, video_path, start_time: datetime,
                 evidence_dir, vehicle_tracker=None, plate_detector=None,
                 write_video: bool = True, output_dir=None):
        if camera_id not in CAMERAS:
            raise KeyError(f"Unknown camera {camera_id}")
        self.cam = CAMERAS[camera_id]
        self.camera_id = camera_id
        self.video_path = Path(video_path)
        self.start_time = start_time
        self.evidence_dir = Path(evidence_dir) / camera_id
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.write_video = write_video
        self.output_dir = Path(output_dir) if output_dir else Path("data/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vehicles = vehicle_tracker or VehicleTracker()
        self.plates = plate_detector or PlateDetector()
        self.ocr = get_ocr()

        self.readings = {}
        self.track_meta = {}
        self.best_label = {}
        # Unique-vehicle counting. class_votes[tid] tallies the class YOLO gave
        # this track on each frame; the majority wins, so a car that flickers to
        # "truck" for two frames is still counted once, as a car.
        self.class_votes: dict = {}
        self.track_frames: dict = {}
        self.stats = {"frames": 0, "vehicle_dets": 0, "plate_dets": 0,
                      "ocr_calls": 0, "ocr_success": 0}

    # ---------------- counting ----------------

    def _note_tracks(self, tracks) -> None:
        for t in tracks:
            self.class_votes.setdefault(t.track_id, {})
            self.class_votes[t.track_id][t.cls_name] = \
                self.class_votes[t.track_id].get(t.cls_name, 0) + 1
            self.track_frames[t.track_id] = self.track_frames.get(t.track_id, 0) + 1

    def unique_counts(self) -> dict:
        """Cumulative UNIQUE vehicles seen so far, by majority class per track."""
        counts = {c: 0 for c in COUNT_CLASSES}
        other = 0
        for tid, votes in self.class_votes.items():
            if self.track_frames.get(tid, 0) < MIN_FRAMES_TO_COUNT:
                continue
            cls = max(votes.items(), key=lambda kv: kv[1])[0]
            if cls in counts:
                counts[cls] += 1
            else:
                other += 1
        counts["total"] = sum(counts[c] for c in COUNT_CLASSES) + other
        if other:
            counts["other"] = other
        return counts

    def _draw_counter(self, frame):
        counts = self.unique_counts()
        rows = [(c, counts[c]) for c in COUNT_CLASSES if counts.get(c)]
        if counts.get("other"):
            rows.append(("other", counts["other"]))
        rows.append(("total", counts["total"]))

        pad, lh = 10, 24
        w = 200
        h = pad * 2 + lh * (len(rows) + 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + w, 10 + h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + w, 10 + h), (70, 70, 70), 1)

        y = 10 + pad + 16
        cv2.putText(frame, "VEHICLES COUNTED", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        y += lh - 4
        for name, val in rows:
            col = COUNT_COLORS.get(name, (220, 220, 220))
            if name == "total":
                cv2.line(frame, (20, y - 15), (10 + w - 10, y - 15), (70, 70, 70), 1)
            cv2.putText(frame, f"{name}:", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, col, 2, cv2.LINE_AA)
            cv2.putText(frame, str(val), (10 + w - 22 - 12 * len(str(val)), y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, col, 2, cv2.LINE_AA)
            y += lh
        return frame

    # ---------------- per-frame ----------------

    def _handle_plate(self, frame, track, plate, frame_no, t_s):
        tid = track.track_id if track else -1
        buf = self.readings.setdefault(tid, [])
        if len(buf) >= MAX_READINGS_PER_TRACK:
            return
        x1, y1, x2, y2 = plate.box
        if (x2 - x1) * (y2 - y1) < MIN_PLATE_PIXELS:
            return
        crop = crop_with_margin(frame, plate.box)
        if crop is None:
            return

        self.stats["ocr_calls"] += 1
        res = self.ocr.read(crop)
        if res is None or not res.text_final:
            return
        self.stats["ocr_success"] += 1

        buf.append(Reading(
            frame_number=frame_no, timestamp_s=round(t_s, 3),
            text_raw=res.text_raw, text_final=res.text_final,
            ocr_confidence=res.confidence, plate_confidence=plate.confidence,
            is_valid_format=res.is_valid_format, repaired=res.repaired,
            variant=res.variant, char_confidences=res.char_confidences,
        ))
        self.best_label[tid] = res.text_final

        if track is not None:
            m = self.track_meta.setdefault(tid, {
                "cls": track.cls_name, "first_frame": frame_no,
                "first_t": t_s, "best_plate_conf": 0.0, "crop": None})
            m["last_frame"], m["last_t"] = frame_no, t_s
            m["cls"] = track.cls_name
            # Keep the single highest-confidence crop as evidence.
            if plate.confidence > m["best_plate_conf"]:
                m["best_plate_conf"] = plate.confidence
                path = self.evidence_dir / f"track{tid:04d}_f{frame_no:06d}.jpg"
                cv2.imwrite(str(path), crop)
                m["crop"] = str(path)

    def _annotate(self, frame, tracks, pairs):
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"#{t.track_id} {t.cls_name}"
            plate = self.best_label.get(t.track_id)
            if plate:
                label += f" | {plate}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 200, 255), -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        for _t, p in pairs:
            x1, y1, x2, y2 = [int(v) for v in p.box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 255, 60), 2)
        return self._draw_counter(frame)

    # ---------------- run ----------------

    def run(self, max_frames=None, frame_stride: int = 1, progress_every: int = 200):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[{self.camera_id}] {self.video_path.name}  {w}x{h} @ {fps:.2f}fps  frames={total}")

        writer = None
        if self.write_video:
            out_path = self.output_dir / f"{self.camera_id}_annotated.mp4"
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / max(1, frame_stride), (w, h))

        self.vehicles.reset()
        frame_no, t0 = 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_no >= max_frames:
                break
            if frame_stride > 1 and frame_no % frame_stride:
                frame_no += 1
                continue

            t_s = frame_no / fps
            tracks = self.vehicles.update(frame)
            self._note_tracks(tracks)
            plates = self.plates.detect(frame)
            self.stats["vehicle_dets"] += len(tracks)
            self.stats["plate_dets"] += len(plates)

            pairs = bind_plates_to_tracks(plates, tracks)
            for track, plate in pairs:
                self._handle_plate(frame, track, plate, frame_no, t_s)

            if writer is not None:
                writer.write(self._annotate(frame, tracks, pairs))

            frame_no += 1
            self.stats["frames"] += 1
            if progress_every and frame_no % progress_every == 0:
                el = time.time() - t0
                print(f"  [{self.camera_id}] frame {frame_no}/{total} "
                      f"({frame_no/el:.1f} fps proc) tracks={len(tracks)} "
                      f"plates={len(plates)} ocr_ok={self.stats['ocr_success']}")

        cap.release()
        if writer is not None:
            writer.release()
        elapsed = max(1e-6, time.time() - t0)
        self.stats["wall_seconds"] = round(elapsed, 1)
        self.stats["processing_fps"] = round(self.stats["frames"] / elapsed, 2)

        # Surface component failures so an empty result is never ambiguous:
        # "no plates found" and "the OCR engine was broken" look identical in
        # the output unless these are reported.
        self.stats["ocr_engine_failures"] = getattr(self.ocr, "failures", 0)
        self.stats["ocr_last_error"] = getattr(self.ocr, "last_error", None)
        self.stats["plate_detector_failures"] = getattr(self.plates, "failures", 0)
        self.stats["plate_detector_last_error"] = getattr(self.plates, "last_error", None)
        from . import preprocess as _pp
        self.stats["preprocess_variant_failures"] = dict(_pp.VARIANT_FAILURES)

        # Unique vehicles counted (each track once, majority class).
        counts = self.unique_counts()
        self.stats["unique_vehicles"] = counts.pop("total")
        self.stats["counts_by_class"] = counts
        self.stats["count_min_frames"] = MIN_FRAMES_TO_COUNT
        print(f"  [{self.camera_id}] unique vehicles counted: "
              f"{self.stats['unique_vehicles']}  {counts}")

        if self.stats["ocr_engine_failures"]:
            print(f"  [{self.camera_id}] WARNING: OCR engine failed "
                  f"{self.stats['ocr_engine_failures']} time(s): {self.stats['ocr_last_error']}")
        if self.stats["plate_detector_failures"]:
            print(f"  [{self.camera_id}] WARNING: plate detector failed "
                  f"{self.stats['plate_detector_failures']} time(s): "
                  f"{self.stats['plate_detector_last_error']}")
        return self.finalize()

    def finalize(self):
        """Collapse each track's readings into a single observation."""
        observations = []
        for tid, buf in sorted(self.readings.items()):
            if tid < 0 or not buf:
                continue
            agg = aggregate(buf, min_readings=1)
            if agg is None:
                continue
            meta = self.track_meta.get(tid, {})
            # The observation is timed at the track's FIRST confident sighting.
            t_s = agg.first_timestamp_s
            ts = self.start_time + timedelta(seconds=t_s)
            observations.append(Observation(
                observation_id=f"OBS_{self.camera_id}_{tid:04d}",
                camera_id=self.camera_id,
                plate_number=agg.plate,
                ocr_confidence=agg.confidence,
                plate_detection_confidence=round(meta.get("best_plate_conf", 0.0), 4),
                local_track_id=tid,
                vehicle_type=meta.get("cls", "unknown"),
                frame_number=agg.first_frame,
                video_time_s=round(t_s, 3),
                timestamp=_fmt(ts),
                clock_source="video_start + frame_offset",
                latitude=self.cam.latitude, longitude=self.cam.longitude,
                direction=self.cam.direction,
                aggregation_method=agg.method, support=agg.support,
                total_readings=agg.total_readings, agreement=agg.agreement,
                is_valid_format=agg.is_valid_format,
                evidence_crop=meta.get("crop"),
                raw_readings=[r.to_dict() for r in buf],
            ))
        return observations


def save_observations(observations, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([o.to_dict() for o in observations], indent=2))
    return path
