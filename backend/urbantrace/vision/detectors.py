"""Vehicle detection + multi-object tracking, and license-plate detection.

Two local YOLO models run per frame:
  1. yolo11m (COCO)          -> vehicles, with ByteTrack IDs persisted per camera
  2. license-plate-finetune  -> plate boxes (full frame, then assigned to tracks)

Plate detection is run on the FULL frame rather than per-vehicle crop: it is a
single inference regardless of vehicle count, and the plate model was trained on
full scenes. Plates are then bound to vehicle tracks by containment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..core.runtime import device

WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "models" / "weights"
VEHICLE_WEIGHTS = WEIGHTS_DIR / "yolo11m.pt"
PLATE_WEIGHTS = WEIGHTS_DIR / "license-plate-finetune-v1m.pt"

# COCO ids we treat as vehicles.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}


@dataclass
class VehicleTrack:
    track_id: int
    cls_name: str
    confidence: float
    box: tuple          # (x1, y1, x2, y2)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlateBox:
    confidence: float
    box: tuple

    def to_dict(self) -> dict:
        return asdict(self)


def _xyxy(t) -> tuple:
    return tuple(float(v) for v in t)


class VehicleTracker:
    """YOLO detection + ByteTrack. One instance per camera (state is per-video)."""

    def __init__(self, weights: Path = VEHICLE_WEIGHTS, conf: float = 0.30,
                 imgsz: int = 960, tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO
        if not weights.exists():
            raise FileNotFoundError(f"Vehicle weights missing: {weights}")
        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.tracker = conf, imgsz, tracker
        self.device = 0 if device() == "cuda" else "cpu"

    def update(self, frame: np.ndarray) -> list:
        res = self.model.track(
            frame, persist=True, conf=self.conf, imgsz=self.imgsz,
            classes=list(VEHICLE_CLASSES), tracker=self.tracker,
            device=self.device, verbose=False,
        )
        out = []
        if not res:
            return out
        boxes = res[0].boxes
        if boxes is None or boxes.id is None:
            return out
        ids = boxes.id.int().cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        for tid, c, cf, bb in zip(ids, clss, confs, xyxy):
            out.append(VehicleTrack(int(tid), VEHICLE_CLASSES.get(int(c), str(c)),
                                    round(float(cf), 4), _xyxy(bb)))
        return out

    def reset(self) -> None:
        """Drop tracker state between videos so track IDs restart per camera."""
        try:
            self.model.predictor.trackers[0].reset()
        except Exception:
            self.model.predictor = None


class PlateDetector:
    """Local YOLOv11 license-plate detector."""

    def __init__(self, weights: Path = PLATE_WEIGHTS, conf: float = 0.25,
                 imgsz: int = 960):
        from ultralytics import YOLO
        if not weights.exists():
            raise FileNotFoundError(f"Plate weights missing: {weights}")
        self.model = YOLO(str(weights))
        self.conf, self.imgsz = conf, imgsz
        self.device = 0 if device() == "cuda" else "cpu"

    def detect(self, frame: np.ndarray) -> list:
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)
        out = []
        if not res or res[0].boxes is None:
            return out
        b = res[0].boxes
        for cf, bb in zip(b.conf.cpu().tolist(), b.xyxy.cpu().tolist()):
            out.append(PlateBox(round(float(cf), 4), _xyxy(bb)))
        return out


def containment(inner: tuple, outer: tuple) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (iw * ih) / area if area > 0 else 0.0


def bind_plates_to_tracks(plates: list, tracks: list, min_containment: float = 0.60):
    """Assign each plate box to the smallest vehicle track that contains it.

    Smallest-containing-box wins, which resolves the common case of a plate
    falling inside both a car box and an overlapping bus/truck box behind it.
    Returns [(VehicleTrack | None, PlateBox)].
    """
    pairs = []
    for p in plates:
        best, best_area = None, float("inf")
        for t in tracks:
            if containment(p.box, t.box) >= min_containment:
                area = (t.box[2] - t.box[0]) * (t.box[3] - t.box[1])
                if area < best_area:
                    best, best_area = t, area
        pairs.append((best, p))
    return pairs
