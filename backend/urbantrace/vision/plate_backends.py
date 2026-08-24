"""Pluggable license-plate detection backends.

Two interchangeable implementations of the same interface:

  RoboflowPlateDetector  - the user's own trained model
                           (indian-license-plate-detection-6tmbr/1, YOLOv8n,
                           Roboflow-reported mAP@50 96.04 / P 94.11 / R 93.30)
                           served over Roboflow hosted inference.

  LocalPlateDetector     - local YOLOv11 weights on GPU, no network, no credits.

`get_plate_detector()` probes Roboflow once and uses it when the workspace can
actually serve inference; otherwise it falls back to local weights and says so.
The rest of the system is agnostic to which one is active.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .detectors import PlateBox, PlateDetector as LocalPlateDetector, PLATE_WEIGHTS

ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID",
                                   "indian-license-plate-detection-6tmbr/1")
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")


def _api_key() -> str | None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key
    env = Path(__file__).resolve().parents[3] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ROBOFLOW_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


@dataclass
class BackendStatus:
    name: str
    available: bool
    detail: str
    model_id: str | None = None


class RoboflowPlateDetector:
    """The user's trained Roboflow model, via hosted inference.

    Note on cost/latency: this issues one HTTP request per frame. For video at
    25 fps that is 25 requests/second of billable inference, so `frame_stride`
    should normally be > 1 when this backend is active.
    """

    def __init__(self, model_id: str = ROBOFLOW_MODEL_ID, conf: float = 0.25):
        from inference_sdk import InferenceHTTPClient
        key = _api_key()
        if not key:
            raise RuntimeError("ROBOFLOW_API_KEY not set (env or .env file)")
        self.model_id = model_id
        self.conf = conf
        self.client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=key)
        self.name = "roboflow-hosted"
        # Hosted inference can start failing mid-run (credits, rate limit,
        # network). Those failures are counted and surfaced, not hidden behind
        # an empty detection list.
        self.failures = 0
        self.calls = 0
        self.last_error = None

    def detect(self, frame: np.ndarray) -> list:
        # The hosted API takes a file path or base64; encode in-memory to avoid
        # writing a temp file for every frame.
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return []
        payload = base64.b64encode(buf.tobytes()).decode("ascii")
        self.calls += 1
        try:
            res = self.client.infer(payload, model_id=self.model_id)
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
            if self.failures in (1, 5, 25) or self.failures % 100 == 0:
                import warnings
                warnings.warn(f"Roboflow inference failed {self.failures}/{self.calls} "
                              f"calls (latest: {self.last_error})", RuntimeWarning)
            return []

        out = []
        for p in (res or {}).get("predictions", []):
            c = float(p.get("confidence", 0.0))
            if c < self.conf:
                continue
            # Roboflow returns centre-x/centre-y/width/height.
            cx, cy = float(p["x"]), float(p["y"])
            w, h = float(p["width"]), float(p["height"])
            out.append(PlateBox(round(c, 4),
                                (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        return out


def probe_roboflow() -> BackendStatus:
    """One cheap call to decide whether the hosted model is actually usable."""
    key = _api_key()
    if not key:
        return BackendStatus("roboflow-hosted", False,
                             "No ROBOFLOW_API_KEY configured", ROBOFLOW_MODEL_ID)
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError:
        return BackendStatus("roboflow-hosted", False,
                             "inference-sdk not installed", ROBOFLOW_MODEL_ID)

    probe = np.full((64, 128, 3), 127, np.uint8)
    ok, buf = cv2.imencode(".jpg", probe)
    try:
        client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=key)
        client.infer(base64.b64encode(buf.tobytes()).decode("ascii"),
                     model_id=ROBOFLOW_MODEL_ID)
        return BackendStatus("roboflow-hosted", True,
                             "Hosted inference reachable", ROBOFLOW_MODEL_ID)
    except Exception as e:
        msg = str(e)
        if "credit_cap" in msg or "402" in msg:
            detail = ("Workspace out of inference credits (402 credit_cap_exceeded). "
                      "Restore billing at app.roboflow.com/settings/plan to enable "
                      "this backend.")
        elif "401" in msg or "403" in msg:
            detail = "API key rejected (401/403)."
        else:
            detail = msg[:180]
        return BackendStatus("roboflow-hosted", False, detail, ROBOFLOW_MODEL_ID)


@lru_cache(maxsize=1)
def get_plate_detector(prefer: str = "auto"):
    """Return (detector, BackendStatus).

    prefer: "auto" | "roboflow" | "local"
    """
    if prefer in ("auto", "roboflow"):
        status = probe_roboflow()
        if status.available:
            return RoboflowPlateDetector(), status
        if prefer == "roboflow":
            raise RuntimeError(f"Roboflow backend unavailable: {status.detail}")
    else:
        status = BackendStatus("roboflow-hosted", False, "Not requested", ROBOFLOW_MODEL_ID)

    det = LocalPlateDetector()
    det.name = "local-yolov11"
    local_status = BackendStatus(
        "local-yolov11", True,
        f"Local weights {PLATE_WEIGHTS.name} on GPU. "
        f"(Roboflow unavailable: {status.detail})" if not status.available
        else f"Local weights {PLATE_WEIGHTS.name}",
        None)
    return det, local_status
