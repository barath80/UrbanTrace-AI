"""Smooth load shedding for the analysis rate.

When the detectors cannot keep up, the frame bus drops its oldest frames. That
is safe but uneven: whichever cameras happen to lose the race lose coverage.
This controller watches the drop ratio and lowers the analysis rate for every
camera together, then raises it again when there is headroom. Cameras sample
less often, none go dark.

Multiplicative decrease and gentle increase, so it backs off quickly under
overload and recovers without oscillating.
"""
from __future__ import annotations

import threading

from .config import SCALE


class RateController:
    def __init__(self, target_fps: float | None = None):
        self.target_fps = SCALE.target_fps_per_camera if target_fps is None else target_fps
        self.floor_fps = min(SCALE.min_fps_per_camera, self.target_fps)
        self._fps = self.target_fps
        self._last_submitted = 0
        self._last_dropped = 0
        self._lock = threading.Lock()
        self.cuts = 0
        self.recoveries = 0

    def fps(self) -> float:
        return self._fps

    def tick(self, submitted: int, dropped: int) -> float:
        """Feed cumulative bus counters; returns the rate to use next."""
        with self._lock:
            ds = submitted - self._last_submitted
            dd = dropped - self._last_dropped
            self._last_submitted, self._last_dropped = submitted, dropped
            if not SCALE.adaptive_fps or ds <= 0:
                return self._fps
            ratio = dd / ds
            if ratio > SCALE.adapt_drop_high and self._fps > self.floor_fps:
                self._fps = max(self.floor_fps, self._fps * 0.8)
                self.cuts += 1
            elif ratio < SCALE.adapt_drop_low and self._fps < self.target_fps:
                self._fps = min(self.target_fps, self._fps * 1.1 + 0.05)
                self.recoveries += 1
            return self._fps

    def stats(self) -> dict:
        return {"current_fps": round(self._fps, 2), "target_fps": self.target_fps,
                "floor_fps": self.floor_fps, "cuts": self.cuts,
                "recoveries": self.recoveries}
