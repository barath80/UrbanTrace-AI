"""License-plate crop extraction and preprocessing.

Preprocessing variants are generated, not blindly stacked: the OCR engine is
run over several candidate renderings of the same plate and the highest
confidence reading wins. This is measurable, unlike a fixed filter chain.
"""
from __future__ import annotations

import cv2
import numpy as np

# Target height for OCR input; plates below this are upscaled.
OCR_TARGET_H = 64
MIN_PLATE_H = 12
MIN_PLATE_W = 32


def crop_with_margin(frame: np.ndarray, box, margin: float = 0.08) -> np.ndarray | None:
    """Crop a plate box with a safe margin, clipped to frame bounds."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    mx, my = bw * margin, bh * margin
    x1 = max(0, int(round(x1 - mx)));  y1 = max(0, int(round(y1 - my)))
    x2 = min(w, int(round(x2 + mx)));  y2 = min(h, int(round(y2 + my)))
    if x2 - x1 < MIN_PLATE_W or y2 - y1 < MIN_PLATE_H:
        return None
    return frame[y1:y2, x1:x2].copy()


def _upscale(img: np.ndarray, target_h: int = OCR_TARGET_H) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    if h >= target_h:
        return img
    scale = target_h / h
    # INTER_CUBIC preserves character strokes better than LINEAR when upscaling.
    return cv2.resize(img, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_CUBIC)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def variant_raw(img: np.ndarray) -> np.ndarray:
    return _upscale(img)


def variant_clahe(img: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation - helps low light."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(g)
    return _to_bgr(_upscale(g))


def variant_sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp mask - recovers edges lost to motion blur / low resolution."""
    up = _upscale(img)
    blur = cv2.GaussianBlur(up, (0, 0), 3)
    return cv2.addWeighted(up, 1.6, blur, -0.6, 0)


def variant_denoise(img: np.ndarray) -> np.ndarray:
    """Bilateral filter - suppresses sensor noise while keeping edges."""
    up = _upscale(img)
    return cv2.bilateralFilter(up, 7, 60, 60)


def variant_binarize(img: np.ndarray) -> np.ndarray:
    """Otsu threshold - helps high-contrast plates with uneven illumination."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Plates are dark-on-light in India; invert if we got light-on-dark.
    if th.mean() < 127:
        th = cv2.bitwise_not(th)
    return _to_bgr(_upscale(th))


def deskew(img: np.ndarray) -> np.ndarray:
    """Correct in-plane rotation using the dominant text edge orientation."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=max(12, img.shape[1] // 3), maxLineGap=8)
    if lines is None:
        return img
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -30 < a < 30:           # near-horizontal only
            angles.append(a)
    if not angles:
        return img
    angle = float(np.median(angles))
    if abs(angle) < 1.0:           # not worth resampling
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# Order matters only for tie-breaking; all are scored by the OCR engine.
VARIANTS = {
    "raw": variant_raw,
    "clahe": variant_clahe,
    "sharpen": variant_sharpen,
    "denoise": variant_denoise,
    "binarize": variant_binarize,
}

DEFAULT_VARIANTS = ("raw", "clahe", "sharpen")

# Variants that failed to render (counted so a systematically broken variant
# cannot silently reduce OCR quality without anyone noticing).
VARIANT_FAILURES: dict = {}


def build_variants(plate_img: np.ndarray, names=DEFAULT_VARIANTS,
                   apply_deskew: bool = True) -> dict:
    """Return {variant_name: image} for OCR scoring."""
    base = deskew(plate_img) if apply_deskew else plate_img
    out = {}
    for n in names:
        fn = VARIANTS.get(n)
        if fn is None:
            continue
        try:
            out[n] = fn(base)
        except cv2.error as e:
            VARIANT_FAILURES[n] = VARIANT_FAILURES.get(n, 0) + 1
            if VARIANT_FAILURES[n] in (1, 50):
                import warnings
                warnings.warn(f"preprocessing variant '{n}' failed "
                              f"{VARIANT_FAILURES[n]}x: {e}", RuntimeWarning)
            continue
    return out or {"raw": _upscale(plate_img)}
