"""Measure OCR robustness under controlled image degradation.

Method
------
1. Take REAL plate crops that the OCR reads with high confidence when clean.
   Their clean reading is used as the reference string for that crop.
2. Apply controlled degradations at increasing severity: motion blur, defocus
   blur, downscale (distance), sensor noise, low light.
3. Re-run OCR and measure how often the reference string is recovered, with
   single-variant ("raw only") vs multi-variant preprocessing.

What this IS: a robustness curve showing how preprocessing holds up as image
quality falls, measured on real plate imagery.

What this is NOT: an absolute accuracy figure. The reference strings are the
model's own confident readings, not human-verified ground truth. Absolute
exact-plate accuracy requires a hand-labelled test set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from urbantrace.vision.ocr_engine import PlateOCR  # noqa: E402
from urbantrace.vision import preprocess as pp     # noqa: E402

BASELINE_CONF = 0.90


# ---------------- degradations ----------------

def motion_blur(img, k: int):
    if k < 3:
        return img
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k          # horizontal streak, as in a passing car
    return cv2.filter2D(img, -1, kern)


def defocus_blur(img, k: int):
    if k < 3:
        return img
    k = k if k % 2 else k + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def downscale(img, factor: float):
    if factor <= 1.0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w / factor)), max(4, int(h / factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def add_noise(img, sigma: float):
    if sigma <= 0:
        return img
    n = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def low_light(img, gain: float):
    if gain >= 1.0:
        return img
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


DEGRADATIONS = {
    "motion_blur":  (motion_blur,  [3, 5, 7, 9, 11]),
    "defocus_blur": (defocus_blur, [3, 5, 7, 9, 11]),
    "downscale":    (downscale,    [1.5, 2.0, 2.5, 3.0, 4.0]),
    "noise":        (add_noise,    [5, 10, 20, 30, 40]),
    "low_light":    (low_light,    [0.7, 0.5, 0.35, 0.25, 0.15]),
}


def collect_baselines(evidence_dir: Path, ocr: PlateOCR, limit: int = 40):
    """Real crops the OCR reads confidently when undegraded."""
    out = []
    for f in sorted(evidence_dir.rglob("*.jpg"))[:limit * 4]:
        img = cv2.imread(str(f))
        if img is None:
            continue
        r = ocr.read(img)
        if r and r.confidence >= BASELINE_CONF and len(r.text_final) >= 6:
            out.append((f.name, img, r.text_final, r.confidence))
        if len(out) >= limit:
            break
    return out


def char_accuracy(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0
    n = min(len(ref), len(hyp))
    same = sum(1 for i in range(n) if ref[i] == hyp[i])
    return same / max(len(ref), len(hyp))


def main() -> int:
    evidence = ROOT / "data" / "evidence"
    if not any(evidence.rglob("*.jpg")):
        print("No plate crops in data/evidence/. Run the pipeline on a video first.")
        return 1

    raw_only = PlateOCR(variants=("raw",), consensus=False)
    multi = PlateOCR(variants=("raw", "clahe", "sharpen", "denoise"), consensus=True)

    baselines = collect_baselines(evidence, multi)
    print(f"Baseline crops (clean read conf >= {BASELINE_CONF}): {len(baselines)}")
    if not baselines:
        print("No high-confidence baseline crops available.")
        return 1
    for name, img, ref, conf in baselines:
        print(f"   {name:32} {str(img.shape[:2]):12} -> {ref:10} ({conf:.3f})")

    print("\n" + "=" * 86)
    print(f"{'degradation':14}{'level':>8}{'raw exact':>12}{'multi exact':>13}"
          f"{'raw char':>11}{'multi char':>12}{'delta':>9}")
    print("=" * 86)

    totals = {"raw": [], "multi": []}
    for deg_name, (fn, levels) in DEGRADATIONS.items():
        for lv in levels:
            r_ex = m_ex = 0
            r_ch, m_ch = [], []
            for _name, img, ref, _c in baselines:
                bad = fn(img.copy(), lv)
                a = raw_only.read(bad)
                b = multi.read(bad)
                at = a.text_final if a else ""
                bt = b.text_final if b else ""
                r_ex += int(at == ref)
                m_ex += int(bt == ref)
                r_ch.append(char_accuracy(ref, at))
                m_ch.append(char_accuracy(ref, bt))
            n = len(baselines)
            r_rate, m_rate = r_ex / n, m_ex / n
            totals["raw"].append(r_rate)
            totals["multi"].append(m_rate)
            print(f"{deg_name:14}{lv:>8}{r_rate:>11.1%}{m_rate:>13.1%}"
                  f"{np.mean(r_ch):>11.1%}{np.mean(m_ch):>12.1%}"
                  f"{m_rate - r_rate:>+9.1%}")

    print("=" * 86)
    print(f"MEAN exact-recovery across all degradations: "
          f"raw-only {np.mean(totals['raw']):.1%}  |  "
          f"multi-variant {np.mean(totals['multi']):.1%}  |  "
          f"delta {np.mean(totals['multi']) - np.mean(totals['raw']):+.1%}")
    print("\nNote: reference strings are the model's own clean-image readings, not")
    print("human-verified ground truth. This measures ROBUSTNESS, not absolute accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
