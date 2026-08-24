"""Does temporal aggregation recover plates that single frames cannot?

This is the decisive test for the "it must read the plate even when blurred"
requirement. Single-frame preprocessing has a hard ceiling on motion blur
(measured in benchmark_ocr_robustness.py: 0% exact recovery at kernel >= 7).

Reality is kinder than that worst case: a vehicle is seen over many frames and
blur VARIES across them - it is worst when the car is far and moving fast,
better as it approaches. The system therefore should not need any single frame
to be readable, only that SOME frames are.

Simulated track: N frames of one real plate, each independently degraded with a
random blur severity drawn from a realistic distribution. We compare:
    best-single-frame  - the most confident individual frame reading
    temporal aggregate - the consensus produced by aggregate.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from urbantrace.vision.ocr_engine import PlateOCR       # noqa: E402
from urbantrace.vision.aggregate import Reading, aggregate  # noqa: E402
from benchmark_ocr_robustness import (motion_blur, defocus_blur, downscale,   # noqa: E402
                                      add_noise, low_light, collect_baselines)

TRACK_LENGTH = 12
TRIALS_PER_PLATE = 6
SEED = 7

# Degradation severity across a track. Calibrated so that only a MINORITY of
# frames are individually readable - the regime where consensus should matter.
# (At SEV_MEAN=0.75 essentially no frame is readable and aggregation cannot
# help; that measurement is reported in the notes rather than hidden.)
SEV_MEAN = 0.42
SEV_SLOPE = 0.30
SEV_SD = 0.16


def degrade_frame(img, rng: random.Random, severity: float):
    """One frame of a track: blur severity scales with `severity` in [0,1]."""
    out = img.copy()
    mk = int(round(3 + severity * 8))
    if mk >= 3:
        out = motion_blur(out, mk if mk % 2 else mk + 1)
    if rng.random() < 0.5:
        dk = int(round(1 + severity * 6))
        if dk >= 3:
            out = defocus_blur(out, dk)
    out = downscale(out, 1.0 + severity * 1.8)
    out = add_noise(out, severity * 18)
    out = low_light(out, 1.0 - severity * 0.45)
    return out


def main() -> int:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    ocr = PlateOCR(variants=("raw", "clahe", "sharpen"), consensus=True)
    evidence = ROOT / "data" / "evidence"
    baselines = collect_baselines(evidence, ocr, limit=10)
    if not baselines:
        print("No baseline crops. Run the pipeline on a video first.")
        return 1

    print(f"Simulating {TRACK_LENGTH}-frame tracks, {TRIALS_PER_PLATE} trials x "
          f"{len(baselines)} plates = {TRIALS_PER_PLATE*len(baselines)} tracks")
    print("Each frame independently degraded (motion+defocus blur, distance, noise, low light)\n")

    single_hits = agg_hits = trials = 0
    readable_frac = []

    for _name, img, ref, _c in baselines:
        for _t in range(TRIALS_PER_PLATE):
            readings, best_single, best_conf = [], "", -1.0
            n_correct_frames = 0

            for f in range(TRACK_LENGTH):
                # Blur worst at the start of the track, easing as the car nears.
                sev = max(0.0, min(1.0, rng.gauss(SEV_MEAN - SEV_SLOPE * (f / TRACK_LENGTH),
                                                  SEV_SD)))
                r = ocr.read(degrade_frame(img, rng, sev))
                if r is None or not r.text_final:
                    continue
                readings.append(Reading(
                    frame_number=f, timestamp_s=f / 25.0,
                    text_raw=r.text_raw, text_final=r.text_final,
                    ocr_confidence=r.confidence, plate_confidence=0.9,
                    is_valid_format=r.is_valid_format, repaired=r.repaired,
                    variant=r.variant, char_confidences=r.char_confidences))
                n_correct_frames += int(r.text_final == ref)
                if r.confidence > best_conf:
                    best_conf, best_single = r.confidence, r.text_final

            agg = aggregate(readings) if readings else None
            trials += 1
            single_hits += int(best_single == ref)
            agg_hits += int(bool(agg) and agg.plate == ref)
            readable_frac.append(n_correct_frames / max(1, TRACK_LENGTH))

    print("=" * 74)
    print(f"tracks simulated                     : {trials}")
    print(f"mean fraction of frames individually")
    print(f"  correct within a track             : {np.mean(readable_frac):.1%}")
    print("-" * 74)
    print(f"best-single-frame  exact plate       : {single_hits/trials:.1%}")
    print(f"TEMPORAL AGGREGATE exact plate       : {agg_hits/trials:.1%}")
    print(f"improvement                          : "
          f"{(agg_hits-single_hits)/trials:+.1%}")
    print("=" * 74)
    print("\nReference strings are the model's own clean-image readings, so this")
    print("measures RECOVERY UNDER DEGRADATION, not absolute plate accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
