"""Plate OCR engine.

Primary engine: fast-plate-ocr `cct-s-v2-global-model` (CCT transformer trained
on global plates), executed on GPU via onnxruntime.

For each detected plate we run OCR over several preprocessing variants and keep
the highest-confidence reading. Confidence is the mean per-character
probability, which the engine returns natively -- it is a real model output,
not a heuristic score.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from functools import lru_cache

import numpy as np

from ..core.runtime import device, onnx_providers, enable_gpu_dlls
from . import preprocess as pp
from .plate_format import validate, PlateValidation

PAD_CHARS = set("_ -")


@dataclass
class OCRResult:
    text_raw: str                 # engine output, untouched
    text_final: str               # after Indian-plate grammar validation
    confidence: float             # mean per-character probability
    min_char_confidence: float    # weakest character - flags partial reads
    variant: str                  # which preprocessing variant won
    is_valid_format: bool
    repaired: bool
    corrections: list = field(default_factory=list)
    char_confidences: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=2)
def _load(model_name: str = "cct-s-v2-global-model"):
    enable_gpu_dlls()          # must precede onnxruntime session creation
    from fast_plate_ocr import LicensePlateRecognizer
    return LicensePlateRecognizer(
        model_name,
        device="cuda" if device() == "cuda" else "cpu",
        providers=onnx_providers(),
    )


def _clean(text: str) -> str:
    return "".join(c for c in (text or "") if c not in PAD_CHARS).strip()


def _score(pred, text: str) -> tuple[float, float, list]:
    """Mean/min character confidence over the non-padding characters."""
    probs = np.asarray(getattr(pred, "char_probs", []), dtype=float).ravel()
    n = len(text)
    if probs.size == 0 or n == 0:
        return 0.0, 0.0, []
    used = probs[:n] if probs.size >= n else probs
    return float(used.mean()), float(used.min()), [round(float(x), 4) for x in used]


class PlateOCR:
    """Runs OCR across preprocessing variants and returns the best reading."""

    def __init__(self, model_name: str = "cct-s-v2-global-model",
                 variants=pp.DEFAULT_VARIANTS, consensus: bool = True):
        self.model_name = model_name
        self.variants = tuple(variants)
        self.consensus = consensus
        self._engine = _load(model_name)
        # Failures are COUNTED, never swallowed silently. A pipeline that reads
        # zero plates must be able to say whether that is because there were no
        # plates or because the OCR engine was failing on every call.
        self.failures = 0
        self.last_error = None
        self.calls = 0

    def read(self, plate_img: np.ndarray) -> OCRResult | None:
        if plate_img is None or plate_img.size == 0:
            return None
        rendered = pp.build_variants(plate_img, names=self.variants)
        names = list(rendered.keys())
        images = [rendered[n] for n in names]

        self.calls += 1
        try:
            preds = self._engine.run(images, return_confidence=True)
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            if self.failures in (1, 10, 100) or self.failures % 500 == 0:
                import warnings
                warnings.warn(f"OCR engine failed {self.failures}x "
                              f"(latest: {self.last_error})", RuntimeWarning)
            return None
        if not preds:
            return None
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        scored = []
        for name, pred in zip(names, preds):
            raw = _clean(getattr(pred, "plate", ""))
            if not raw:
                continue
            mean_c, min_c, chars = _score(pred, raw)
            v: PlateValidation = validate(raw)
            # A grammar-valid reading is preferred over a marginally more
            # confident but malformed one; format_confidence encodes repair cost.
            rank = mean_c * (0.55 + 0.45 * v.format_confidence) + (0.05 if v.is_valid else 0.0)
            scored.append((rank, name, OCRResult(
                text_raw=raw, text_final=v.final, confidence=round(mean_c, 4),
                min_char_confidence=round(min_c, 4), variant=name,
                is_valid_format=v.is_valid, repaired=v.repaired,
                corrections=v.corrections, char_confidences=chars)))
        if not scored:
            return None

        # CROSS-VARIANT CONSENSUS. Preprocessing variants are independent views
        # of the same plate, so a string that several variants agree on is more
        # trustworthy than one variant that happens to be confident. Without
        # this, a single confidently-wrong variant can outrank the correct
        # reading (measured on motion-blurred crops).
        if self.consensus and len(scored) > 1:
            agree = {}
            for rank, _n, r in scored:
                agree[r.text_final] = agree.get(r.text_final, 0.0) + rank
            best_text = max(agree.items(), key=lambda kv: kv[1])[0]
            supporting = [s for s in scored if s[2].text_final == best_text]
            top = max(supporting, key=lambda s: s[0])
            res = top[2]
            # Confidence is tempered by how many variants agreed.
            res.variant = f"{top[1]}+{len(supporting)}/{len(scored)}"
            res.confidence = round(res.confidence * (0.80 + 0.20 * len(supporting) / len(scored)), 4)
            return res

        return max(scored, key=lambda s: s[0])[2]

    def _rank_variants(self, names: list, preds: list) -> OCRResult | None:
        """Score one plate's preprocessing variants and pick the winner.

        Shared by `read` and `read_many` so the single-plate and batched paths
        can never drift apart in how they choose a reading.
        """
        scored = []
        for name, pred in zip(names, preds):
            raw = _clean(getattr(pred, "plate", ""))
            if not raw:
                continue
            mean_c, min_c, chars = _score(pred, raw)
            v: PlateValidation = validate(raw)
            rank = mean_c * (0.55 + 0.45 * v.format_confidence) + (0.05 if v.is_valid else 0.0)
            scored.append((rank, name, OCRResult(
                text_raw=raw, text_final=v.final, confidence=round(mean_c, 4),
                min_char_confidence=round(min_c, 4), variant=name,
                is_valid_format=v.is_valid, repaired=v.repaired,
                corrections=v.corrections, char_confidences=chars)))
        if not scored:
            return None

        if self.consensus and len(scored) > 1:
            agree = {}
            for rank, _n, r in scored:
                agree[r.text_final] = agree.get(r.text_final, 0.0) + rank
            best_text = max(agree.items(), key=lambda kv: kv[1])[0]
            supporting = [s for s in scored if s[2].text_final == best_text]
            top = max(supporting, key=lambda s: s[0])
            res = top[2]
            res.variant = f"{top[1]}+{len(supporting)}/{len(scored)}"
            res.confidence = round(res.confidence * (0.80 + 0.20 * len(supporting) / len(scored)), 4)
            return res
        return max(scored, key=lambda s: s[0])[2]

    def read_many(self, plate_images: list) -> list:
        """OCR several plates in ONE engine call.

        `read()` issues one call per plate. Across a camera network that is the
        dominant per-frame cost, because each call carries fixed dispatch
        overhead and leaves the accelerator mostly idle. Here every variant of
        every plate is flattened into a single batch, run once, and scattered
        back to its plate, which is the same arithmetic the detector already
        benefits from.

        Returns a list aligned with `plate_images`; entries are None where the
        crop was unusable or the engine returned nothing.
        """
        out: list = [None] * len(plate_images)
        flat: list = []
        spans: list = []          # (index, variant names, slice start, slice end)
        for i, img in enumerate(plate_images):
            if img is None or getattr(img, "size", 0) == 0:
                continue
            rendered = pp.build_variants(img, names=self.variants)
            names = list(rendered.keys())
            if not names:
                continue
            start = len(flat)
            flat.extend(rendered[n] for n in names)
            spans.append((i, names, start, len(flat)))

        if not flat:
            return out

        self.calls += len(spans)
        try:
            preds = self._engine.run(flat, return_confidence=True)
        except Exception as e:
            self.failures += len(spans)
            self.last_error = f"{type(e).__name__}: {e}"
            if self.failures in (1, 10, 100) or self.failures % 500 == 0:
                import warnings
                warnings.warn(f"OCR engine failed {self.failures}x "
                              f"(latest: {self.last_error})", RuntimeWarning)
            return out
        if not preds:
            return out
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        for idx, names, a, b in spans:
            chunk = preds[a:b]
            if len(chunk) != len(names):
                continue
            out[idx] = self._rank_variants(names, chunk)
        return out

    def read_batch(self, plate_images: list) -> list:
        return self.read_many(plate_images)


@lru_cache(maxsize=1)
def get_ocr() -> PlateOCR:
    return PlateOCR()
