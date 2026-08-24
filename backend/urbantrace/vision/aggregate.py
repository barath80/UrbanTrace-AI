"""Temporal OCR aggregation.

A vehicle is seen over many frames, so a single OCR reading is never trusted.
Two independent consensus methods are combined:

  1. STRING VOTE     - confidence-weighted vote over whole plate strings.
  2. CHARACTER VOTE  - per-position weighted vote using the per-character
                       probabilities the OCR model emits.

Character voting recovers plates that NO single frame read correctly (frame A
gets position 3 right, frame B gets position 7 right). The result is then run
through the Indian-plate grammar validator. All raw readings are retained.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict

from .plate_format import validate


@dataclass
class Reading:
    frame_number: int
    timestamp_s: float
    text_raw: str
    text_final: str
    ocr_confidence: float
    plate_confidence: float
    is_valid_format: bool
    repaired: bool
    variant: str
    char_confidences: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def weight(self) -> float:
        """Evidence weight: OCR certainty x detector certainty x grammar bonus."""
        w = self.ocr_confidence * (0.5 + 0.5 * self.plate_confidence)
        if self.is_valid_format:
            w *= 1.25
        if self.repaired:
            w *= 0.90
        return w


@dataclass
class AggregatedPlate:
    plate: str                     # final accepted identity
    confidence: float              # aggregated confidence [0,1]
    method: str                    # "string-vote" | "character-vote"
    support: int                   # readings backing this plate
    total_readings: int
    agreement: float               # support / total_readings
    is_valid_format: bool
    alternatives: list = field(default_factory=list)   # [(plate, weight)]
    first_frame: int = 0
    last_frame: int = 0
    first_timestamp_s: float = 0.0
    last_timestamp_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# How much the vote leans on consensus (summed evidence) versus peak evidence
# (the single best frame). Pure consensus fails when only a small minority of
# frames are readable: many wrong-but-plausible reads outvote one sharp frame.
# Pure peak fails when one blurred frame is confidently wrong. Blending beats
# both -- see scripts/benchmark_temporal_blur.py.
CONSENSUS_WEIGHT = 0.55
PEAK_CONFIDENCE_FLOOR = 0.90   # a read this strong is never simply outvoted


def _string_vote(readings: list) -> tuple:
    scores = defaultdict(float)
    peaks = defaultdict(float)
    counts = defaultdict(int)
    for r in readings:
        if not r.text_final:
            continue
        scores[r.text_final] += r.weight
        peaks[r.text_final] = max(peaks[r.text_final], r.ocr_confidence)
        counts[r.text_final] += 1
    if not scores:
        return None, 0.0, 0, []

    total_w = sum(scores.values()) or 1.0
    max_peak = max(peaks.values()) or 1.0

    blended = {}
    for plate, w in scores.items():
        consensus = w / total_w
        peak = peaks[plate] / max_peak
        score = CONSENSUS_WEIGHT * consensus + (1.0 - CONSENSUS_WEIGHT) * peak
        # A very sharp single frame carries its own authority.
        if peaks[plate] >= PEAK_CONFIDENCE_FLOOR:
            score += 0.15 * (peaks[plate] - PEAK_CONFIDENCE_FLOOR) / (1.0 - PEAK_CONFIDENCE_FLOOR)
        blended[plate] = score

    ranked = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)
    best = ranked[0][0]
    return (best, scores[best] / total_w, counts[best],
            [(p, round(s, 3)) for p, s in ranked[1:4]])


def _character_vote(readings: list) -> tuple:
    """Per-position weighted vote among readings of the modal plate length."""
    by_len = defaultdict(list)
    for r in readings:
        if r.text_final:
            by_len[len(r.text_final)].append(r)
    if not by_len:
        return None, 0.0
    # Modal length weighted by evidence, not raw count.
    target_len = max(by_len, key=lambda L: sum(x.weight for x in by_len[L]))
    group = by_len[target_len]
    if len(group) < 2:
        return None, 0.0

    out, pos_conf = [], []
    for i in range(target_len):
        votes = defaultdict(float)
        for r in group:
            ch = r.text_final[i]
            # Prefer the model's own per-character probability when available.
            cw = r.char_confidences[i] if i < len(r.char_confidences) else r.ocr_confidence
            votes[ch] += r.weight * cw
        ch, w = max(votes.items(), key=lambda kv: kv[1])
        total = sum(votes.values()) or 1.0
        out.append(ch)
        pos_conf.append(w / total)
    consensus = "".join(out)
    return consensus, sum(pos_conf) / len(pos_conf) if pos_conf else 0.0


def aggregate(readings: list, min_readings: int = 1) -> AggregatedPlate | None:
    """Fuse a track's readings into one plate identity."""
    readings = [r for r in readings if r and r.text_final]
    if len(readings) < min_readings or not readings:
        return None

    s_plate, s_conf, s_support, alts = _string_vote(readings)
    c_plate, c_conf = _character_vote(readings)

    # Character consensus is adopted only when it is grammar-valid and the
    # string vote is either weak or disagrees while being invalid.
    chosen, method, conf = s_plate, "string-vote", s_conf
    if c_plate and c_plate != s_plate:
        cv, sv = validate(c_plate), validate(s_plate or "")
        if cv.is_valid and (not sv.is_valid or c_conf > s_conf):
            chosen, method, conf = c_plate, "character-vote", c_conf

    if not chosen:
        return None
    v = validate(chosen)
    final = v.final
    support = sum(1 for r in readings if r.text_final == final)

    # Blend consensus strength with the mean OCR confidence of supporting reads.
    backing = [r.ocr_confidence for r in readings if r.text_final == final] or \
              [r.ocr_confidence for r in readings]
    mean_conf = sum(backing) / len(backing)
    final_conf = round(min(1.0, 0.5 * conf + 0.5 * mean_conf), 4)

    return AggregatedPlate(
        plate=final,
        confidence=final_conf,
        method=method,
        support=support,
        total_readings=len(readings),
        agreement=round(support / len(readings), 3),
        is_valid_format=v.is_valid,
        alternatives=alts,
        first_frame=min(r.frame_number for r in readings),
        last_frame=max(r.frame_number for r in readings),
        first_timestamp_s=min(r.timestamp_s for r in readings),
        last_timestamp_s=max(r.timestamp_s for r in readings),
    )
