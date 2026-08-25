"""Unique-vehicle counting that survives contact with real traffic video.

Counting "how many vehicles are in this video" by tallying distinct tracker IDs
is wrong in three independent ways, all of which inflate the number:

  1. FRAGMENTATION. A tracker drops a vehicle behind an occluder for a few
     frames and re-acquires it under a fresh id. One vehicle, two counts.
  2. CLUTTER. A detector fires on 40x40px far-field blobs and on vehicles
     parked at the kerb. Neither is traffic passing the camera.
  3. FLICKER. A one-to-three frame spurious detection gets an id like anything
     else, and a low persistence gate lets it straight through.

This module addresses each: lost tracks are stitched back together by motion
prediction, tracks are gated on persistence / size / motion, and the class is
decided by an area-weighted vote so a distant blob's guess cannot outvote the
frames where the vehicle was close enough to classify.

Two counting modes, because "how many vehicles" has two honest answers at an
intersection where half the vehicles in shot are parked:

  TRAFFIC  - vehicles that actually travel through the scene (the traffic
             engineering number). Parked and stationary vehicles excluded.
  PRESENCE - every distinct real vehicle that appears, moving or not.

Both dedupe and both reject clutter; they differ only on the motion gate. The
counter reports both so the number is never quietly one when you meant the
other. When a counting line is configured, TRAFFIC additionally requires the
vehicle to cross it, which is the most defensible definition available: it is
checkable by watching the video.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

# Classes shown in the counter, in display order.
COUNT_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle")


@dataclass
class CountConfig:
    """Thresholds are fractions of frame size so they port across resolutions."""

    # A track is real if it is EITHER big enough to be unambiguous OR
    # persistent enough that noise cannot explain it. Requiring both at once
    # discards a distant motorcycle tracked for 174 straight frames and a
    # truck that enters two seconds before the clip ends - both plainly real.
    #
    # Ordinary acceptance: this many frames AND this median box area
    # (fraction of frame area; 0.0012 of 1280x720 is ~1100px, a 33x33 box).
    min_frames: int = 8
    min_median_area_frac: float = 0.0012

    # Either alone is also sufficient: a track held this long is not noise,
    # and a box this large is not a far-field artefact.
    persistent_frames: int = 25
    clear_area_frac: float = 0.005

    # Hard floors nothing passes: below these there is no evidence at all.
    floor_frames: int = 3
    floor_area_frac: float = 0.0004

    # Motion (TRAFFIC mode). Pixel displacement scales with 1/depth, so a flat
    # pixel threshold quietly discards all far-field traffic: a distant car
    # genuinely crossing the scene moves fewer pixels than a nearby parked one
    # jitters. Motion is therefore measured in BODY LENGTHS - net displacement
    # over sqrt(median box area) - which is depth-invariant. The pixel floor
    # only exists to beat box jitter on the smallest boxes.
    min_displacement_bodies: float = 1.0
    min_displacement_frac: float = 0.02

    # Fragment stitching: how long a lost track stays claimable, and how close
    # a new track must appear to a lost track's predicted position.
    stitch_max_gap_frames: int = 30
    stitch_max_dist_frac: float = 0.055
    stitch_max_area_ratio: float = 3.0

    # Optional counting line in NORMALISED coords ((x1,y1),(x2,y2)), 0..1.
    # When set, TRAFFIC counts only vehicles whose centre crosses it.
    line: tuple | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CountedVehicle:
    """One deduped vehicle - the unit the counter actually counts."""
    vehicle_id: int
    vehicle_type: str
    first_frame: int
    last_frame: int
    frames_seen: int
    raw_track_ids: list = field(default_factory=list)
    displacement_px: float = 0.0
    median_area_px: float = 0.0
    crossed_line: bool = False
    counted_traffic: bool = False
    counted_presence: bool = False
    # The frame at which this vehicle actually became countable - it crossed
    # the line, or it had travelled far enough. A running on-screen total
    # should increment here, not at the start or end of the track.
    counted_at_frame: int | None = None
    reject_reason: str = ""
    # Frame + box where the vehicle was largest: the best evidence crop.
    best_frame: int = 0
    best_box: tuple = (0, 0, 0, 0)

    def to_dict(self) -> dict:
        return asdict(self)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _first_frame_past(tr, need: float):
    """The frame at which the track first got `need` px from where it started."""
    start = tr.pos[0]
    for p, f in zip(tr.pos, tr.frames):
        if math.dist(start, p) >= need:
            return f
    return tr.frames[-1] if tr.frames else None


def _side(pt, a, b) -> float:
    """Signed side of the line a->b that pt lies on."""
    return (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])


class _Track:
    __slots__ = ("cid", "raw_ids", "first", "last", "n", "pos", "frames",
                 "areas", "votes", "best_area", "best_frame", "best_box",
                 "crossed", "crossed_at", "last_side")

    def __init__(self, cid: int, raw_id: int, frame_no: int):
        self.cid = cid
        self.raw_ids = [raw_id]
        self.first = self.last = frame_no
        self.n = 0
        self.pos: list = []          # centre history
        self.frames: list = []       # frame number for each entry in pos
        self.areas: list = []
        self.votes: dict = {}        # cls -> area-weighted confidence
        self.best_area = -1.0
        self.best_frame = frame_no
        self.best_box = (0, 0, 0, 0)
        self.crossed = False
        self.crossed_at = None
        self.last_side = None

    def velocity(self, lookback: int = 5):
        if len(self.pos) < 2:
            return (0.0, 0.0)
        span = min(lookback, len(self.pos) - 1)
        a = self.pos[-1 - span]
        b = self.pos[-1]
        return ((b[0] - a[0]) / span, (b[1] - a[1]) / span)

    def displacement(self) -> float:
        if len(self.pos) < 2:
            return 0.0
        return math.dist(self.pos[0], self.pos[-1])

    def cls_name(self) -> str:
        if not self.votes:
            return "unknown"
        return max(self.votes.items(), key=lambda kv: kv[1])[0]


class VehicleCounter:
    """Feed it per-frame tracks; ask it for counts.

    Tracks carry `.track_id`, `.cls_name`, `.confidence`, `.box` (x1,y1,x2,y2).
    """

    def __init__(self, frame_w: int, frame_h: int, config: CountConfig | None = None):
        self.cfg = config or CountConfig()
        self.w, self.h = frame_w, frame_h
        self.diag = math.hypot(frame_w, frame_h)
        self.frame_area = float(frame_w * frame_h)

        self._tracks: dict = {}      # canonical id -> _Track
        self._alias: dict = {}       # raw tracker id -> canonical id
        self._next_cid = 1
        self.stitches = 0            # fragments re-joined; a quality signal

        self.line = None
        if self.cfg.line:
            (x1, y1), (x2, y2) = self.cfg.line
            self.line = ((x1 * frame_w, y1 * frame_h), (x2 * frame_w, y2 * frame_h))

    # ---------------- ingest ----------------

    def update(self, tracks, frame_no: int) -> None:
        for t in tracks:
            box = tuple(float(v) for v in t.box)
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))

            cid = self._alias.get(t.track_id)
            if cid is None:
                cid = self._claim(t.track_id, (cx, cy), area, frame_no)

            tr = self._tracks[cid]
            tr.last = frame_no
            tr.n += 1
            tr.pos.append((cx, cy))
            tr.frames.append(frame_no)
            tr.areas.append(area)
            # Area-weighted vote: the frames where the vehicle is close are the
            # frames where the classifier is worth believing.
            tr.votes[t.cls_name] = tr.votes.get(t.cls_name, 0.0) + \
                area * float(getattr(t, "confidence", 1.0) or 1.0)
            if area > tr.best_area:
                tr.best_area, tr.best_frame, tr.best_box = area, frame_no, box

            if self.line is not None:
                s = _side((cx, cy), *self.line)
                if tr.last_side is not None and s != 0 and \
                        (s > 0) != (tr.last_side > 0):
                    tr.crossed = True
                if s != 0:
                    tr.last_side = s

    def _claim(self, raw_id: int, centre, area: float, frame_no: int) -> int:
        """Attach a new tracker id to a lost track if it plausibly IS that track."""
        best_cid, best_d = None, float("inf")
        max_d = self.cfg.stitch_max_dist_frac * self.diag
        for cid, tr in self._tracks.items():
            gap = frame_no - tr.last
            # gap <= 0 means that track is still live this frame; it cannot
            # also be this new id.
            if gap <= 0 or gap > self.cfg.stitch_max_gap_frames:
                continue
            ref_area = _median(tr.areas)
            ratio = max(area / ref_area, ref_area / area)
            if ratio > self.cfg.stitch_max_area_ratio:
                continue
            vx, vy = tr.velocity()
            px, py = tr.pos[-1][0] + vx * gap, tr.pos[-1][1] + vy * gap
            d = math.dist(centre, (px, py))
            # Allow more slack the longer the gap: prediction degrades.
            if d < min(best_d, max_d * (1.0 + gap / self.cfg.stitch_max_gap_frames)):
                best_cid, best_d = cid, d

        if best_cid is not None:
            self._alias[raw_id] = best_cid
            self._tracks[best_cid].raw_ids.append(raw_id)
            self.stitches += 1
            return best_cid

        cid = self._next_cid
        self._next_cid += 1
        self._tracks[cid] = _Track(cid, raw_id, frame_no)
        self._alias[raw_id] = cid
        return cid

    def canonical_id(self, raw_track_id: int):
        """The counted vehicle a tracker id belongs to, or None if unseen.

        Several tracker ids map to one vehicle wherever the tracker fragmented,
        so this is the id to show on screen: labelling boxes with raw tracker
        ids contradicts the count standing next to them.
        """
        return self._alias.get(raw_track_id)

    # ---------------- results ----------------

    def vehicles(self) -> list:
        """Every canonical track, with its gate verdicts - counted or not.

        Rejected tracks are returned too, each with a reason, so a count can be
        audited instead of taken on faith.
        """
        cfg = self.cfg
        min_area = cfg.min_median_area_frac * self.frame_area
        clear_area = cfg.clear_area_frac * self.frame_area
        floor_area = cfg.floor_area_frac * self.frame_area
        min_disp = cfg.min_displacement_frac * self.diag
        out = []
        for cid, tr in sorted(self._tracks.items()):
            med = _median(tr.areas)
            disp = tr.displacement()
            v = CountedVehicle(
                vehicle_id=cid, vehicle_type=tr.cls_name(),
                first_frame=tr.first, last_frame=tr.last, frames_seen=tr.n,
                raw_track_ids=list(tr.raw_ids),
                displacement_px=round(disp, 1), median_area_px=round(med, 1),
                crossed_line=tr.crossed,
                best_frame=tr.best_frame,
                best_box=tuple(round(b, 1) for b in tr.best_box),
            )
            if tr.n < cfg.floor_frames:
                v.reject_reason = (f"seen {tr.n} frames < floor "
                                   f"{cfg.floor_frames}")
            elif med < floor_area:
                v.reject_reason = (f"median area {med:.0f}px < floor "
                                   f"{floor_area:.0f}px")
            elif not (tr.n >= cfg.persistent_frames or med >= clear_area
                      or (tr.n >= cfg.min_frames and med >= min_area)):
                v.reject_reason = (
                    f"too weak: {tr.n} frames / {med:.0f}px - needs "
                    f"{cfg.persistent_frames}+ frames, or {clear_area:.0f}px+, "
                    f"or {cfg.min_frames}+ frames at {min_area:.0f}px+")
            else:
                v.counted_presence = True
                if self.line is not None:
                    v.counted_traffic = tr.crossed
                    v.counted_at_frame = tr.crossed_at
                    if not tr.crossed:
                        v.reject_reason = "never crossed the counting line"
                else:
                    need = max(min_disp, cfg.min_displacement_bodies
                               * math.sqrt(med))
                    v.counted_traffic = disp >= need
                    if v.counted_traffic:
                        v.counted_at_frame = _first_frame_past(tr, need)
                    if not v.counted_traffic:
                        v.reject_reason = (
                            f"stationary: moved {disp:.0f}px "
                            f"({disp / math.sqrt(med):.2f} body lengths) "
                            f"< {need:.0f}px")
            out.append(v)
        return out

    def counts(self, mode: str = "traffic") -> dict:
        """{class: n, ...} plus 'total', for the given mode."""
        field_name = "counted_traffic" if mode == "traffic" else "counted_presence"
        counts = {c: 0 for c in COUNT_CLASSES}
        other = 0
        for v in self.vehicles():
            if not getattr(v, field_name):
                continue
            if v.vehicle_type in counts:
                counts[v.vehicle_type] += 1
            else:
                other += 1
        counts["total"] = sum(counts[c] for c in COUNT_CLASSES) + other
        if other:
            counts["other"] = other
        return counts

    def report(self) -> dict:
        vs = self.vehicles()
        return {
            "mode_default": "traffic",
            "traffic": self.counts("traffic"),
            "presence": self.counts("presence"),
            "raw_tracker_ids": len(self._alias),
            "canonical_tracks": len(self._tracks),
            "fragments_stitched": self.stitches,
            "rejected": len([v for v in vs if not v.counted_presence]),
            "config": self.cfg.to_dict(),
        }
