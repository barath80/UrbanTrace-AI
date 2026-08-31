"""Cross-camera vehicle identity resolution at city scale.

The problem: OCR reads the same plate slightly differently at different
cameras -- TN01AB1234, TNO1AB1234, TN01A81234 -- and the platform has to decide
which readings are the same vehicle.

The batch pipeline answers this by comparing every plate with every other
plate. That is O(n^2): fine for the 5 plates in a test clip, hopeless for a
city. At 100,000 plates a day it is 5 billion comparisons.

This module answers the same question in time proportional to the number of
plausible candidates, using **blocking**. Every plate is reduced to one or more
`blocking keys` chosen so that two plates can only be similar if they share a
key. Lookup then touches a handful of candidates instead of the whole corpus,
and the expensive edit-distance check runs only on those.

Two keys are generated per plate:

  1. The canonical key -- every character mapped to the representative of its
     OCR-confusion group (O, Q and D all become 0; I and L become 1; and so
     on). Any pure substitution *within* confusion groups collapses to the same
     key, which is the overwhelming majority of real OCR disagreement.

  2. Deletion keys -- the canonical key with one character removed, at each
     position. These catch a dropped or spurious character, which is the other
     common failure and the one a substitution-only key would miss.

A candidate that shares a key is then verified properly: confusion-weighted
edit similarity above threshold, and a physically possible transition between
the two sightings. Sharing a key is a hypothesis, never a conclusion.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import SCALE

# Characters OCR genuinely confuses. Substituting within a group is cheap; any
# other substitution is expensive. The first character of each group is its
# canonical representative.
CONFUSION_GROUPS = ("0OQD", "1IL", "2Z", "5S", "8B", "6G", "7T", "4A", "9P")

_CANON: dict[str, str] = {}
for _g in CONFUSION_GROUPS:
    for _c in _g:
        _CANON[_c] = _g[0]


def canonical_key(plate: str) -> str:
    """Fold a plate onto its OCR-confusion equivalence class."""
    return "".join(_CANON.get(c, c) for c in (plate or "").upper())


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return 0.35 if _CANON.get(a, a) == _CANON.get(b, b) else 1.0


def plate_similarity(p1: str, p2: str) -> float:
    """Confusion-weighted normalised edit similarity in [0, 1]."""
    if not p1 or not p2:
        return 0.0
    if p1 == p2:
        return 1.0
    n, m = len(p1), len(p2)
    # Length alone can rule a pair out before any matrix is built.
    if abs(n - m) / max(n, m) > (1.0 - SCALE.min_plate_similarity):
        return 0.0
    prev = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [float(i)] + [0.0] * m
        ci = p1[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1.0,
                         cur[j - 1] + 1.0,
                         prev[j - 1] + _sub_cost(ci, p2[j - 1]))
        prev = cur
    return max(0.0, 1.0 - prev[m] / max(n, m))


def _canonical_edit_distance(a: str, b: str, cap: int = 2) -> int:
    """Edit distance between two ALREADY-canonicalised plates, capped.

    Because canonicalisation has folded every OCR-confusable character onto one
    representative, any substitution that survives it is a genuine difference
    between two plates, not a misread.
    """
    n, m = len(a), len(b)
    if abs(n - m) > cap:
        return cap + 1
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[m]


def same_vehicle(p1: str, p2: str) -> bool:
    """Whether two readings may be treated as the same plate.

    Similarity alone is too permissive to decide this. On a ten-character
    plate a threshold of 0.82 tolerates nearly two full character edits, so
    TN01AB1234 and TN01AB1235 -- two different vehicles -- score 0.90 and would
    be merged. Merging two vehicles into one identity is the worst error this
    system can make: it puts one owner's journey under another's plate.

    So the rule is stricter and stated in terms of what OCR actually gets
    wrong:

      * same length  -> the canonical forms must match EXACTLY. Every
        difference has to be explainable as a confusion-group misread
        (O/0, I/1, B/8 ...). A surviving substitution means a real difference.
      * length differs by one -> one canonical form must be the other with a
        single character removed, which is a dropped or spurious glyph.
      * any greater difference -> not the same vehicle.
    """
    if not p1 or not p2:
        return False
    if p1 == p2:
        return True
    c1, c2 = canonical_key(p1), canonical_key(p2)
    if c1 == c2:
        return True
    if abs(len(c1) - len(c2)) > 1:
        return False
    # Same length and canonical forms differ => a real substitution.
    if len(c1) == len(c2):
        return False
    return _canonical_edit_distance(c1, c2, cap=1) <= 1


def blocking_keys(plate: str) -> list[str]:
    """Keys under which a plate is indexed and looked up.

    Deletion keys are only generated for plates long enough that removing a
    character still leaves something distinctive; on very short strings they
    produce keys that match almost anything.
    """
    canon = canonical_key(plate)
    keys = [canon]
    if len(canon) >= 5:
        seen = {canon}
        for i in range(len(canon)):
            k = canon[:i] + canon[i + 1:]
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


@dataclass
class Sighting:
    """One confirmed plate reading at one camera at one time."""
    plate: str
    camera_id: str
    observed_at: datetime
    confidence: float
    track_id: int
    latitude: float
    longitude: float
    vehicle_type: str = "vehicle"
    support: int = 0
    total_readings: int = 0
    is_valid_format: bool = False
    site_id: str = ""

    def key(self) -> tuple:
        return (self.camera_id, self.track_id)


@dataclass
class Cluster:
    """A set of sightings judged to be one physical vehicle."""
    cluster_id: str
    plate: str                                  # canonical spelling
    sightings: list = field(default_factory=list)
    spellings: dict = field(default_factory=dict)   # spelling -> summed confidence
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    def cameras(self) -> list[str]:
        seen, out = set(), []
        for s in self.sightings:
            if s.camera_id not in seen:
                seen.add(s.camera_id)
                out.append(s.camera_id)
        return out

    def mean_confidence(self) -> float:
        return (sum(s.confidence for s in self.sightings) / len(self.sightings)
                if self.sightings else 0.0)


class IdentityIndex:
    """Blocking-based identity resolution.

    Thread-safe. `add()` is the only mutating entry point and is the operation
    on the hot ingest path, so it is kept to: build keys, gather candidates,
    verify, merge.
    """

    def __init__(self, feasible=None):
        """`feasible(a, b) -> bool` decides whether two sightings could be the
        same vehicle physically. Injected so this module stays independent of
        the camera graph; when omitted, only plate similarity is used."""
        self._lock = threading.RLock()
        self._clusters: dict[str, Cluster] = {}
        self._by_key: dict[str, set[str]] = {}     # blocking key -> cluster ids
        self._by_track: dict[tuple, str] = {}      # (camera, track) -> cluster id
        self._seq = 0
        self._feasible = feasible
        self.comparisons = 0                       # candidates actually scored
        self.additions = 0

    # ------------------------------------------------------------ add

    def add(self, s: Sighting) -> str:
        """Insert a sighting and return the cluster id it belongs to."""
        with self._lock:
            self.additions += 1

            # A track that has already been published simply updates in place;
            # re-reading the same vehicle must not create a second identity.
            existing = self._by_track.get(s.key())
            if existing is not None and existing in self._clusters:
                self._absorb(self._clusters[existing], s, replace_track=True)
                return existing

            keys = blocking_keys(s.plate)
            candidates: set[str] = set()
            for k in keys:
                candidates |= self._by_key.get(k, set())

            best_id, best_sim = None, 0.0
            for cid in candidates:
                c = self._clusters.get(cid)
                if c is None:
                    continue
                self.comparisons += 1
                if not same_vehicle(s.plate, c.plate):
                    continue
                sim = plate_similarity(s.plate, c.plate)
                if sim <= best_sim:
                    continue
                if self._feasible is not None and not self._compatible(c, s):
                    continue
                best_id, best_sim = cid, sim

            if best_id is None:
                self._seq += 1
                cid = f"GV_{self._seq:08d}"
                cluster = Cluster(cluster_id=cid, plate=s.plate)
                self._clusters[cid] = cluster
                self._absorb(cluster, s)
                self._index(cluster, keys)
                return cid

            self._absorb(self._clusters[best_id], s)
            self._index(self._clusters[best_id], keys)
            return best_id

    def _compatible(self, c: Cluster, s: Sighting) -> bool:
        """At least one existing sighting must permit this transition.

        Only the most recent sightings are checked: a vehicle's compatibility
        with a new sighting is decided by where it was last, not by its whole
        history, and bounding the check keeps `add()` constant-time.
        """
        for prev in reversed(c.sightings[-8:]):
            if prev.camera_id == s.camera_id:
                return True
            a, b = (prev, s) if prev.observed_at <= s.observed_at else (s, prev)
            if self._feasible(a, b):
                return True
        return False

    def _absorb(self, c: Cluster, s: Sighting, replace_track: bool = False) -> None:
        if replace_track:
            c.sightings = [x for x in c.sightings if x.key() != s.key()]
        c.sightings.append(s)
        c.sightings.sort(key=lambda x: x.observed_at)
        c.spellings[s.plate] = c.spellings.get(s.plate, 0.0) + s.confidence
        # The canonical spelling is the one with the most total evidence, not
        # the single most confident frame: one lucky misread should not rename
        # a vehicle that many other sightings agree on.
        c.plate = max(c.spellings.items(), key=lambda kv: kv[1])[0]
        c.first_seen = c.sightings[0].observed_at
        c.last_seen = c.sightings[-1].observed_at
        self._by_track[s.key()] = c.cluster_id

    def _index(self, c: Cluster, keys: list[str]) -> None:
        for k in set(keys) | set(blocking_keys(c.plate)):
            self._by_key.setdefault(k, set()).add(c.cluster_id)

    # -------------------------------------------------------- querying

    def get(self, cluster_id: str) -> Cluster | None:
        with self._lock:
            return self._clusters.get(cluster_id)

    def find(self, plate: str) -> list[Cluster]:
        """Clusters whose canonical plate matches or nearly matches."""
        with self._lock:
            out, seen = [], set()
            for k in blocking_keys(plate):
                for cid in self._by_key.get(k, ()):  # noqa: SIM118
                    if cid in seen:
                        continue
                    seen.add(cid)
                    c = self._clusters.get(cid)
                    if c and same_vehicle(plate, c.plate):
                        out.append(c)
            out.sort(key=lambda c: -len(c.sightings))
            return out

    def clusters(self, multi_camera_only: bool = False) -> list[Cluster]:
        with self._lock:
            cs = list(self._clusters.values())
        if multi_camera_only:
            cs = [c for c in cs if len(set(s.camera_id for s in c.sightings)) > 1]
        cs.sort(key=lambda c: (-len(set(s.camera_id for s in c.sightings)),
                               -c.mean_confidence()))
        return cs

    # --------------------------------------------------------- upkeep

    def evict_before(self, cutoff: datetime) -> int:
        """Drop clusters whose last sighting predates `cutoff`.

        Without this the index grows for the lifetime of the process. Evicted
        clusters remain in the database; this only bounds the in-memory
        resolution window.
        """
        with self._lock:
            dead = [cid for cid, c in self._clusters.items()
                    if c.last_seen is not None and c.last_seen < cutoff]
            for cid in dead:
                c = self._clusters.pop(cid)
                for s in c.sightings:
                    self._by_track.pop(s.key(), None)
            if dead:
                dead_set = set(dead)
                empty = []
                for k, ids in self._by_key.items():
                    ids -= dead_set
                    if not ids:
                        empty.append(k)
                for k in empty:
                    del self._by_key[k]
            return len(dead)

    def evict_expired(self, now: datetime | None = None) -> int:
        now = now or datetime.now(tz=None)
        return self.evict_before(now - timedelta(minutes=SCALE.identity_ttl_minutes))

    def stats(self) -> dict:
        with self._lock:
            multi = sum(1 for c in self._clusters.values()
                        if len(set(s.camera_id for s in c.sightings)) > 1)
            return {
                "clusters": len(self._clusters),
                "multi_camera_clusters": multi,
                "sightings": sum(len(c.sightings) for c in self._clusters.values()),
                "blocking_keys": len(self._by_key),
                "candidate_comparisons": self.comparisons,
                "additions": self.additions,
                "comparisons_per_add": round(self.comparisons / self.additions, 2)
                if self.additions else 0.0,
            }
