# 4. Tracking and identity resolution

There are **three different questions** that are easy to confuse, and the system
answers each with a different mechanism:

| Question | Scope | Mechanism |
|---|---|---|
| Is this box the same vehicle as in the previous frame? | one camera, seconds | **ByteTrack** (motion) |
| Which track does this plate belong to? | one frame | **Containment binding** |
| Is the sighting at camera B the same vehicle as at camera A? | many cameras, minutes–hours | **Plate identity** + **spatio-temporal feasibility** |

## 4.1 Why ByteTrack

ByteTrack associates detections to tracks by IoU against Kalman-predicted boxes using
the Hungarian algorithm, in **two passes**: first high-confidence detections, then the
*low-confidence* leftovers against still-unmatched tracks. Those low-confidence boxes
are usually a vehicle that is partly occluded or blurred — exactly the case where
other trackers lose the track and later re-create it under a new id.

Reasons it fits this system **[design rationale]**:

1. **No appearance network.** DeepSORT/StrongSORT run a re-ID CNN on every detection.
   With one shared detector serving many cameras, that would add GPU work proportional
   to *total detections across the whole city*. ByteTrack's per-camera cost is a
   Kalman filter and an IoU matrix on the CPU.
2. **Re-ID is not what links cameras here.** Cross-camera identity comes from the
   *plate*, which is far more discriminative than the appearance of two cars of the
   same colour. The tracker only has to keep one camera's boxes consistent for the few
   seconds a vehicle is in view.
3. **Per-camera state is tiny.** A `BYTETracker` is a few kilobytes, so 10,000 of them
   cost less memory than one extra copy of the detector.
4. **Fixed cameras.** Camera-motion compensation (BoT-SORT) buys nothing.
5. **Pinned behaviour.** Parameters are explicit in `BYTETRACK_ARGS`:

| Parameter | Value | Meaning |
|---|---|---|
| `track_high_thresh` | 0.25 | first-pass detections |
| `track_low_thresh` | 0.10 | second-pass (occluded/blurred) detections |
| `new_track_thresh` | 0.25 | confidence needed to start a track |
| `track_buffer` | 30 | frames a lost track is remembered |
| `match_thresh` | 0.80 | IoU-cost matching threshold |
| `fuse_score` | True | fuse detection score into the cost |

**Failure modes, stated.** ByteTrack uses motion only, so it can swap ids when two
vehicles cross closely and cannot re-acquire a vehicle that is lost for longer than
`track_buffer`. Both cause *fragmentation* (one vehicle, several ids). The batch counter
(`vision/counting.py`) corrects this for counting by stitching lost tracks with motion
prediction, and the temporal vote tolerates it because each fragment can still produce
the same plate.

### The id-reset defect

See document 1 §1.3(b). Across a network, ids must be unique. The stock library resets
a *class-level* id counter every time a tracker is constructed; the fix serialises
allocation and neutralises the reset so ids are monotonic across all cameras.

### Resilience

If a camera's tracker raises, it is replaced with a fresh one and the camera continues
(identities restart — visible in the data — rather than the feed silently dying). Idle
trackers are evicted after 900 s; a hard cap of 20,000 cameras evicts the least recently
used.

## 4.2 Binding a plate to a track

The plate detector finds plates independently of vehicles. `bind_plates_to_tracks`
assigns each plate to the **smallest track box that contains ≥ 60 % of the plate's
area**. "Smallest" resolves the common case of a plate sitting inside both a car box and
an overlapping bus box behind it. Plates matching no track are returned with `None` so
the caller decides what to do rather than losing them silently.

The precondition is measurable: on the held-out plate images only **50.6 %** of plates
fell inside a detected vehicle box (document 3, §3.3). Those images are close-crops;
road-camera footage is far friendlier, but a plate the vehicle detector missed *can* be
read yet not attached to a journey.

## 4.3 From frames to a Sighting

Per track (`InferencePool`):

1. Plate crop with margin, only if the plate is ≥ `UT_MIN_PLATE_PX` (480) pixels in
   area — below that, glyph detail is not recoverable.
2. Several preprocessing variants (raw, CLAHE, sharpen) are flattened across the whole
   batch into **one** OCR call.
3. Each reading is grammar-validated (document 3) and appended to the track's buffer
   (at most `UT_MAX_READINGS` = 14).
4. After `UT_MIN_READINGS_PUB` (3) readings, `aggregate()` votes. A sighting is
   emitted only if the vote clears `UT_MIN_PUB_CONF` (0.60) **and improves on the last
   published confidence** — so one track emits one sighting that gets better, not a
   stream of near-duplicates.
5. A track read at ≥ 0.93 confidence is *settled* and stops consuming OCR.

## 4.4 Plate-grammar validation

Indian plates are `SS DD AA NNNN` (state, district, series, number), with variants
(Bharat series `DD BH NNNN AA`, short forms). `plate_format.py` validates against the
real state/UT code list and, **only where the grammar forces the character class**,
repairs OCR confusions: a letter in a digit slot is mapped (`O→0`, `I→1`, `B→8`, `S→5`…)
and a digit in a letter slot the reverse. It **never invents a character**, caps repairs
at 2, lowers confidence per repair and always keeps the raw string. Measured effect on
the test crops: format-valid share 0.787; raw exact 0.688 → final 0.713.

## 4.5 Cross-camera identity: the blocking index

The problem: OCR reads one plate differently at different cameras (`TN01AB1234`,
`TNO1AB1234`, `TN01A81234`).

**Confusion groups** (first character is the canonical representative):
`0OQD`, `1IL`, `2Z`, `5S`, `8B`, `6G`, `7T`, `4A`, `9P`.

**Blocking keys per plate:** (1) the canonical form — every character mapped to its
group representative; (2) for plates of ≥ 5 characters, the canonical form with one
character deleted at each position (catches a dropped/spurious glyph). Two plates can
only match if they share a key, so a lookup touches a handful of candidates instead of
the whole corpus. Measured: **0.83 comparisons per insert, constant**, 2,000 → 40,000
vehicles.

**Merge rule (`same_vehicle`)**, precision-first:

| Case | Decision |
|---|---|
| identical | same |
| canonical forms equal | same (every difference is a confusion-group misread) |
| same length, canonical forms differ | **different** (a real substitution survives) |
| length differs by 1 and one canonical form is the other minus one character | same (dropped/spurious glyph) |
| anything else | different |

Why not plain edit-distance similarity: on a 10-character plate a 0.82 threshold
tolerates ~two edits, so `TN01AB1234` and `TN01AB1235` (two cars, similarity 0.90)
would merge. Merging two vehicles is the worst error; splitting one vehicle in two is
recoverable. The cost, measured honestly in document 3 §3.7, is that confusable twins
merge unless the feasibility check intervenes.

**Verification before merge:** the candidate cluster is checked with `_compatible`: at
least one of the cluster's last 8 sightings must permit the transition — same camera,
or an edge whose road distance divided by elapsed time is ≤ `UT_MAX_SPEED`
(140 km/h). With no known edge the check does not veto ("do not veto on no evidence").

**Idempotence by track.** A `(camera, track)` already published updates its cluster in
place; re-reading the same vehicle never creates a second identity. The canonical spelling
of a cluster is the one with the most **total confidence**, so one lucky misread cannot
rename a vehicle that many readings agree on.

**Unique cluster ids.** Cluster ids are primary keys in a table shared by every worker, so
they are `GV_<8-hex process nonce>_<6-digit sequence>`. (A bare counter collided across
workers and restarts — fixed in this pass, document 9, defect 1.)

**Bounded memory.** Clusters older than `UT_IDENTITY_TTL_MIN` (720 min) are evicted from
the in-memory index (they remain in the database); track buffers older than 300 s and
idle camera trackers are evicted in housekeeping.

## 4.6 Spatio-temporal validation: legs and verdicts

A *leg* is the hop between a vehicle's first arrivals at consecutive cameras.

| Verdict | Condition |
|---|---|
| `IMPLAUSIBLE` | arrival not after departure, or implied speed > 140 km/h |
| `FAST` | implied speed > 100 km/h |
| `SLOW` | took more than 1.5× the expected maximum travel time |
| `PLAUSIBLE` | otherwise |

Each carries a written `reason`. **Continuity limit:** if two sightings are more than
`UT_MAX_LEG_GAP_MIN` (90 min) apart they are *not one journey* and no leg is written —
without this, a car seen Monday and Friday produces a four-day "leg" at 0.1 km/h that
poisons the origin–destination matrix and the congestion baseline.

**Camera graph.** Edges go to the *k* nearest cameras (default 6) within a radius
(default 12 km), found through a uniform lat/lon grid, not all pairs (7,048 edges vs
499,500 at 1,000 cameras). A non-adjacent pair still gets an edge labelled
`estimate-nonadjacent`. With GraphHopper configured, distances are real driven road
distance (disk-cached, key kept server-side) — and the result always says whether a
distance is a measurement or a great-circle ×1.25 estimate.

## 4.7 Known limits of cross-camera linking

* A leg between two cameras assigned to **different workers** (different affinity cells)
  is not recorded in `net_legs`, because each worker's identity index is in-process.
  Trajectory queries, which join sightings in SQL, still show the journey (document 6).
* Confusable twins are separated only by the feasibility check; two twins at nearby
  cameras within the speed limit would still merge.
* Association quality is bounded by OCR quality (document 3): a plate whose trailing
  character is always dropped is *consistently* wrong, so voting cannot fix it.
