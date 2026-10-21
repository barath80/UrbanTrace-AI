# 3. Models and measured accuracy

Source of every number below: `testing_accuracy.ipynb` (executed copy
`testing_accuracy_executed.ipynb`), unless another source is named. The notebook
recomputes everything when run; nothing is cached or copied from a paper.
Hardware: NVIDIA RTX 5060 Laptop GPU (8 GB), Ultralytics 8.4.102, PyTorch 2.11 + CUDA 12.8.

## 3.0 The models at a glance

| Stage | Model | Size | Role |
|---|---|---|---|
| Vehicle detection | `yolo11m.pt` (COCO) | 20.0 M params, 67.6 GFLOPs fused | car, motorcycle, bus, truck, bicycle |
| Plate detection | `license-plate-finetune-v1m.pt` (YOLO11m fine-tune) | 20.0 M params, 67.6 GFLOPs | one class, `license_plate` |
| Plate detection (alt.) | `license-plate-finetune-v1l.pt` | 25.3 M params, 86.6 GFLOPs | measured, not default |
| Tracking | ByteTrack | kilobytes per camera, no weights | per-camera track ids |
| OCR | `cct-xs-v2-global-model` (fast-plate-ocr, ONNX) | 128×64 input, 10 character slots | plate string + per-character confidence |
| Format repair | Indian plate grammar (rules, no weights) | — | forced-position repair, state-code check |

## 3.1 The evaluation data

**Plate detection dataset:** Roboflow *Indian License Plate Detection v1*, YOLOv8
format, **833 images**, one class. Boxes only — no vehicle boxes and no plate text.

| Split | Images | Boxes | Median box area | Boxes covering > ½ the frame |
|---|---|---|---|---|
| train | 583 | 688 | 0.0325 | 70 |
| valid | 167 | 194 | 0.0344 | 24 |
| **test (used for all held-out numbers)** | **83** | **99** | 0.0304 | 10 |

**Two honesty notes about this data.** Ten of the 99 test boxes cover more than half
the frame — a plate never does; these are annotation errors and each is scored as a
miss no matter how good the model. And the detectors were trained on a *different*
collection (*License-Plate-Recognition-11*), so the test set is genuinely held out.

**OCR ground truth:** `data/datasets/plate_ocr_test_labels.csv` — one row per test
plate box, **transcribed by hand from enlarged crops with no model involved**, with a
legibility grade:

| Legibility | Plates | Treatment |
|---|---|---|
| clear | 77 | scored |
| uncertain | 3 | scored |
| illegible | 9 (10.1 %) | **excluded** from accuracy; count always reported |

Scored: **80 plates** (length 9: 15, length 10: 65). Excluding illegible plates
*raises* the score, which is why their count is printed beside every result.

## 3.2 Plate detector

### Training record (recorded, not measured here)

| | v1m | v1l |
|---|---|---|
| Dataset | License-Plate-Recognition-11 | same |
| Epochs / batch / imgsz | 300 / 16 / 640 | same |
| Optimiser / lr0 / patience | auto / 0.01 / 15 | same |
| Own validation mAP50 | 0.9796 | 0.9811 |
| Own validation mAP50-95 | 0.7327 | 0.7318 |

### Held-out result (measured, 83 images, 99 boxes, `ultralytics val`)

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Inference |
|---|---|---|---|---|---|---|
| **v1m (default)** | 0.8036 | 0.5423 | 0.9607 | **0.8081** | **0.8778** | 12.1 ms/img |
| v1l | 0.8291 | 0.5492 | 1.0000 | 0.7542 | 0.8599 | 15.4 ms/img |

**The training score is not the deployment score.** v1m scored 0.980 on its own
validation split but 0.804 here — a drop of **0.176 mAP50** — and v1l dropped 0.152.
That gap is what a new camera collection costs. Quote the held-out figure.

### Operating point (conf = 0.25, what the pipeline actually runs)

| conf | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 0.10 | 80 | 6 | 19 | 0.9302 | 0.8081 | 0.8649 |
| **0.25** | **78** | **4** | **21** | **0.9512** | **0.7879** | **0.8619** |
| 0.40 | 78 | 3 | 21 | 0.9630 | 0.7879 | 0.8667 |
| 0.50 | 77 | 2 | 22 | 0.9747 | 0.7778 | 0.8652 |
| 0.70 | 62 | 0 | 37 | 1.0000 | 0.6263 | 0.7702 |

At 0.25: **precision 0.951, recall 0.788; 21 of 99 plates missed.** Raising the
threshold buys precision at the cost of recall that ANPR cannot afford, and F1 is
flat between 0.10 and 0.50. 13 of 83 test images have at least one miss or false
positive.

### What the bad annotations cost

Removing the 10 whole-frame boxes (89 left): **mAP50 0.8935, mAP50-95 0.6025,
precision 0.961, recall 0.899.** That is the performance on plates that are really
there. Both figures are reported; the lower is the one to quote unless you also say
the annotations were cleaned.

## 3.3 Vehicle detector

* **Recorded (COCO, 80 classes, from the checkpoint):** mAP50 0.684, mAP50-95 0.514,
  precision 0.738, recall 0.617; trained 600 epochs, batch 128, imgsz 640.
* **Not measured on Indian traffic.** The dataset has no vehicle boxes, so a COCO-style
  mAP for this project's footage **cannot be computed here and is not claimed.**
  The notebook ships a cell that computes it if you supply labelled frames
  (`data/datasets/vehicle_det`) and refuses to invent a figure otherwise.
* **What *was* measured** is the property the pipeline depends on, on plate images:
  vehicle detected in image 0.674; annotated plate falls inside a detected vehicle box
  **0.506** (the plate→track binding precondition); 2.30 vehicles/image; classes
  assigned: car 95.6 %, motorcycle 3.3 %, truck 1.1 %. These images are mostly
  close-crops with the vehicle cut off, so this is **not** how a road camera sees
  traffic — treat 0.506 as a stress figure, not a deployment estimate.
* **Known limit:** no autorickshaw class in COCO, so the per-class breakdown is
  unreliable on Indian traffic; unique-vehicle *totals* are unaffected.

## 3.4 Plate OCR

### Single-frame accuracy (measured, 80 scored plates)

| Metric | Value |
|---|---|
| Exact match (final) | **0.713 (57/80)** |
| Exact match (raw engine, before repair) | 0.688 |
| Character error rate | 0.0777 |
| Character accuracy | **0.9223** |
| Format-valid share | 0.787 |
| Engine failures | 0 / 80 |
| Latency | 8 ms per plate (all preprocessing variants, GPU) |

### Anatomy of the errors

Of 80 plates, **23 are wrong**. Of those, **17 are shorter than the truth** and
**4 are right except for the last character** (17 % of all errors). By length: of 65
ten-character plates, 52 were emitted at length 10; of 15 nine-character plates, 11 at
length 9. The reading cap is a property of the model (10 slots, 128×64 input): when a
character is dropped, no amount of preprocessing recovers it. Examples:
`HR26BW8871 → HR268W887`, `KA03AG6307 → KA03A6630`, `DL4CAW4204 → DL2CAW2204`.

### Does a different OCR model fix it? (measured)

Same 80 plates, every published `fast-plate-ocr` recogniser: see the table in
document 2, §2.3. Winner: `cct-xs-v2-global-model` at 0.7125 exact / 0.0777 CER —
the smaller model beats the larger (`cct-s-v2`: 0.275 / 0.195). That is why it is the
default.

### Is the confidence number trustworthy? (measured calibration)

| OCR confidence | Plates | Exact match | Mean conf | CER |
|---|---|---|---|---|
| [0.0, 0.5) | 3 | 0.000 | 0.259 | 0.897 |
| [0.5, 0.7) | 3 | 0.000 | 0.566 | 0.200 |
| [0.7, 0.8) | 3 | 0.333 | 0.736 | 0.172 |
| [0.8, 0.9) | 9 | 0.333 | 0.867 | 0.124 |
| [0.9, 0.95) | 10 | 0.800 | 0.925 | 0.030 |
| [0.95, 1.01) | 52 | 0.865 | 0.992 | 0.020 |

Confidence **is informative** — it rises with correctness — but it is **over-confident**
at the top: plates the model calls 0.99 are right 86.5 % of the time, not 99 %. At
confidence ≥ 0.90: 62 plates, **85.5 % exactly correct**, CER 0.0214. The pipeline's
publish gate (`UT_MIN_PUB_CONF = 0.60`) is a floor, not a guarantee.

### Do the preprocessing variants earn their cost?

On the **clean** test crops: no. `raw` alone and `raw + clahe + sharpen` both score
0.7125 exact / 0.0777 CER (5.3 vs 6.4 ms). The variants win on **degraded** input
(measured by `scripts/benchmark_ocr_robustness.py`, multi-variant vs raw): low light
(darkest) **+12.5 %**; defocus blur **+10.0 / +7.5 / +7.5 %**; sensor noise **+7.5 %**;
motion blur kernel ≥ 7 and downscale ≥ 3×: **0 % either way** — a real physical limit.
Variant that won most often: `sharpen+3/3` (25), `raw+3/3` (18), `clahe+3/3` (14).

### Robustness to blur, night, rain and distance (measured, character accuracy)

Clean character accuracy is 0.922. Character accuracy stays ≥ 0.90 at: motion blur
severity 3, defocus 3 and 5, distance 1.5×, sensor noise 5 and 10, rain 1 and 2.
Anything harsher falls below 0.90. Selected points:

| Condition | Mild | Severe |
|---|---|---|
| Motion blur (px) | 0.921 @3 | 0.618 @9, 0.462 @11 |
| Defocus (px) | 0.916 @3 | 0.823 @9, 0.732 @11 |
| Night (gain) | 0.888 @0.70 | 0.595 @0.15 |
| Distance (×) | 0.911 @1.5 | 0.773 @3.0 |
| Sensor noise | 0.930 @5 | 0.675 @40 |
| Rain (drops) | 0.922 @1.0 | 0.871–0.874 @6–8 |

The degradations are synthetic (but physical) and applied to real crops with real
ground truth. They bound behaviour; they do not replace a field trial.

## 3.5 End to end: image in, plate string out (measured)

Detect the plate, crop what the detector returned, read it, compare with the truth.

| Stage | Value |
|---|---|
| Legible plates in test split | 80 |
| Found by the detector (conf 0.25) | 0.912 |
| Read exactly, given found | 0.795 |
| **End-to-end exact plate read** | **0.725 (58/80)** |
| Mean IoU of found boxes | 0.829 |

**This is single-frame accuracy.** In service a plate is read on every frame a vehicle
is tracked and the readings are voted (§3.6).

## 3.6 Temporal voting (what is and is not proven)

`vision/aggregate.py` combines a confidence-weighted **string vote** and a per-position
**character vote**; character voting can recover a plate that *no single frame* read
correctly. Evidence weight per reading = OCR confidence × (0.5 + 0.5·plate confidence),
×1.25 if grammar-valid, ×0.90 if repaired.

On the supplied footage one vehicle was read in 14 frames, 12 agreed on
`TN01AB1234` (agreement 0.857, confidence 0.943), against **4 distinct raw strings**
(`135119`, `TI01AA1334`, `TNO1AB1234`, `TN01AB1234`) — a single-frame system would have
emitted a wrong plate. **This demonstrates that the vote changes the answer. It is not
an accuracy figure**: scoring it needs video with plate ground truth, which does not
exist for this footage. It also cannot fix an error the model makes identically on every
frame, such as the trailing-character truncation.

## 3.7 Identity resolution (measured by construction)

Test: 2,000 vehicles, 8,000 sightings, 1,000-camera estate.

| OCR error rate | Correctly merged | Fragmented | Clusters (ideal 2,000) | Sightings/s |
|---|---|---|---|---|
| 0 % | 1.00 | 0 | 2,000 | 74,837 |
| 25 % | 1.00 | 0 | 2,000 | 47,052 |
| 45 % | 1.00 | 0 | 2,000 | 37,935 |
| 70 % | 1.00 | 0 | 2,000 | 24,205 |

**Read this fairly.** The corruption used is confusion-group substitution — exactly
what `same_vehicle()` is built to absorb — so perfect recall is the *design case*,
not a surprise. The harder, honest test follows.

**The twin test.** Every vehicle is given a twin whose plate differs by one
*confusable* character (`TN01AB1234` vs `TN01AB123B`) — two different cars.
Result: 990 vehicles → 495 clusters, each holding two vehicles: **false-merge rate
1.00 by design.** Plate logic alone cannot separate plates that differ only by an OCR
confusion (the alternative is splitting one real vehicle in two). In the live service
the camera graph's **feasibility check** (`IdentityIndex(feasible=…)`) separates them
because one car cannot be in two places. This is a real limitation of plate-only
identity and is stated, not hidden.

## 3.8 Throughput

| Measurement | Value | Source / conditions |
|---|---|---|
| Per-stage (60 frames, 1280×720) | vehicle detect + track 20.9 ms, plate detect 18.2 ms, OCR 6.5 ms/plate; **23.2 frames/s** end to end | notebook §8.3 |
| Cameras per GPU from that | 23 @ 1 fps, 12 @ 2 fps, 5 @ 5 fps | arithmetic on the measured rate |
| Fleet benchmark (16 looping clips) | **52.3 analysed frames/s** | `scale/README.md`, earlier session |
| Fleet benchmark, **re-run for this documentation** | **7.2–14.2 frames/s** (runs: 14.2, 13.0, 7.7, 10.4, 7.2) | `scripts/bench_fleet.py --cameras 16`, same machine, 21 Oct 2026 |
| Same benchmark on a clean checkout of the previous commit | 12.0, 11.5 frames/s | git worktree of `HEAD`, same weights and clips |

**Read the last three rows together.** The documented ~50 frames/s was **not
reproduced today**. The clean previous commit gives 11.5–12 frames/s on the same
machine, so the gap is **not caused by the fixes made in this documentation pass**;
it is environmental or workload-related (run-to-run spread of 7–14 frames/s on
identical code points to the machine's current state, e.g. thermal/power management or
other GPU load, which this session could not isolate). **Re-run `bench_fleet.py` on
your deployment hardware before sizing workers**, and treat any single figure as
machine- and footage-specific.

Other measured facts that do not depend on that: peak VRAM **0.75–0.78 GiB** for 16
cameras; plate-pass skipping ~40 %; batched OCR 3.2× faster with byte-identical output;
camera registry at 1,000 cameras: graph built in 46 ms, 7,048 edges (vs 499,500
all-pairs), radius query 29.9 µs.

## 3.9 Counting accuracy

`vision/counting.py` counts unique vehicles per class with fragmentation stitching,
persistence/size/motion gates and an area-weighted class vote. On `camera1.mp4`
(1280×720, 24 fps, 240 frames): 4,130 vehicle detections, **110 raw tracker ids
(not a vehicle count)**, **28 unique traffic vehicles** (car 10, motorcycle 16,
truck 2) and 45 present (adds parked/stationary). Processing 18.2 fps. **No
human-labelled count exists for this footage, so these counts are auditable
(checked against a contact sheet) but not ground truth.**

## 3.10 Summary table — what may be quoted

| Claim | Value | Qualifier to attach |
|---|---|---|
| Plate detection mAP50 | 0.804 | 83 held-out images, 99 boxes; 0.894 with the 10 bad annotations removed |
| Plate detection precision / recall @0.25 | 0.951 / 0.788 | same set |
| OCR exact plate (single frame) | 0.713 | 80 hand-transcribed plates; 9 illegible excluded |
| OCR character accuracy | 0.922 | same |
| End-to-end single-frame plate read | 0.725 | detector + OCR, 80 plates |
| Identity merge accuracy | 1.00 at ≤ 70 % confusion noise | by construction; 100 % false merge on confusable twins unless feasibility separates them |
| Vehicle detection mAP on Indian traffic | **not measured** | COCO 0.684 is the checkpoint's COCO figure |
| Accuracy after temporal voting | **not measured** | demonstrably changes answers; no video ground truth |
| Fleet throughput | machine-specific (7–52 fps observed) | re-run on target hardware |
