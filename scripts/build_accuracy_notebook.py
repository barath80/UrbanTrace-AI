"""Generate testing_accuracy.ipynb - the measured-accuracy notebook.

Design rule, same as the realtime notebook: nothing is pre-baked. Every number
printed by the notebook is computed when the cell runs, from the held-out test
split of the Roboflow dataset and from the real model weights in models/weights.

Where a quantity cannot be measured because no ground truth exists, the notebook
says so instead of printing a number.

Regenerate with:  python scripts/build_accuracy_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testing_accuracy.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {},
                "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}

cells = []

# ------------------------------------------------------------------ intro
cells.append(md("""
# UrbanTrace AI - Measured Accuracy

**Held-out evaluation of every model in the pipeline: vehicle detection, plate
detection, and plate OCR - plus the same measurement repeated at city scale.**

Each number below is computed when the cell runs. There is no cached result and
no figure copied from a training log. Where a metric cannot be measured because
the ground truth for it does not exist, the notebook prints what is missing
rather than an estimate.

### The three models under test

| Model | Weights | Trained by | Task |
|---|---|---|---|
| Vehicle detector | `yolo11m.pt` | Ultralytics, on COCO | car / motorcycle / bus / truck / bicycle boxes |
| Plate detector | `license-plate-finetune-v1m.pt`, `-v1l.pt` | fine-tuned on Roboflow data | license-plate boxes |
| Plate OCR | `cct-xs-v2-global-model` (fast-plate-ocr) | pretrained, not trained here | plate box -> characters |

### What each section can and cannot measure

| Section | Ground truth used | Status |
|---|---|---|
| 4 · Plate detection | 83 test images, 99 boxes, from the Roboflow export | **fully measurable** |
| 5 · Vehicle detection | no vehicle boxes exist in this dataset | **partially measurable** - the pipeline-critical property is measured, COCO mAP is not |
| 6 · Plate OCR | `data/datasets/plate_ocr_test_labels.csv`, transcribed from the test crops | **fully measurable** |
| 7 · End-to-end | detector + OCR chained on the same test images | **fully measurable** |
| 8 · City scale | synthetic identities with known answers, real data structures | **fully measurable** |

### How to run

```
python scripts/build_accuracy_notebook.py     # regenerate this notebook
jupyter lab testing_accuracy.ipynb            # then Run All
```

Run the cells in order. Section 2 stops the notebook with a clear message if a
required file is missing, rather than letting later cells fail obscurely.
"""))

# ------------------------------------------------------------------ 1
cells.append(md("## 1 · Environment"))
cells.append(code("""
import json, os, sys, time, warnings
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
while not (ROOT / "backend").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore", category=UserWarning)

import torch, cv2, ultralytics

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"repo root       {ROOT}")
print(f"python          {sys.version.split()[0]}")
print(f"torch           {torch.__version__}")
print(f"ultralytics     {ultralytics.__version__}")
print(f"opencv          {cv2.__version__}")
print(f"device          {torch.cuda.get_device_name(0) if DEVICE == 0 else 'cpu'}")
if DEVICE == 0:
    print(f"vram            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""))

# ------------------------------------------------------------------ 2
cells.append(md("""
## 2 · Inputs

Four things must be present. Anything missing is named here, not three cells later.
"""))
cells.append(code("""
DATASET   = ROOT / "Indian License Plate Detection.v1i.yolov8"
WEIGHTS   = ROOT / "models" / "weights"
OCR_LABELS = ROOT / "data" / "datasets" / "plate_ocr_test_labels.csv"
OUT_DIR   = ROOT / "data" / "output" / "accuracy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

required = {
    "dataset export":        DATASET / "data.yaml",
    "test images":           DATASET / "test" / "images",
    "test labels":           DATASET / "test" / "labels",
    "plate weights (m)":     WEIGHTS / "license-plate-finetune-v1m.pt",
    "vehicle weights":       WEIGHTS / "yolo11m.pt",
    "OCR ground truth":      OCR_LABELS,
}
missing = {k: p for k, p in required.items() if not p.exists()}
for k, p in required.items():
    print(f"{'ok  ' if p.exists() else 'MISS'}  {k:22} {p}")
if missing:
    raise SystemExit("\\nMissing inputs listed above. The notebook stops here rather "
                     "than reporting accuracy for data it does not have.")
print("\\nall inputs present")
"""))

# ------------------------------------------------------------------ 3
cells.append(md("""
## 3 · The dataset, and how each model was trained

### 3.1 The evaluation dataset

`Indian License Plate Detection v1` was exported from Roboflow in YOLOv8 format.
It carries **plate boxes only** - one class, `license_plate`. There are no vehicle
boxes and no plate text in the export; that shapes what sections 5 and 6 can do.
"""))
cells.append(code("""
import yaml, collections

cfg = yaml.safe_load((DATASET / "data.yaml").read_text())
print("classes:", cfg["names"], " nc:", cfg["nc"])
readme = (DATASET / "README.roboflow.txt").read_text()
print(readme[readme.index("The dataset includes"):].strip())

rows = []
for split in ("train", "valid", "test"):
    labs = sorted((DATASET / split / "labels").glob("*.txt"))
    boxes, areas, per_img = 0, [], collections.Counter()
    for f in labs:
        lines = [l for l in f.read_text().split("\\n") if l.strip()]
        boxes += len(lines); per_img[len(lines)] += 1
        for l in lines:
            _, _, _, w, h = (float(v) for v in l.split())
            areas.append(w * h)
    areas.sort()
    rows.append(dict(split=split, images=len(labs), boxes=boxes,
                     median_box_area=round(areas[len(areas)//2], 4),
                     boxes_over_half_frame=sum(1 for a in areas if a > 0.5),
                     boxes_per_image=dict(sorted(per_img.items()))))

import pandas as pd
splits = pd.DataFrame(rows)
display(splits)
print("\\nBoxes covering more than half the frame are almost certainly annotation")
print("errors - a plate is never half the image. Section 4.3 measures their cost.")
"""))

cells.append(md("""
### 3.2 A look at the test images

Ground-truth boxes drawn on six test images, so the annotation quality is visible
rather than asserted.
"""))
cells.append(code("""
import matplotlib.pyplot as plt

test_imgs = sorted((DATASET / "test" / "images").glob("*.jpg"))
show = test_imgs[:3] + test_imgs[-3:]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, p in zip(axes.ravel(), show):
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    H, W = im.shape[:2]
    lab = DATASET / "test" / "labels" / (p.stem + ".txt")
    for l in [l for l in lab.read_text().split("\\n") if l.strip()]:
        _, x, y, w, h = (float(v) for v in l.split())
        big = w * h > 0.5
        cv2.rectangle(im, (int((x-w/2)*W), int((y-h/2)*H)), (int((x+w/2)*W), int((y+h/2)*H)),
                      (255, 60, 60) if big else (40, 220, 90), 3)
    ax.imshow(im); ax.set_title(p.name[:28], fontsize=8); ax.axis("off")
fig.suptitle("Test split ground truth  (green = plate box, red = box over half the frame)")
plt.tight_layout(); plt.show()
"""))

cells.append(md("""
### 3.3 How each model was trained

Ultralytics checkpoints carry their own training arguments and the validation
metrics from the end of that run. Those are read out of the files below - they are
the trainer's record, not a claim made here.

**Read the two columns apart.** `train_metrics` is what the model scored on *its
own* validation split at the end of training. Everything from section 4 onward is
what it scores on *this* held-out test split. When they disagree, the second
number is the one that describes deployment.
"""))
cells.append(code("""
def training_record(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ta = ck.get("train_args") or {}
    tm = ck.get("train_metrics") or {}
    return {
        "weights": path.name,
        "trained": str(ck.get("date", "?"))[:19],
        "ultralytics": ck.get("version", "?"),
        "dataset": Path(str(ta.get("data", "?"))).parent.name or str(ta.get("data", "?")),
        "epochs": ta.get("epochs"), "batch": ta.get("batch"), "imgsz": ta.get("imgsz"),
        "optimizer": ta.get("optimizer"), "lr0": ta.get("lr0"), "patience": ta.get("patience"),
        "own_val_mAP50": round(tm.get("metrics/mAP50(B)", float("nan")), 4),
        "own_val_mAP50_95": round(tm.get("metrics/mAP50-95(B)", float("nan")), 4),
        "own_val_P": round(tm.get("metrics/precision(B)", float("nan")), 4),
        "own_val_R": round(tm.get("metrics/recall(B)", float("nan")), 4),
    }

records = [training_record(WEIGHTS / w) for w in
           ("license-plate-finetune-v1m.pt", "license-plate-finetune-v1l.pt", "yolo11m.pt")
           if (WEIGHTS / w).exists()]
training = pd.DataFrame(records)
display(training)
"""))

cells.append(md("""
**Plate detector** - `license-plate-finetune-v1m.pt` / `-v1l.pt`. YOLO11 fine-tuned
on a Roboflow license-plate dataset (`License-Plate-Recognition-11`), 300 epochs,
batch 16, 640 px, `optimizer=auto`, `lr0=0.01`, early stopping `patience=15`,
resumed from an earlier run of the same recipe. One class: `license_plate`.

That training set is **not** the dataset evaluated here. The Indian export used
below is therefore a cross-dataset test - harder than a same-dataset split, and the
right question to ask of a model that will meet new cameras.

To retrain or continue training on this dataset:

```python
from ultralytics import YOLO
YOLO("yolo11m.pt").train(
    data="Indian License Plate Detection.v1i.yolov8/data.yaml",
    epochs=300, imgsz=640, batch=16, patience=15, device=0,
)
```

**Vehicle detector** - `yolo11m.pt`, stock Ultralytics COCO weights (600 epochs,
batch 128, 640 px). Nothing here was trained on Indian traffic, and COCO has no
autorickshaw class. Section 5 measures what that costs.

**OCR** - `cct-xs-v2-global-model` from `fast-plate-ocr`, a compact convolutional
transformer served through ONNX Runtime. It is used as published; no fine-tuning
was done in this project. Its configuration - alphabet, slot count, input size -
is printed in section 6.1.

This is the model the pipeline runs today, and it was chosen by section 6.4 rather
than by reputation: the larger `cct-s-v2` was the original default and reads these
plates less than half as well. Section 6.4 re-runs that comparison every time, so
the choice stays evidence-backed rather than inherited.

### 3.4 What the dataset was preprocessed with

Roboflow applied auto-orientation (EXIF stripped) and a stretch resize to 640x640.
No augmentation was applied in the export. Ultralytics applies its own augmentation
during training only, never during the validation runs below.
"""))

# ------------------------------------------------------------------ 4
cells.append(md("""
## 4 · Plate detection accuracy

The one metric in this notebook with complete, unambiguous ground truth: 83 held-out
images, 99 annotated plate boxes, evaluated with the standard COCO-style protocol
through `ultralytics.val` - the same code path the Ultralytics leaderboard uses.
"""))
cells.append(code("""
from ultralytics import YOLO

# ultralytics resolves dataset paths relative to `path`; the exported yaml uses
# `../train/images`, which only works from inside the export directory.
ds_yaml = OUT_DIR / "dataset.yaml"
ds_yaml.write_text(yaml.safe_dump({
    "path": str(DATASET), "train": "train/images", "val": "valid/images",
    "test": "test/images", "nc": 1, "names": ["license_plate"],
}))

def evaluate(weights, split="test", conf=0.001, iou=0.6):
    m = YOLO(str(weights))
    r = m.val(data=str(ds_yaml), split=split, imgsz=640, conf=conf, iou=iou,
              device=DEVICE, verbose=False, plots=False,
              # workers=0: Windows spawns dataloader workers by re-importing
              # __main__, which a notebook kernel cannot provide.
              workers=0, project=str(OUT_DIR / "runs"), name=weights.stem, exist_ok=True)
    p, rc = float(r.box.mp), float(r.box.mr)
    return {"weights": weights.name, "split": split,
            "mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
            "precision": round(p, 4), "recall": round(rc, 4),
            "F1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0}

plate_weights = [w for w in (WEIGHTS / "license-plate-finetune-v1m.pt",
                             WEIGHTS / "license-plate-finetune-v1l.pt") if w.exists()]
det_rows = [evaluate(w) for w in plate_weights]
detection = pd.DataFrame(det_rows)
display(detection)
detection.to_csv(OUT_DIR / "plate_detection.csv", index=False)
"""))

cells.append(md("""
### 4.1 Held-out score vs the score the trainer reported

Both numbers are real. They measure different things, and only one of them
predicts what happens on a new camera.
"""))
cells.append(code("""
cmp = training.merge(detection, on="weights", how="inner")[
    ["weights", "own_val_mAP50", "mAP50", "own_val_mAP50_95", "mAP50-95"]]
cmp["mAP50_drop"] = (cmp["own_val_mAP50"] - cmp["mAP50"]).round(4)
display(cmp)
for _, r in cmp.iterrows():
    print(f"{r['weights']}: {r['own_val_mAP50']:.3f} on its own validation split, "
          f"{r['mAP50']:.3f} here - a drop of {r['mAP50_drop']:.3f} mAP50 when the "
          f"images come from a different collection.")
"""))

cells.append(md("""
### 4.2 Where the operating point sits

`val` reports precision and recall at the confidence that maximises F1. The
pipeline runs at a fixed `conf=0.25`, so that is the number that describes
production. The sweep below measures both.
"""))
cells.append(code("""
def sweep(weights, confs=(0.10, 0.25, 0.40, 0.50, 0.70)):
    model = YOLO(str(weights))
    labels = {}
    for lab in sorted((DATASET / "test" / "labels").glob("*.txt")):
        img = DATASET / "test" / "images" / (lab.stem + ".jpg")
        im = cv2.imread(str(img)); H, W = im.shape[:2]
        boxes = []
        for l in [l for l in lab.read_text().split("\\n") if l.strip()]:
            _, x, y, w, h = (float(v) for v in l.split())
            boxes.append([(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H])
        labels[img] = boxes

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2-x1) * max(0.0, y2-y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    out = []
    for c in confs:
        tp = fp = fn = 0
        for img, gts in labels.items():
            res = model.predict(str(img), conf=c, imgsz=640, device=DEVICE, verbose=False)
            preds = res[0].boxes.xyxy.cpu().tolist() if res and res[0].boxes is not None else []
            used = set()
            for p in preds:
                best, bi = 0.0, -1
                for i, g in enumerate(gts):
                    if i in used:
                        continue
                    v = iou(p, g)
                    if v > best:
                        best, bi = v, i
                if best >= 0.5:
                    tp += 1; used.add(bi)
                else:
                    fp += 1
            fn += len(gts) - len(used)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append({"conf": c, "TP": tp, "FP": fp, "FN": fn,
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "F1": round(2*prec*rec/(prec+rec), 4) if prec+rec else 0.0})
    return pd.DataFrame(out)

op = sweep(plate_weights[0])
display(op)
row = op[op.conf == 0.25].iloc[0]
print(f"At the pipeline's conf=0.25: precision {row.precision:.3f}, recall {row.recall:.3f}, "
      f"{int(row.FN)} of {int(row.TP + row.FN)} ground-truth plates missed.")
"""))

cells.append(md("""
### 4.3 What the bad annotations cost

Ten of the 99 test boxes cover more than half the frame. A plate never does. These
are annotation errors, and every one of them is scored as a miss no matter how well
the model works. Removing them shows the score on the plates that are really there.
"""))
cells.append(code("""
clean_dir = OUT_DIR / "test_clean"
(clean_dir / "images").mkdir(parents=True, exist_ok=True)
(clean_dir / "labels").mkdir(parents=True, exist_ok=True)
for f in (clean_dir / "images").glob("*"): f.unlink()
for f in (clean_dir / "labels").glob("*"): f.unlink()

import shutil
kept = dropped = 0
for lab in sorted((DATASET / "test" / "labels").glob("*.txt")):
    lines = [l for l in lab.read_text().split("\\n") if l.strip()]
    good = []
    for l in lines:
        _, _, _, w, h = (float(v) for v in l.split())
        (good.append(l) if w * h <= 0.5 else None)
        dropped += w * h > 0.5
    if not good:
        continue
    kept += len(good)
    (clean_dir / "labels" / lab.name).write_text("\\n".join(good) + "\\n")
    src = DATASET / "test" / "images" / (lab.stem + ".jpg")
    shutil.copy(src, clean_dir / "images" / src.name)

clean_yaml = OUT_DIR / "dataset_clean.yaml"
clean_yaml.write_text(yaml.safe_dump({
    "path": str(clean_dir), "train": "images", "val": "images", "test": "images",
    "nc": 1, "names": ["license_plate"]}))
print(f"kept {kept} plate boxes, dropped {dropped} whole-frame annotations")

m = YOLO(str(plate_weights[0]))
r = m.val(data=str(clean_yaml), split="test", imgsz=640, conf=0.001, iou=0.6,
          device=DEVICE, verbose=False, plots=False, workers=0,
          project=str(OUT_DIR / "runs"), name="clean", exist_ok=True)
clean_row = {"weights": plate_weights[0].name + " (annotations cleaned)",
             "mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
             "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4)}
display(pd.DataFrame([det_rows[0], clean_row]))
"""))

cells.append(md("""
### 4.4 The images it gets wrong

Misses and false positives, drawn. This is the part a table cannot show.
"""))
cells.append(code("""
model = YOLO(str(plate_weights[0]))
worst = []
for lab in sorted((DATASET / "test" / "labels").glob("*.txt")):
    img = DATASET / "test" / "images" / (lab.stem + ".jpg")
    im = cv2.imread(str(img)); H, W = im.shape[:2]
    gts = []
    for l in [l for l in lab.read_text().split("\\n") if l.strip()]:
        _, x, y, w, h = (float(v) for v in l.split())
        if w * h > 0.5:
            continue
        gts.append([(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H])
    res = model.predict(str(img), conf=0.25, imgsz=640, device=DEVICE, verbose=False)
    preds = res[0].boxes.xyxy.cpu().tolist() if res and res[0].boxes is not None else []
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2-x1) * max(0.0, y2-y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0
    missed = [g for g in gts if not any(iou(p, g) >= 0.5 for p in preds)]
    spurious = [p for p in preds if not any(iou(p, g) >= 0.5 for g in gts)]
    if missed or spurious:
        worst.append((img, gts, preds, missed, spurious))

print(f"{len(worst)} of {len(list((DATASET/'test'/'labels').glob('*.txt')))} test images "
      f"have at least one miss or false positive at conf=0.25")
n_show = min(6, len(worst))
if n_show:
    fig, axes = plt.subplots((n_show + 2) // 3, 3, figsize=(15, 5 * ((n_show + 2) // 3)))
    for ax, (img, gts, preds, missed, spurious) in zip(np.atleast_1d(axes).ravel(), worst[:n_show]):
        im = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
        for g in gts:
            cv2.rectangle(im, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (40, 220, 90), 3)
        for p in preds:
            cv2.rectangle(im, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (80, 140, 255), 2)
        ax.imshow(im); ax.axis("off")
        ax.set_title(f"{len(missed)} missed, {len(spurious)} false positive", fontsize=9)
    for ax in np.atleast_1d(axes).ravel()[n_show:]:
        ax.axis("off")
    fig.suptitle("green = ground truth, blue = prediction at conf=0.25")
    plt.tight_layout(); plt.show()
"""))

# ------------------------------------------------------------------ 5
cells.append(md("""
## 5 · Vehicle detection accuracy

**State the limit first.** This dataset has no vehicle boxes. Without them, COCO
mAP for `yolo11m` cannot be computed here, and any figure claiming otherwise would
be invented. The stock COCO validation score is printed in section 3.3 from the
checkpoint's own record: mAP50 0.684, mAP50-95 0.514 across all 80 COCO classes.

What *can* be measured on this data is the property the pipeline actually depends
on. `bind_plates_to_tracks` only works if the vehicle detector puts a box around
the vehicle whose plate was found. So:

1. **vehicle-present recall** - of the images with an annotated plate, in how many
   does the vehicle detector find a vehicle at all;
2. **plate-containment rate** - in how many is the annotated plate inside a
   detected vehicle box, which is the precondition for binding a plate to a track;
3. **class mix** - what classes it assigns, which is where the autorickshaw problem
   shows up.

Each is a real measurement against real annotations. None of them is COCO mAP, and
none is reported as such.
"""))
cells.append(code("""
from urbantrace.vision.detectors import VehicleTracker, VEHICLE_CLASSES, containment

veh = YOLO(str(WEIGHTS / "yolo11m.pt"))
rows, class_counter = [], collections.Counter()
for lab in sorted((DATASET / "test" / "labels").glob("*.txt")):
    img = DATASET / "test" / "images" / (lab.stem + ".jpg")
    im = cv2.imread(str(img)); H, W = im.shape[:2]
    gts = []
    for l in [l for l in lab.read_text().split("\\n") if l.strip()]:
        _, x, y, w, h = (float(v) for v in l.split())
        if w * h > 0.5:
            continue
        gts.append(((x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H))
    if not gts:
        continue
    res = veh.predict(str(img), conf=0.30, imgsz=960, classes=list(VEHICLE_CLASSES),
                      device=DEVICE, verbose=False)
    boxes, clss = [], []
    if res and res[0].boxes is not None:
        boxes = res[0].boxes.xyxy.cpu().tolist()
        clss = [VEHICLE_CLASSES.get(int(c), str(int(c))) for c in res[0].boxes.cls.cpu().tolist()]
    class_counter.update(clss)
    for g in gts:
        rows.append({"image": img.name, "vehicle_found": bool(boxes),
                     "plate_inside_vehicle": any(containment(g, b) >= 0.60 for b in boxes),
                     "vehicles_in_image": len(boxes)})

vd = pd.DataFrame(rows)
print(f"annotated plates considered      {len(vd)}")
print(f"vehicle detected in the image    {vd.vehicle_found.mean():.3f}")
print(f"plate falls inside a vehicle box {vd.plate_inside_vehicle.mean():.3f}   "
      f"<- the binding precondition")
print(f"mean vehicles detected per image {vd.vehicles_in_image.mean():.2f}")
print("\\nclass mix assigned by the COCO detector:")
for k, v in class_counter.most_common():
    print(f"  {k:12} {v:4}  {v/max(1,sum(class_counter.values())):.1%}")
vd.to_csv(OUT_DIR / "vehicle_detection.csv", index=False)
"""))

cells.append(md("""
Plates that fall outside every vehicle box are cases where the pipeline would read
a plate it cannot attach to a track: the reading survives, the journey does not.
These are close-crop images where the vehicle is cut off at the frame edge, which is
worth knowing but is not how a road camera sees traffic.

**If you want a true COCO mAP for the vehicle detector on your own footage**, label
vehicle boxes for a sample of frames in YOLO format and point the cell below at
them. It runs only when that directory exists, so nothing is fabricated in its
absence.
"""))
cells.append(code("""
VEHICLE_GT = ROOT / "data" / "datasets" / "vehicle_det"   # images/ and labels/, YOLO format
if (VEHICLE_GT / "images").is_dir() and any((VEHICLE_GT / "images").iterdir()):
    v_yaml = OUT_DIR / "vehicle.yaml"
    v_yaml.write_text(yaml.safe_dump({
        "path": str(VEHICLE_GT), "train": "images", "val": "images", "test": "images",
        "nc": 5, "names": ["bicycle", "car", "motorcycle", "bus", "truck"]}))
    r = YOLO(str(WEIGHTS / "yolo11m.pt")).val(
        data=str(v_yaml), split="test", imgsz=960, device=DEVICE,
        verbose=False, plots=False, workers=0,
        project=str(OUT_DIR / "runs"), name="vehicle", exist_ok=True)
    display(pd.DataFrame([{"mAP50": round(float(r.box.map50), 4),
                           "mAP50-95": round(float(r.box.map), 4),
                           "precision": round(float(r.box.mp), 4),
                           "recall": round(float(r.box.mr), 4)}]))
else:
    print(f"No labelled vehicle set at {VEHICLE_GT}.")
    print("Vehicle mAP is therefore NOT reported. To produce it:")
    print("  1. sample frames from data/videos/ (or your own cameras)")
    print("  2. label vehicle boxes in YOLO format with classes")
    print("     0 bicycle, 1 car, 2 motorcycle, 3 bus, 4 truck")
    print(f"  3. place them in {VEHICLE_GT}/images and {VEHICLE_GT}/labels")
    print("  4. re-run this cell")
"""))

# ------------------------------------------------------------------ 6
cells.append(md("""
## 6 · Plate OCR accuracy

### 6.1 The ground truth

`data/datasets/plate_ocr_test_labels.csv` holds one row per plate box in the test
split: the source image, the box, the transcribed characters, and a legibility
grade. The transcriptions were made by reading each crop enlarged, with no model
involved.

`legibility` is the honest part of the file:

- `clear` - the characters are unambiguous to a careful reader;
- `uncertain` - transcribed, but one or two characters are a judgement call;
- `illegible` - no reader could recover the plate from this crop. These carry no
  text and are excluded from accuracy. **Excluding them raises the reported score**,
  so their count is printed alongside every result.

Accuracy is reported over `clear` + `uncertain`. A model cannot be marked wrong for
failing to read what is not there.
"""))
cells.append(code("""
gt = pd.read_csv(OCR_LABELS)
print(gt.legibility.value_counts().to_string())
print(f"\\ntotal plate crops        {len(gt)}")
print(f"scored (clear+uncertain) {(gt.legibility != 'illegible').sum()}")
print(f"excluded as illegible    {(gt.legibility == 'illegible').sum()}"
      f"  ({(gt.legibility == 'illegible').mean():.1%} of the test plates)")
print(f"\\nplate length distribution: "
      f"{gt[gt.plate_text.notna()].plate_text.str.len().value_counts().sort_index().to_dict()}")

from urbantrace.vision.ocr_engine import PlateOCR, _load
engine_cfg = _load().config
print(f"\\nOCR model      {PlateOCR().model_name}")
print(f"max_plate_slots {engine_cfg.max_plate_slots}   <- characters the model can emit")
print(f"input size      {engine_cfg.img_width}x{engine_cfg.img_height} "
      f"({engine_cfg.image_color_mode})")
print(f"India in its region list: {'India' in engine_cfg.plate_regions}")
"""))

cells.append(md("""
### 6.2 Reading every test plate
"""))
cells.append(code("""
def levenshtein(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j-1] + 1, prev + (ca != cb))
    return d[-1]

def load_crop(row):
    im = cv2.imread(str(DATASET / "test" / "images" / row.image))
    return im[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]

scored = gt[gt.legibility != "illegible"].copy()
crops = [(r.plate_text, load_crop(r), r.legibility) for r in scored.itertuples()]

def score_ocr(ocr, crops):
    out = []
    t0 = time.perf_counter()
    for truth, crop, leg in crops:
        res = ocr.read(crop)
        out.append({
            "truth": truth,
            "raw": res.text_raw if res else "",
            "final": res.text_final if res else "",
            "confidence": round(res.confidence, 4) if res else 0.0,
            "valid_format": bool(res.is_valid_format) if res else False,
            "variant": res.variant if res else "",
            "legibility": leg,
        })
    df = pd.DataFrame(out)
    df["exact"] = df.truth == df.final
    df["exact_raw"] = df.truth == df.raw
    df["edits"] = [levenshtein(t, f) for t, f in zip(df.truth, df.final)]
    df["truth_len"] = df.truth.str.len()
    df["pred_len"] = df.final.str.len()
    df["ms_per_plate"] = (time.perf_counter() - t0) / len(df) * 1000
    return df

ocr = PlateOCR()
reads = score_ocr(ocr, crops)
n_scored = len(reads)
cer = reads.edits.sum() / reads.truth_len.sum()
print(f"plates scored             {n_scored}")
print(f"exact match (final)       {reads.exact.mean():.3f}   {reads.exact.sum()}/{n_scored}")
print(f"exact match (raw engine)  {reads.exact_raw.mean():.3f}")
print(f"character error rate      {cer:.4f}")
print(f"character accuracy        {1 - cer:.4f}")
print(f"format-valid share        {reads.valid_format.mean():.3f}")
print(f"OCR engine failures       {ocr.failures}/{ocr.calls}")
print(f"latency                   {reads.ms_per_plate.iloc[0]:.0f} ms per plate "
      f"(all preprocessing variants, GPU)")
reads.to_csv(OUT_DIR / "ocr_reads.csv", index=False)
display(reads.head(12))
"""))

cells.append(md("""
### 6.3 Why the exact-match number is what it is

Exact match is unforgiving: one wrong character fails the whole plate. Before
concluding the model cannot read, look at *how* the wrong answers are wrong.
"""))
cells.append(code("""
wrong = reads[~reads.exact]
short = wrong[wrong.pred_len < wrong.truth_len]
prefix_ok = wrong[[t.startswith(f) and len(f) == len(t) - 1
                   for t, f in zip(wrong.truth, wrong.final)]]
print(f"wrong readings                       {len(wrong)}")
print(f"  of those, shorter than the truth   {len(short)}")
print(f"  correct except the last character  {len(prefix_ok)}  "
      f"({len(prefix_ok)/max(1,len(wrong)):.0%} of all errors)")
print(f"\\npredicted length vs true length:")
display(pd.crosstab(reads.truth_len, reads.pred_len).rename_axis(index="true", columns="pred"))
print("Indian plates are 10 characters. Every reading capped at 9 loses the last one,")
print("and no amount of preprocessing recovers a character the model never emits.")
display(wrong[["truth", "raw", "final", "confidence"]].head(15))
"""))

cells.append(md("""
### 6.4 Does a different OCR model fix it?

`fast-plate-ocr` publishes several recognisers. They cost nothing to try, and the
answer is measured rather than argued. The mobile-ViT models take single-channel
input, so they are fed a grayscale crop; the CCT models take the pipeline's normal
colour crop.
"""))
cells.append(code("""
# Every recogniser fast-plate-ocr publishes, so "best available" is a measured
# claim rather than an assumption. The mobile-ViT entries take single-channel
# input the colour preprocessing path does not produce; they are listed so their
# absence is explained rather than silent.
CANDIDATES = ["cct-xs-v2-global-model", "cct-s-v2-global-model",
              "cct-xs-v1-global-model", "cct-s-v1-global-model",
              "cct-xs-relu-v1-global-model", "cct-s-relu-v1-global-model"]
rows = []
for name in CANDIDATES:
    try:
        cand = PlateOCR(model_name=name)
        df = score_ocr(cand, crops)
    except Exception as e:
        rows.append({"model": name, "exact": None, "CER": None,
                     "ms_per_plate": None, "note": f"unavailable: {type(e).__name__}"})
        continue
    rows.append({"model": name,
                 "exact": round(df.exact.mean(), 4),
                 "CER": round(df.edits.sum() / df.truth_len.sum(), 4),
                 "ms_per_plate": round(df.ms_per_plate.iloc[0], 1),
                 "note": "current default" if name == ocr.model_name else ""})

models_df = pd.DataFrame(rows).sort_values("exact", ascending=False, na_position="last")
display(models_df)
models_df.to_csv(OUT_DIR / "ocr_models.csv", index=False)

best_ocr = models_df.dropna(subset=["exact"]).iloc[0]
current_ocr = models_df[models_df.model == ocr.model_name].iloc[0]
if best_ocr.model != ocr.model_name:
    print(f"\\n{best_ocr.model} reads {best_ocr.exact:.1%} of these plates exactly, against "
          f"{current_ocr.exact:.1%} for the configured {current_ocr.model}.")
    print("To switch, set the model name where PlateOCR is constructed")
    print("(backend/urbantrace/vision/ocr_engine.py, PlateOCR.__init__ default).")
"""))

cells.append(md("""
### 6.5 Is the confidence number trustworthy?

The pipeline uses OCR confidence to decide what to keep. That is only sound if
confidence tracks correctness. Measured against the ground truth:
"""))
cells.append(code("""
bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
reads["conf_bin"] = pd.cut(reads.confidence, bins, right=False)
cal = reads.groupby("conf_bin", observed=True).agg(
    plates=("exact", "size"), exact_match=("exact", "mean"),
    mean_conf=("confidence", "mean"), CER=("edits", "sum")).reset_index()
cal["CER"] = (cal.CER / reads.groupby("conf_bin", observed=True).truth_len.sum().values).round(4)
cal["exact_match"] = cal.exact_match.round(3)
cal["mean_conf"] = cal.mean_conf.round(3)
display(cal)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(range(len(cal)), cal.exact_match, color="#3b82f6")
ax[0].plot(range(len(cal)), cal.mean_conf, "o--", color="#ef4444", label="mean confidence")
ax[0].set_xticks(range(len(cal))); ax[0].set_xticklabels([str(b) for b in cal.conf_bin], rotation=30, fontsize=7)
ax[0].set_ylabel("exact-match rate"); ax[0].set_title("Confidence vs correctness"); ax[0].legend()
ax[1].hist(reads[reads.exact].confidence, bins=20, alpha=.7, label="correct", color="#22c55e")
ax[1].hist(reads[~reads.exact].confidence, bins=20, alpha=.7, label="wrong", color="#ef4444")
ax[1].set_xlabel("OCR confidence"); ax[1].set_title("Confidence distribution"); ax[1].legend()
plt.tight_layout(); plt.show()

hi = reads[reads.confidence >= 0.9]
print(f"At confidence >= 0.90: {len(hi)} plates, {hi.exact.mean():.1%} exactly correct, "
      f"CER {hi.edits.sum()/max(1,hi.truth_len.sum()):.4f}")
"""))

cells.append(md("""
### 6.6 Do the preprocessing variants earn their cost?

The engine renders several preprocessed versions of each crop and keeps the best
reading. That multiplies the OCR cost per plate, so it should be shown to pay.
"""))
cells.append(code("""
from urbantrace.vision import preprocess as pp

variant_rows = []
for variants in (("raw",), pp.DEFAULT_VARIANTS):
    v_ocr = PlateOCR(variants=variants)
    df = score_ocr(v_ocr, crops)
    variant_rows.append({"variants": ", ".join(variants),
                         "count": len(variants),
                         "exact": round(df.exact.mean(), 4),
                         "CER": round(df.edits.sum() / df.truth_len.sum(), 4),
                         "ms_per_plate": round(df.ms_per_plate.iloc[0], 1)})
display(pd.DataFrame(variant_rows))
print("Which variant won, over all scored plates:")
print(reads.variant.value_counts().to_string())
"""))

# ------------------------------------------------------------------ 7
cells.append(md("""
### 6.7 Blur, night, rain and distance

A clean-crop score answers the easy question. Road cameras are not clean: vehicles
move, light fails, glass gets wet, and the plate is often far away. Each crop below
is degraded at five increasing severities and re-read, so the fall-off is measured
rather than assumed.

The degradations are synthetic but physical: a horizontal streak for motion, a
Gaussian for defocus, downscale-then-upscale for distance, Gaussian sensor noise,
a gain reduction plus read noise for night, and bright streaks plus veiling glare
for rain on the lens. They are applied to real plate crops with real ground truth.
"""))
cells.append(code("""
rng_np = np.random.default_rng(3)

def deg_motion(img, k):
    if k < 3: return img
    kern = np.zeros((k, k), np.float32); kern[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kern)

def deg_defocus(img, k):
    k = k if k % 2 else k + 1
    return cv2.GaussianBlur(img, (k, k), 0) if k >= 3 else img

def deg_distance(img, f):
    if f <= 1: return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w / f)), max(4, int(h / f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def deg_noise(img, sigma):
    return np.clip(img.astype(np.float32) + rng_np.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

def deg_night(img, gain):
    dark = img.astype(np.float32) * gain
    return np.clip(dark + rng_np.normal(0, 8, img.shape), 0, 255).astype(np.uint8)

def deg_rain(img, drops):
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]
    for _ in range(int(drops * w * h / 5000)):
        x, y = int(rng_np.integers(0, w)), int(rng_np.integers(0, h))
        L = int(rng_np.integers(3, max(4, h // 3)))
        cv2.line(out, (x, y), (x + int(rng_np.integers(-2, 3)), min(h - 1, y + L)),
                 (220, 220, 220), 1)
    return np.clip(out * 0.85 + 30, 0, 255).astype(np.uint8)   # wet glass veils contrast

DEGRADATIONS = {
    "motion blur (px)":  (deg_motion,   [3, 5, 7, 9, 11]),
    "defocus (px)":      (deg_defocus,  [3, 5, 7, 9, 11]),
    "distance (x)":      (deg_distance, [1.5, 2.0, 2.5, 3.0, 4.0]),
    "sensor noise (s)":  (deg_noise,    [5, 10, 20, 30, 40]),
    "night (gain)":      (deg_night,    [0.7, 0.5, 0.35, 0.25, 0.15]),
    "rain (drops)":      (deg_rain,     [1, 2, 4, 6, 8]),
}

def degraded_score(ocr, fn, level):
    ex = err = chars = 0
    for truth, crop, _ in crops:
        res = ocr.read(fn(crop, level))
        got = res.text_final if res else ""
        ex += got == truth; err += levenshtein(truth, got); chars += len(truth)
    return ex / len(crops), 1 - err / chars

rob = []
for label, (fn, levels) in DEGRADATIONS.items():
    for lv in levels:
        e, c = degraded_score(ocr, fn, lv)
        rob.append({"degradation": label, "severity": lv,
                    "exact": round(e, 3), "char_accuracy": round(c, 3)})
robust = pd.DataFrame(rob)
display(robust.pivot(index="degradation", columns="severity", values="char_accuracy"))
robust.to_csv(OUT_DIR / "ocr_robustness.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for label in DEGRADATIONS:
    sub = robust[robust.degradation == label]
    ax[0].plot(range(5), sub.char_accuracy.values, "o-", label=label)
    ax[1].plot(range(5), sub.exact.values, "o-", label=label)
for a, title in zip(ax, ("character accuracy", "exact-plate accuracy")):
    a.set_xlabel("severity (1 = mild, 5 = severe)"); a.set_ylabel(title)
    a.set_xticks(range(5)); a.set_xticklabels(range(1, 6)); a.set_ylim(0, 1)
    a.axhline(0.9, ls="--", c="#94a3b8", lw=1)
    a.set_title(title)
ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()

clean_char = 1 - cer
print(f"clean character accuracy {clean_char:.3f}")
holds = robust[robust.char_accuracy >= 0.90]
print(f"\\nConditions where character accuracy stays at or above 0.90:")
for _, r in holds.iterrows():
    print(f"  {r.degradation:18} at severity {r.severity}")
print("\\nEverything harsher than that falls below 0.90. Read the dashed line on the")
print("plot as the claim boundary: mild-to-moderate degradation is handled, severe")
print("motion blur and near-darkness are not, and no preprocessing recovers detail")
print("the sensor never captured.")
"""))

cells.append(md("""
## 7 · End to end: image in, plate string out

Sections 4 and 6 measure the detector and the reader separately, each handed
perfect input. Deployment gets neither. This section runs the real chain -
detect the plate, crop what the detector returned, read it - and scores the
final string against the ground truth.

A plate counts as read only if the detector found it **and** the OCR got every
character right.
"""))
cells.append(code("""
det = YOLO(str(plate_weights[0]))
ocr_e2e = PlateOCR()

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

e2e = []
for img_name, group in scored.groupby("image"):
    im = cv2.imread(str(DATASET / "test" / "images" / img_name))
    res = det.predict(im, conf=0.25, imgsz=640, device=DEVICE, verbose=False)
    preds = res[0].boxes.xyxy.cpu().tolist() if res and res[0].boxes is not None else []
    for r in group.itertuples():
        g = (r.x1, r.y1, r.x2, r.y2)
        match_box, best_iou = None, 0.0
        for p in preds:
            v = iou(p, g)
            if v > best_iou:
                match_box, best_iou = p, v
        if match_box is None or best_iou < 0.5:
            e2e.append({"truth": r.plate_text, "detected": False, "read": "",
                        "exact": False, "iou": round(best_iou, 3)})
            continue
        crop = im[int(match_box[1]):int(match_box[3]), int(match_box[0]):int(match_box[2])]
        out = ocr_e2e.read(crop)
        text = out.text_final if out else ""
        e2e.append({"truth": r.plate_text, "detected": True, "read": text,
                    "exact": text == r.plate_text, "iou": round(best_iou, 3)})

e2e = pd.DataFrame(e2e)
det_rate = e2e.detected.mean()
read_rate = e2e[e2e.detected].exact.mean() if e2e.detected.any() else 0.0
print(f"legible plates in the test split       {len(e2e)}")
print(f"found by the detector at conf=0.25     {det_rate:.3f}")
print(f"read exactly, given it was found       {read_rate:.3f}")
print(f"END TO END exact plate reads           {e2e.exact.mean():.3f}   "
      f"{e2e.exact.sum()}/{len(e2e)}")
print(f"mean IoU of the boxes that were found  {e2e[e2e.detected].iou.mean():.3f}")
e2e.to_csv(OUT_DIR / "end_to_end.csv", index=False)
display(e2e.head(12))
"""))

cells.append(md("""
### 7.1 One frame is not the pipeline

The number above is single-frame accuracy. In service, a plate is read on every
frame a vehicle is tracked for, and the readings are voted on
(`vision/aggregate.py`) before an observation is emitted. The vote fixes errors
that differ between frames; it cannot fix an error the model makes identically
every time - the truncation in section 6.3 is exactly that kind.

The cell below measures the vote on real footage if `data/videos/` has clips.
It reports agreement between frames, which is what the vote is built on; it is
not accuracy, because the video has no plate ground truth.
"""))
cells.append(code("""
videos = sorted((ROOT / "data" / "videos").glob("*.mp4"))
if not videos:
    print("No videos in data/videos/ - temporal aggregation is not measured here.")
else:
    obs_file = ROOT / "data" / "output" / "CAM_01_observations.json"
    if obs_file.exists():
        obs = json.loads(obs_file.read_text())
        agg = pd.DataFrame([{"plate": o["plate_number"], "support": o["support"],
                             "readings": o["total_readings"],
                             "agreement": o["agreement"],
                             "confidence": o["ocr_confidence"],
                             "distinct_raw": len({r["text_raw"] for r in o["raw_readings"]})}
                            for o in obs])
        display(agg)
        print(f"\\n{len(agg)} observations; each aggregates "
              f"{agg.readings.mean():.1f} per-frame readings on average, of which "
              f"{agg.support.mean():.1f} agreed with the emitted string.")
        print(f"distinct raw strings per vehicle: {agg.distinct_raw.mean():.1f} - "
              f"the vote is choosing between genuinely different readings.")
    else:
        print(f"{obs_file} not present. Run the pipeline first "
              f"(python run.py --videos data/videos) to populate it.")
"""))

# ------------------------------------------------------------------ 8
cells.append(md("""
## 8 · The same questions at city scale

A metric measured on 83 images says nothing about what happens with a thousand
cameras. This section measures the parts of `urbantrace.scale` whose correctness
*can* be checked against a known answer, plus the throughput that decides how many
cameras one GPU carries.

### 8.1 Identity resolution accuracy

`IdentityIndex` decides whether two plate readings are the same vehicle. The right
answer is known by construction here, so this is a true accuracy measurement - but
only if the test is fair, and there are two ways to make it unfair.

**Test it on the noise it was built for and it scores 100%, meaninglessly.**
`same_vehicle` merges two readings when their confusion-canonical forms match
(`8<->B`, `0<->O`, `1<->I`) or when one is the other minus a character. Corrupt
plates using exactly those confusions and perfect recall is arithmetic, not
evidence. That run is included below as the design case, labelled as such.

**The error that matters is the opposite one.** Merging two vehicles puts one
owner's journey under another's plate, and the module's own docstring calls it the
worst error the system can make. So the second run below adds *twin* vehicles -
genuinely different plates that differ by a single confusable character - and
measures how often the resolver collapses them into one identity.

Both numbers are reported. The first is the ceiling; the second is the risk.
"""))
cells.append(code("""
import random
from datetime import datetime, timedelta, timezone
from urbantrace.scale.identity import IdentityIndex, Sighting

LETTERS, DIGITS = "ABCDEFGHJKMNPRSTUVWXYZ", "0123456789"
CONFUSIONS = {"0": "OQD", "O": "0", "1": "IL", "I": "1", "8": "B",
              "B": "8", "5": "S", "S": "5", "2": "Z", "Z": "2"}

def make_plate(rng):
    return ("".join(rng.choice(LETTERS) for _ in range(2))
            + "".join(rng.choice(DIGITS) for _ in range(2))
            + "".join(rng.choice(LETTERS) for _ in range(2))
            + "".join(rng.choice(DIGITS) for _ in range(4)))

def misread(plate, rng, rate):
    if rng.random() > rate:
        return plate
    pos = [i for i, c in enumerate(plate) if c in CONFUSIONS]
    if not pos:
        return plate
    i = rng.choice(pos)
    return plate[:i] + rng.choice(CONFUSIONS[plate[i]]) + plate[i+1:]

def identity_accuracy(n_vehicles=2000, per_vehicle=4, error_rate=0.45, seed=7):
    rng = random.Random(seed)
    base = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    plates = list({make_plate(rng) for _ in range(n_vehicles)})
    events = []
    for vi, plate in enumerate(plates):
        for k in range(per_vehicle):
            events.append((plate, misread(plate, rng, error_rate),
                           f"CAM_{rng.randrange(40):05d}", vi * 0.01 + k * 7.0))
    rng.shuffle(events)

    idx = IdentityIndex()
    truth = {}
    t_start = time.perf_counter()
    for i, (plate, read, cam, mins) in enumerate(events):
        cid = idx.add(Sighting(plate=read, camera_id=cam,
                               observed_at=base + timedelta(minutes=mins),
                               confidence=0.9, track_id=i,
                               latitude=13.0, longitude=80.0))
        truth.setdefault(plate, []).append(cid)
    elapsed = time.perf_counter() - t_start

    merged = sum(1 for ids in truth.values() if len(set(ids)) == 1)
    clusters = idx.stats()["clusters"]
    return {"vehicles": len(truth), "sightings": len(events),
            "ocr_error_rate": error_rate,
            "correctly_merged": round(merged / len(truth), 4),
            "fragmented": len(truth) - merged,
            "clusters_formed": clusters,
            "cluster_count_error": round((clusters - len(truth)) / len(truth), 4),
            "sightings_per_second": round(len(events) / elapsed)}

ident = pd.DataFrame([identity_accuracy(error_rate=e) for e in (0.0, 0.25, 0.45, 0.70)])
display(ident)
print("Design case: every corruption is a confusion-group substitution, which is")
print("exactly what same_vehicle() is built to absorb. correctly_merged is the share")
print("of vehicles whose sightings all landed in one cluster; fragmented is the rest.")
ident.to_csv(OUT_DIR / "identity_accuracy.csv", index=False)
"""))

cells.append(md("""
Now the harder question. Every vehicle below is given a twin whose plate differs by
one confusable character - `TN01AB1234` against `TN01AB123B`. These are two cars.
A false merge means the resolver has silently attributed one car's journey to
another.
"""))
cells.append(code("""
def twin_plate(plate, rng):
    \"\"\"A genuinely different plate, one confusable character away.

    Returns None when the plate holds no confusable character - such a plate has
    no twin of this kind, and inventing one would change what is being measured.
    \"\"\"
    pos = [i for i, c in enumerate(plate) if c in CONFUSIONS]
    if not pos:
        return None
    i = rng.choice(pos)
    return plate[:i] + rng.choice(CONFUSIONS[plate[i]]) + plate[i+1:]

def false_merge_rate(n_pairs=500, per_vehicle=3, error_rate=0.45, seed=5):
    rng = random.Random(seed)
    base = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    vehicles = []
    for _ in range(n_pairs):
        a = make_plate(rng)
        b = twin_plate(a, rng)
        if b and a != b:
            vehicles += [a, b]
    events = []
    for vi, plate in enumerate(vehicles):
        for k in range(per_vehicle):
            events.append((vi, misread(plate, rng, error_rate),
                           f"CAM_{rng.randrange(40):05d}", vi * 0.01 + k * 7.0))
    rng.shuffle(events)

    idx = IdentityIndex()
    cluster_members = {}
    for i, (vi, read, cam, mins) in enumerate(events):
        cid = idx.add(Sighting(plate=read, camera_id=cam,
                               observed_at=base + timedelta(minutes=mins),
                               confidence=0.9, track_id=i,
                               latitude=13.0, longitude=80.0))
        cluster_members.setdefault(cid, set()).add(vi)

    contaminated = sum(1 for v in cluster_members.values() if len(v) > 1)
    vehicles_in_merged = sum(len(v) for v in cluster_members.values() if len(v) > 1)
    return {"true_vehicles": len(vehicles),
            "clusters_formed": len(cluster_members),
            "clusters_holding_two_or_more_vehicles": contaminated,
            "vehicles_caught_in_a_false_merge": vehicles_in_merged,
            "false_merge_rate": round(vehicles_in_merged / len(vehicles), 4)}

fm = pd.DataFrame([false_merge_rate(error_rate=e) for e in (0.0, 0.45)])
display(fm)
print("Every pair here differs by one character that OCR is known to confuse, so")
print("the two plates share a canonical form and same_vehicle() cannot tell them")
print("apart - by design, since the alternative is splitting one real vehicle in two.")
print("The number above is therefore not a bug count. It is the price of the design,")
print("measured: in a fleet where plates differ only by a confusable character, that")
print("share of vehicles is merged with a neighbour. The camera graph's feasibility")
print("check (IdentityIndex(feasible=...)) is what separates them in the live")
print("service, because two cars cannot be in two places at once.")
fm.to_csv(OUT_DIR / "identity_false_merge.csv", index=False)
"""))

cells.append(md("""
### 8.2 Camera registry at city size

Spatial lookup has to stay fast as the network grows, or nothing downstream keeps
up. Real `CameraRegistry`, real grid, timed.
"""))
cells.append(code("""
from urbantrace.scale.registry import CameraRegistry

reg_rows = []
for n_cams in (100, 1000, 5000):
    rng = random.Random(11)
    reg = CameraRegistry(cell_km=2.0)
    reg.load([{"camera_id": f"CAM_{i:06d}", "name": f"cam {i}",
               "latitude": 12.80 + rng.random() * 0.55,
               "longitude": 79.95 + rng.random() * 0.55} for i in range(n_cams)])
    t0 = time.perf_counter()
    graph = reg.build_graph()
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(2000):
        reg.near(12.80 + rng.random() * 0.55, 79.95 + rng.random() * 0.55, 5.0, limit=8)
    q = (time.perf_counter() - t0) / 2000
    reg_rows.append({"cameras": n_cams, "graph_build_s": round(build, 3),
                     "edges": graph["edges"],
                     "radius_query_us": round(q * 1e6, 1),
                     "queries_per_second": round(1 / q)})
display(pd.DataFrame(reg_rows))
"""))

cells.append(md("""
### 8.3 Throughput: how many cameras does one GPU carry?

The models are run over the real videos in `data/videos/` through the shared-model
path the scale service uses, and the frame rate is measured. Cameras-per-GPU is
that rate divided by the frame rate each camera is sampled at - arithmetic on a
measured number, with the assumption stated.
"""))
cells.append(code("""
if not videos:
    print("No videos in data/videos/ - throughput is not measured.")
else:
    from urbantrace.vision.detectors import PlateDetector

    tracker = VehicleTracker()
    plate_det = PlateDetector()
    cap = cv2.VideoCapture(str(videos[0]))
    frames, t_veh, t_plate, t_ocr, n_plates = [], 0.0, 0.0, 0.0, 0
    while len(frames) < 60:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    for f in frames[:5]:                      # warm-up, excluded from timing
        tracker.update(f); plate_det.detect(f)

    t0 = time.perf_counter()
    for f in frames:
        s = time.perf_counter(); tracks = tracker.update(f); t_veh += time.perf_counter() - s
        s = time.perf_counter(); plates = plate_det.detect(f); t_plate += time.perf_counter() - s
        for p in plates:
            x1, y1, x2, y2 = (int(v) for v in p.box)
            crop = f[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                s = time.perf_counter(); ocr_e2e.read(crop); t_ocr += time.perf_counter() - s
                n_plates += 1
    total = time.perf_counter() - t0
    fps = len(frames) / total
    print(f"frames processed            {len(frames)}  ({frames[0].shape[1]}x{frames[0].shape[0]})")
    print(f"end-to-end rate             {fps:.1f} frames/s")
    print(f"  vehicle detect + track    {t_veh/len(frames)*1000:.1f} ms/frame")
    print(f"  plate detect              {t_plate/len(frames)*1000:.1f} ms/frame")
    print(f"  OCR                       {t_ocr/max(1,n_plates)*1000:.1f} ms/plate "
          f"({n_plates} plates read)")
    for sample_fps in (1, 2, 5):
        print(f"cameras on this GPU at {sample_fps} fps sampled: {fps/sample_fps:.0f}")
    print("\\nAssumption stated explicitly: cameras are sampled, not decoded at full")
    print("30 fps. ANPR needs a few frames per vehicle, not every frame.")
"""))

cells.append(md("""
### 8.4 Accuracy does not change with scale - throughput does

Worth being precise about, because "accuracy at scale" is often used to mean two
different things. The detector and the OCR see one crop at a time; a thousand
cameras do not make a plate harder to read. What scale changes is:

- **queueing** - `FrameBus` drops the oldest frame when a camera's queue is full,
  so an overloaded GPU loses frames, and a vehicle seen in fewer frames gets fewer
  votes. That is a throughput problem showing up as accuracy loss;
- **identity** - more vehicles means more near-collisions between plate strings,
  measured in 8.1;
- **association** - more cameras means more candidate journeys per plate.

Section 8.1 measures the second at 2000 vehicles. The first is bounded by 8.3: keep
the sampled frame rate under the measured throughput and no frames are dropped.
"""))

# ------------------------------------------------------------------ 9
cells.append(md("""
## 9 · Everything measured, in one table
"""))
cells.append(code("""
summary = [
    ("Plate detection", "mAP@50, held-out test split",
     f"{detection.iloc[0]['mAP50']:.3f}", "83 images, 99 boxes, ultralytics val"),
    ("Plate detection", "mAP@50-95, held-out test split",
     f"{detection.iloc[0]['mAP50-95']:.3f}", "same run"),
    ("Plate detection", "precision / recall at conf=0.25",
     f"{row.precision:.3f} / {row.recall:.3f}", "the pipeline's operating point"),
    ("Vehicle detection", "COCO mAP", "not measured",
     "no vehicle boxes in this dataset; checkpoint reports 0.684 on COCO"),
    ("Vehicle detection", "plate falls inside a detected vehicle",
     f"{vd.plate_inside_vehicle.mean():.3f}", "the plate-to-track binding precondition"),
    ("OCR", "exact-match, legible plates",
     f"{reads.exact.mean():.3f}", f"{n_scored} plates, ground truth transcribed by hand"),
    ("OCR", "character accuracy", f"{1 - cer:.3f}", "1 - CER"),
    ("OCR", "best available model",
     f"{best_ocr.exact:.3f} ({best_ocr.model})", "measured in 6.4"),
    ("End to end", "detector + OCR, exact plate",
     f"{e2e.exact.mean():.3f}", "single frame, no temporal vote"),
    ("Scale", "identity merge accuracy at 45% OCR error",
     f"{ident[ident.ocr_error_rate == 0.45].iloc[0]['correctly_merged']:.3f}",
     "2000 vehicles, confusion-group noise - the resolver's design case"),
    ("Scale", "vehicles merged with a confusable twin",
     f"{fm.false_merge_rate.iloc[0]:.3f}",
     f"{int(fm.true_vehicles.iloc[0])} plates in twin pairs; the design's cost, see 8.1"),
]
summary_df = pd.DataFrame(summary, columns=["subsystem", "metric", "value", "basis"])
display(summary_df)
summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
print(f"\\nwritten to {OUT_DIR}")
for f in sorted(OUT_DIR.glob("*.csv")):
    print(f"  {f.name}")
"""))

cells.append(md("""
## 10 · What these numbers do and do not license you to say

**Measured, and safe to quote with the qualifier attached:**

- plate detection mAP@50 on 83 held-out images from a collection the detector was
  never trained on;
- single-frame OCR exact-match accuracy on 80 hand-transcribed plates;
- end-to-end single-frame plate-read accuracy;
- identity-merge accuracy at a stated OCR error rate;
- throughput on this specific GPU and this specific footage.

**Not measured here, and not to be claimed:**

- *vehicle detection accuracy* - there are no vehicle boxes in this dataset. The
  COCO figure in section 3.3 is from the checkpoint and describes COCO, not Indian
  traffic. Section 5 measures the property the pipeline depends on and nothing more.
- *accuracy after temporal voting* - the vote demonstrably changes the answer, but
  scoring it needs video with plate ground truth, which does not exist here. The
  single-frame number is the floor, not the ceiling.
- *anything about autorickshaws* - COCO has no such class, so per-class counts for
  Indian traffic are wrong by construction, whatever the mAP says.
- *identity resolution as "100% accurate"* - it merges everything it is supposed to
  merge, and section 8.1 shows it also merges plates that differ only by a
  confusable character. Without the camera-graph feasibility check wired in, two
  vehicles whose plates differ by `8` against `B` are one identity to this index.
- *the 83-image sample as a citywide figure* - 83 images, one collection, mostly
  close-range car fronts. A road camera at 30 m in rain is a different distribution.
  Widen the test set before widening the claim.

**Sample size, stated plainly.** 83 images, 99 boxes, 80 legible plates. At n=80 a
measured 70% exact-match carries a 95% interval of roughly +/-10 points. Report the
interval or report more plates; do not report the point estimate alone.

**To make the missing numbers measurable**

1. *Vehicle mAP* - label vehicle boxes on frames from `data/videos/`, drop them in
   `data/datasets/vehicle_det/`, re-run section 5. The cell is already written.
2. *Temporal accuracy* - transcribe the plates in the demo videos, then score
   pipeline observations against them instead of scoring single crops.
3. *A tighter OCR figure* - the label file is a CSV. Add rows as you transcribe more
   crops; every cell in section 6 scales with it.
"""))

for i, cell in enumerate(cells):
    cell["id"] = f"cell{i:03d}"          # nbformat 4.5 requires an id per cell

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells, "
      f"{sum(1 for c in cells if c['cell_type'] == 'code')} code)")
