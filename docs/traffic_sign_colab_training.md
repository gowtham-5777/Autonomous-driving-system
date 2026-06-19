# Traffic Sign Detection — Google Colab Training Guide

**Model:** YOLOv8n  
**Dataset:** GTSRB-derived `traffic_sign_yolo` (ADAS-aligned 7-class labels)  
**Target GPU:** Colab T4 (16 GB VRAM)  
**Output weights:** `traffic_signs_yolov8n.pt`

---

## Overview

| Step | Where | Action |
|------|-------|--------|
| 1 | Local PC | Download GTSRB + run `prepare_traffic_sign_dataset.py` |
| 2 | Local PC | Export dataset zip |
| 3 | Google Drive | Upload zip |
| 4 | Colab | Extract, train, save weights |
| 5 | Local PC | Download `traffic_signs_yolov8n.pt` into project |

---

## Part 0 — Obtain GTSRB (one-time)

Download and extract the **German Traffic Sign Recognition Benchmark (GTSRB)** so the layout matches:

```
datasets/gtsrb/
├── Train/
│   ├── 00001/          # speed limit 30 (mapped)
│   ├── 00005/          # speed limit 60
│   ├── 00012/          # pedestrians
│   ├── 00014/          # stop
│   ├── 00034/          # turn right
│   ├── 00036/          # keep right
│   ├── 00038/          # turn left
│   └── ...             # other classes ignored in v1
├── Test/
│   └── *.ppm
└── Test.csv
```

Default local path (override with `--gtsrb` or env `ADAS_GTSRB_ROOT`):

```
C:\Users\gauth\Desktop\ADAS_DATASETS\gtsrb
```

Prepare the YOLO dataset:

```powershell
cd "c:\Users\gauth\Desktop\Autonomous Driving Car"
python scripts/prepare_traffic_sign_dataset.py --gtsrb "C:\path\to\gtsrb"
```

By default, **validation uses the official GTSRB test split** (`Test/` + `Test.csv`) with ROI boxes. Use `--stratified-val` for an 80/20 hold-out from train crops instead.

---

## Part 1 — Export dataset locally (Windows)

Run from the project root **after** `prepare_traffic_sign_dataset.py` has completed:

```powershell
python scripts/export_traffic_sign_dataset.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `data/traffic_sign_yolo` | Source dataset directory |
| `--output` | `data/traffic_sign_yolo_colab.zip` | Output zip path |
| `--colab-path` | `/content/traffic_sign_yolo` | Path written into `dataset.yaml` |
| `--no-stats` | off | Skip `dataset_stats.json` |
| `--compresslevel` | `1` | 1=fast, 9=smaller zip |

### Zip contents

```
traffic_sign_yolo/
├── dataset.yaml
├── dataset_stats.json
├── images/
│   ├── train/    # ~11k mapped GTSRB crops (JPG)
│   └── val/      # official GTSRB test subset (~2.5k mapped)
└── labels/
    ├── train/
    └── val/
```

### Expected size

| Item | Approximate size |
|------|------------------|
| Zip file | **150–350 MB** (depends on JPG conversion) |
| Upload time | Use Drive desktop app for large files |

Output files:

- `data/traffic_sign_yolo_colab.zip`
- `data/traffic_sign_yolo_colab.manifest.json`

---

## Part 2 — Upload to Google Drive

1. Create a folder on Drive, e.g.:

   ```
   My Drive/adas-project/
   ```

2. Upload `traffic_sign_yolo_colab.zip`.

3. Create a weights output folder:

   ```
   My Drive/adas-project/models/trained/yolov8_sign/
   ```

Recommended Drive layout:

```
My Drive/adas-project/
├── traffic_sign_yolo_colab.zip
├── datasets/gtsrb/              # optional raw archive backup
└── models/trained/yolov8_sign/
    └── traffic_signs_yolov8n.pt   # saved after training
```

---

## Part 3 — Colab notebook

Create a new **Python 3** notebook with **GPU** runtime:

**Runtime → Change runtime type → T4 GPU**

### Cell 1 — Mount Drive and install dependencies

```python
from google.colab import drive
drive.mount("/content/drive")

!pip install -q ultralytics>=8.0
```

### Cell 2 — Paths (edit `DRIVE_ROOT` if needed)

```python
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/adas-project")
ZIP_PATH = DRIVE_ROOT / "traffic_sign_yolo_colab.zip"
DATASET_DIR = Path("/content/traffic_sign_yolo")
WEIGHTS_DIR = DRIVE_ROOT / "models" / "trained" / "yolov8_sign"
RUNS_DIR = Path("/content/runs/traffic_sign")

DATASET_YAML = DATASET_DIR / "dataset.yaml"
FINAL_WEIGHTS = WEIGHTS_DIR / "traffic_signs_yolov8n.pt"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
print("Zip exists:", ZIP_PATH.is_file())
```

### Cell 3 — Extract dataset

```python
import zipfile
import shutil

if DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall("/content")

assert DATASET_YAML.is_file(), f"Missing {DATASET_YAML}"
print(DATASET_YAML.read_text())
```

`dataset.yaml` inside the zip is preconfigured with:

```yaml
path: /content/traffic_sign_yolo
train: images/train
val: images/val
nc: 7
names:
  0: stop
  1: speed_limit_30
  2: speed_limit_60
  3: turn_left
  4: turn_right
  5: keep_right
  6: pedestrian_crossing
```

### Cell 4 — Train YOLOv8n on T4

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data=str(DATASET_YAML),
    epochs=100,
    imgsz=640,
    batch=16,          # use batch=8 if CUDA OOM on T4
    patience=20,
    workers=2,
    device=0,
    seed=42,
    project=str(RUNS_DIR),
    name="yolov8n_gtsrb",
    exist_ok=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    save=True,
    plots=True,
    verbose=True,
)

run_dir = Path(results.save_dir)
print("Run directory:", run_dir)
```

### Expected training time (Colab T4)

| Setting | Estimate |
|---------|----------|
| 100 epochs, batch 16, ~11k train images | **1–3 hours** |
| With early stopping (patience 20) | Often **45–90 minutes** |

---

## Part 4 — Resume interrupted training

```python
from pathlib import Path
from ultralytics import YOLO

RUN_DIR = Path("/content/runs/traffic_sign/yolov8n_gtsrb")
LAST_PT = RUN_DIR / "weights" / "last.pt"

if not LAST_PT.is_file():
    raise FileNotFoundError(f"No checkpoint at {LAST_PT}")

model = YOLO(str(LAST_PT))
results = model.train(resume=True)
print("Resumed run:", results.save_dir)
```

Backup run folder to Drive periodically:

```python
import shutil
backup = DRIVE_ROOT / "runs_backup" / "yolov8n_gtsrb"
backup.parent.mkdir(parents=True, exist_ok=True)
if RUN_DIR.exists():
    shutil.copytree(RUN_DIR, backup, dirs_exist_ok=True)
    print("Backed up to", backup)
```

---

## Part 5 — Save best weights to Drive

```python
import shutil
from pathlib import Path

run_dir = Path("/content/runs/traffic_sign/yolov8n_gtsrb")
best_pt = run_dir / "weights" / "best.pt"

if not best_pt.is_file():
    raise FileNotFoundError(f"best.pt not found at {best_pt}")

shutil.copy2(best_pt, FINAL_WEIGHTS)
print("Saved:", FINAL_WEIGHTS)
print("Size MB:", round(FINAL_WEIGHTS.stat().st_size / 1024 / 1024, 2))
```

Target path on Drive:

```
/content/drive/MyDrive/adas-project/models/trained/yolov8_sign/traffic_signs_yolov8n.pt
```

---

## Part 6 — Download to local project

1. Download from Drive or sync via Drive desktop app.
2. Place the file at:

   ```
   models/trained/yolov8_sign/traffic_signs_yolov8n.pt
   ```

3. Verify ADAS config (`config/default.yaml`) already points to this filename under `weight_files.yolov8_sign`.

4. Run local evaluation:

   ```powershell
   python scripts/evaluate_traffic_sign.py `
     --weights models/trained/yolov8_sign/traffic_signs_yolov8n.pt
   ```

5. Run the ADAS gate:

   ```powershell
   python scripts/verify_traffic_sign_detection.py --real
   ```

---

## Part 7 — Quick validation on Colab (optional)

```python
from ultralytics import YOLO

model = YOLO(str(FINAL_WEIGHTS))
metrics = model.val(
    data=str(DATASET_YAML),
    split="val",
    imgsz=640,
    batch=16,
    device=0,
)
print("mAP@0.5:", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM on T4 | Set `batch=8` or `batch=4` |
| `dataset.yaml` not found | Re-run extract cell; check `DATASET_DIR` |
| Colab disconnected | Backup `runs/` to Drive; resume from `last.pt` |
| GTSRB folder missing | Download GTSRB; run `prepare_traffic_sign_dataset.py` locally first |
| Class mismatch at inference | Confirm 7-class order matches `src/modules/yolov8_sign/class_map.py` |
| Poor dashcam localization | Expected limitation — GTSRB crops teach classification; fine-tune on full-frame data later |

---

## Class mapping reference

| YOLO ID | Name | GTSRB ID | Regulatory |
|---------|------|----------|------------|
| 0 | stop | 14 | Yes |
| 1 | speed_limit_30 | 1 | Yes |
| 2 | speed_limit_60 | 5 | Yes |
| 3 | turn_left | 38 | No |
| 4 | turn_right | 34 | No |
| 5 | keep_right | 36 | Yes |
| 6 | pedestrian_crossing | 12 | Warning |

Training labels match `SIGN_CLASS_ID_TO_LABEL` in `src/modules/yolov8_sign/class_map.py` — no remapping needed at deployment.

---

## Related files

| File | Purpose |
|------|---------|
| `scripts/export_traffic_sign_dataset.py` | Build Colab zip |
| `scripts/prepare_traffic_sign_dataset.py` | Build local YOLO dataset |
| `scripts/train_traffic_sign.py` | Local YOLOv8n training |
| `scripts/evaluate_traffic_sign.py` | Local evaluation after download |
| `docs/traffic_sign_training_report.md` | Full training pipeline report |
| `docs/traffic_sign_detection_design.md` | Module design rationale |
