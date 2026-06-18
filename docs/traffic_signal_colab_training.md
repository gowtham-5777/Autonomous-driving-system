# Traffic Signal Detection — Google Colab Training Guide

**Model:** YOLOv8n  
**Dataset:** BDD100K-derived `traffic_signal_yolo` (ADAS-aligned labels: red=0, yellow=1, green=2)  
**Target GPU:** Colab T4 (16 GB VRAM)  
**Output weights:** `traffic_signals_yolov8n.pt`

---

## Overview

| Step | Where | Action |
|------|-------|--------|
| 1 | Local PC | Export dataset zip |
| 2 | Google Drive | Upload zip |
| 3 | Colab | Extract, train, save weights |
| 4 | Local PC | Download `traffic_signals_yolov8n.pt` into project |

---

## Part 1 — Export dataset locally (Windows)

Run from the project root **after** `prepare_traffic_signal_dataset.py` has completed:

```powershell
cd "c:\Users\gauth\Desktop\Autonomous Driving Car"
python scripts/export_traffic_signal_dataset.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `data/traffic_signal_yolo` | Source dataset directory |
| `--output` | `data/traffic_signal_yolo_colab.zip` | Output zip path |
| `--colab-path` | `/content/traffic_signal_yolo` | Path written into `dataset.yaml` |
| `--no-stats` | off | Skip `dataset_stats.json` |
| `--compresslevel` | `1` | 1=fast, 9=smaller zip |

### Zip contents

```
traffic_signal_yolo/
├── dataset.yaml
├── dataset_stats.json
├── images/
│   ├── train/    # 36,765 JPG
│   └── val/      # 5,283 JPG
└── labels/
    ├── train/    # 36,765 TXT
    └── val/      # 5,283 TXT
```

### Expected size

| Item | Approximate size |
|------|------------------|
| Zip file | **6–10 GB** (42,048 images at 1280×720) |
| Upload time | Depends on connection (use Drive desktop app for large files) |

Output files:

- `data/traffic_signal_yolo_colab.zip`
- `data/traffic_signal_yolo_colab.manifest.json`

---

## Part 2 — Upload to Google Drive

1. Create a folder on Drive, e.g.:

   ```
   My Drive/adas-project/
   ```

2. Upload `traffic_signal_yolo_colab.zip` to that folder.

3. (Optional) Create a weights output folder:

   ```
   My Drive/adas-project/models/trained/yolov8_signal/
   ```

Recommended Drive layout:

```
My Drive/adas-project/
├── traffic_signal_yolo_colab.zip
├── datasets/                    # optional alternate location
└── models/trained/yolov8_signal/
    └── traffic_signals_yolov8n.pt   # saved after training
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
ZIP_PATH = DRIVE_ROOT / "traffic_signal_yolo_colab.zip"
DATASET_DIR = Path("/content/traffic_signal_yolo")
WEIGHTS_DIR = DRIVE_ROOT / "models" / "trained" / "yolov8_signal"
RUNS_DIR = Path("/content/runs/traffic_signal")

DATASET_YAML = DATASET_DIR / "dataset.yaml"
FINAL_WEIGHTS = WEIGHTS_DIR / "traffic_signals_yolov8n.pt"

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
path: /content/traffic_signal_yolo
train: images/train
val: images/val
nc: 3
names:
  0: red
  1: yellow
  2: green
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
    name="yolov8n_bdd100k",
    exist_ok=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
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
| 100 epochs, batch 16, ~37k train images | **6–9 hours** |
| With early stopping (patience 20) | Often **4–7 hours** |

---

## Part 4 — Resume interrupted training

If Colab disconnects, remount Drive, re-extract only if `/content` was cleared, then resume from `last.pt`:

### Cell — Resume

```python
from pathlib import Path
from ultralytics import YOLO

RUN_DIR = Path("/content/runs/traffic_signal/yolov8n_bdd100k")
LAST_PT = RUN_DIR / "weights" / "last.pt"

if not LAST_PT.is_file():
    raise FileNotFoundError(
        f"No checkpoint at {LAST_PT}. Re-run training or copy last.pt from a backup."
    )

model = YOLO(str(LAST_PT))
results = model.train(resume=True)
print("Resumed run:", results.save_dir)
```

**Tips for resume:**

- Colab wipes `/content` on disconnect — keep a Drive backup of the run folder:

  ```python
  # Run periodically during training or after disconnect
  import shutil
  backup = DRIVE_ROOT / "runs_backup" / "yolov8n_bdd100k"
  backup.parent.mkdir(parents=True, exist_ok=True)
  if RUN_DIR.exists():
      shutil.copytree(RUN_DIR, backup, dirs_exist_ok=True)
      print("Backed up to", backup)
  ```

- To resume from Drive backup, copy back to `/content/runs/traffic_signal/yolov8n_bdd100k/` before calling `resume=True`.

---

## Part 5 — Save best weights to Drive

After training completes (or after the final resume session):

### Cell — Copy and rename best.pt

```python
import shutil
from pathlib import Path

run_dir = Path("/content/runs/traffic_signal/yolov8n_bdd100k")
best_pt = run_dir / "weights" / "best.pt"

if not best_pt.is_file():
    raise FileNotFoundError(f"best.pt not found at {best_pt}")

shutil.copy2(best_pt, FINAL_WEIGHTS)
print("Saved:", FINAL_WEIGHTS)
print("Size MB:", round(FINAL_WEIGHTS.stat().st_size / 1024 / 1024, 2))
```

This writes:

```
/content/drive/MyDrive/adas-project/models/trained/yolov8_signal/traffic_signals_yolov8n.pt
```

---

## Part 6 — Download to local project

1. Download from Drive, or copy via Drive desktop sync.
2. Place the file at:

   ```
   models/trained/yolov8_signal/traffic_signals_yolov8n.pt
   ```

3. Verify ADAS config (`config/default.yaml`) already points to this filename.

4. Run local evaluation (optional):

   ```powershell
   python scripts/evaluate_traffic_signal.py `
     --weights models/trained/yolov8_signal/traffic_signals_yolov8n.pt
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
| Zip upload slow | Use Google Drive desktop app; upload overnight |
| Class mismatch at inference | Confirm `dataset.yaml` uses `0:red, 1:yellow, 2:green` (ADAS-aligned) |
| `best.pt` missing | Training may not have finished; check `last.pt` and resume |

---

## Class mapping reference

| YOLO ID | Name | ADAS label |
|---------|------|------------|
| 0 | red | red_light |
| 1 | yellow | yellow_light |
| 2 | green | green_light |

Training labels match `src/modules/yolov8_signal/class_map.py` — no remapping needed at deployment.

---

## Related files

| File | Purpose |
|------|---------|
| `scripts/export_traffic_signal_dataset.py` | Build Colab zip |
| `scripts/prepare_traffic_signal_dataset.py` | Build local YOLO dataset |
| `scripts/evaluate_traffic_signal.py` | Local evaluation after download |
| `docs/traffic_signal_training_report.md` | Full training pipeline report |
| `docs/traffic_signal_label_alignment_report.md` | ADAS label compatibility |
