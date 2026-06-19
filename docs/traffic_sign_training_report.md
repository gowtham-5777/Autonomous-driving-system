# Traffic Sign Detection — Training Report

**Repository:** Autonomous Driving Car  
**Date:** June 2026  
**Model:** YOLOv8n fine-tuned on GTSRB (7-class ADAS subset)  
**Status:** Training pipeline complete — run prepare → train → evaluate after GTSRB download

---

## 1. Executive Summary

This report documents the **GTSRB → YOLOv8n** training pipeline for 7-class traffic-sign detection. The YOLO class order matches the ADAS inference head in `src/modules/yolov8_sign/class_map.py` — **no inference code changes required** after training.

| Stage | Script | Output |
|-------|--------|--------|
| Dataset preparation | `scripts/prepare_traffic_sign_dataset.py` | `data/traffic_sign_yolo/` |
| Training | `scripts/train_traffic_sign.py` | `models/trained/yolov8_sign/traffic_signs_yolov8n.pt` |
| Evaluation | `scripts/evaluate_traffic_sign.py` | `runs/traffic_sign/evaluation_metrics.json` |
| Colab export | `scripts/export_traffic_sign_dataset.py` | `data/traffic_sign_yolo_colab.zip` |
| Colab guide | `docs/traffic_sign_colab_training.md` | GPU training workflow |

---

## 2. Source Dataset Structure (GTSRB)

### 2.1 Layout

```
{gtsrb_root}/
├── Train/
│   └── {ClassId:05d}/     # 00000–00042 (43 GTSRB classes)
│       └── *.ppm
├── Test/
│   └── *.ppm
└── Test.csv               # Filename, ROI, ClassId for test images
```

Default path: `C:\Users\gauth\Desktop\ADAS_DATASETS\gtsrb` (override via `--gtsrb` or `ADAS_GTSRB_ROOT`).

Config reference: `config/default.yaml` → `paths.gtsrb: "${data_root}/datasets/gtsrb"`

### 2.2 v1 ADAS subset

Only **7 of 43** GTSRB classes are exported. All other GTSRB folders are skipped.

| GTSRB ID | GTSRB meaning (approx.) | YOLO ID | ADAS label |
|----------|-------------------------|---------|------------|
| 14 | Stop | 0 | stop |
| 1 | Speed limit 30 | 1 | speed_limit_30 |
| 5 | Speed limit 60 | 2 | speed_limit_60 |
| 38 | Turn left ahead | 3 | turn_left |
| 34 | Turn right ahead | 4 | turn_right |
| 36 | Pass on right | 5 | keep_right |
| 12 | Pedestrians | 6 | pedestrian_crossing |

Mapping defined in `training/traffic_sign/gtsrb_converter.py` (`GTSRB_TO_YOLO_CLASS`) and verified against `src/modules/yolov8_sign/class_map.py` (`GTSRB_CLASS_ID_TO_ADAS_LABEL`).

### 2.3 Bounding-box strategy

| Split | Image type | Label strategy |
|-------|------------|----------------|
| **Train** | GTSRB cropped patches | Full-frame box `(cx=0.5, cy=0.5, w=1.0, h=1.0)` — sign fills crop |
| **Val (default)** | GTSRB official test | ROI from `Test.csv` converted to normalized YOLO bbox |
| **Val (optional)** | `--stratified-val` | 20% hold-out from train crops (seed=42) |

**Known limitation:** Train crops teach **sign classification** strongly but **scene localization** weakly until full-frame fine-tuning (Bdd100K signs, synthetic paste) is added in a future phase.

---

## 3. Dataset Statistics

Statistics are written to `data/traffic_sign_yolo/dataset_stats.json` when `prepare_traffic_sign_dataset.py` runs.

### 3.1 Expected corpus counts (official GTSRB mapped subset)

| Split | Source | Approx. images | Notes |
|-------|--------|----------------|-------|
| Train | GTSRB `Train/` mapped folders | **~11,150** | All mapped class crops |
| Val | GTSRB `Test/` + `Test.csv` | **~2,500** | Mapped test rows only |
| **Total** | | **~13,650** | Exact counts in `dataset_stats.json` |

### 3.2 Expected per-class train distribution (GTSRB benchmark)

| ADAS label | GTSRB ID | Approx. train images |
|------------|----------|----------------------|
| speed_limit_30 | 1 | 2,220 |
| speed_limit_60 | 5 | 1,860 |
| stop | 14 | 2,100 |
| turn_right | 34 | 1,920 |
| keep_right | 36 | 1,410 |
| turn_left | 38 | 1,380 |
| pedestrian_crossing | 12 | 1,260 |

> **Note:** Run `python scripts/prepare_traffic_sign_dataset.py` after downloading GTSRB to generate exact counts on your machine.

---

## 4. YOLO Output Dataset

### 4.1 Directory layout

```
data/traffic_sign_yolo/
├── images/
│   ├── train/          # JPG (converted from PPM)
│   └── val/
├── labels/
│   ├── train/          # one box per train crop
│   └── val/            # ROI boxes from Test.csv
├── dataset.yaml
└── dataset_stats.json
```

Images under `images/` are gitignored; labels and metadata are tracked/regenerated via the prepare script.

### 4.2 Class mapping (ADAS-aligned)

| YOLO class ID | YOLO name | ADAS runtime label |
|---------------|-----------|-------------------|
| 0 | stop | stop |
| 1 | speed_limit_30 | speed_limit_30 |
| 2 | speed_limit_60 | speed_limit_60 |
| 3 | turn_left | turn_left |
| 4 | turn_right | turn_right |
| 5 | keep_right | keep_right |
| 6 | pedestrian_crossing | pedestrian_crossing |

For signs, YOLO training names **equal** ADAS parser labels (unlike traffic signals where `red` → `red_light`).

### 4.3 `dataset.yaml`

```yaml
path: C:/Users/gauth/Desktop/Autonomous Driving Car/data/traffic_sign_yolo
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

### 4.4 YOLO label format

Each line: `class_id x_center y_center width height` (normalized 0–1).

---

## 5. Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `yolov8n.pt` | Ultralytics nano variant |
| Epochs | 100 | Early stop via patience |
| Image size | 640 | Matches inference `yolov8_sign.imgsz` |
| Batch size | 16 | Reduce to 8 on 6 GB GPU |
| Patience | 20 | Early stopping |
| Output weights | `models/trained/yolov8_sign/traffic_signs_yolov8n.pt` | Copied from `best.pt` |
| Run directory | `runs/traffic_sign/yolov8n_gtsrb/` | Ultralytics artifacts |

### 5.1 Augmentation (sign-tuned)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `degrees` | 10.0 | Sign rotation invariance |
| `hsv_h/s/v` | 0.015 / 0.7 / 0.4 | Color jitter for sign recognition |
| `mosaic` | 1.0 | Standard YOLOv8 multi-image mosaic |
| `fliplr` | 0.5 | Horizontal flip |

Defined in `training/traffic_sign/config.py` → `TrafficSignTrainingConfig`.

### 5.2 Hardware requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 6 GB VRAM (batch=8) | 8+ GB VRAM (batch=16) |
| CPU RAM | 8 GB | 16 GB |
| Disk | ~500 MB YOLO dataset + GTSRB source | SSD preferred |

### 5.3 Expected training time

| Hardware | Approx. time (100 epochs, ~11k images) |
|----------|----------------------------------------|
| NVIDIA RTX 3060 (12 GB) | 30–60 minutes |
| Google Colab T4 | 45–90 minutes |
| CPU only | 4–8 hours (not recommended) |

---

## 6. Evaluation Methodology

### 6.1 Script

```bash
python scripts/evaluate_traffic_sign.py \
  --weights models/trained/yolov8_sign/traffic_signs_yolov8n.pt \
  --data data/traffic_sign_yolo/dataset.yaml
```

### 6.2 Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | Mean precision across classes |
| **Recall** | Mean recall across classes |
| **F1** | `2 × P × R / (P + R)` |
| **mAP@0.5** | Mean AP at IoU 0.50 |
| **mAP@0.5:0.95** | COCO-style AP (IoU 0.50–0.95) |

Per-class AP@0.5 is reported under both YOLO names and ADAS labels.

### 6.3 Expected validation performance

On GTSRB cropped train / ROI test data, expect **high mAP@0.5 (>0.90)** because train crops use full-frame boxes and val ROIs are tight. **Dashcam full-frame performance will be lower** until domain fine-tuning — document this gap in interviews.

Record actual metrics in `runs/traffic_sign/evaluation_metrics.json` after training on your hardware.

---

## 7. Quick Start

```bash
# 0. Download GTSRB to ADAS_DATASETS/gtsrb (or set --gtsrb)

# 1. Prepare YOLO dataset (7-class ADAS-aligned labels)
python scripts/prepare_traffic_sign_dataset.py

# 2. Train YOLOv8n
python scripts/train_traffic_sign.py --device 0 --epochs 100 --batch 16

# 3. Evaluate
python scripts/evaluate_traffic_sign.py --device 0

# 4. Export for Colab (optional — if training locally first)
python scripts/export_traffic_sign_dataset.py
```

Colab workflow: see `docs/traffic_sign_colab_training.md`.

---

## 8. Generated Files

| File | Purpose |
|------|---------|
| `training/traffic_sign/gtsrb_converter.py` | GTSRB → YOLO conversion + ADAS mapping |
| `training/traffic_sign/config.py` | Hyperparameters and default paths |
| `scripts/prepare_traffic_sign_dataset.py` | Dataset builder |
| `scripts/train_traffic_sign.py` | YOLOv8n training |
| `scripts/evaluate_traffic_sign.py` | Val metrics with ADAS class names |
| `scripts/export_traffic_sign_dataset.py` | Colab zip export |
| `tests/test_traffic_sign_training.py` | Alignment unit tests (5 tests) |

### Not modified

- `src/modules/traffic_sign.py`
- `src/modules/yolov8_sign/*`
- `config/default.yaml` / `config/classes.yaml`

---

## 9. ADAS Integration

Training class order **matches** `SIGN_CLASS_ID_TO_LABEL` in `class_map.py`:

| Class ID | Training / YOLO | ADAS parser |
|----------|-----------------|-------------|
| 0 | stop | stop |
| 1 | speed_limit_30 | speed_limit_30 |
| 2 | speed_limit_60 | speed_limit_60 |
| 3 | turn_left | turn_left |
| 4 | turn_right | turn_right |
| 5 | keep_right | keep_right |
| 6 | pedestrian_crossing | pedestrian_crossing |

After training, copy weights to:

```
models/trained/yolov8_sign/traffic_signs_yolov8n.pt
```

The ADAS `YOLOv8SignOutputParser` maps class IDs directly — no `class_map.py` changes needed.

### Deployment workflow

1. Train locally or on Colab (`docs/traffic_sign_colab_training.md`)
2. Place `traffic_signs_yolov8n.pt` under `models/trained/yolov8_sign/`
3. Set `ADAS_DATA_ROOT` to project root (or use Colab `data_root`)
4. Run `python scripts/verify_traffic_sign_detection.py --real`
5. Enable in pipeline: `config/default.yaml` → `pipeline.run_signs: true`
6. Streamlit real mode requires all four weight files (`app.py` → `real_weights_available()`)

---

## 10. References

- GTSRB: https://benchmark.ini.rub.de/gtsrb.html
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- Design: `docs/traffic_sign_detection_design.md`
- Implementation: `docs/traffic_sign_implementation_report.md`
- Signal pipeline (mirror): `docs/traffic_signal_training_report.md`
