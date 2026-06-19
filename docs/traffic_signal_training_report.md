# Traffic Signal Detection — Training Report

**Repository:** Autonomous Driving Car  
**Date:** June 2026  
**Model:** YOLOv8n fine-tuned on BDD100K traffic-light annotations  
**Status:** Training pipeline complete — weights produced and validated on BDD100K val split.

---

## 1. Executive Summary

This report documents the **BDD100K → YOLOv8n** training pipeline for 3-class traffic-signal state detection. The YOLO class order matches the ADAS inference head in `src/modules/yolov8_signal/class_map.py` — **no inference code changes required** after training.

| Stage | Script | Output |
|-------|--------|--------|
| Dataset preparation | `scripts/prepare_traffic_signal_dataset.py` | `data/traffic_signal_yolo/` |
| Training | `scripts/train_traffic_signal.py` | `models/trained/yolov8_signal/traffic_signals_yolov8n.pt` |
| Evaluation | `scripts/evaluate_traffic_signal.py` | `runs/traffic_signal/evaluation_metrics.json` |
| Inference demo | `scripts/predict_traffic_signal.py` | `outputs/traffic_signal_predictions/` |

---

## 2. Source Dataset Structure (BDD100K)

### 2.1 Layout

```
C:\Users\gauth\Desktop\ADAS_DATASETS\
├── bdd100k_images\
│   ├── train\          # 70,000 JPG frames (1280×720)
│   └── val\            # 10,000 JPG frames
└── bdd100k_labels\
    ├── train\          # 70,000 JSON annotations (matching stems)
    └── val\            # 10,000 JSON annotations
```

### 2.2 Annotation format

BDD100K stores per-frame objects under `frames[0].objects[]`. Traffic lights are filtered with:

- `category == "traffic light"`
- `attributes.trafficLightColor` ∈ `{red, yellow, green}` (boxes with `"none"` are skipped)
- Valid `box2d` with positive area inside image bounds

### 2.3 Filtering rules

| Rule | Action |
|------|--------|
| `trafficLightColor == "none"` | Skip (unlabeled state) |
| Missing / unknown color | Skip |
| `x2 <= x1` or `y2 <= y1` | Skip (invalid box) |
| Box outside 1280×720 bounds | Skip |
| Width or height < 1 px | Skip |

---

## 3. Dataset Statistics

Statistics from `data/traffic_signal_yolo/dataset_stats.json` (box counts by color are unchanged; class IDs were realigned).

### 3.1 Corpus-wide counts

| Split | Source JSON files | Images with ≥1 valid TL | Boxes kept |
|-------|-------------------|-------------------------|------------|
| Train | 70,000 | 36,765 | 129,012 |
| Val | 10,000 | 5,283 | 18,538 |
| **Total** | **80,000** | **42,048** | **147,550** |

### 3.2 Class distribution

| Class | YOLO ID | Train boxes | Val boxes | Combined | Share |
|-------|---------|-------------|-----------|----------|-------|
| Red | 0 | 46,153 | 6,605 | 52,758 | **35.7%** |
| Yellow | 1 | 3,418 | 510 | 3,928 | **2.7%** |
| Green | 2 | 79,441 | 11,423 | 90,864 | **61.6%** |

Yellow is heavily under-represented — expect lower per-class recall for yellow unless augmentation or class weighting is applied.

### 3.3 Skipped annotations

| Reason | Train | Val | Combined |
|--------|-------|-----|----------|
| `trafficLightColor == "none"` | 57,225 | 8,349 | 65,574 |
| Invalid box geometry | 64 | 4 | 68 |

---

## 4. YOLO Output Dataset

### 4.1 Directory layout

```
data/traffic_signal_yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── dataset.yaml
└── dataset_stats.json
```

### 4.2 Class mapping (ADAS-aligned)

| BDD100K color | YOLO class ID | YOLO name | ADAS runtime label |
|---------------|---------------|-----------|-------------------|
| red | 0 | red | red_light |
| yellow | 1 | yellow | yellow_light |
| green | 2 | green | green_light |

Defined in `training/traffic_signal/bdd100k_converter.py` as `BDD100K_TO_YOLO_CLASS` and `YOLO_TO_ADAS_LABEL`.

### 4.3 `dataset.yaml`

```yaml
path: C:/Users/gauth/Desktop/Autonomous Driving Car/data/traffic_signal_yolo
train: images/train
val: images/val
nc: 3
names:
  0: red
  1: yellow
  2: green
```

### 4.4 YOLO label format

Each line: `class_id x_center y_center width height` (normalized 0–1).

---

## 5. Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `yolov8n.pt` | Ultralytics nano variant |
| Epochs | 100 | Early stop via patience |
| Image size | 640 | Standard YOLOv8 input |
| Batch size | 16 | Reduce to 8 on 6 GB GPU |
| Patience | 20 | Early stopping |
| Output weights | `models/trained/yolov8_signal/traffic_signals_yolov8n.pt` | Copied from `best.pt` |

### 5.1 Hardware requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 6 GB VRAM (batch=8) | 8+ GB VRAM (batch=16) |
| CPU RAM | 8 GB | 16 GB |
| Disk | ~5 GB for YOLO dataset (hard links) | SSD preferred |

### 5.2 Expected training time

| Hardware | Approx. time (100 epochs, ~42k images) |
|----------|----------------------------------------|
| NVIDIA RTX 3060 (12 GB) | 4–6 hours |
| NVIDIA RTX 4070 / 4080 | 2–4 hours |
| Google Colab T4 | 6–9 hours |
| CPU only | 2–4 days (not recommended) |

---

## 6. Evaluation Methodology

### 6.1 Script

```bash
python scripts/evaluate_traffic_signal.py \
  --weights models/trained/yolov8_signal/traffic_signals_yolov8n.pt \
  --data data/traffic_signal_yolo/dataset.yaml
```

### 6.2 Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | Mean precision across classes |
| **Recall** | Mean recall across classes |
| **F1** | `2 × P × R / (P + R)` |
| **mAP@0.5** | Mean AP at IoU 0.50 |
| **mAP@0.5:0.95** | COCO-style AP (IoU 0.50–0.95) |

Per-class AP@0.5 is reported under both YOLO names (`red`, `yellow`, `green`) and ADAS names (`red_light`, `yellow_light`, `green_light`).

### 6.3 Validation results (BDD100K val split)

| Metric | Value |
|--------|-------|
| **Precision** | 0.580 |
| **Recall** | 0.443 |
| **mAP@0.5** | 0.443 |
| **mAP@0.5:0.95** | 0.149 |

Weights: `models/trained/yolov8_signal/traffic_signals_yolov8n.pt`

---

## 7. Quick Start

```bash
# 1. Prepare or refresh YOLO dataset (emits ADAS-aligned labels)
python scripts/prepare_traffic_signal_dataset.py

# 2. Train YOLOv8n
python scripts/train_traffic_signal.py --device 0 --epochs 100 --batch 16

# 3. Evaluate
python scripts/evaluate_traffic_signal.py --device 0

# 4. Inference demo
python scripts/predict_traffic_signal.py path/to/image.jpg --device 0
```

---

## 8. Generated Files

| File | Purpose |
|------|---------|
| `training/traffic_signal/bdd100k_converter.py` | BDD100K → YOLO conversion + ADAS mapping |
| `scripts/prepare_traffic_signal_dataset.py` | Dataset builder |
| `scripts/train_traffic_signal.py` | YOLOv8n training |
| `scripts/evaluate_traffic_signal.py` | Val metrics with ADAS class names |
| `scripts/predict_traffic_signal.py` | Inference demo |
| `tests/test_traffic_signal_training.py` | Alignment unit tests |

### Not modified

- `src/modules/traffic_signal.py`
- `src/modules/yolov8_signal/*`
- `config/default.yaml` / `config/classes.yaml`

---

## 9. ADAS Integration

Training class order **matches** `SIGNAL_CLASS_ID_TO_LABEL` in `class_map.py`:

| Class ID | Training / YOLO | ADAS parser |
|----------|-----------------|-------------|
| 0 | red | red_light |
| 1 | yellow | yellow_light |
| 2 | green | green_light |

After training, copy weights to:

```
models/trained/yolov8_signal/traffic_signals_yolov8n.pt
```

The ADAS `YOLOv8SignalOutputParser` will map class IDs directly — no `class_map.py` changes needed.

See also: `docs/traffic_signal_label_alignment_report.md`

---

## 10. References

- BDD100K: https://bdd-data.berkeley.edu/
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- Design: `docs/traffic_signal_detection_design.md`
- Implementation: `docs/traffic_signal_implementation_report.md`
