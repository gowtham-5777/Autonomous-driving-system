# Traffic Signal Label Alignment — Compatibility Report

**Date:** June 2026  
**Task:** Align training pipeline class IDs with ADAS runtime  
**Training started:** No (alignment and verification only)

---

## 1. Problem (resolved)

| Layer | Before fix | After fix |
|-------|------------|-----------|
| Training (`bdd100k_converter.py`) | 0=green, 1=yellow, 2=red | **0=red, 1=yellow, 2=green** |
| ADAS runtime (`class_map.py`) | 0=red_light, 1=yellow_light, 2=green_light | unchanged |
| `dataset.yaml` | names 0:green … | **names 0:red, 1:yellow, 2:green** |
| On-disk YOLO labels | legacy IDs | **relabeled to ADAS-aligned IDs** |

---

## 2. Final canonical mapping

| Class ID | BDD100K color | YOLO `dataset.yaml` | ADAS `SIGNAL_CLASS_ID_TO_LABEL` |
|----------|---------------|---------------------|--------------------------------|
| **0** | red | red | red_light |
| **1** | yellow | yellow | yellow_light |
| **2** | green | green | green_light |

### Source definitions

**Training** (`training/traffic_signal/bdd100k_converter.py`):

```python
BDD100K_TO_YOLO_CLASS = {"red": 0, "yellow": 1, "green": 2}
YOLO_CLASS_NAMES = ["red", "yellow", "green"]
YOLO_TO_ADAS_LABEL = {0: "red_light", 1: "yellow_light", 2: "green_light"}
```

**ADAS runtime** (`src/modules/yolov8_signal/class_map.py`) — **unchanged**:

```python
SIGNAL_CLASS_ID_TO_LABEL = {0: "red_light", 1: "yellow_light", 2: "green_light"}
BDD100K_LABEL_TO_ADAS_LABEL = {"red": "red_light", "yellow": "yellow_light", "green": "green_light"}
```

Automated cross-check result: **all three layers match** for IDs 0–2.

---

## 3. Files changed

| File | Change |
|------|--------|
| `training/traffic_signal/bdd100k_converter.py` | Updated `BDD100K_TO_YOLO_CLASS`, `YOLO_CLASS_NAMES`, added `YOLO_TO_ADAS_LABEL`; `write_dataset_yaml()` uses dynamic names |
| `training/traffic_signal/__init__.py` | Export `YOLO_TO_ADAS_LABEL` |
| `scripts/prepare_traffic_signal_dataset.py` | Stats use `YOLO_CLASS_NAMES`; class count dict order updated |
| `scripts/evaluate_traffic_signal.py` | Emits `class_mapping`, `adas_class_mapping`, `per_class_ap50_adas` |
| `tests/test_traffic_signal_training.py` | Tests updated + ADAS alignment test added (5 tests) |
| `data/traffic_signal_yolo/dataset.yaml` | Regenerated with aligned names |
| `data/traffic_signal_yolo/labels/**/*.txt` | Relabeled (42,048 files) |
| `docs/traffic_signal_training_report.md` | Regenerated with aligned mapping |
| `docs/traffic_signal_label_alignment_report.md` | This report |

### ADAS inference — no modifications required

| File | Modified? | Reason |
|------|-----------|--------|
| `src/modules/yolov8_signal/class_map.py` | **No** | Already defined red=0, yellow=1, green=2 |
| `src/modules/yolov8_signal/output_parser.py` | **No** | Uses `SIGNAL_CLASS_ID_TO_LABEL[class_id]` |
| `src/modules/yolov8_signal/inference.py` | **No** | Passes through raw `class_ids` from Ultralytics |
| `src/modules/traffic_signal.py` | **No** | Orchestrator unchanged |
| `config/classes.yaml` | **No** | `traffic_signal_classes` order already red/yellow/green |

---

## 4. Verification results

| Check | Result |
|-------|--------|
| Unit tests (`test_traffic_signal_training.py`) | **5/5 PASS** |
| `dataset.yaml` names | **0:red, 1:yellow, 2:green** |
| Sample label files contain IDs {0,1,2} only | **PASS** |
| Training `YOLO_TO_ADAS_LABEL` vs `SIGNAL_CLASS_ID_TO_LABEL` | **PASS (3/3)** |
| Training names vs `BDD100K_LABEL_TO_ADAS_LABEL` keys | **PASS** |
| ADAS inference code modified | **No** |

---

## 5. Evaluation script behavior

`scripts/evaluate_traffic_signal.py` now writes:

```json
{
  "class_mapping": {"0": "red", "1": "yellow", "2": "green"},
  "adas_class_mapping": {"0": "red_light", "1": "yellow_light", "2": "green_light"},
  "per_class_ap50": { "red": 0.0, "yellow": 0.0, "green": 0.0 },
  "per_class_ap50_adas": { "red_light": 0.0, "yellow_light": 0.0, "green_light": 0.0 }
}
```

(Metric values populate after a model is trained and evaluated.)

---

## 6. Deployment path

1. Train: `python scripts/train_traffic_signal.py --device 0`
2. Weights land at: `models/trained/yolov8_signal/traffic_signals_yolov8n.pt`
3. ADAS loads via `get_traffic_signal_weights_path()` — parser maps class IDs directly

**No post-training label remapping or `class_map.py` edits needed.**

---

## 7. If dataset is rebuilt from scratch

Run:

```bash
python scripts/prepare_traffic_signal_dataset.py
```

New exports will emit ADAS-aligned class IDs automatically. No separate relabel step required.

---

## 8. Summary

| Item | Status |
|------|--------|
| Label alignment | **Complete** |
| Dataset regenerated | **Yes** (yaml + 42,048 label files) |
| ADAS compatibility | **Verified — no inference changes** |
| Training | **Not started** (per instructions) |
