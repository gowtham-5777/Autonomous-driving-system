# Vehicle Detection — Implementation Report

**Date:** June 2026  
**Design reference:** `docs/vehicle_detection_design.md`  
**Status:** Implemented and verified

---

## 1. Summary

Vehicle detection is implemented using **Ultralytics YOLOv8** (`yolov8s.pt` default) following the same architectural pattern as `LaneDetectionModule`. The module detects six ADAS road-user classes (person, bicycle, car, motorcycle, bus, truck), returns frame-space bounding boxes, and includes visualization, tests, and a gate script.

| Verification item | Result |
|-------------------|--------|
| Model loads | **PASS** (stub + real `yolov8s.pt`) |
| Inference runs | **PASS** |
| Detections produced | **PASS** (stub pipeline; real on synthetic road = 0 objects, expected) |
| Visualization works | **PASS** |
| All tests pass | **PASS** (16/16) |

---

## 2. Files Created

| File | Purpose |
|------|---------|
| `src/modules/yolov8/__init__.py` | Package exports |
| `src/modules/yolov8/output_schema.py` | `VehicleDetectionResult`, `DetectedObject`, `BoundingBoxData`, etc. |
| `src/modules/yolov8/model_loader.py` | `YOLOv8ModelLoader` — Ultralytics weight loading |
| `src/modules/yolov8/inference.py` | `YOLOv8InferenceEngine` — forward pass wrapper |
| `src/modules/yolov8/output_parser.py` | COCO class filter + frame-space box validation |
| `tests/test_vehicle_detection_pipeline.py` | Integration tests (5 cases) |
| `scripts/verify_vehicle_detection.py` | Gate script (stub default, `--real` optional) |
| `docs/vehicle_detection_implementation_report.md` | This report |

---

## 3. Files Modified

| File | Change |
|------|--------|
| `src/modules/vehicle_detection.py` | Full `VehicleDetectionModule` orchestrator |
| `src/modules/__init__.py` | Export `VehicleDetectionResult`, `VEHICLE_OUTPUT_KEYS` |
| `src/utils/model_paths.py` | `get_yolov8_weights_path()`, `get_yolov8_config()` |
| `src/visualization/overlays.py` | `draw_vehicle_detections()` |
| `config/default.yaml` | YOLOv8 weights, thresholds, `yolov8` section |
| `requirements.txt` | `ultralytics>=8.0,<9.0` |
| `tests/conftest.py` | Stub YOLOv8 loader/engine + `vehicle_detection_module` fixture |

### Not modified (per requirements)

- `src/modules/yolop/*`
- `src/modules/lane_detection.py`

---

## 4. Architecture

```
BGR Frame (H, W, 3)
    │
    ▼
VehicleDetectionModule.predict()
    │
    ├─► YOLOv8ModelLoader.load_model()     [initialize]
    ├─► YOLOv8InferenceEngine.run(frame)   [Ultralytics predict]
    ├─► YOLOv8OutputParser.parse()         [COCO filter → frame coords]
    └─► VehicleDetectionResult             [summary + detections]
```

### Layer responsibilities

| Layer | Class | File |
|-------|-------|------|
| Orchestrator | `VehicleDetectionModule` | `vehicle_detection.py` |
| Loader | `YOLOv8ModelLoader` | `yolov8/model_loader.py` |
| Inference | `YOLOv8InferenceEngine` | `yolov8/inference.py` |
| Parser | `YOLOv8OutputParser` | `yolov8/output_parser.py` |
| Schema | `VehicleDetectionResult` | `yolov8/output_schema.py` |
| Visualization | `draw_vehicle_detections()` | `visualization/overlays.py` |

### Detected classes (COCO filter)

| Label | COCO ID |
|-------|---------|
| person | 0 |
| bicycle | 1 |
| car | 2 |
| motorcycle | 3 |
| bus | 5 |
| truck | 7 |

### Default configuration

- Model variant: **`s`** → `yolov8s.pt`
- Config path: `{data_root}/models/pretrained/yolov8/yolov8s.pt`
- Fallback: Ultralytics auto-downloads `yolov8s.pt` when configured path missing (Colab-friendly)
- Confidence: `0.5` (`thresholds.object_confidence`)
- IoU NMS: `0.45` (`thresholds.object_iou`)
- Input size: `640`

---

## 5. Verification Evidence

### 5.1 Pytest (all modules)

```bash
python -m pytest tests/ -v
```

**Result:** `16 passed in ~10s`

Vehicle detection tests:

| Test | Status |
|------|--------|
| `test_module_initializes_with_stub_loader` | PASS |
| `test_end_to_end_predict_pipeline` | PASS |
| `test_predict_from_uninitialized_module_auto_inits` | PASS |
| `test_visualize_returns_annotated_frame` | PASS |
| `test_only_allowed_classes_in_output` | PASS |

### 5.2 Gate script (stub — CI default)

```bash
python scripts/verify_vehicle_detection.py
```

**Output:**

```
PASS: model loads (stub)
PASS: inference runs
PASS: detections produced — count=1
PASS: all boxes in original frame coordinates
PASS: visualization works
PASS: cleanup releases module state
VEHICLE DETECTION GATE: ALL CHECKS PASSED
  mode            = stub
  raw_status      = verify_stub
  total_count     = 1
  labels          = {'car': 1}
```

### 5.3 Gate script (real Ultralytics weights)

```bash
python scripts/verify_vehicle_detection.py --real
```

**Output:**

```
PASS: model loads (real)
PASS: inference runs
WARN: real inference returned 0 ADAS objects on synthetic road fixture (expected)
PASS: real inference completed (0 objects on synthetic scene)
PASS: visualization works
PASS: cleanup releases module state
VEHICLE DETECTION GATE: ALL CHECKS PASSED
  mode            = real
  raw_status      = ok
  inference_ms    = ~702 ms (CPU, first run)
```

Real weights (`yolov8s.pt`, 21.5 MB) downloaded and loaded successfully. Zero detections on the synthetic `road_sample.jpg` fixture is expected — it contains drawn lane lines only, no COCO vehicles.

### 5.4 Frame-coordinate guarantee

`_assert_detections_within_frame()` in `vehicle_detection.py` raises `RuntimeError` if any box exceeds frame bounds. Parser clips boxes via `_clip_bbox_to_frame()`.

---

## 6. Usage

### Basic inference

```python
from src.modules.vehicle_detection import VehicleDetectionModule

module = VehicleDetectionModule(device="cpu")  # or "cuda" on Colab
module.initialize()

result = module.predict(frame)  # BGR uint8 (H, W, 3)
print(result.summary.count_by_label)
print(result.detections[0].bbox.to_list())

annotated = module.visualize(frame, result)
module.cleanup()
```

### Configurable variant

```python
module = VehicleDetectionModule(model_variant="n", device="cuda")
```

Or via `config/default.yaml`:

```yaml
yolov8:
  model_variant: "s"   # n | s | m
  imgsz: 640
  device: "cpu"
```

### Colab setup

1. `pip install -r requirements.txt`
2. Copy weights to Drive (optional):  
   `{data_root}/models/pretrained/yolov8/yolov8s.pt`
3. Or let Ultralytics auto-download on first `initialize()`
4. Set `yolov8.device: "cuda"` in config for GPU inference

### Import paths (package-safe)

```python
from src.modules import VehicleDetectionModule, VehicleDetectionResult
from src.modules.yolov8 import YOLOv8InferenceEngine
from src.utils.model_paths import get_yolov8_weights_path, get_yolov8_config
```

---

## 7. Output schema

`VehicleDetectionResult` fields:

| Field | Type | Description |
|-------|------|-------------|
| `detections` | `list[DetectedObject]` | Filtered road-user detections |
| `summary` | `VehicleDetectionSummary` | Counts, nearest object, highest confidence |
| `frame_shape` | `(H, W)` | Input frame dimensions |
| `inference_time_ms` | `float` | Forward pass timing |
| `model_variant` | `str` | `n`, `s`, or `m` |
| `confidence_threshold` | `float` | Applied threshold |
| `raw_status` | `str` | `ok`, `stub`, `init_failed`, etc. |

`to_prediction_dict()` keys: `detections`, `count_by_label`, `total_count`, `nearest_object`, `raw_status`.

---

## 8. Known limitations

1. **No multi-object tracking** — `track_id` reserved for future use.
2. **Synthetic test image** — Real YOLOv8 returns 0 objects on `road_sample.jpg`; use dashcam video for demo detections.
3. **README** — Still lists SSD MobileNetV2 in module list (not updated in this PR scope).
4. **Orchestrator / decision engine** — Not wired; modules run independently.
5. **Dual-model memory** — Running YOLOP + YOLOv8 together requires GPU memory management via `cleanup()`.

---

## 9. Next steps (optional)

- Wire `VehicleDetectionModule` into `src/pipeline/orchestrator.py`
- Add `vehicles` field to `src/decision/scene_state.py`
- Evaluation script with COCO/Bdd100K metrics
- README update: Module 2 → YOLOv8

---

## 10. Conclusion

Vehicle detection is **fully implemented** per `docs/vehicle_detection_design.md`:

- YOLOv8 subpackage with loader, inference, parser, schema
- `VehicleDetectionModule` with `initialize`, `predict`, `visualize`, `cleanup`
- Six-class COCO filter, frame-space boxes
- Tests and gate script passing
- Colab-compatible imports and auto-download fallback

**Verification status: PASS**
