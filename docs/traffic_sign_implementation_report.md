# Traffic Sign Detection — Implementation Report

**Date:** June 2026  
**Design reference:** `docs/traffic_sign_detection_design.md`  
**Status:** Implemented and verified

---

## 1. Summary

Traffic sign detection is implemented using **Ultralytics YOLOv8n** fine-tuned on a 7-class ADAS sign subset (GTSRB-derived labels), following the same architectural pattern as `VehicleDetectionModule`. The module detects stop, speed limits, turn guidance, keep right, and pedestrian crossing signs; returns frame-space bounding boxes with enriched metadata; and includes visualization, stub-based tests, and a gate script.

| Verification item | Result |
|-------------------|--------|
| Model loads | **PASS** (stub; real requires `traffic_signs_yolov8n.pt`) |
| Inference runs | **PASS** |
| Detections produced | **PASS** (stub pipeline) |
| Visualization works | **PASS** |
| Module imports | **PASS** |
| All tests pass | **PASS** (22/22) |

---

## 2. Files Created

| File | Purpose |
|------|---------|
| `src/modules/yolov8_sign/__init__.py` | Package exports |
| `src/modules/yolov8_sign/class_map.py` | Sign class IDs, GTSRB mapping, speed-limit helpers |
| `src/modules/yolov8_sign/output_schema.py` | `TrafficSignDetectionResult`, `DetectedSign`, `SignBoundingBoxData` |
| `src/modules/yolov8_sign/model_loader.py` | `YOLOv8SignModelLoader` — fine-tuned weight loading (no COCO fallback) |
| `src/modules/yolov8_sign/inference.py` | `YOLOv8SignInferenceEngine` — forward pass wrapper |
| `src/modules/yolov8_sign/output_parser.py` | Sign class filter, bbox clip, speed-limit enrichment |
| `tests/test_traffic_sign_pipeline.py` | Integration tests (6 cases) |
| `scripts/verify_traffic_sign_detection.py` | Gate script (stub default, `--real` optional) |
| `docs/traffic_sign_implementation_report.md` | This report |

---

## 3. Files Modified

| File | Change |
|------|--------|
| `src/modules/traffic_sign.py` | Full `TrafficSignModule` orchestrator |
| `src/modules/__init__.py` | Export `TrafficSignDetectionResult`, `TRAFFIC_SIGN_OUTPUT_KEYS` |
| `src/utils/model_paths.py` | `get_traffic_sign_weights_path()`, `get_yolov8_sign_config()` |
| `src/visualization/overlays.py` | `draw_traffic_signs()` |
| `config/default.yaml` | `yolov8_sign` weights, thresholds, `yolov8_sign` section |
| `config/classes.yaml` | Comment updated to YOLOv8 |
| `tests/conftest.py` | Stub YOLOv8 sign loader/engine + `traffic_sign_module` fixture |

### Not modified (per requirements)

- `src/modules/yolop/*`
- `src/modules/lane_detection.py`
- `src/modules/vehicle_detection.py`
- `src/modules/yolov8/*`

---

## 4. Architecture

```
BGR Frame (H, W, 3)
    │
    ▼
TrafficSignModule.predict()
    │
    ├─► YOLOv8SignModelLoader.load_model()     [initialize]
    ├─► YOLOv8SignInferenceEngine.run(frame)   [Ultralytics predict]
    ├─► YOLOv8SignOutputParser.parse()         [sign filter → frame coords]
    └─► TrafficSignDetectionResult             [summary + detections]
```

### Layer responsibilities

| Layer | Class | File |
|-------|-------|------|
| Orchestrator | `TrafficSignModule` | `traffic_sign.py` |
| Loader | `YOLOv8SignModelLoader` | `yolov8_sign/model_loader.py` |
| Inference | `YOLOv8SignInferenceEngine` | `yolov8_sign/inference.py` |
| Parser | `YOLOv8SignOutputParser` | `yolov8_sign/output_parser.py` |
| Schema | `TrafficSignDetectionResult` | `yolov8_sign/output_schema.py` |
| Class map | `SIGN_CLASS_ID_TO_LABEL`, GTSRB helpers | `yolov8_sign/class_map.py` |
| Visualization | `draw_traffic_signs()` | `visualization/overlays.py` |

### Detected classes (7-class ADAS head)

| `class_id` | `sign_label` | Regulatory |
|------------|--------------|------------|
| 0 | `stop` | Yes |
| 1 | `speed_limit_30` | Yes |
| 2 | `speed_limit_60` | Yes |
| 3 | `turn_left` | No |
| 4 | `turn_right` | No |
| 5 | `keep_right` | Yes |
| 6 | `pedestrian_crossing` | Warning |

### GTSRB → ADAS mapping (v1)

| GTSRB ID | ADAS label |
|----------|------------|
| 14 | `stop` |
| 1 | `speed_limit_30` |
| 5 | `speed_limit_60` |
| 38 | `turn_left` |
| 34 | `turn_right` |
| 36 | `keep_right` |
| 12 | `pedestrian_crossing` |

### Default configuration

- Model variant: **`n`** → `traffic_signs_yolov8n.pt`
- Config path: `{data_root}/models/trained/yolov8_sign/traffic_signs_yolov8n.pt`
- **No COCO fallback** — fine-tuned weights are required for real inference
- Confidence: `0.5` (`thresholds.sign_confidence`)
- IoU NMS: `0.45` (`thresholds.sign_iou`)
- Input size: `640`
- Max detections: `50`

### Summary heuristics

- **`nearest_sign`:** Highest `center_y` (closest to ego in image)
- **`active_speed_limit_kmh`:** Minimum numeric limit among visible `speed_limit_*` signs (most restrictive)

---

## 5. Verification Evidence

### 5.1 Pytest (all modules)

```bash
python -m pytest tests/ -v
```

**Result:** `22 passed in ~14s`

Traffic sign tests:

| Test | Status |
|------|--------|
| `test_module_initializes_with_stub_loader` | PASS |
| `test_end_to_end_predict_pipeline` | PASS |
| `test_predict_auto_inits` | PASS |
| `test_visualize_returns_annotated_frame` | PASS |
| `test_only_allowed_sign_classes` | PASS |
| `test_active_speed_limit_in_summary` | PASS |

### 5.2 Gate script (stub — CI default)

```bash
python scripts/verify_traffic_sign_detection.py
```

**Output:**

```
PASS: model loads (stub)
PASS: inference runs
PASS: detections produced — count=1
PASS: all boxes in original frame coordinates
PASS: visualization works
PASS: cleanup releases module state
TRAFFIC SIGN DETECTION GATE: ALL CHECKS PASSED
  mode            = stub
  raw_status      = verify_stub
  total_count     = 1
  labels          = {'stop': 1}
  active_limit    = None
  inference_ms    = 2.5
```

### 5.3 Module import

```bash
python -c "from src.modules import TrafficSignModule, TrafficSignDetectionResult, TRAFFIC_SIGN_OUTPUT_KEYS; print('OK')"
```

**Result:** `OK`

### 5.4 Real weights (`--real`)

Requires `traffic_signs_yolov8n.pt` at the configured trained-models path. Gate script fails fast with a clear message if weights are missing. Use after training:

```bash
python scripts/verify_traffic_sign_detection.py --real
```

---

## 6. Dependency Injection (testing)

Mirrors `VehicleDetectionModule`:

```python
TrafficSignModule(
    weights_path=None,
    model_loader=None,       # inject stub loader
    inference_engine=None,   # inject stub engine
    output_parser=None,
    device="cpu",
    model_variant="n",
    confidence_threshold=0.5,
    iou_threshold=0.45,
    imgsz=640,
)
```

Stub fixtures in `tests/conftest.py`:

- `stub_yolov8_sign_model_loader` — skips Ultralytics
- `stub_yolov8_sign_inference_engine` — returns one `stop` box at upper-center (conf=0.92)
- `traffic_sign_module` — pre-initialized module

---

## 7. Output Schema

### `TrafficSignDetectionResult`

| Field | Type | Purpose |
|-------|------|---------|
| `detections` | `list[DetectedSign]` | Filtered sign detections |
| `summary` | `TrafficSignDetectionSummary` | Aggregates including `active_speed_limit_kmh` |
| `frame_shape` | `(H, W) \| None` | Input dimensions |
| `inference_time_ms` | `float \| None` | Timing |
| `model_variant` | `str` | `"n"` default |
| `confidence_threshold` | `float` | Applied threshold |
| `raw_status` | `str` | `ok`, `stub`, `init_failed`, etc. |

### Serialized detection example

```json
{
  "sign_label": "speed_limit_30",
  "class_id": 1,
  "confidence": 0.91,
  "bbox": [800, 120, 860, 180],
  "speed_limit_kmh": 30,
  "is_regulatory": true
}
```

---

## 8. Error Handling

| Scenario | Behavior |
|----------|----------|
| Fine-tuned weights missing | `WeightsNotFoundError` on init (no public fallback) |
| `ultralytics` not installed | `WeightsLoadError` on init |
| `predict()` before init, init fails | `TrafficSignDetectionResult.empty(raw_status="init_failed")` |
| Invalid frame | `ValueError` from `_validate_input()` |
| Inference failure | `empty(raw_status="pipeline_error")` |
| Engine not ready | `empty(raw_status="inference_not_ready")` |
| Invalid bbox after clip | Dropped + warning logged |
| Bbox outside frame post-parse | `RuntimeError` from `_assert_detections_within_frame()` |

---

## 9. Remaining Work (out of scope for this PR)

| Item | Notes |
|------|-------|
| Train `traffic_signs_yolov8n.pt` | GTSRB + Bdd100K or synthetic paste pipeline |
| `evaluation/evaluation_traffic_sign_detection.py` | mAP@0.5, per-class P/R/F1 |
| Unit tests for parser/class_map/schema | Optional granular coverage |
| Decision engine integration | `SceneState.signs`, speed-limit rules |
| README module 3 update | Still references YOLOv5 in places |

---

## 10. Design Compliance Checklist

- [x] `yolov8_sign/` package (loader, inference, parser, schema, class_map)
- [x] `TrafficSignModule` with `initialize`, `predict`, `visualize`, `cleanup`
- [x] Same architecture pattern as `VehicleDetectionModule`
- [x] Dependency injection + stub loaders in `conftest.py`
- [x] `tests/test_traffic_sign_pipeline.py` (6 integration tests)
- [x] `scripts/verify_traffic_sign_detection.py` (stub + `--real`)
- [x] Config and `model_paths` updates
- [x] `draw_traffic_signs()` visualization
- [x] Did not modify `lane_detection.py`, `yolop/*`, `vehicle_detection.py`, `yolov8/*`
- [x] Pytest passes (22/22)
- [x] Gate script passes (stub mode)
- [x] Module imports successfully

---

*End of Traffic Sign Detection Implementation Report.*
