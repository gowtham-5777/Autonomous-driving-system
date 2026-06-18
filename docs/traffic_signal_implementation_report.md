# Traffic Signal Detection — Implementation Report

**Date:** June 2026  
**Design reference:** `docs/traffic_signal_detection_design.md`  
**Status:** Implemented and verified

---

## 1. Summary

Traffic signal detection is implemented using **Ultralytics YOLOv8n** fine-tuned for 3-class traffic-light state detection, following the same architectural pattern as `VehicleDetectionModule` and `TrafficSignModule`. The module detects `red_light`, `yellow_light`, and `green_light`; returns frame-space bounding boxes with state flags; computes `controlling_signal` and `dominant_state` summaries; and includes visualization, stub-based tests, and a gate script.

| Verification item | Result |
|-------------------|--------|
| Model loads | **PASS** (stub; real requires `traffic_signals_yolov8n.pt`) |
| Inference runs | **PASS** |
| Detections produced | **PASS** (stub pipeline) |
| Visualization works | **PASS** |
| Module imports | **PASS** |
| All tests pass | **PASS** (29/29) |

---

## 2. Files Created

| File | Purpose |
|------|---------|
| `src/modules/yolov8_signal/__init__.py` | Package exports |
| `src/modules/yolov8_signal/class_map.py` | Signal class IDs, Bdd100K mapping, state priority |
| `src/modules/yolov8_signal/output_schema.py` | `TrafficSignalDetectionResult`, `DetectedSignal`, `TrafficSignalSummary` |
| `src/modules/yolov8_signal/model_loader.py` | `YOLOv8SignalModelLoader` — fine-tuned weight loading (no COCO fallback) |
| `src/modules/yolov8_signal/inference.py` | `YOLOv8SignalInferenceEngine` — forward pass wrapper |
| `src/modules/yolov8_signal/output_parser.py` | Class filter, bbox clip, controlling/dominant state heuristics |
| `tests/test_traffic_signal_pipeline.py` | Integration tests (7 cases) |
| `scripts/verify_traffic_signal_detection.py` | Gate script (stub default, `--real` optional) |
| `docs/traffic_signal_implementation_report.md` | This report |

---

## 3. Files Modified

| File | Change |
|------|--------|
| `src/modules/traffic_signal.py` | Full `TrafficSignalModule` orchestrator |
| `src/modules/__init__.py` | Export `TrafficSignalDetectionResult`, `TRAFFIC_SIGNAL_OUTPUT_KEYS` |
| `src/utils/model_paths.py` | `get_traffic_signal_weights_path()`, `get_yolov8_signal_config()` |
| `src/visualization/overlays.py` | `draw_traffic_signals()` |
| `config/default.yaml` | `yolov8_signal` weights, `signal_confidence`/`signal_iou`, `yolov8_signal` section |
| `config/classes.yaml` | `traffic_signal_classes` with `_light` suffix; legacy `traffic_light_classes` retained |
| `tests/conftest.py` | Stub YOLOv8 signal loader/engine + `traffic_signal_module` fixture |

### Not modified (per requirements)

- `src/modules/yolop/*`
- `src/modules/lane_detection.py`
- `src/modules/vehicle_detection.py`
- `src/modules/traffic_sign.py`
- `src/modules/yolov8/*`
- `src/modules/yolov8_sign/*`

---

## 4. Architecture

```
BGR Frame (H, W, 3)
    │
    ▼
TrafficSignalModule.predict()
    │
    ├─► YOLOv8SignalModelLoader.load_model()     [initialize]
    ├─► YOLOv8SignalInferenceEngine.run(frame)   [Ultralytics predict]
    ├─► YOLOv8SignalOutputParser.parse()         [signal filter → frame coords]
    └─► TrafficSignalDetectionResult             [summary + detections]
```

### Layer responsibilities

| Layer | Class | File |
|-------|-------|------|
| Orchestrator | `TrafficSignalModule` | `traffic_signal.py` |
| Loader | `YOLOv8SignalModelLoader` | `yolov8_signal/model_loader.py` |
| Inference | `YOLOv8SignalInferenceEngine` | `yolov8_signal/inference.py` |
| Parser | `YOLOv8SignalOutputParser` | `yolov8_signal/output_parser.py` |
| Schema | `TrafficSignalDetectionResult` | `yolov8_signal/output_schema.py` |
| Class map | `SIGNAL_CLASS_ID_TO_LABEL`, Bdd100K helpers | `yolov8_signal/class_map.py` |
| Visualization | `draw_traffic_signals()` | `visualization/overlays.py` |

### Detected classes (3-class ADAS head)

| `class_id` | `signal_label` | Stop | Caution | Proceed |
|------------|----------------|------|---------|---------|
| 0 | `red_light` | Yes | No | No |
| 1 | `yellow_light` | No | Yes | No |
| 2 | `green_light` | No | No | Yes |

### Default configuration

- Model variant: **`n`** → `traffic_signals_yolov8n.pt`
- Config path: `{data_root}/models/trained/yolov8_signal/traffic_signals_yolov8n.pt`
- **No COCO fallback** — fine-tuned weights required for real inference
- Confidence: `0.5` (`thresholds.signal_confidence`, falls back to `traffic_light_confidence`)
- IoU NMS: `0.45` (`thresholds.signal_iou`)
- Input size: `640`
- Max detections: `20`

### Summary heuristics

- **`nearest_signal`:** Highest `center_y` (closest to ego in image)
- **`controlling_signal`:** Among detections in upper 60% of frame, pick highest `center_y`; tie-break by confidence; fallback to `nearest_signal`
- **`dominant_state`:** State of `controlling_signal`, or highest `state_priority` among all detections (red > yellow > green)

---

## 5. Verification Evidence

### 5.1 Pytest (all modules)

```bash
python -m pytest tests/ -v
```

**Result:** `29 passed in ~5s`

Traffic signal tests:

| Test | Status |
|------|--------|
| `test_module_initializes_with_stub_loader` | PASS |
| `test_end_to_end_predict_pipeline` | PASS |
| `test_predict_auto_inits` | PASS |
| `test_visualize_returns_annotated_frame` | PASS |
| `test_only_allowed_signal_classes` | PASS |
| `test_controlling_signal_in_summary` | PASS |
| `test_dominant_state_priority` | PASS |

### 5.2 Gate script (stub — CI default)

```bash
python scripts/verify_traffic_signal_detection.py
```

**Output:**

```
PASS: model loads (stub)
PASS: inference runs
PASS: detections produced — count=1
PASS: all boxes in original frame coordinates
PASS: visualization works
PASS: cleanup releases module state
TRAFFIC SIGNAL DETECTION GATE: ALL CHECKS PASSED
  mode            = stub
  raw_status      = verify_stub
  total_count     = 1
  labels          = {'red_light': 1}
  dominant_state  = red_light
  inference_ms    = 2.5
```

### 5.3 Module import

```bash
python -c "from src.modules import TrafficSignalModule, TrafficSignalDetectionResult, TRAFFIC_SIGNAL_OUTPUT_KEYS; print('OK')"
```

**Result:** `OK`

### 5.4 Real weights (`--real`)

Requires `traffic_signals_yolov8n.pt` at the configured trained-models path:

```bash
python scripts/verify_traffic_signal_detection.py --real
```

---

## 6. Dependency Injection (testing)

Mirrors `TrafficSignModule` / `VehicleDetectionModule`:

```python
TrafficSignalModule(
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

- `stub_yolov8_signal_model_loader` — skips Ultralytics
- `stub_yolov8_signal_inference_engine` — returns one `red_light` box at upper-center (conf=0.90)
- `traffic_signal_module` — pre-initialized module

---

## 7. Output Schema

### `TrafficSignalDetectionResult`

| Field | Type | Purpose |
|-------|------|---------|
| `detections` | `list[DetectedSignal]` | Filtered signal detections |
| `summary` | `TrafficSignalSummary` | Aggregates including `dominant_state` |
| `frame_shape` | `(H, W) \| None` | Input dimensions |
| `inference_time_ms` | `float \| None` | Timing |
| `model_variant` | `str` | `"n"` default |
| `confidence_threshold` | `float` | Applied threshold |
| `raw_status` | `str` | `ok`, `stub`, `init_failed`, etc. |

### Serialized detection example

```json
{
  "signal_label": "red_light",
  "class_id": 0,
  "confidence": 0.91,
  "bbox": [612, 48, 648, 128],
  "is_stop_state": true,
  "is_caution_state": false,
  "is_proceed_state": false
}
```

---

## 8. Error Handling

| Scenario | Behavior |
|----------|----------|
| Fine-tuned weights missing | `WeightsNotFoundError` on init (no public fallback) |
| `ultralytics` not installed | `WeightsLoadError` on init |
| `predict()` before init, init fails | `TrafficSignalDetectionResult.empty(raw_status="init_failed")` |
| Invalid frame | `ValueError` from `_validate_input()` |
| Inference failure | `empty(raw_status="pipeline_error")` |
| Engine not ready | `empty(raw_status="inference_not_ready")` |
| Invalid bbox after clip | Dropped + warning logged |
| Bbox outside frame post-parse | `RuntimeError` from `_assert_detections_within_frame()` |
| Conflicting red + green | Warning logged; `dominant_state` from controlling/priority rules |

---

## 9. Remaining Work (out of scope for this PR)

| Item | Notes |
|------|-------|
| Train `traffic_signals_yolov8n.pt` | Bdd100K + BSTLD fine-tune pipeline |
| `evaluation/evaluation_traffic_signal_detection.py` | mAP@0.5, per-class P/R/F1, confusion matrix |
| Temporal smoothing (v2) | Majority vote / hysteresis for flicker reduction |
| Decision engine integration | `SceneState.signals`, STOP/CAUTION/PROCEED rules |
| README module 4 update | Still references CNN in places |

---

## 10. Design Compliance Checklist

- [x] `yolov8_signal/` package (loader, inference, parser, schema, class_map)
- [x] `TrafficSignalModule` with `initialize`, `predict`, `visualize`, `cleanup`
- [x] YOLOv8n architecture with classes `red_light`, `yellow_light`, `green_light`
- [x] Same architecture pattern as vehicle/sign modules
- [x] Dependency injection + stub loaders in `conftest.py`
- [x] `tests/test_traffic_signal_pipeline.py` (7 integration tests)
- [x] `scripts/verify_traffic_signal_detection.py` (stub + `--real`)
- [x] Config and `model_paths` updates
- [x] `draw_traffic_signals()` visualization
- [x] Did not modify protected modules/packages
- [x] Pytest passes (29/29)
- [x] Gate script passes (stub mode)
- [x] Module imports successfully

---

*End of Traffic Signal Detection Implementation Report.*
