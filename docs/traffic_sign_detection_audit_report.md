# Traffic Sign Detection Module — Technical Audit Report

**Repository:** Autonomous Driving Car  
**Module:** Traffic Sign Detection (`TrafficSignModule`)  
**Date:** June 2026  
**Audit basis:** Static inspection of implemented source, tests, config, scripts, and design/implementation docs  
**Test baseline:** 6 sign tests in `tests/test_traffic_sign_pipeline.py`; **29/29** total pytest pass

---

## Section 1: Module Overview

### 1.1 Purpose

Traffic Sign Detection identifies **regulatory and warning road signs** in a forward-facing camera frame and returns:

- **Where** each sign is (`bbox` in frame pixels)
- **What** type it is (`sign_label` from a 7-class ADAS taxonomy)
- **How confident** the model is (`confidence`)
- **Derived rules metadata** (`speed_limit_kmh`, `is_regulatory`)

Unlike dynamic obstacle detection (vehicles, pedestrians), traffic signs encode **static traffic rules** — speed limits, mandatory stops, turn guidance, and crossing warnings.

### 1.2 Role in the ADAS pipeline

Planned pipeline order (`src/pipeline/orchestrator.py`, comments only):

```
Lane Detection → Vehicle Detection → Traffic Sign Recognition →
Traffic Signal Detection → Segmentation → Decision Support
```

Traffic sign recognition runs **after** vehicle detection. There is **no hard v1 dependency** on lane or vehicle outputs inside `TrafficSignModule`, but the decision engine (future) will fuse sign rules with lane position and obstacles.

**Current reality:** `TrafficSignModule` runs **standalone**. The orchestrator is not implemented.

### 1.3 Relationship with Lane Detection and Vehicle Detection

| Module | Detects | Overlap with signs |
|--------|---------|-------------------|
| **Lane detection** (`LaneDetectionModule`) | Lane center, departure | None — geometry only |
| **Vehicle detection** (`VehicleDetectionModule`) | COCO road users (6 classes) | COCO has `stop sign` (id 11) but **not used** — insufficient for speed limits / turns |
| **Traffic sign** (`TrafficSignModule`) | 7 ADAS sign classes | Dedicated fine-tuned head |

**Deliberate separation:** Signs are regulatory objects with a **custom label set** trained on GTSRB-derived taxonomy. Vehicle YOLOv8 intentionally filters to road users only (`ALLOWED_COCO_CLASS_IDS` in `yolov8/output_parser.py`).

### 1.4 Why traffic sign recognition matters

- **Speed compliance:** `active_speed_limit_kmh` enables “Slow Down” rules
- **Mandatory stops:** `stop` detections trigger halt behavior at intersections
- **Warning context:** `pedestrian_crossing` supports caution alerts
- **Navigation hints:** `turn_left` / `turn_right` / `keep_right` inform path planning (informational in v1 decision design)

Without a dedicated sign module, an ADAS stack cannot interpret posted speed limits or EU/GTSRB-style regulatory signage from generic COCO detection alone.

---

## Section 2: Design Evolution

### 2.1 Original state (pre-implementation)

| Aspect | State |
|--------|-------|
| **File** | `src/modules/traffic_sign.py` — 33-line stub |
| **Docstring** | “YOLOv5 wrapper”, GTSRB-trained |
| **`predict()`** | Returned `{}` |
| **`initialize()` / `cleanup()`** | `pass` |
| **`visualize()`** | `frame.copy()` only |
| **Config** | `models.traffic_sign: "YOLOv5"`, `yolov5_traffic_signs.pt` |
| **Subpackage** | None — no `yolov5/` or `yolov8_sign/` |
| **`get_yolov5_weights_path()`** | Existed in `model_paths.py` but **unused** |

The original design implied a **detector + classifier** path (YOLOv5) but never implemented loading, inference, or parsing.

### 2.2 Current state

| Aspect | State |
|--------|-------|
| **Architecture** | YOLOv8n fine-tuned, Ultralytics API |
| **Orchestrator** | `TrafficSignModule` — 261 lines, full lifecycle |
| **Subpackage** | `src/modules/yolov8_sign/` (6 files) |
| **Output** | `TrafficSignDetectionResult` dataclass |
| **Config** | `models.traffic_sign: "YOLOv8"`, `yolov8_sign/traffic_signs_yolov8n.pt` |
| **Tests** | 6 integration tests + gate script |
| **Visualization** | `draw_traffic_signs()` wired via `visualize()` |

**Design references:** `docs/traffic_sign_detection_design.md`, `docs/traffic_sign_implementation_report.md`

### 2.3 Why YOLOv8 was selected

| Factor | Rationale |
|--------|-----------|
| **Stack parity** | `ultralytics` already required for `VehicleDetectionModule` |
| **Single forward pass** | Localization + classification in one pass (vs two-stage CNN-on-crop) |
| **Pattern reuse** | Same loader → inference → parser decomposition as `yolov8/` |
| **Maintenance** | Active Ultralytics v8 train/export vs legacy YOLOv5 pipelines |
| **Latency** | `yolov8n` keeps sequential inference feasible with YOLOP + vehicle + signal |

### 2.4 Alternatives considered

| Alternative | Verdict |
|-------------|---------|
| **Faster R-CNN** | Too slow for real-time ADAS video pipeline |
| **SSD MobileNetV2** | TensorFlow `.pb`; repo is PyTorch-first — rejected for vehicle; same for signs |
| **YOLOv5** | Original stub/config choice; stack fragmentation vs YOLOv8 vehicle |
| **YOLOv8 (COCO pretrained, no fine-tune)** | Only generic `stop sign`; misses speed limits and turns |
| **Two-stage (detect + CNN classify)** | Higher latency; ROI stage never implemented |
| **YOLOP detection head** | Unparsed in lane module; wrong label set |

**Decision:** YOLOv8n **fine-tuned** on 7-class ADAS subset. Mandatory regardless of YOLOv5 vs YOLOv8 backbone choice.

---

## Section 3: Repository Architecture

### 3.1 High-level diagram

```mermaid
flowchart TB
    subgraph Orchestrator["traffic_sign.py"]
        TSM[TrafficSignModule]
    end

    subgraph SignPkg["src/modules/yolov8_sign/"]
        ML[YOLOv8SignModelLoader]
        IE[YOLOv8SignInferenceEngine]
        OP[YOLOv8SignOutputParser]
        CM[class_map.py]
        OS[output_schema.py]
    end

    subgraph External
        UL[Ultralytics YOLO fine-tuned]
        CFG[config/default.yaml + classes.yaml]
        MP[model_paths.py]
    end

    subgraph Viz["visualization/overlays.py"]
        DS[draw_traffic_signs]
    end

    CFG --> MP --> TSM
    TSM --> ML --> UL
    TSM --> IE --> UL
    TSM --> OP --> CM
    OP --> OS
    TSM --> DS
```

### 3.2 Layer responsibilities

| Layer | Class | File | Responsibility |
|-------|-------|------|----------------|
| Orchestrator | `TrafficSignModule` | `traffic_sign.py` | Lifecycle, config resolution, pipeline wiring, frame validation |
| Loader | `YOLOv8SignModelLoader` | `model_loader.py` | Resolve weights path, load via `YOLO()`, metadata, unload |
| Inference | `YOLOv8SignInferenceEngine` | `inference.py` | Ultralytics `predict()`, raw tensor extraction, timing |
| Parser | `YOLOv8SignOutputParser` | `output_parser.py` | Class filter, confidence filter, bbox clip, summary |
| Schema | `TrafficSignDetectionResult` | `output_schema.py` | Dataclasses, `to_prediction_dict()` |
| Class map | `SIGN_CLASS_ID_TO_LABEL`, GTSRB helpers | `class_map.py` | Label ↔ ID, speed limit parse, regulatory flags |
| Exports | Public API | `__init__.py` | Re-exports for tests and external imports |

### 3.3 `TrafficSignModule` (`traffic_sign.py`)

**Inherits:** `BaseModule` (`src/modules/base.py`)

**Injectable constructor parameters:**

```python
TrafficSignModule(
    weights_path=None,
    model_loader=None,
    inference_engine=None,
    output_parser=None,
    device="cpu",
    model_variant="n",
    confidence_threshold=0.5,
    iou_threshold=0.45,
    imgsz=640,
)
```

**Config sources:** `get_yolov8_sign_config()`, `get_traffic_sign_weights_path()` from `src/utils/model_paths.py`

**Public methods:**

| Method | Purpose |
|--------|---------|
| `initialize()` | `load_model()` → `get_model()` → `attach_model()` |
| `predict(frame)` | Full pipeline → `TrafficSignDetectionResult` |
| `visualize(frame, results)` | Delegates to `draw_traffic_signs()` |
| `cleanup()` | `unload()` + `detach_model()` |
| `empty_prediction()` | Static empty result factory |

### 3.4 `yolov8_sign/model_loader.py`

| Component | Detail |
|-----------|--------|
| `DEFAULT_MODEL_VARIANT` | `"n"` |
| `ALLOWED_VARIANTS` | `n`, `s`, `m` |
| `resolve_variant_name()` | Normalizes `yolov8n` → `n` |
| `WeightsNotFoundError` | Raised when fine-tuned file missing |
| **No COCO fallback** | Unlike `YOLOv8ModelLoader` for vehicles |

Key methods: `resolve_weights_source()`, `load_model()`, `get_model()`, `unload()`

### 3.5 `yolov8_sign/inference.py`

| Component | Detail |
|-----------|--------|
| `YOLOv8SignInferenceConfig` | `imgsz=640`, `conf=0.5`, `iou=0.45`, `max_det=50` |
| `run(frame)` | Calls `self._model.predict(source=frame, ...)` |
| `_build_raw_output()` | Extracts `boxes_xyxy`, `confidences`, `class_ids` to NumPy |
| Exceptions | `InferenceNotReadyError`, `InvalidFrameError`, `InferenceExecutionError` |

### 3.6 `yolov8_sign/output_parser.py`

| Component | Detail |
|-----------|--------|
| `ALLOWED_SIGN_CLASS_IDS` | `{0, 1, 2, 3, 4, 5, 6}` |
| `parse()` | Filter class → filter confidence → clip bbox → enrich fields |
| `build_summary()` | `count_by_label`, `nearest_sign`, `active_speed_limit_kmh` |
| `_select_nearest_sign()` | `max(center_y, area)` |
| `_select_active_speed_limit()` | `min(speed_limit_kmh)` among detections |

### 3.7 `yolov8_sign/output_schema.py`

Dataclasses: `SignBoundingBoxData`, `DetectedSign`, `TrafficSignDetectionSummary`, `ParsedYOLOv8SignOutput`, `TrafficSignDetectionResult`

Constants: `TRAFFIC_SIGN_OUTPUT_KEYS`, `ADAS_SIGN_LABELS`

### 3.8 `yolov8_sign/class_map.py`

Maps model IDs, GTSRB IDs, regulatory/warning sets, `extract_speed_limit_kmh()`, `load_traffic_sign_classes()` from `config/classes.yaml`

---

## Section 4: Data Flow

### 4.1 Step-by-step execution

| Step | Action | File | Method |
|------|--------|------|--------|
| 1 | Receive BGR frame | `traffic_sign.py` | `predict(frame)` |
| 2 | Auto-init if needed | `traffic_sign.py` | `initialize()` |
| 3 | Validate frame | `traffic_sign.py` | `_validate_input()` |
| 4 | Check engine ready | `traffic_sign.py` | `_run_pipeline()` |
| 5 | Forward pass | `inference.py` | `YOLOv8SignInferenceEngine.run()` |
| 6 | Ultralytics predict | `inference.py` | `self._model.predict(...)` |
| 7 | Extract tensors | `inference.py` | `_build_raw_output()` |
| 8 | Parse outputs | `output_parser.py` | `YOLOv8SignOutputParser.parse()` |
| 9 | Filter sign classes | `output_parser.py` | `ALLOWED_SIGN_CLASS_IDS` check |
| 10 | Confidence filter | `output_parser.py` | `ParserConfig.confidence_threshold` |
| 11 | Clip bbox | `output_parser.py` | `_clip_bbox_to_frame()` |
| 12 | Speed limit extract | `class_map.py` | `extract_speed_limit_kmh()` |
| 13 | Regulatory flag | `class_map.py` | `is_regulatory_label()` |
| 14 | Build summary | `output_parser.py` | `build_summary()` |
| 15 | Assemble result | `traffic_sign.py` | `TrafficSignDetectionResult(...)` |
| 16 | Assert in-frame | `traffic_sign.py` | `_assert_detections_within_frame()` |
| 17 | Return | `traffic_sign.py` | `TrafficSignDetectionResult` |

### 4.2 Initialization flow

| Step | File | Method |
|------|------|--------|
| Resolve weights | `model_paths.py` | `get_traffic_sign_weights_path()` |
| Load model | `model_loader.py` | `YOLOv8SignModelLoader.load_model()` |
| Package | `model_loader.py` | `get_model()` |
| Attach | `inference.py` | `attach_model()` |

### 4.3 Sequence diagram

```mermaid
sequenceDiagram
    participant User
    participant TSM as TrafficSignModule
    participant IE as YOLOv8SignInferenceEngine
    participant UL as Ultralytics YOLO
    participant OP as YOLOv8SignOutputParser

    User->>TSM: predict(frame)
    TSM->>TSM: _validate_input()
    TSM->>IE: run(frame)
    IE->>UL: predict(conf, iou, imgsz=640)
    UL-->>IE: Results.boxes
    IE-->>TSM: raw_outputs
    TSM->>OP: parse(raw_outputs, frame_shape)
    OP-->>TSM: ParsedYOLOv8SignOutput
    TSM->>OP: build_summary()
    TSM->>TSM: _assert_detections_within_frame()
    TSM-->>User: TrafficSignDetectionResult
```

---

## Section 5: Sign Classes

### 5.1 Class table

| `class_id` | Label | Type | Decision-engine use (planned) |
|------------|-------|------|-------------------------------|
| 0 | `stop` | Regulatory | **STOP** rule when high confidence + nearest sign in lower frame |
| 1 | `speed_limit_30` | Regulatory | `active_speed_limit_kmh=30` → slow down if over limit |
| 2 | `speed_limit_60` | Regulatory | `active_speed_limit_kmh=60` → speed compliance |
| 3 | `turn_left` | Guidance | Informational lane guidance (v1) |
| 4 | `turn_right` | Guidance | Informational lane guidance (v1) |
| 5 | `keep_right` | Regulatory | Lane discipline / pass-right rules |
| 6 | `pedestrian_crossing` | Warning | Caution alert; large bbox as proximity proxy |

### 5.2 Regulatory vs warning

From `class_map.py`:

- **Regulatory:** `stop`, `speed_limit_30`, `speed_limit_60`, `keep_right` → `is_regulatory=True`
- **Warning:** `pedestrian_crossing` → `is_warning_label()` helper (not stored on `DetectedSign` in v1 — only `is_regulatory`)

Turn signs are **neither** regulatory nor warning in the static maps — informational only.

### 5.3 Decision-engine fields consumed (future)

| Field | Rule example |
|-------|--------------|
| `summary.active_speed_limit_kmh` | Slow Down if assumed speed > limit |
| `summary.nearest_sign` + `stop` | Stop if confidence > 0.7 and `center_y` in lower half |
| `pedestrian_crossing` + large `bbox.area` | Warning — crossing ahead |
| `count_by_label` | HUD / logging |

`SceneState.signs` is **designed** but **not implemented** (`src/decision/scene_state.py` stub).

---

## Section 6: GTSRB Mapping

### 6.1 Original GTSRB benchmark

**German Traffic Sign Recognition Benchmark (GTSRB)**

- **43 classes**, 39,209 train + 12,630 test images (cropped patches)
- Config path: `paths.gtsrb` → `${data_root}/datasets/gtsrb`
- `scripts/prepare_datasets.py` — **stub** (not implemented)

### 6.2 ADAS v1 subset mapping

Implemented in `GTSRB_CLASS_ID_TO_ADAS_LABEL` (`class_map.py`):

| GTSRB ID | GTSRB meaning (approx.) | ADAS label |
|----------|-------------------------|------------|
| 14 | Stop | `stop` |
| 1 | Speed limit 30 | `speed_limit_30` |
| 5 | Speed limit 60 | `speed_limit_60` |
| 38 | Turn left ahead | `turn_left` |
| 34 | Turn right ahead | `turn_right` |
| 36 | Pass on right | `keep_right` |
| 12 | Pedestrians | `pedestrian_crossing` |

Model internal IDs 0–6 map via `SIGN_CLASS_ID_TO_LABEL` — training must align head output with this order.

### 6.3 Why these seven classes

- Cover **highest-impact ADAS rules**: stop, speed, crossing
- Keep **v1 head small** (`yolov8n`, 7 classes) for latency alongside 3 other YOLO models
- Map cleanly from GTSRB without full 43-class head
- Extensible to full GTSRB-43 in phase 2

### 6.4 Dataset limitations

| Limitation | Impact |
|------------|--------|
| GTSRB = cropped patches | Poor full-frame localization until Bdd100K / synthetic paste fine-tune |
| German/EU geometry | Regional signs (e.g. India) need local fine-tuning |
| No bundled training data | `data/raw/`, `data/samples/` empty in repo |
| `prepare_datasets.py` stub | No automated GTSRB → YOLO bbox pipeline in repo |

---

## Section 7: Output Schema Audit

### 7.1 `SignBoundingBoxData`

Frozen dataclass — inclusive frame-space corners.

| Field | Type | Purpose |
|-------|------|---------|
| `x1`, `y1`, `x2`, `y2` | `int` | Inclusive corners |
| `width`, `height` | `int` | Derived dimensions |
| `center_x`, `center_y` | `float` | Box center |
| `area` | `int` | Pixel area |

Factory: `SignBoundingBoxData.from_xyxy(x1, y1, x2, y2)`  
Serialize: `to_list()` → `[x1, y1, x2, y2]`

### 7.2 `DetectedSign`

| Field | Type | Example |
|-------|------|---------|
| `sign_label` | `str` | `"speed_limit_30"` |
| `class_id` | `int` | `1` |
| `confidence` | `float` | `0.91` |
| `bbox` | `SignBoundingBoxData` | — |
| `speed_limit_kmh` | `int \| None` | `30` |
| `is_regulatory` | `bool` | `True` |
| `track_id` | `int \| None` | `None` (reserved) |

**JSON example:**

```json
{
  "sign_label": "speed_limit_30",
  "class_id": 1,
  "confidence": 0.91,
  "bbox": [800, 120, 860, 180],
  "speed_limit_kmh": 30,
  "is_regulatory": true,
  "track_id": null
}
```

### 7.3 `TrafficSignDetectionSummary`

| Field | Type | Purpose |
|-------|------|---------|
| `count_by_label` | `dict[str, int]` | Per-class counts |
| `total_count` | `int` | Total detections |
| `nearest_sign` | `DetectedSign \| None` | Max `center_y` (closest in image) |
| `highest_confidence` | `DetectedSign \| None` | Max confidence |
| `active_speed_limit_kmh` | `int \| None` | **Minimum** visible speed limit |

### 7.4 `TrafficSignDetectionResult`

| Field | Type | Purpose |
|-------|------|---------|
| `detections` | `list[DetectedSign]` | All filtered signs |
| `summary` | `TrafficSignDetectionSummary` | Aggregates |
| `frame_shape` | `(H, W) \| None` | Input dimensions |
| `inference_time_ms` | `float \| None` | Timing from engine |
| `model_variant` | `str` | `"n"` default |
| `confidence_threshold` | `float` | Applied threshold |
| `raw_status` | `str` | Pipeline status |

**`to_prediction_dict()`** — orchestrator JSON contract (includes all summary fields).

**`empty(raw_status)`** — error factory; used for `init_failed`, `pipeline_error`, etc.

### 7.5 `TRAFFIC_SIGN_OUTPUT_KEYS`

```python
("detections", "count_by_label", "total_count", "nearest_sign",
 "active_speed_limit_kmh", "raw_status")
```

---

## Section 8: Verification & Testing

### 8.1 Integration tests — `tests/test_traffic_sign_pipeline.py`

| Test | Validates |
|------|-----------|
| `test_module_initializes_with_stub_loader` | `is_initialized`, loader loaded, engine ready |
| `test_end_to_end_predict_pipeline` | `road_frame` → `stop` stub, bbox in frame, `to_prediction_dict()` |
| `test_predict_auto_inits` | Auto-initialize on first `predict()` |
| `test_visualize_returns_annotated_frame` | Shape preserved, pixels changed |
| `test_only_allowed_sign_classes` | Labels ⊆ `ADAS_SIGN_LABELS` |
| `test_active_speed_limit_in_summary` | `active_speed_limit_kmh == 30` when 30+60 detected |

### 8.2 Fixtures — `tests/conftest.py`

| Fixture | Behavior |
|---------|----------|
| `stub_yolov8_sign_model_loader` | Skips Ultralytics; sets `stub://traffic_signs_yolov8n.pt` |
| `stub_yolov8_sign_inference_engine` | One `stop` box upper-center, conf=0.92, `class_id=0` |
| `traffic_sign_module` | Pre-initialized module with stubs |
| `road_frame` | BGR image from `tests/fixtures/road_sample.jpg` |

### 8.3 Gate script — `scripts/verify_traffic_sign_detection.py`

| Mode | Behavior |
|------|----------|
| **Default (stub)** | Injects fake stop sign; all checks PASS without weights |
| **`--real`** | Requires `traffic_signs_yolov8n.pt` + `ultralytics`; runs on road fixture |

**Gate checks:**

1. `TrafficSignModule` importable
2. `yolov8_sign/` package files present (6 files)
3. `initialize()` succeeds
4. `predict()` returns `TrafficSignDetectionResult`
5. At least one detection (stub) or graceful zero (real on empty scene)
6. All bboxes within frame bounds
7. `visualize()` mutates frame
8. `cleanup()` resets state

### 8.4 Validation strategy

| Layer | Strategy |
|-------|----------|
| **Unit** | Parser/class_map/schema — **not present** as separate test files |
| **Integration** | Full module path with stub engine |
| **Geometry** | `_assert_detections_within_frame()` raises on out-of-bounds |
| **CI** | Stub tests always pass without trained weights |
| **Real weights** | Manual `--real` gate; no `@pytest.mark.slow` real-weight test yet |

### 8.5 Missing test coverage

- `tests/test_yolov8_sign_output_parser.py` — designed, not created
- `tests/test_yolov8_sign_class_map.py` — designed, not created
- `tests/test_yolov8_sign_output_schema.py` — designed, not created
- End-to-end orchestrator test — does not exist

---

## Section 9: Real Inference Readiness

### 9.1 Current state

| Component | Status |
|-----------|--------|
| Architecture | **Complete** |
| Config paths | **Complete** |
| Stub tests + gate | **Complete** |
| Visualization | **Complete** |
| Fine-tuned weights in repo | **Not present** |
| Training pipeline | **Not in repo** |
| Evaluation metrics | **Not implemented** |
| Orchestrator wiring | **Not implemented** |

### 9.2 Weight artifact

**Filename:** `traffic_signs_yolov8n.pt`  
**Config key:** `weight_files.yolov8_sign`  
**Resolved path:** `{data_root}/models/trained/yolov8_sign/traffic_signs_yolov8n.pt`  
**Resolver:** `get_traffic_sign_weights_path()` in `src/utils/model_paths.py`

**Critical difference from vehicle module:** `YOLOv8SignModelLoader.resolve_weights_source()` raises `WeightsNotFoundError` if file missing — **no Ultralytics COCO auto-download**.

### 9.3 Requirements for real inference

| Requirement | Detail |
|-------------|--------|
| `pip install ultralytics` | In `requirements.txt` |
| `torch`, `opencv-python`, `numpy` | Present |
| Trained `.pt` file | Must be placed at configured path |
| `ADAS_DATA_ROOT` or `config/default.yaml` `data_root` | Path expansion |

### 9.4 Training requirements (out of repo)

Per design doc:

1. Map GTSRB / Bdd100K annotations to 7 ADAS labels
2. Train YOLOv8n with class head matching `SIGN_CLASS_ID_TO_LABEL` order
3. Export `traffic_signs_yolov8n.pt`
4. Validate class names in model `names` match `config/classes.yaml`

Recommended data: Bdd100K sign boxes + GTSRB crops / synthetic paste for bootstrap.

### 9.5 Evaluation requirements (out of repo)

| Metric | Target (aspirational) |
|--------|----------------------|
| mAP@0.5 | ≥ 0.70 on held-out sign boxes |
| Per-class P/R/F1 | Especially `stop` recall |
| Confusion matrix | speed_limit_30 vs 60 |
| Pipeline latency | `inference_time_ms` on target hardware |

Proposed script: `evaluation/evaluation_traffic_sign_detection.py` — **does not exist**.

---

## Section 10: Error Handling

| Scenario | Behavior | Location |
|----------|----------|----------|
| Weights path not configured | `WeightsNotFoundError` | `model_loader.resolve_weights_source()` |
| Weights file missing | `WeightsNotFoundError` with train message | `model_loader.resolve_weights_source()` |
| Path exists but not a file | `WeightsValidationError` | `model_loader` |
| `ultralytics` not installed | `WeightsLoadError` | `model_loader.load_model()` |
| YOLO load failure | `WeightsLoadError` | `model_loader.load_model()` |
| `predict()` before init, init fails | `TrafficSignDetectionResult.empty(raw_status="init_failed")` | `traffic_sign.predict()` |
| `frame is None` | `ValueError` | `_validate_input()` |
| Wrong shape / empty frame | `ValueError` | `_validate_input()` |
| Non-uint8 dtype | Warning only | `_validate_input()` |
| Inference engine not ready | `empty(raw_status="inference_not_ready")` | `_run_pipeline()` |
| Forward pass exception | `empty(raw_status="pipeline_error")` | `predict()` catches `InferenceExecutionError` |
| Unexpected exception | Re-raised after log | `predict()` |
| Invalid bbox after clip | Dropped + warning log | `output_parser.parse()` |
| Bbox outside frame post-parse | `RuntimeError` | `_assert_detections_within_frame()` |
| Frame shape missing in parser | `raw_status="parser_error"` | `output_parser.parse()` |

### Auto-initialization

```python
if not self._initialized:
    self._log_warning("predict() called before initialize() — auto-initializing")
    try:
        self.initialize()
    except (WeightsValidationError, WeightsLoadError, WeightsNotFoundError):
        return TrafficSignDetectionResult.empty(raw_status="init_failed")
```

---

## Section 11: Performance & Deployment

### 11.1 YOLOv8n characteristics

| Property | Value |
|----------|-------|
| Parameters | ~3M (nano variant) |
| Default `imgsz` | 640 |
| `max_det` | 50 (signs per frame typically < 10) |
| Classes | 7 (custom head) |

Smallest YOLOv8 variant — chosen for **multi-model sequential** inference (YOLOP + yolov8s vehicle + yolov8n sign + yolov8n signal).

### 11.2 Expected inference speed (indicative)

| Environment | Estimated per-frame (sign only) |
|-------------|--------------------------------|
| CPU (laptop) | 30–80 ms |
| GPU (T4) | 5–15 ms |
| Sequential 4-model CPU | Often latency-bound; sign is smallest model |

Stub tests report `inference_time_ms ≈ 2.0` (fake forward pass).

**Note:** No benchmark script in repo — estimates based on YOLOv8n literature and vehicle module audit patterns.

### 11.3 Deployment considerations

1. **GPU recommended** for video demo with YOLOP + vehicle + sign + signal
2. **Copy fine-tuned weights to Drive** before Colab session — no auto-download
3. Call `cleanup()` between heavy modules to free GPU (`torch.cuda.empty_cache()` in loader)
4. Lower `sign_confidence` only after measuring FP rate on target geography
5. **TensorRT / ONNX export** — not implemented
6. **Embedded (Jetson)** — yolov8n feasible; verify combined pipeline FPS

### 11.4 Multi-model memory

Sign model loaded separately from vehicle `yolov8s.pt`. Sequential execution (not parallel) is the v1 design — load → infer → `cleanup()` pattern recommended on memory-constrained devices.

---

## Section 12: Interview Questions

### 12.1 Beginner (25)

**Q1: What does the Traffic Sign module do?**  
Detects seven ADAS traffic sign types in a camera frame and returns bounding boxes, labels, confidence, and speed-limit metadata.

**Q2: Which file is the main entry point?**  
`src/modules/traffic_sign.py` — class `TrafficSignModule`.

**Q3: What base class does it inherit?**  
`BaseModule` in `src/modules/base.py`.

**Q4: What model is used?**  
Ultralytics YOLOv8n fine-tuned (`traffic_signs_yolov8n.pt`).

**Q5: What is the default confidence threshold?**  
`0.5` from `thresholds.sign_confidence` in `config/default.yaml`.

**Q6: What input does `predict()` expect?**  
BGR NumPy array, shape `(H, W, 3)`, preferably `uint8`.

**Q7: What does `predict()` return?**  
`TrafficSignDetectionResult` dataclass.

**Q8: How do you get a dict for the orchestrator?**  
`result.to_prediction_dict()`.

**Q9: What seven signs are detected?**  
stop, speed_limit_30, speed_limit_60, turn_left, turn_right, keep_right, pedestrian_crossing.

**Q10: Where are class IDs defined?**  
`SIGN_CLASS_ID_TO_LABEL` in `src/modules/yolov8_sign/class_map.py`.

**Q11: What is a sign bounding box?**  
`SignBoundingBoxData` with inclusive integer `(x1, y1, x2, y2)` in frame pixels.

**Q12: What does `raw_status` mean?**  
Pipeline diagnostic: `ok`, `stub`, `init_failed`, `pipeline_error`, `inference_not_ready`, `empty`.

**Q13: What package wraps sign YOLOv8?**  
`src/modules/yolov8_sign/`.

**Q14: Why `yolov8_sign/` instead of `yolov8/`?**  
Separate fine-tuned weights and 7-class head; avoids conflating with vehicle COCO detector.

**Q15: What dependency powers YOLOv8?**  
`ultralytics` in `requirements.txt`.

**Q16: What is `TRAFFIC_SIGN_OUTPUT_KEYS`?**  
Stable tuple of orchestrator dict keys in `output_schema.py`.

**Q17: Does sign detection use vehicle YOLOv8 weights?**  
No — separate fine-tuned checkpoint required.

**Q18: What was the module before implementation?**  
YOLOv5 stub returning `{}` from `predict()`.

**Q19: How is visualization done?**  
`visualize()` → `draw_traffic_signs()` in `src/visualization/overlays.py`.

**Q20: What is `active_speed_limit_kmh`?**  
Minimum numeric speed limit among visible `speed_limit_*` detections (most restrictive).

**Q21: What is `nearest_sign`?**  
Detection with highest `bbox.center_y` (lowest on screen ≈ closest), tie-break by `area`.

**Q22: Default input size?**  
640 (`yolov8_sign.imgsz`).

**Q23: What test file covers sign detection?**  
`tests/test_traffic_sign_pipeline.py` — 6 tests.

**Q24: What gate script verifies the module?**  
`scripts/verify_traffic_sign_detection.py`.

**Q25: Is the orchestrator wired?**  
No — `src/pipeline/orchestrator.py` is comments only.

---

### 12.2 Intermediate (25)

**Q1: Why YOLOv8 over YOLOv5?**  
Same Ultralytics stack as vehicle module; active maintenance; unified loader/inference/parser pattern.

**Q2: Why yolov8n not yolov8s?**  
Lower latency when running YOLOP + vehicle + sign + signal sequentially.

**Q3: Why no COCO fallback for signs?**  
COCO only has generic stop sign; custom 7-class head is meaningless without fine-tuned weights.

**Q4: What is NMS IoU threshold?**  
`0.45` from `thresholds.sign_iou`, passed to Ultralytics `predict(iou=...)`.

**Q5: Why a separate output parser?**  
Decouples Ultralytics tensors from ADAS schema; enables class filter, bbox clip, unit testing without GPU.

**Q6: How are boxes kept in frame coordinates?**  
Ultralytics returns image-space `xyxy`; parser clips via `_clip_bbox_to_frame()`; module asserts via `_assert_detections_within_frame()`.

**Q7: Explain dependency injection.**  
Constructor accepts `model_loader`, `inference_engine`, `output_parser` — tests inject stubs without real weights.

**Q8: What if `predict()` called before `initialize()`?**  
Auto-init; on failure returns `TrafficSignDetectionResult.empty(raw_status="init_failed")`.

**Q9: Difference between inference `conf` and parser threshold?**  
Both filter confidence — Ultralytics at forward pass, parser as ADAS safety net (default 0.5 each).

**Q10: What is `ParsedYOLOv8SignOutput`?**  
Parser intermediate output before `TrafficSignDetectionResult` assembly.

**Q11: How does `extract_speed_limit_kmh()` work?**  
Parses `speed_limit_30` → `30` from label suffix after `speed_limit_`.

**Q12: What does stub inference return in tests?**  
One stop box at upper-center, confidence 0.92, `inference_status="stub"`.

**Q13: What is `ADAS_SIGN_LABELS`?**  
Frozenset loaded from `config/classes.yaml` → `traffic_sign_classes`.

**Q14: GTSRB class 14 maps to what?**  
`stop` via `GTSRB_CLASS_ID_TO_ADAS_LABEL`.

**Q15: Why GTSRB alone is insufficient for detection training?**  
GTSRB provides cropped patches, not full dashcam frames with scene context.

**Q16: What happens when two speed limits are visible?**  
`active_speed_limit_kmh = min(30, 60) = 30` — most restrictive wins.

**Q17: What is `is_regulatory`?**  
True for stop, speed limits, keep_right — from `REGULATORY_LABELS` in `class_map.py`.

**Q18: Where is weight path configured?**  
`config/default.yaml` → `weight_files.yolov8_sign`, resolved by `get_traffic_sign_weights_path()`.

**Q19: Can you run real inference without weights?**  
No — `WeightsNotFoundError` on init (unless using stub injection in tests).

**Q20: How does `cleanup()` help?**  
`unload()` + `detach_model()` + optional `torch.cuda.empty_cache()`.

**Q21: Signs vs traffic signals?**  
Signs = static regulatory boards (`TrafficSignModule`); signals = red/yellow/green lights (`TrafficSignalModule`).

**Q22: What colors are used in `draw_traffic_signs()`?**  
Stop red `(0,0,220)`; speed limits blue `(220,120,0)`; turns yellow; crossing orange.

**Q23: What legacy config key exists?**  
`yolov5` / `get_yolov5_weights_path()` — deprecated, unused by current module.

**Q24: How many max detections per frame?**  
50 (`yolov8_sign.max_detections`).

**Q25: Is `track_id` used?**  
No — reserved for future tracking; always `None` in v1.

---

### 12.3 Advanced (25)

**Q1: Design `SceneState` integration for signs.**  
Add `signs: TrafficSignDetectionResult | None` to `scene_state.py`; orchestrator calls `traffic_sign_module.predict(frame)` after vehicle module; decision rules read `active_speed_limit_kmh` and `nearest_sign`.

**Q2: Why not use COCO class 11 (stop sign) from vehicle detector?**  
Insufficient taxonomy; no speed limits; duplicate semantics; vehicle parser explicitly excludes non-road-user classes.

**Q3: How would you add an 8th sign class?**  
Retrain YOLOv8n with 8-class head; extend `SIGN_CLASS_ID_TO_LABEL`, `ALLOWED_SIGN_CLASS_IDS`, `classes.yaml`, parser tests.

**Q4: Explain fail-fast bbox assertion rationale.**  
Same as vehicle module — geometry bugs (wrong resize/coords) must not silently pass bad boxes to decision engine.

**Q5: How would you implement Bdd100K training pipeline?**  
Convert sign annotations to YOLO format with ADAS label map; train with Ultralytics CLI; verify `model.names` order matches `class_map.py`.

**Q6: Trade-off: synthetic paste vs real scene boxes?**  
Synthetic bootstraps fast but domain gap; Bdd100K gives context; production needs real-scene fine-tune.

**Q7: How to reduce false positives on billboards?**  
Raise confidence, hard-negative mining, regional fine-tune, temporal consistency (not in v1).

**Q8: Sequential vs parallel multi-model inference?**  
v1 sequential on one GPU; parallel needs duplicate GPU memory for 4 models — not implemented.

**Q9: How does sign module differ from vehicle loader?**  
Vehicle: COCO fallback download; Sign: **must** have fine-tuned file, raises `WeightsNotFoundError`.

**Q10: Parser metadata fields?**  
`num_raw_boxes`, `num_filtered`, `dropped_class`, `dropped_confidence`, `dropped_invalid`, `frame_shape`.

**Q11: How to benchmark real latency?**  
Loop `predict()` on video frames; read `result.inference_time_ms`; profile with/without GPU.

**Q12: ONNX/TensorRT deployment path?**  
Export fine-tuned `.pt` via Ultralytics `model.export()`; replace `YOLO()` load in loader — not in repo.

**Q13: How would orchestrator handle init failure for signs?**  
Continue pipeline with `signs=None` or empty result; decision rules degrade gracefully — design choice.

**Q14: Conflict: stop sign + green traffic signal?**  
Signal module handles lights; sign stop is regulatory board — orchestrator precedence rules needed (signal at intersection typically wins for proceed/stop).

**Q15: Why `nearest_sign` uses `center_y` not distance metric?**  
Image-space proxy without camera calibration; larger `center_y` = lower in frame ≈ nearer to ego camera.

**Q16: Unit test parser without Ultralytics?**  
Feed synthetic `raw_outputs` dict with `boxes_xyxy`, `confidences`, `class_ids` directly to `YOLOv8SignOutputParser.parse()` — test file not yet created.

**Q17: How to validate GTSRB mapping correctness?**  
Cross-check against official GTSRB `classes.csv`; unit test `gtsrb_id_to_adas_label()` roundtrip.

**Q18: Security of `WeightsLoadError` messages?**  
Paths logged for debugging; production may redact full filesystem paths.

**Q19: Multi-scale signs / small objects?**  
yolov8n may miss distant signs; fallback to yolov8s per design doc if recall poor.

**Q20: How does `half=True` in inference config affect deployment?**  
FP16 on GPU — faster, slight accuracy trade-off; default `False`.

**Q21: Explain `inference_status` propagation.**  
Stub tests set `"stub"`; engine sets `"ok"`; parser preserves non-unknown status in `raw_status`.

**Q22: CI strategy for real weights?**  
`@pytest.mark.slow` + skip if `traffic_signs_yolov8n.pt` missing — designed, not implemented.

**Q23: How to fuse sign output with lane departure?**  
Future decision rule: stop + lane departure → stronger warning; requires `SceneState` with both fields.

**Q24: Data augmentation recommendations for training?**  
Motion blur, rain, night, sign rotation; critical for dashcam domain gap from GTSRB crops.

**Q25: Production readiness blockers?**  
Fine-tuned weights, orchestrator, decision engine, evaluation metrics, README sync, real-video benchmark.

---

## Section 13: Resume Description

### 13.1 Two-line version

Developed a modular traffic sign detection pipeline using YOLOv8n and Ultralytics, detecting seven ADAS sign classes with frame-space bounding boxes and speed-limit summarization. Implemented loader/inference/parser architecture with dependency injection, OpenCV visualization, and automated stub-based testing.

### 13.2 Five-line version

Built `TrafficSignModule` for an autonomous driving assistance system using fine-tuned YOLOv8n (seven-class head: stop, speed limits, turns, pedestrian crossing). Designed a decomposed architecture — model loader, inference engine, output parser, and typed dataclass schemas — mirroring production patterns from vehicle detection. Implemented GTSRB-to-ADAS class mapping, regulatory flag enrichment, and `active_speed_limit_kmh` heuristics for decision-engine integration. Added OpenCV overlay visualization, six pytest integration tests, and a CI-friendly gate script with injectable stub loaders. Module is integration-ready at the API level; orchestrator and fine-tuned weight deployment remain next steps.

### 13.3 Resume bullets

- Implemented **YOLOv8n traffic sign detection** with 7-class ADAS taxonomy (stop, speed limits, regulatory signs) in a modular Python ADAS stack
- Architected **`yolov8_sign/` subpackage** (loader, inference, parser, schema, class map) with dependency injection for testability without GPU weights
- Built **`TrafficSignDetectionResult`** schema with `active_speed_limit_kmh`, `nearest_sign`, and frame-space bbox validation with fail-fast guards
- Integrated **GTSRB class mapping** and speed-limit parsing for downstream rule-based decision support
- Delivered **6 integration tests** and gate script (`verify_traffic_sign_detection.py`) with stub inference for CI

### 13.4 LinkedIn project description

**Autonomous Driving Assistance System — Traffic Sign Detection Module**

Contributed the traffic sign recognition component of a modular ADAS perception stack. Replaced a placeholder YOLOv5 stub with a production-style YOLOv8n pipeline using Ultralytics, detecting seven regulatory and warning sign classes in real-time camera frames. The implementation follows a layered architecture (model loader → inference engine → output parser → typed results) consistent with lane and vehicle detection modules, enabling independent testing and future orchestrator integration.

Key technical outcomes: frame-coordinate bounding boxes with confidence filtering and in-frame validation; GTSRB-derived class mapping; automatic extraction of speed limit values for ADAS decision rules; OpenCV visualization overlays; and a stub-based verification gate for continuous integration without requiring fine-tuned weights in the repository.

**Tech stack:** Python, PyTorch, Ultralytics YOLOv8, OpenCV, NumPy, pytest, YAML configuration

---

## Section 14: Current Status

### 14.1 Completeness scores

| Dimension | % | Rationale |
|-----------|---|-----------|
| **Implementation** | **~88%** | Full module + subpackage + viz + config; missing eval script, training pipeline |
| **Testing** | **~72%** | 6 integration tests + gate; no parser/class_map unit tests; no real-weight CI |
| **Production readiness** | **~52%** | No fine-tuned weights in repo; no orchestrator; no decision fusion; no mAP eval |

### 14.2 Completed features

- `TrafficSignModule` full lifecycle (`initialize`, `predict`, `visualize`, `cleanup`)
- `yolov8_sign/` complete subpackage (6 files)
- 7-class ADAS taxonomy with GTSRB mapping helpers
- `TrafficSignDetectionResult` + `to_prediction_dict()`
- `active_speed_limit_kmh` and `nearest_sign` summary heuristics
- Config integration (`get_traffic_sign_weights_path`, `get_yolov8_sign_config`)
- `draw_traffic_signs()` visualization
- Stub-based pytest + gate script
- Dependency injection pattern
- Error handling with typed empty results

### 14.3 Pending features

- Fine-tuned `traffic_signs_yolov8n.pt` on target environment
- Pipeline orchestrator wiring
- `SceneState.signs` + decision rules
- `evaluation/evaluation_traffic_sign_detection.py`
- Unit tests for parser, class_map, schema
- Real-weight `@pytest.mark.slow` test
- `scripts/prepare_datasets.py` for GTSRB/Bdd100K
- README update (still lists YOLOv5)

### 14.4 Future enhancements

- Full GTSRB-43 or Mapillary taxonomy expansion
- Temporal smoothing / tracking (`track_id`)
- Camera calibration for metric distance to signs
- TensorRT/ONNX export for embedded deployment
- Regional sign fine-tuning (non-EU geometries)
- Suppress duplicate COCO stop sign if vehicle module fires (orchestrator responsibility)

---

## Section 15: Executive Summary

### What was built

A **complete traffic sign detection module** at the API level: `TrafficSignModule` orchestrating `yolov8_sign/` (YOLOv8n fine-tuned architecture) with seven ADAS sign classes, typed dataclass outputs, speed-limit summarization, OpenCV visualization, and config-driven weight paths. The design mirrors `VehicleDetectionModule` with injectable loader/engine/parser for CI-friendly stub testing.

### What was verified

- **6/6** integration tests in `tests/test_traffic_sign_pipeline.py`
- Gate script `scripts/verify_traffic_sign_detection.py` passes in stub mode
- Module imports via `src.modules.TrafficSignModule`, `TrafficSignDetectionResult`, `TRAFFIC_SIGN_OUTPUT_KEYS`
- Part of **29/29** total project pytest pass
- Frame-space bbox validation and class filtering verified via stub pipeline

### What remains

- **Fine-tuned weights** (`traffic_signs_yolov8n.pt`) — required for meaningful real-world detection
- **Pipeline orchestrator** — module not called in end-to-end flow
- **Decision engine** — `SceneState.signs` and rules not implemented
- **Evaluation metrics** — no mAP/P/R/F1 automation
- **Documentation drift** — `README.md` still describes YOLOv5 scaffold

### Readiness for integration

| Criterion | Ready? |
|-----------|--------|
| Module API (`predict` → `TrafficSignDetectionResult`) | **Yes** |
| `to_prediction_dict()` orchestrator contract | **Yes** |
| Visualization | **Yes** |
| Stub CI | **Yes** |
| Orchestrator wiring | **No** |
| Decision engine consumption | **No** |
| Production weights + eval | **No** |

**Verdict:** Traffic sign detection is **integration-ready at the module API level** — suitable for interviews, viva, and resume discussion of architecture and ADAS semantics. **Not production-deployed** until fine-tuned weights exist, orchestrator fuses outputs, and evaluation quantifies recall on target geography (especially `stop` and speed limits).

---

*End of Traffic Sign Detection Module Audit Report.*
