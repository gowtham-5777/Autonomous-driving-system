# Autonomous Driving Car — Final Project Master Report

**Repository:** Autonomous Driving Car  
**Audit date:** June 2026  
**Method:** Read-only inspection of source, config, tests, scripts, data/, models/, docs/  
**Test baseline:** **75 passed** (`python -m pytest tests/ -q`)  
**Primary UI:** `app.py` (Streamlit)  
**Pipeline entry:** `src/pipeline/orchestrator.py` → `PipelineOrchestrator.run_frame()`

> Facts come from files on disk. **Stub**, **Missing on disk**, and **Not implemented** are labeled explicitly.

---

## Table of Contents

1. Executive Summary  
2. Complete Repository Architecture  
3. System Architecture  
4. Lane Detection Deep Audit  
5. Vehicle Detection Deep Audit  
6. Traffic Sign Deep Audit  
7. Traffic Signal Deep Audit  
8. Decision Engine Audit  
9. Pipeline Orchestrator Audit  
10. UI Layer Audit  
11. Dataset Audit  
12. Model Selection Justification  
13. Verification & Testing  
14. Technical Challenges  
15. Production Readiness  
16. Resume & Placement Narrative  
17. Interview Preparation Guide (300 Q&A)  
18. Viva Preparation Guide  
19. Ownership Narrative  
20. Final Completion Report  

---

# SECTION 1 — Executive Summary

This repository implements a **modular Autonomous Driving Assistance System (ADAS)** in Python. A single BGR camera frame flows through four perception modules (lane, vehicle, traffic sign, traffic signal), aggregates into `SceneState`, and produces an explainable driving recommendation via a **12-rule** `DecisionEngine`.

| Area | Status |
|------|--------|
| Core pipeline software | **Implemented** — `PipelineOrchestrator`, four modules, decision engine |
| Automated tests | **75/75 pass** in `tests/` |
| Pretrained weights (YOLOP, YOLOv8s) | **Present** under `models/pretrained/` |
| Trained sign/signal `.pt` | **Missing on disk** at audit (paths in `config/default.yaml`) |
| Traffic signal YOLO dataset | **Present** — `data/traffic_signal_yolo/` (42,048 images, 147,550 boxes) |
| Traffic sign YOLO dataset | **Not prepared** — GTSRB external, no `data/traffic_sign_yolo/` |
| Sign training pipeline | **Implemented** — `training/traffic_sign/` + scripts |
| Signal training pipeline | **Implemented** — metrics P=0.58, R=0.443, mAP50=0.443 |
| UI | **Streamlit** `app.py`; `src/app.py` Gradio **stub** |
| Segmentation | **Stub** — not in pipeline (`run_segmentation: false`) |

---

# SECTION 2 — Complete Repository Architecture

```
Autonomous Driving Car/
├── app.py                          # Streamlit UI (primary)
├── config/
│   ├── default.yaml                # Paths, thresholds, models
│   └── classes.yaml                # Sign/signal class names
├── data/
│   ├── traffic_signal_yolo/        # Prepared BDD100K YOLO dataset
│   ├── traffic_signal_yolo_colab_sample.zip
│   ├── raw/ processed/ samples/    # Placeholders (.gitkeep)
├── docs/                           # Design, audit, training reports
├── models/
│   ├── pretrained/yolop/End-to-end.pth   # YES on disk
│   ├── pretrained/yolov8/yolov8s.pt        # YES on disk
│   └── trained/                            # sign/signal .pt MISSING
├── scripts/                        # verify, train, prepare, evaluate
├── src/
│   ├── pipeline/orchestrator.py
│   ├── decision/                   # engine, rules, scene_state, types
│   ├── modules/                    # lane, vehicle, sign, signal, yolop, yolov8*
│   ├── visualization/              # overlays, hud
│   ├── preprocessing/
│   └── app.py                      # Gradio stub
├── tests/                          # 75 pytest tests
└── training/
    ├── traffic_signal/             # bdd100k_converter, config
    └── traffic_sign/               # gtsrb_converter, config
```

**Key entry points:** `app.py`, `scripts/verify_pipeline.py`, `PipelineOrchestrator.run_frame()`.

---

# SECTION 3 — System Architecture

```
BGR Frame (H,W,3)
    │
    ▼
PipelineOrchestrator.run_frame()
    ├── LaneDetectionModule.predict()      → LaneDetectionResult
    ├── VehicleDetectionModule.predict()   → VehicleDetectionResult
    ├── TrafficSignModule.predict()        → TrafficSignDetectionResult
    ├── TrafficSignalModule.predict()      → TrafficSignalDetectionResult
    │
    ▼
SceneState.from_perception()
    │
    ▼
DecisionEngine.evaluate() → DecisionResult
    │
    ▼
PipelineOrchestrator.visualize() → annotated frame + HUD
```

Reference order: `REFERENCE_ORDER` in `orchestrator.py` — lane_detection, vehicle_detection, traffic_sign, traffic_signal.

---

# SECTION 4 — Lane Detection Deep Audit

**Module:** `src/modules/lane_detection.py` — `LaneDetectionModule`  
**Model:** YOLOP MCnet vendored in `src/modules/yolop/vendor/`  
**Weights:** `get_yolop_weights_path()` → `models/pretrained/yolop/End-to-end.pth` (**exists**)

**Pipeline (`_run_pipeline`):**
1. `LanePreprocessor.preprocess()` — classical edges/ROI  
2. `YOLOPInferenceEngine.run()`  
3. `YOLOPOutputParser.parse()` — lane + drivable masks  
4. Optional `postprocess_lane_mask()`  
5. **`_resize_masks_to_frame_shape()`** — `LANE_DETECTION_PIPELINE_VERSION = 2`  
6. `LaneGeometryExtractor` — lane_center_x, vehicle_offset, lane_departure  

**Outputs:** `LaneDetectionResult` — left/right lane polylines, masks, offset, departure flag.

**Tests:** `tests/test_lane_detection_pipeline.py`, `tests/test_mask_resize_geometry.py`, `tests/test_yolop_output_parser.py`.

---

# SECTION 5 — Vehicle Detection Deep Audit

**Module:** `src/modules/vehicle_detection.py` — `VehicleDetectionModule`  
**Model:** Ultralytics YOLOv8s — `config/default.yaml` yolov8.model_variant: s  
**Weights:** `models/pretrained/yolov8/yolov8s.pt` (**exists**)

**COCO filter** (`src/modules/yolov8/output_parser.py`):

| COCO ID | Label |
|---------|-------|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |

**Thresholds:** confidence 0.5, IoU 0.45 from `config/default.yaml`.

**Tests:** `tests/test_vehicle_detection_pipeline.py`.

---

# SECTION 6 — Traffic Sign Deep Audit

**Module:** `src/modules/traffic_sign.py` — `TrafficSignModule`  
**Subpackage:** `src/modules/yolov8_sign/` (loader, inference, parser, class_map, schema)  
**Classes (7):** stop, speed_limit_30/60, turn_left/right, keep_right, pedestrian_crossing — `class_map.py`  
**Weights:** `models/trained/yolov8_sign/traffic_signs_yolov8n.pt` — **MISSING on disk**

**Training pipeline (NEW — implemented):**

| Script | Role |
|--------|------|
| `scripts/prepare_traffic_sign_dataset.py` | GTSRB → YOLO |
| `scripts/train_traffic_sign.py` | YOLOv8n training |
| `scripts/evaluate_traffic_sign.py` | Validation metrics |
| `scripts/export_traffic_sign_dataset.py` | Export bundle |
| `training/traffic_sign/gtsrb_converter.py` | Conversion logic |
| `training/traffic_sign/config.py` | `TrafficSignTrainingConfig` |

**GTSRB:** NOT on disk in repo; default root `ADAS_GTSRB_ROOT` or `C:\Users\gauth\Desktop\ADAS_DATASETS\gtsrb`.  
**Prepared dataset:** `data/traffic_sign_yolo/` — **not prepared yet**.

**Tests:** `tests/test_traffic_sign_pipeline.py`, `tests/test_traffic_sign_training.py`.

---

# SECTION 7 — Traffic Signal Deep Audit

**Module:** `src/modules/traffic_signal.py` — `TrafficSignalModule`  
**Subpackage:** `src/modules/yolov8_signal/`  
**Classes (3):** red_light, yellow_light, green_light — aligned with training  
**Weights:** `models/trained/yolov8_signal/traffic_signals_yolov8n.pt` — **MISSING on disk**

**Dataset:** `data/traffic_signal_yolo/` — 42,048 images, 147,550 boxes (`dataset_stats.json`)

**Validation metrics** (`docs/traffic_signal_training_report.md`):

| Metric | Value |
|--------|-------|
| Precision | 0.580 |
| Recall | 0.443 |
| mAP@0.5 | 0.443 |
| mAP@0.5:0.95 | 0.149 |

**Training scripts:** prepare/train/evaluate/predict_traffic_signal.py; `training/traffic_signal/bdd100k_converter.py`.

**Tests:** `tests/test_traffic_signal_pipeline.py`, `tests/test_traffic_signal_training.py`.

---

# SECTION 8 — Decision Engine Audit

**Files:** `src/decision/decision_engine.py`, `rules.py`, `scene_state.py`, `types.py`

**Rules R01–R12:**

| ID | Function | Recommendation | Priority | Source |
|----|----------|----------------|----------|--------|
| R01 | rule_r01_red_light_stop | STOP | 100 | traffic_signal |
| R02 | rule_r02_stop_sign | STOP | 95 | traffic_sign |
| R03 | rule_r03_pedestrian_on_drivable | STOP | 90 | vehicle_detection |
| R04 | rule_r04_yellow_light_caution | SLOW_DOWN | 70 | traffic_signal |
| R05 | rule_r05_active_speed_limit | SLOW_DOWN | 65 | traffic_sign |
| R06 | rule_r06_vulnerable_road_user | SLOW_DOWN | 60 | vehicle_detection |
| R07 | rule_r07_pedestrian_crossing_sign | WARNING | 55 | traffic_sign |
| R08 | rule_r08_large_vehicle_proximity | WARNING | 50 | vehicle_detection |
| R09 | rule_r09_lane_departure | WARNING | 45 | lane_detection |
| R10 | rule_r10_lane_offset_correct | KEEP_LANE | 40 | lane_detection |
| R11 | rule_r11_green_proceed | PROCEED | 10 | traffic_signal |
| R12 | rule_r12_default_proceed | PROCEED | 1 | decision_engine |

**Arbitration:** `max(hits, key=(priority, confidence))`; R12 excluded from non-default set unless alone.

**Config thresholds:** `config/default.yaml` decision section → `DecisionConfig`.

---

# SECTION 9 — Pipeline Orchestrator Audit

**Class:** `PipelineOrchestrator` — `src/pipeline/orchestrator.py`

- `initialize()` / `_ensure_initialized()` — lazy init when `auto_initialize`  
- `run_frame()` — runs enabled modules, builds SceneState, evaluates decision  
- `visualize()` — `draw_scene_overlays` + `draw_decision_hud`  
- `cleanup()` — module cleanup  
- `create_default_orchestrator(device, config)` — factory  

**Config:** `PipelineConfig` mirrors `config/default.yaml` pipeline section; segmentation disabled.

**Tests:** `tests/test_pipeline_orchestrator.py`, `tests/test_verify_pipeline.py`.

---

# SECTION 10 — UI Layer Audit

**Primary:** `app.py` — Streamlit  
- Image + video tabs  
- `load_orchestrator()` cached with `@st.cache_resource`  
- `real_weights_available()` checks all four weight paths  
- Falls back to `build_stub_orchestrator()` from `scripts/verify_pipeline.py`  
- Outputs: `outputs/streamlit/`  

**Stub:** `src/app.py` — Gradio TODO only.

**Note:** `README.md` still lists SSD/YOLOv5/CNN — **stale**; actual code uses YOLOP/YOLOv8.

---

# SECTION 11 — Dataset Audit

| Dataset | On disk? | Location | Notes |
|---------|----------|----------|-------|
| traffic_signal_yolo | **Yes** | `data/traffic_signal_yolo/` | BDD100K derived, stats in dataset_stats.json |
| traffic_signal colab sample | **Yes** | `data/traffic_signal_yolo_colab_sample.zip` | Subset for Colab |
| traffic_sign_yolo | **No** | Expected `data/traffic_sign_yolo/` | Not prepared |
| GTSRB raw | **No** | External path in training config | Not in repo |
| COCO/Cityscapes | **No** | Config paths only | prepare_datasets.py stub |

---

# SECTION 12 — Model Selection Justification

| Task | Selected | Config key | Rationale in repo |
|------|----------|------------|-------------------|
| Lane | YOLOP | models.lane_detection | Joint lane + drivable segmentation for geometry and R03 |
| Vehicles | YOLOv8s | models.object_detection | COCO pretrained, filtered road users |
| Signs | YOLOv8n | models.traffic_sign | Fine-tuned head, 7 ADAS classes |
| Signals | YOLOv8n | models.traffic_signal | BDD100K state detection, 3 classes |
| Segmentation | U-Net | models.segmentation | **Stub only** — not active |

---

# SECTION 13 — Verification & Testing

**Command:** `python -m pytest tests/ -q` → **75 passed**

| Test file | Focus |
|-----------|-------|
| test_decision_rules.py | R01,R02,R04,R09,R11,R12, spatial helpers |
| test_decision_engine.py | Arbitration |
| test_scene_state.py | Module health flags |
| test_pipeline_orchestrator.py | End-to-end stub orchestrator |
| test_lane_detection_pipeline.py | Lane module |
| test_vehicle_detection_pipeline.py | Vehicle module |
| test_traffic_sign_pipeline.py | Sign module |
| test_traffic_signal_pipeline.py | Signal module |
| test_mask_resize_geometry.py | Mask resize regression |
| test_yolop_output_parser.py | Parser frame_shape |
| test_traffic_signal_training.py | BDD converter alignment |
| test_traffic_sign_training.py | GTSRB converter alignment |
| test_verify_pipeline.py | verify_pipeline.py gate |

**Gate scripts:** `scripts/verify_pipeline.py`, verify_*_detection.py, verify_mask_resize.py.

---

# SECTION 14 — Technical Challenges

1. **Lane mask resize (v2):** Geometry at model resolution broke offset — fixed in `lane_detection.py`, tested in `test_mask_resize_geometry.py`.  
2. **YOLOP vendor import portability:** Patched under `src/modules/yolop/vendor/` (see docs).  
3. **Class ID alignment:** Signal/sign training matched `class_map.py` — no inference changes.  
4. **Yellow class imbalance:** 2.7% of signal boxes — documented in training report.  
5. **Missing trained weights locally:** Sign/signal modules return `init_failed` without .pt files.  
6. **README drift:** Documents old SSD/YOLOv5 stack vs implemented YOLOv8.

---

# SECTION 15 — Production Readiness

| Criterion | Assessment |
|-----------|------------|
| Real-time on CPU | Demo-oriented; timing collected but not SLA |
| Model artifacts | Sign/signal weights missing locally |
| Security | No auth on Streamlit uploads |
| Monitoring | Logging only |
| Segmentation | Not integrated |
| CI | Tests pass; no CI config required in audit |

**Verdict:** Research/demo portfolio quality — **not production-deployed ADAS**.

---

# SECTION 16 — Resume & Placement Narrative

Built a modular Python ADAS integrating YOLOP lane/drivable segmentation and three YOLOv8 detectors with a 12-rule explainable decision engine. Delivered Streamlit demo, 75 automated tests, BDD100K traffic-signal training pipeline (mAP50 0.443), and GTSRB sign training tooling. Fixed lane geometry via frame-aligned mask resize (pipeline v2).

---

# SECTION 17 — Interview Preparation Guide (300 Q&A)

All questions reference files under this repository only.

---

## 17.A — Beginner (100 Questions)

#### Beginner Q1. What is the project name in config?

**Answer:** Autonomous Driving Assistance System — `config/default.yaml` project.name.

**Follow-up:** Where is version stored?

**Common mistake:** Inventing a different product name not in config.

#### Beginner Q2. What command runs the UI?

**Answer:** `streamlit run app.py` from project root — see `app.py` module docstring.

**Follow-up:** What is `src/app.py`?

**Common mistake:** Claiming Gradio is the primary UI.

#### Beginner Q3. How many pytest tests exist?

**Answer:** 75 tests in `tests/` — verified via `python -m pytest tests/ --collect-only -q`.

**Follow-up:** Which file tests rules?

**Common mistake:** Citing a number without running collect-only.

#### Beginner Q4. What class orchestrates the pipeline?

**Answer:** `PipelineOrchestrator` in `src/pipeline/orchestrator.py`.

**Follow-up:** What does `run_frame` return?

**Common mistake:** Confusing orchestrator with DecisionEngine.

#### Beginner Q5. What is the pipeline order?

**Answer:** Lane → Vehicle → Sign → Signal → SceneState → DecisionEngine — `REFERENCE_ORDER` in orchestrator.

**Follow-up:** Is segmentation enabled?

**Common mistake:** Reordering modules arbitrarily.

#### Beginner Q6. What frame format is required?

**Answer:** BGR `numpy.ndarray` shape `(H, W, 3)` uint8 — `_validate_frame` in orchestrator.

**Follow-up:** RGB vs BGR?

**Common mistake:** Passing PIL images without conversion.

#### Beginner Q7. What is SceneState?

**Answer:** Unified dataclass in `src/decision/scene_state.py` aggregating module outputs per frame.

**Follow-up:** What is `lane_ok`?

**Common mistake:** Treating SceneState as a dict.

#### Beginner Q8. How many decision rules?

**Answer:** 12 rules R01–R12 registered in `src/decision/rules.py` via `@_register`.

**Follow-up:** Which is default?

**Common mistake:** Saying 10 rules or ML policy.

#### Beginner Q9. What recommendations exist?

**Answer:** `ADASRecommendation` enum: PROCEED, STOP, SLOW_DOWN, KEEP_LANE, WARNING — `src/decision/types.py`.

**Follow-up:** Highest priority action?

**Common mistake:** Inventing new recommendation types.

#### Beginner Q10. What model does lane detection use?

**Answer:** YOLOP (MCnet) — `LaneDetectionModule` in `src/modules/lane_detection.py`.

**Follow-up:** Checkpoint filename?

**Common mistake:** Calling it YOLOv8 lanes.

#### Beginner Q11. Where are YOLOP weights?

**Answer:** `models/pretrained/yolop/End-to-end.pth` via `get_yolop_weights_path()`.

**Follow-up:** Env override?

**Common mistake:** Assuming weights auto-download for YOLOP.

#### Beginner Q12. What vehicle model is used?

**Answer:** YOLOv8s — `config/default.yaml` yolov8.model_variant: s.

**Follow-up:** Default confidence?

**Common mistake:** Saying SSD MobileNetV2 is implemented (README stale).

#### Beginner Q13. Which COCO IDs are kept?

**Answer:** 0,1,2,3,5,7 — person, bicycle, car, motorcycle, bus, truck in `yolov8/output_parser.py`.

**Follow-up:** Why filter?

**Common mistake:** Using all 80 COCO classes.

#### Beginner Q14. How many traffic sign classes?

**Answer:** 7 — `SIGN_CLASS_ID_TO_LABEL` in `src/modules/yolov8_sign/class_map.py`.

**Follow-up:** List stop class ID?

**Common mistake:** Confusing with 43 GTSRB classes at runtime.

#### Beginner Q15. How many traffic signal classes?

**Answer:** 3 — red_light, yellow_light, green_light in `yolov8_signal/class_map.py`.

**Follow-up:** Class 0 meaning?

**Common mistake:** Using legacy CNN 3-color names without _light suffix.

#### Beginner Q16. Where is traffic sign module?

**Answer:** `TrafficSignModule` in `src/modules/traffic_sign.py`.

**Follow-up:** Subpackage?

**Common mistake:** Single generic YOLOv8 for signs.

#### Beginner Q17. Where is traffic signal module?

**Answer:** `TrafficSignalModule` in `src/modules/traffic_signal.py`.

**Follow-up:** Parser file?

**Common mistake:** Merging sign and signal parsers.

#### Beginner Q18. Is segmentation implemented?

**Answer:** No — `SegmentationModule` in `src/modules/segmentation.py` is a TODO stub; `run_segmentation: false`.

**Follow-up:** Planned model?

**Common mistake:** Claiming U-Net runs in pipeline.

#### Beginner Q19. What is BaseModule?

**Answer:** Abstract contract: initialize, predict, visualize, cleanup — `src/modules/base.py`.

**Follow-up:** Frame type alias?

**Common mistake:** Modules without initialize pattern.

#### Beginner Q20. What env var sets data root?

**Answer:** `ADAS_DATA_ROOT` overrides `data_root` in `src/utils/model_paths.py`.

**Follow-up:** Default data_root?

**Common mistake:** Hardcoding paths only from yaml Google Drive default.

#### Beginner Q21. Are sign weights on disk?

**Answer:** No at audit — `models/trained/yolov8_sign/traffic_signs_yolov8n.pt` missing.

**Follow-up:** Inference without weights?

**Common mistake:** Claiming sign model is trained locally.

#### Beginner Q22. Are signal weights on disk?

**Answer:** No at audit — `models/trained/yolov8_signal/traffic_signals_yolov8n.pt` missing.

**Follow-up:** Documented metrics?

**Common mistake:** Claiming signal inference works with real weights locally.

#### Beginner Q23. What is R01?

**Answer:** `rule_r01_red_light_stop` — STOP when dominant red_light confidence ≥ 0.70.

**Follow-up:** Priority?

**Common mistake:** STOP on any red pixel.

#### Beginner Q24. What is R12?

**Answer:** `rule_r12_default_proceed` — fallback PROCEED priority 1.

**Follow-up:** When skipped?

**Common mistake:** Thinking no rule always fires without R12.

#### Beginner Q25. How does arbitration work?

**Answer:** `DecisionEngine.arbitrate` picks max (priority, confidence) — `decision_engine.py`.

**Follow-up:** Tie-break?

**Common mistake:** Averaging recommendations.

#### Beginner Q26. What is stub mode in Streamlit?

**Answer:** When weights missing, `build_stub_orchestrator()` from `scripts/verify_pipeline.py`.

**Follow-up:** Function name?

**Common mistake:** Stub mode runs without any modules.

#### Beginner Q27. What does verify_pipeline.py do?

**Answer:** Gate script for end-to-end pipeline with stub engines by default.

**Follow-up:** Real flag?

**Common mistake:** Requires all weights always.

#### Beginner Q28. Where are overlays drawn?

**Answer:** `src/visualization/overlays.py` and HUD in `src/visualization/hud.py`.

**Follow-up:** Called from?

**Common mistake:** Drawing inside decision engine.

#### Beginner Q29. What config has thresholds?

**Answer:** `config/default.yaml` sections thresholds and decision.

**Follow-up:** red_light_confidence?

**Common mistake:** Hardcoding thresholds in rules only.

#### Beginner Q30. What is LANE_DETECTION_PIPELINE_VERSION?

**Answer:** Constant `2` in `lane_detection.py` — mask resize before geometry.

**Follow-up:** Why version 2?

**Common mistake:** Ignoring mask/frame shape mismatch bug.

#### Beginner Q31. What does lane_departure mean?

**Answer:** Boolean when |vehicle_offset| > parser departure threshold.

**Follow-up:** Source module for R09?

**Common mistake:** Same as R10 KEEP_LANE.

#### Beginner Q32. What is drivable_mask used for?

**Answer:** R03 pedestrian-on-drivable via `overlaps_drivable_mask` in rules.py.

**Follow-up:** From which module?

**Common mistake:** Using lane lines only for R03.

#### Beginner Q33. What is active_speed_limit_kmh?

**Answer:** Parsed from sign summary in sign output_parser for R05.

**Follow-up:** Which signs?

**Common mistake:** Any sign triggers speed rule.

#### Beginner Q34. What dataset for signals?

**Answer:** BDD100K converted to `data/traffic_signal_yolo/` — 42048 images, 147550 boxes.

**Follow-up:** Source converter?

**Common mistake:** COCO for traffic lights.

#### Beginner Q35. GTSRB on disk?

**Answer:** Not in repo — `DEFAULT_GTSRB_ROOT` points external path in `training/traffic_sign/config.py`.

**Follow-up:** traffic_sign_yolo prepared?

**Common mistake:** Claiming GTSRB is bundled.

#### Beginner Q36. Sign training scripts?

**Answer:** `scripts/prepare_traffic_sign_dataset.py`, `train_traffic_sign.py`, `evaluate_traffic_sign.py`, `export_traffic_sign_dataset.py`.

**Follow-up:** Package?

**Common mistake:** Saying no train script exists.

#### Beginner Q37. Signal training scripts?

**Answer:** `prepare/train/evaluate/predict_traffic_signal.py` — see training report.

**Follow-up:** Metrics?

**Common mistake:** No evaluation script.

#### Beginner Q38. What is mAP50 for signal model?

**Answer:** 0.443 — `docs/traffic_signal_training_report.md` validation.

**Follow-up:** Precision/recall?

**Common mistake:** Quoting training loss as mAP.

#### Beginner Q39. What tests decision engine?

**Answer:** `tests/test_decision_engine.py`, `tests/test_decision_rules.py`.

**Follow-up:** Rule count tests?

**Common mistake:** Only integration tests.

#### Beginner Q40. What is PipelineResult?

**Answer:** Dataclass: scene_state, decision, total_time_ms — orchestrator.py.

**Follow-up:** Timing flag?

**Common mistake:** Only returning decision.

#### Beginner Q41. What is raw_status?

**Answer:** Module status string e.g. parsed, init_failed, stub — scene_state health.

**Follow-up:** When lane_ok false?

**Common mistake:** Ignoring init_failed.

#### Beginner Q42. What imgsz for YOLOv8 modules?

**Answer:** 640 default in config for vehicle/sign/signal.

**Follow-up:** YOLOP input?

**Common mistake:** Same 640 assumption for all without checking YOLOP config.

#### Beginner Q43. What is create_default_orchestrator?

**Answer:** Factory in orchestrator.py wiring default modules with device.

**Follow-up:** Config param?

**Common mistake:** Manual module wiring only.

#### Beginner Q44. Where is classes.yaml?

**Answer:** `config/classes.yaml` — traffic_sign_classes and traffic_signal_classes lists.

**Follow-up:** Legacy key?

**Common mistake:** Only hardcoded class_map.

#### Beginner Q45. What does download_weights.py do?

**Answer:** Script in `scripts/download_weights.py` for pretrained fetch.

**Follow-up:** Sign weights?

**Common mistake:** Assuming it downloads trained sign/signal .pt.

#### Beginner Q46. What is prepare_datasets.py?

**Answer:** Stub with TODO only — not implemented.

**Follow-up:** COCO prep?

**Common mistake:** Claiming full dataset prep works.

#### Beginner Q47. What package is yolop vendor?

**Answer:** Vendored MCnet under `src/modules/yolop/vendor/`.

**Follow-up:** Model file?

**Common mistake:** pip install yolop only.

#### Beginner Q48. What does auto_initialize do?

**Answer:** PipelineConfig flag — `_ensure_initialized` before predict.

**Follow-up:** Default?

**Common mistake:** Always calling initialize manually.

#### Beginner Q49. What file has RECOMMENDATION_COLORS?

**Answer:** `app.py` for Streamlit decision panel styling.

**Follow-up:** STOP color?

**Common mistake:** Colors in decision types.

#### Beginner Q50. What is `collect_timing`?

**Answer:** PipelineConfig flag enabling perf_counter total_time_ms.

**Follow-up:** Logged where?

**Common mistake:** Disabling timing always.

#### Beginner Q51. What is `run_lane`?

**Answer:** PipelineConfig toggle for lane module.

**Follow-up:** All toggles?

**Common mistake:** Cannot disable modules.

#### Beginner Q52. What is nearest_object?

**Answer:** Vehicle summary field — largest bbox area proxy.

**Follow-up:** Used in rules?

**Common mistake:** R08 large vehicle.

#### Beginner Q53. What is controlling_signal?

**Answer:** Signal summary — upper-frame signal for decisions.

**Follow-up:** Fraction?

**Common mistake:** CONTROLLING_SIGNAL_UPPER_FRACTION 0.60 in class_map.

#### Beginner Q54. What is has_stop_state?

**Answer:** Signal summary flag blocking R11 green proceed.

**Follow-up:** R01 relation?

**Common mistake:** Green always proceeds.

#### Beginner Q55. What is stop_sign_lower_frame_fraction?

**Answer:** 0.40 — R02 requires sign in lower 60% of frame.

**Follow-up:** Why spatial?

**Common mistake:** Any stop sign anywhere.

#### Beginner Q56. What is vulnerable_user_confidence?

**Answer:** 0.60 for R06 person/bicycle in lower third.

**Follow-up:** Labels?

**Common mistake:** All vehicles.

#### Beginner Q57. What is large_vehicle_area_ratio?

**Answer:** 0.08 minimum bbox/frame area for R08 truck/bus.

**Follow-up:** Labels?

**Common mistake:** Car triggers R08.

#### Beginner Q58. What is lane_offset_warn_px?

**Answer:** 35.0 px threshold for R10 KEEP_LANE.

**Follow-up:** R09 difference?

**Common mistake:** Same rule.

#### Beginner Q59. What is drivable_overlap_threshold?

**Answer:** 0.15 for R03 mask overlap.

**Follow-up:** Function?

**Common mistake:** overlaps_drivable_mask.

#### Beginner Q60. What is stop sign class ID?

**Answer:** 0 maps to label stop in SIGN_CLASS_ID_TO_LABEL.

**Follow-up:** R02 label check?

**Common mistake:** Using string stop vs class id.

#### Beginner Q61. What is red_light class ID?

**Answer:** 0 in SIGNAL_CLASS_ID_TO_LABEL.

**Follow-up:** Training alignment?

**Common mistake:** Swapping red/green IDs.

#### Beginner Q62. What does sign raw_status parsed mean?

**Answer:** Module considered ok in SceneState when raw_status in {parsed, stub}.

**Follow-up:** init_failed?

**Common mistake:** Treating all statuses as ok.

#### Beginner Q63. What is RuleHit?

**Answer:** Dataclass: rule_id, recommendation, priority, message, source_module, confidence — types.py.

**Follow-up:** Serialized?

**Common mistake:** to_dict() for JSON.

#### Beginner Q64. What is DecisionResult?

**Answer:** Final recommendation + rule_hits list + explanation — decision_engine output.

**Follow-up:** primary_message?

**Common mistake:** Only enum without message.

#### Beginner Q65. Where is pipeline __init__ export?

**Answer:** src/pipeline/__init__.py exports PipelineOrchestrator, PipelineConfig, create_default_orchestrator.

**Follow-up:** Import in app?

**Common mistake:** from src.pipeline import ...

#### Beginner Q66. What is Frame type alias?

**Answer:** numpy.ndarray BGR — src/modules/base.py.

**Follow-up:** RGB upload?

**Common mistake:** Streamlit converts via bgr_to_rgb for display only.

#### Beginner Q67. What is max_detections for vehicles?

**Answer:** 100 — config default.yaml yolov8.max_detections.

**Follow-up:** Signs?

**Common mistake:** 50 for signs, 20 for signals.

#### Beginner Q68. What weight file key for sign?

**Answer:** yolov8_sign: traffic_signs_yolov8n.pt in weight_files.

**Follow-up:** Location?

**Common mistake:** trained not pretrained.

#### Beginner Q69. What weight file key for signal?

**Answer:** yolov8_signal: traffic_signals_yolov8n.pt.

**Follow-up:** Legacy?

**Common mistake:** traffic_light_cnn.pt deprecated in model_paths.

#### Beginner Q70. What is extract_speed_limit_kmh?

**Answer:** Parses speed_limit_30 → 30 in sign class_map.py.

**Follow-up:** R05?

**Common mistake:** Used in sign summary active_speed_limit_kmh.

#### Beginner Q71. What is enrich_state_flags?

**Answer:** Returns stop/caution/proceed booleans per signal label — signal class_map.py.

**Follow-up:** Summary?

**Common mistake:** Used in signal output_parser build_summary.

#### Beginner Q72. What is CONTROLLING_SIGNAL_UPPER_FRACTION?

**Answer:** 0.60 — signals in upper 60% of frame preferred as controlling.

**Follow-up:** R01 confidence?

**Common mistake:** From controlling_signal detection.

#### Beginner Q73. What does draw_scene_overlays do?

**Answer:** Composites lane, vehicle, sign, signal overlays — visualization/overlays.py.

**Follow-up:** Flags?

**Common mistake:** show_lane etc. in orchestrator.visualize.

#### Beginner Q74. What is OUTPUT_DIR in app.py?

**Answer:** outputs/streamlit/ for saved images and annotated videos.

**Follow-up:** Auto-save?

**Common mistake:** On image run and video process.

#### Beginner Q75. What video codecs used?

**Answer:** mp4v fourcc in process_video — app.py.

**Follow-up:** Formats accepted?

**Common mistake:** mp4, avi, mov, mkv, webm.

#### Beginner Q76. What is test conftest role?

**Answer:** Shared pytest fixtures — tests/conftest.py.

**Follow-up:** Stub weights?

**Common mistake:** Fixtures for pipeline tests.

#### Beginner Q77. What is road_sample.jpg?

**Answer:** Test fixture at tests/fixtures/road_sample.jpg used by verify_pipeline.

**Follow-up:** Synthetic?

**Common mistake:** verify_pipeline also builds synthetic frame.

#### Beginner Q78. What is stub_yolop_weights.pth?

**Answer:** Minimal torch checkpoint in tests/fixtures for gate tests.

**Follow-up:** Real YOLOP?

**Common mistake:** Different from End-to-end.pth.

#### Beginner Q79. What is verify_mask_resize.py?

**Answer:** Gate script for mask resize regression — scripts/.

**Follow-up:** Related tests?

**Common mistake:** test_mask_resize_geometry.py.

#### Beginner Q80. What is verify_environment.py?

**Answer:** Environment check script in scripts/.

**Follow-up:** Dependencies?

**Common mistake:** requirements.txt.

#### Beginner Q81. What is traffic_sign_colab_training.md?

**Answer:** Colab training doc in docs/ for sign workflow.

**Follow-up:** Local?

**Common mistake:** Complements scripts/train_traffic_sign.py.

#### Beginner Q82. What is decision_engine_design.md?

**Answer:** Design doc in docs/ for rule engine.

**Follow-up:** Implementation?

**Common mistake:** decision_engine_implementation_report.md.

#### Beginner Q83. What is classes.yaml legacy key?

**Answer:** traffic_light_classes deprecated — maps via LEGACY_LABEL_ALIASES in signal class_map.

**Follow-up:** Current key?

**Common mistake:** traffic_signal_classes.

#### Beginner Q84. What is project version in config?

**Answer:** 1.0 — config/default.yaml project.version.

**Follow-up:** Name?

**Common mistake:** Autonomous Driving Assistance System.

#### Beginner Q85. What is object_confidence threshold?

**Answer:** 0.5 — config/default.yaml thresholds.object_confidence.

**Follow-up:** Sign?

**Common mistake:** sign_confidence 0.5.

#### Beginner Q86. What is sign_iou threshold?

**Answer:** 0.45 — thresholds.sign_iou.

**Follow-up:** Vehicle?

**Common mistake:** object_iou 0.45.

#### Beginner Q87. What is run_segmentation default?

**Answer:** false — pipeline.run_segmentation in config.

**Follow-up:** Module exists?

**Common mistake:** SegmentationModule stub only.

#### Beginner Q88. What is display_hud config?

**Answer:** output.display_hud: true in config — Streamlit uses show_hud checkbox.

**Follow-up:** Override?

**Common mistake:** Sidebar checkbox in app.

#### Beginner Q89. What is save_annotated_video?

**Answer:** output.save_annotated_video: true in config.

**Follow-up:** Streamlit?

**Common mistake:** Video writer always saves annotated mp4.

#### Beginner Q90. What is PipelineConfig dataclass?

**Answer:** Orchestrator runtime flags — orchestrator.py mirrors yaml.

**Follow-up:** From yaml?

**Common mistake:** _pipeline_config_from_yaml().

#### Beginner Q91. What is ModuleStatus?

**Answer:** Per-module health in SceneState — scene_state.py.

**Follow-up:** Fields?

**Common mistake:** module_name, raw_status, ok, inference_time_ms.

#### Beginner Q92. What is perception_dict?

**Answer:** SceneState method returning module to_prediction_dict payloads.

**Follow-up:** to_dict?

**Common mistake:** Masks as shape metadata in to_dict.

#### Beginner Q93. What is empty_prediction?

**Answer:** Static helpers on modules returning empty result schemas.

**Follow-up:** Status?

**Common mistake:** raw_status empty or init_failed.

#### Beginner Q94. What is WeightsNotFoundError?

**Answer:** Raised by sign/signal model_loader when .pt missing.

**Follow-up:** Vehicle?

**Common mistake:** WeightsLoadError without NotFound in vehicle loader.

#### Beginner Q95. What is resolve_variant_name?

**Answer:** Normalizes yolov8 variant string in model_loaders.

**Follow-up:** Default sign?

**Common mistake:** n nano.

#### Beginner Q96. What is ADAS_VEHICLE_LABELS?

**Answer:** Frozenset of six labels in yolov8/output_schema.py.

**Follow-up:** Parser?

**Common mistake:** Subset of COCO.

#### Beginner Q97. What is TRAFFIC_SIGN_OUTPUT_KEYS?

**Answer:** Exported schema keys in traffic_sign module.

**Follow-up:** Signal?

**Common mistake:** TRAFFIC_SIGNAL_OUTPUT_KEYS.

#### Beginner Q98. What is bbox_lower_fraction?

**Answer:** Helper in rules.py — center_y / frame_height.

**Follow-up:** R02?

**Common mistake:** Filters stop signs too high in frame.

#### Beginner Q99. What is bbox_area_ratio?

**Answer:** Detection area / frame area — rules.py for R08.

**Follow-up:** Threshold?

**Common mistake:** large_vehicle_area_ratio 0.08.

#### Beginner Q100. What is _STOP_CLASS_RECOMMENDATIONS?

**Answer:** frozenset {STOP} in decision_engine for logging.

**Follow-up:** Arbitration?

**Common mistake:** STOP same as other recommendations in max().

---

## 17.B — Intermediate (100 Questions)

#### Intermediate Q1. Explain YOLOP pipeline steps in LaneDetectionModule.

**Answer:** _run_pipeline: preprocess → inference → parse → postprocess masks → resize to frame → geometry — `lane_detection.py`.

**Follow-up:** Failure paths?

**Common mistake:** Skipping resize step.

#### Intermediate Q2. Why separate yolov8_sign model_loader?

**Answer:** No COCO fallback — requires fine-tuned weights — comment in model_loader.py.

**Follow-up:** Vehicle loader difference?

**Common mistake:** Using yolov8s.pt for signs.

#### Intermediate Q3. How does YOLOv8OutputParser filter classes?

**Answer:** Skips coco_id not in ALLOWED_COCO_CLASS_IDS — output_parser.py.

**Follow-up:** Label mapping?

**Common mistake:** Raw COCO names without filter.

#### Intermediate Q4. How is SceneState.from_perception built?

**Answer:** Sets lane_ok via _lane_module_ok; others via raw_status in _MODULE_OK_STATUSES.

**Follow-up:** Stub ok?

**Common mistake:** lane_ok true on empty lane.

#### Intermediate Q5. Explain evaluate_all vs arbitrate.

**Answer:** evaluate_all runs RULE_REGISTRY; engine filters R12 for arbitration unless only default.

**Follow-up:** Non-default empty?

**Common mistake:** All hits in final message.

#### Intermediate Q6. How does sign build_summary work?

**Answer:** YOLOv8SignOutputParser.build_summary — nearest_sign, active_speed_limit_kmh.

**Follow-up:** R05 trigger?

**Common mistake:** Speed from turn_left sign.

#### Intermediate Q7. How does signal build_summary work?

**Answer:** Dominant state via STATE_PRIORITY; controlling_signal upper 60%.

**Follow-up:** Yellow dominant?

**Common mistake:** R04 caution.

#### Intermediate Q8. What does mask resize fix address?

**Answer:** Geometry on 640×640 caused wrong offset — `_resize_masks_to_frame_shape`.

**Follow-up:** Tests?

**Common mistake:** test_mask_resize_geometry.py.

#### Intermediate Q9. BDD100K to YOLO class alignment?

**Answer:** red/yellow/green → class 0/1/2 matching SIGNAL_CLASS_ID_TO_LABEL.

**Follow-up:** Converter file?

**Common mistake:** bdd100k_converter.py.

#### Intermediate Q10. GTSRB to ADAS sign mapping?

**Answer:** GTSRB_CLASS_ID_TO_ADAS_LABEL 7-class subset — class_map.py.

**Follow-up:** Unmapped classes?

**Common mistake:** gtsrb_id_to_adas_label returns None.

#### Intermediate Q11. TrafficSignTrainingConfig defaults?

**Answer:** 100 epochs, batch 16, YOLOv8n — training/traffic_sign/config.py.

**Follow-up:** Output path?

**Common mistake:** DEFAULT_MODEL_OUTPUT path.

#### Intermediate Q12. What does gtsrb_converter.py do?

**Answer:** Converts GTSRB crops/CSV to YOLO layout — training/traffic_sign/.

**Follow-up:** Tests?

**Common mistake:** test_traffic_sign_training.py.

#### Intermediate Q13. Signal dataset_stats location?

**Answer:** data/traffic_signal_yolo/dataset_stats.json.

**Follow-up:** Image count?

**Common mistake:** 42048 with ≥1 TL box.

#### Intermediate Q14. Yellow class imbalance?

**Answer:** 3928 of 147550 boxes (2.7%) — training report.

**Follow-up:** Impact?

**Common mistake:** Lower yellow recall expected.

#### Intermediate Q15. What does test_pipeline_orchestrator verify?

**Answer:** Stub modules, SceneState fields, decision output.

**Follow-up:** Injection?

**Common mistake:** Real weights required.

#### Intermediate Q16. What does conftest provide?

**Answer:** Shared fixtures for tests — tests/conftest.py.

**Follow-up:** Stub engines?

**Common mistake:** Used across pipeline tests.

#### Intermediate Q17. How does Streamlit cache orchestrator?

**Answer:** @st.cache_resource load_orchestrator in app.py.

**Follow-up:** Re-init on device change?

**Common mistake:** New orchestrator every frame.

#### Intermediate Q18. Video processing flow?

**Answer:** process_video: capture → run_frame per frame → VideoWriter — app.py.

**Follow-up:** Output dir?

**Common mistake:** outputs/streamlit/.

#### Intermediate Q19. Decision HUD content?

**Answer:** draw_decision_hud — recommendation, message — hud.py.

**Follow-up:** Overlay order?

**Common mistake:** Overlays then HUD in visualize.

#### Intermediate Q20. Weight path resolution algorithm?

**Answer:** _get_weight_path reads weight_files + weight_locations — model_paths.py.

**Follow-up:** Creates dirs?

**Common mistake:** mkdir parents, not weight file.

#### Intermediate Q21. YOLOP checkpoint errors?

**Answer:** CheckpointNotFoundError, ValidationError, LoadError — model_loader.

**Follow-up:** Auto-init fail?

**Common mistake:** LaneDetectionResult.empty init_failed.

#### Intermediate Q22. Vehicle bbox bounds check?

**Answer:** _assert_detections_within_frame raises RuntimeError if OOB.

**Follow-up:** Why?

**Common mistake:** Parser coordinate bugs.

#### Intermediate Q23. Sign regulatory vs warning labels?

**Answer:** REGULATORY_LABELS, WARNING_LABELS in sign class_map.

**Follow-up:** pedestrian_crossing?

**Common mistake:** Warning — R07.

#### Intermediate Q24. Signal STATE_PRIORITY values?

**Answer:** red_light=3, yellow=2, green=1 — conflict resolution.

**Follow-up:** R11 guard?

**Common mistake:** has_stop_state blocks green.

#### Intermediate Q25. R02 nearest_sign fallback?

**Answer:** Uses summary.nearest_sign if loop finds none — rules.py.

**Follow-up:** Confidence?

**Common mistake:** stop_sign_confidence 0.70.

#### Intermediate Q26. R03 requires both modules?

**Answer:** lane_ok and vehicles_ok and drivable mask — rules.py.

**Follow-up:** Person only?

**Common mistake:** label person.

#### Intermediate Q27. R06 spatial filter?

**Answer:** center_y >= lower_third (2/3 frame height).

**Follow-up:** Bicycle?

**Common mistake:** Included label.

#### Intermediate Q28. R08 nearest_object labels?

**Answer:** truck or bus only with area ratio.

**Follow-up:** Car?

**Common mistake:** Not R08.

#### Intermediate Q29. R10 suppressed when?

**Answer:** lane_departure true — rules.py.

**Follow-up:** Offset None?

**Common mistake:** No hit.

#### Intermediate Q30. Pipeline visualize flags?

**Answer:** show_lane, show_vehicles, show_signs, show_signals, show_hud.

**Follow-up:** Segmentation?

**Common mistake:** Not in visualize.

#### Intermediate Q31. How does LaneGeometryExtractor compute offset?

**Answer:** compare lane_center_x to image center — yolop/lane_geometry.py.

**Follow-up:** Departure?

**Common mistake:** Compared to departure_threshold_px in parser config.

#### Intermediate Q32. What does postprocess_lane_mask do?

**Answer:** Morphological/CC refinement — yolop/postprocess.py before resize.

**Follow-up:** Optional?

**Common mistake:** apply_mask_postprocess flag on LaneDetectionModule.

#### Intermediate Q33. How does YOLOPModelLoader validate checkpoint?

**Answer:** Checkpoint format and tensor keys — yolop/model_loader.py.

**Follow-up:** Errors?

**Common mistake:** CheckpointNotFoundError, ValidationError, LoadError.

#### Intermediate Q34. How does YOLOv8InferenceEngine run?

**Answer:** Ultralytics predict wrapper — yolov8/inference.py.

**Follow-up:** Config?

**Common mistake:** YOLOv8InferenceConfig imgsz, conf, iou, device.

#### Intermediate Q35. How are sign boxes filtered?

**Answer:** class_id in ALLOWED_SIGN_CLASS_IDS and conf threshold — sign output_parser.

**Follow-up:** Speed limit?

**Common mistake:** extract_speed_limit_kmh on label.

#### Intermediate Q36. How is nearest_sign chosen?

**Answer:** Largest area or closest lower in frame in build_summary — sign output_parser.

**Follow-up:** R02?

**Common mistake:** Must be stop label in lower frame.

#### Intermediate Q37. How is dominant_state chosen for signals?

**Answer:** STATE_PRIORITY max among detections — signal output_parser build_summary.

**Follow-up:** R04?

**Common mistake:** yellow_light dominant triggers caution.

#### Intermediate Q38. What is has_proceed_state?

**Answer:** Signal summary flag when green present without blocking stop.

**Follow-up:** R11?

**Common mistake:** Requires not has_stop_state.

#### Intermediate Q39. How does bdd100k_converter map colors?

**Answer:** bdd100k_label_to_adas_label — training/traffic_signal/bdd100k_converter.py.

**Follow-up:** Skip none?

**Common mistake:** trafficLightColor none skipped.

#### Intermediate Q40. What does test_green_maps_to_class_two verify?

**Answer:** Green → YOLO class 2 — test_traffic_signal_training.py.

**Follow-up:** Alignment?

**Common mistake:** Matches SIGNAL_CLASS_ID_TO_LABEL.

#### Intermediate Q41. What does test_class_mapping_matches_adas_runtime verify?

**Answer:** Both sign and signal training class order vs class_map.

**Follow-up:** Inference change?

**Common mistake:** Should be zero after training.

#### Intermediate Q42. How does verify_pipeline --real work?

**Answer:** Uses create_default_orchestrator when weights available — verify_pipeline.py.

**Follow-up:** Default?

**Common mistake:** Stub engines without --real.

#### Intermediate Q43. What stub vehicle engine returns?

**Answer:** Fixed boxes with COCO labels for gate tests — verify_pipeline _build_stub_yolov8_engine.

**Follow-up:** Orchestrator test?

**Common mistake:** test_verify_pipeline.py.

#### Intermediate Q44. How does DecisionEngine load config?

**Answer:** DecisionConfig(**get_decision_config()) in __init__.

**Follow-up:** Override?

**Common mistake:** Pass DecisionConfig to constructor.

#### Intermediate Q45. Why filter R12 before arbitrate?

**Answer:** non_default list excludes R12 unless only hits — evaluate() in decision_engine.

**Follow-up:** Explanation?

**Common mistake:** All hits still in sorted rule_hits message.

#### Intermediate Q46. What lane raw_status values are ok?

**Answer:** parsed, stub_segmentation, stub — _LANE_OK_STATUSES scene_state.

**Follow-up:** init_failed?

**Common mistake:** lane_ok false.

#### Intermediate Q47. What vehicle raw_status ok?

**Answer:** parsed, stub — _MODULE_OK_STATUSES.

**Follow-up:** pipeline_error?

**Common mistake:** Not ok for rules.

#### Intermediate Q48. How does overlaps_drivable_mask clip bbox?

**Answer:** Clips to mask bounds before overlap ratio — rules.py.

**Follow-up:** Threshold?

**Common mistake:** drivable_overlap_threshold 0.15 default.

#### Intermediate Q49. What is SignBoundingBoxData?

**Answer:** Sign bbox dataclass with center_y — yolov8_sign/output_schema.py.

**Follow-up:** R02 spatial?

**Common mistake:** Uses center_y vs frame height.

#### Intermediate Q50. What is DetectedSignal?

**Answer:** Signal detection with signal_label field — yolov8_signal/output_schema.py.

**Follow-up:** Controlling?

**Common mistake:** Referenced in TrafficSignalSummary.

#### Intermediate Q51. How does app read_uploaded_image work?

**Answer:** np.frombuffer + cv2.imdecode IMREAD_COLOR — app.py.

**Follow-up:** Error?

**Common mistake:** ValueError if decode fails.

#### Intermediate Q52. How does session_state store results?

**Answer:** image_result, image_annotated keys — render_image_tab app.py.

**Follow-up:** Video?

**Common mistake:** video_summary VideoSummary dataclass.

#### Intermediate Q53. What is VideoSummary dataclass?

**Answer:** Aggregates frame_count, fps, pipeline times, recommendation_counts — app.py.

**Follow-up:** Dominant?

**Common mistake:** most_common recommendation.

#### Intermediate Q54. What is RECOMMENDATION_COLORS STOP?

**Answer:** #dc3545 red — app.py dict.

**Follow-up:** PROCEED?

**Common mistake:** #28a745 green.

#### Intermediate Q55. How does get_yolov8_config merge thresholds?

**Answer:** Pulls object_confidence/iou from thresholds section — model_paths.py.

**Follow-up:** Sign config?

**Common mistake:** get_yolov8_sign_config similar.

#### Intermediate Q56. What is weight_locations yolop?

**Answer:** pretrained — config/default.yaml.

**Follow-up:** yolov8_sign?

**Common mistake:** trained.

#### Intermediate Q57. What is unet weight path key?

**Answer:** unet: unet_cityscapes.pt pretrained — config.

**Follow-up:** Used?

**Common mistake:** Segmentation stub only.

#### Intermediate Q58. What is yolov5 legacy path?

**Answer:** yolov5_traffic_signs.pt trained — get_yolov5_weights_path legacy.

**Follow-up:** Active module?

**Common mistake:** TrafficSignModule uses yolov8_sign.

#### Intermediate Q59. What does LanePreprocessor do?

**Answer:** Classical edge/ROI prep — preprocessing/lane_preprocess.py.

**Follow-up:** Before YOLOP?

**Common mistake:** First step in _run_pipeline.

#### Intermediate Q60. What does YOLOPOutputParser need frame_shape for?

**Answer:** Lane center/offset in frame space — output_parser.py.

**Follow-up:** Bug fixed?

**Common mistake:** test_yolop_output_parser frame_shape tests.

#### Intermediate Q61. What is departure_threshold_px?

**Answer:** Parser config threshold for lane_departure boolean.

**Follow-up:** R09?

**Common mistake:** lane_departure flag not offset magnitude directly.

#### Intermediate Q62. What files in yolop vendor?

**Answer:** YOLOP.py, common.py under vendor/models/.

**Follow-up:** README?

**Common mistake:** yolop/vendor if documented in docs.

#### Intermediate Q63. What is mask_resize.py?

**Answer:** Helper module under yolop/ — may complement inline resize in lane_detection.

**Follow-up:** Active path?

**Common mistake:** _resize_masks_to_frame_shape in lane_detection.py.

#### Intermediate Q64. What is image_ops preprocessing?

**Answer:** src/preprocessing/image_ops.py shared ops.

**Follow-up:** Lane?

**Common mistake:** lane_preprocess separate.

#### Intermediate Q65. What is test_decision_engine arbitration?

**Answer:** Tests max priority/confidence selection — test_decision_engine.py.

**Follow-up:** Multiple hits?

**Common mistake:** Sorted in explanation.

#### Intermediate Q66. What is test_scene_state lane_ok?

**Answer:** Requires parsed status and lane_center or lane_mask — test_scene_state.py.

**Follow-up:** None lane?

**Common mistake:** lane_ok false.

#### Intermediate Q67. What is test_traffic_sign_pipeline?

**Answer:** Module wiring and schema tests — test_traffic_sign_pipeline.py.

**Follow-up:** Real weights?

**Common mistake:** Stub/mock pattern.

#### Intermediate Q68. What is test_traffic_signal_pipeline?

**Answer:** Signal module tests — test_traffic_signal_pipeline.py.

**Follow-up:** Dominant state?

**Common mistake:** Summary builder coverage.

#### Intermediate Q69. What is evaluate_traffic_sign.py?

**Answer:** Runs val metrics on sign model — scripts/.

**Follow-up:** Output?

**Common mistake:** runs/ directory pattern like signal.

#### Intermediate Q70. What is train_traffic_sign.py?

**Answer:** Ultralytics training entry — scripts/.

**Follow-up:** Config?

**Common mistake:** TrafficSignTrainingConfig defaults.

#### Intermediate Q71. What is prepare_traffic_signal_dataset.py?

**Answer:** Rebuilds data/traffic_signal_yolo from BDD JSON.

**Follow-up:** Stats?

**Common mistake:** Writes dataset_stats.json.

#### Intermediate Q72. What is export_traffic_signal_dataset.py?

**Answer:** Export bundle for Colab — scripts/.

**Follow-up:** Sample zip?

**Common mistake:** Related colab sample in data/.

#### Intermediate Q73. What is verify_traffic_sign_detection.py?

**Answer:** Gate script for sign module — scripts/.

**Follow-up:** Vehicle?

**Common mistake:** verify_vehicle_detection.py parallel.

#### Intermediate Q74. What is verify_traffic_signal_detection.py?

**Answer:** Gate script for signal module — scripts/.

**Follow-up:** Pipeline?

**Common mistake:** verify_pipeline full stack.

#### Intermediate Q75. What is requirements.txt?

**Answer:** Python dependencies at project root.

**Follow-up:** Ultralytics?

**Common mistake:** Listed for YOLOv8.

#### Intermediate Q76. What is .gitignore for models?

**Answer:** Large weights may be gitignored — check .gitignore.

**Follow-up:** On disk?

**Common mistake:** pretrained present locally at audit.

#### Intermediate Q77. What is pipeline __all__?

**Answer:** Check src/pipeline/__init__.py exports.

**Follow-up:** Decision?

**Common mistake:** src/decision/__init__.py exports engine types.

#### Intermediate Q78. What is hud draw_decision_hud?

**Answer:** Renders recommendation text on frame — visualization/hud.py.

**Follow-up:** scene_state?

**Common mistake:** Optional scene_state param.

#### Intermediate Q79. What is VEHICLE_COLORS?

**Answer:** Label color map in overlays.py for draw_vehicle_detections.

**Follow-up:** Sign colors?

**Common mistake:** Separate overlay functions per module.

#### Intermediate Q80. What is LANE_OUTPUT_KEYS?

**Answer:** Tuple exported from lane_detection.py.

**Follow-up:** Schema?

**Common mistake:** LaneDetectionResult fields.

#### Intermediate Q81. What is inference_time_ms?

**Answer:** Optional timing on vehicle/sign/signal results.

**Follow-up:** Orchestrator?

**Common mistake:** total_time_ms on PipelineResult not per-module in orchestrator log.

#### Intermediate Q82. What is logger name adas.modules?

**Answer:** BaseModule logger pattern fadas.modules.{module_name}.

**Follow-up:** Orchestrator?

**Common mistake:** adas pipeline logger in orchestrator.

#### Intermediate Q83. What is strict mask assert?

**Answer:** _assert_result_masks_match_frame raises RuntimeError on shape mismatch.

**Follow-up:** Version?

**Common mistake:** Message cites LANE_DETECTION_PIPELINE_VERSION 2.

#### Intermediate Q84. What is test_unmapped_gtsrb_class?

**Answer:** gtsrb_id_to_adas_label None for unmapped — test_traffic_sign_training.

**Follow-up:** Converter?

**Common mistake:** Skips unmapped classes.

#### Intermediate Q85. What is test_ignores_none_color?

**Answer:** BDD none color skipped — test_traffic_signal_training.

**Follow-up:** Stats?

**Common mistake:** none_color skip count in dataset_stats.

#### Intermediate Q86. What is test_yolo_line_is_normalized?

**Answer:** YOLO label format cx cy w h in [0,1] — signal training test.

**Follow-up:** Converter?

**Common mistake:** bdd100k_converter output format.

#### Intermediate Q87. What is test_stub_pipeline_emits_stop?

**Answer:** verify_pipeline stub can produce STOP decision — test_verify_pipeline.

**Follow-up:** Rules?

**Common mistake:** Stub signal/sign boxes configured in script.

#### Intermediate Q88. What is create_default_orchestrator device?

**Answer:** Passes device to each module constructor — orchestrator.py.

**Follow-up:** CUDA?

**Common mistake:** Selectable in Streamlit sidebar.

#### Intermediate Q89. What is collect_timing perf_counter?

**Answer:** time.perf_counter delta * 1000 — orchestrator run_frame.

**Follow-up:** Logged?

**Common mistake:** logger.info pipeline frame line.

#### Intermediate Q90. What is frame_index param?

**Answer:** Passed to run_frame for logging and SceneState.

**Follow-up:** Video?

**Common mistake:** Incremented per frame in process_video.

#### Intermediate Q91. What is timestamp_ms param?

**Answer:** Optional ms timestamp for SceneState.

**Follow-up:** Video calc?

**Common mistake:** frame_index * (1000/fps).

#### Intermediate Q92. What is cleanup on orchestrator?

**Answer:** Calls cleanup on each enabled module.

**Follow-up:** Streamlit cache?

**Common mistake:** Orchestrator persists for session.

#### Intermediate Q93. What is reference aligned models block?

**Answer:** config/default.yaml models: YOLOP, YOLOv8, UNet names.

**Follow-up:** SSD?

**Common mistake:** Legacy README only.

#### Intermediate Q94. What is ssd_mobilenetv2 weight key?

**Answer:** `ssd_mobilenet_v2_coco.pb` under pretrained in `config/default.yaml` — not used by implemented vehicle module.

**Follow-up:** Active detector?

**Common mistake:** Citing README SSD as running code path.

#### Intermediate Q95. What is traffic_light_cnn legacy path?

**Answer:** `get_traffic_light_cnn_path()` in `model_paths.py` — deprecated in favor of YOLOv8 signal weights.

**Follow-up:** Still in config?

**Common mistake:** Claiming CNN runs in pipeline.

#### Intermediate Q96. What does test_train_crop_uses_full_frame_bbox verify?

**Answer:** GTSRB converter emits full-frame boxes for crop images — `tests/test_traffic_sign_training.py`.

**Follow-up:** ROI signs?

**Common mistake:** Assuming tight crop boxes only.

#### Intermediate Q97. What does test_is_valid_traffic_light_box_reject_oob verify?

**Answer:** BDD converter rejects out-of-bounds boxes — `tests/test_traffic_signal_training.py`.

**Follow-up:** Stats skipped?

**Common mistake:** Counting invalid boxes as kept.

#### Intermediate Q98. What is draw_lane_results?

**Answer:** Overlay helper in `visualization/overlays.py` called from `LaneDetectionModule.visualize`.

**Follow-up:** Masks drawn?

**Common mistake:** Drawing raw 640 masks without resize check.

#### Intermediate Q99. What is draw_traffic_signs?

**Answer:** Sign bbox overlay function in `visualization/overlays.py`.

**Follow-up:** Labels?

**Common mistake:** Using vehicle overlay for signs.

#### Intermediate Q100. What is draw_traffic_signals?

**Answer:** Signal state overlay in `visualization/overlays.py` with state-colored boxes.

**Follow-up:** dominant_state HUD?

**Common mistake:** Confusing with sign overlay colors.

---

## 17.C — Advanced (100 Questions)

#### Advanced Q1. Why rule engine vs learned policy?

**Answer:** Explainability via RuleHit rule_id — no end-to-end policy in repo.

**Follow-up:** Trade-off?

**Common mistake:** Claiming neural planner exists.

#### Advanced Q2. Temporal smoothing for signals?

**Answer:** Not implemented — would need state above predict().

**Follow-up:** Design approach?

**Common mistake:** Saying already in TrafficSignalModule.

#### Advanced Q3. Edge deployment memory?

**Answer:** Four models sequential — not optimized in repo.

**Follow-up:** TensorRT?

**Common mistake:** Only mentioned in vendor docs if present.

#### Advanced Q4. Improve yellow recall?

**Answer:** Oversample, class weights — data 2.7% yellow.

**Follow-up:** Loss change?

**Common mistake:** Ignoring imbalance.

#### Advanced Q5. False green R11 mitigation?

**Answer:** has_stop_state and R01 higher priority.

**Follow-up:** Multi-signal?

**Common mistake:** dominant_state logic.

#### Advanced Q6. Strict checkpoint load YOLOP?

**Answer:** YOLOPModelLoader validation — model_loader.py.

**Follow-up:** Stub checkpoint?

**Common mistake:** verify_pipeline stub .pth.

#### Advanced Q7. Package portability fixes?

**Answer:** Documented in docs/package_portability_fix_report.md.

**Follow-up:** yolop vendor?

**Common mistake:** Import path patches.

#### Advanced Q8. Colab training workflow signals?

**Answer:** docs/traffic_signal_colab_training.md and sample zip in data/.

**Follow-up:** Full dataset?

**Common mistake:** 42048 images on disk.

#### Advanced Q9. Sign Colab doc?

**Answer:** docs/traffic_sign_colab_training.md if present.

**Follow-up:** Local train?

**Common mistake:** scripts/train_traffic_sign.py.

#### Advanced Q10. Production gaps?

**Answer:** No auth, no SLA, CPU default, missing trained weights locally.

**Follow-up:** Streamlit?

**Common mistake:** Demo only.

#### Advanced Q11. LiDAR fusion?

**Answer:** Not in repository.

**Follow-up:** Future?

**Common mistake:** Do not claim implemented.

#### Advanced Q12. Imitation learning?

**Answer:** Not in repository.

**Follow-up:** DecisionEngine?

**Common mistake:** Rules only.

#### Advanced Q13. Multi-camera?

**Answer:** Not in repository.

**Follow-up:** Orchestrator?

**Common mistake:** Single frame API.

#### Advanced Q14. Async video pipeline?

**Answer:** Synchronous loop in app.py process_video.

**Follow-up:** Optimization?

**Common mistake:** Not implemented.

#### Advanced Q15. Batch inference?

**Answer:** Per-frame predict only.

**Follow-up:** YOLOv8 batch?

**Common mistake:** Not exposed in modules.

#### Advanced Q16. Security of uploaded media?

**Answer:** Local temp files, deleted after video — app.py.

**Follow-up:** Network?

**Common mistake:** Local Streamlit.

#### Advanced Q17. ADAS_DATA_ROOT on Windows?

**Answer:** Resolves under project models/ when set to repo root.

**Follow-up:** Audit paths?

**Common mistake:** Verified End-to-end.pth exists.

#### Advanced Q18. Register new rule?

**Answer:** Add @_register function to rules.py append RULE_REGISTRY.

**Follow-up:** Priority scheme?

**Common mistake:** Must arbitrate with existing.

#### Advanced Q19. Test stub YOLOP output?

**Answer:** Returns drivable/lane tensors inference_status stub_segmentation.

**Follow-up:** verify_pipeline?

**Common mistake:** _StubYOLOPInferenceEngine.

#### Advanced Q20. Why LANE ok accepts stub?

**Answer:** _LANE_OK_STATUSES includes stub_segmentation — scene_state.py.

**Follow-up:** Tests?

**Common mistake:** Stub pipeline tests.

#### Advanced Q21. Failure mode: sign init_failed cascade?

**Answer:** signs_ok false → R02/R05/R07 skip; R12 may still PROCEED.

**Follow-up:** Safe?

**Common mistake:** Document as limitation without sign weights.

#### Advanced Q22. Failure mode: all modules init_failed?

**Answer:** Only R12 default if no higher rules; scene flags all false.

**Follow-up:** Streamlit?

**Common mistake:** Shows module health warnings.

#### Advanced Q23. Could R11 and R01 both fire?

**Answer:** Both can hit; arbitration picks R01 priority 100.

**Follow-up:** Explanation?

**Common mistake:** Both messages in explanation sorted by priority.

#### Advanced Q24. Why not fuse sign+signal one model?

**Answer:** Separate heads/tasks and training data sources in repo design.

**Follow-up:** Latency?

**Common mistake:** Sequential predict four forwards.

#### Advanced Q25. Why YOLOv8n for sign/signal not s?

**Answer:** Config yolov8_sign/signal model_variant n for speed.

**Follow-up:** Vehicle s?

**Common mistake:** Different accuracy/speed tradeoff.

#### Advanced Q26. How to add R13 rule?

**Answer:** Implement function, @_register, assign priority, add test in test_decision_rules.

**Follow-up:** Config?

**Common mistake:** Extend DecisionConfig if new thresholds.

#### Advanced Q27. Thread safety of cached orchestrator?

**Answer:** Streamlit cache_resource single process; not designed multi-user server.

**Follow-up:** Production?

**Common mistake:** Would need per-request instances.

#### Advanced Q28. Determinism of stub engines?

**Answer:** Fixed tensors/boxes in verify_pipeline stubs.

**Follow-up:** Tests?

**Common mistake:** Deterministic assertions.

#### Advanced Q29. ONNX export in repo?

**Answer:** Not implemented in src modules.

**Follow-up:** Ultralytics?

**Common mistake:** Possible via external export not in tree.

#### Advanced Q30. TensorRT YOLOP?

**Answer:** Mentioned in vendor docs only if yolop_vendor_plan.md; not in src.

**Follow-up:** Claim?

**Common mistake:** Do not say implemented.

#### Advanced Q31. Quantization?

**Answer:** Not in repository.

**Follow-up:** INT8?

**Common mistake:** Future work.

#### Advanced Q32. Multi-frame stop sign hold?

**Answer:** Not implemented; per-frame R02 only.

**Follow-up:** Hysteresis?

**Common mistake:** Would need temporal state.

#### Advanced Q33. Tracking IDs for vehicles?

**Answer:** Not in YOLOv8 parser output.

**Follow-up:** Nearest?

**Common mistake:** Spatial heuristic only.

#### Advanced Q34. Rain/night domain gap?

**Answer:** Not addressed in training docs on disk.

**Follow-up:** BDD100K?

**Common mistake:** Diverse but not exhaustive.

#### Advanced Q35. Class confusion green vs yellow?

**Answer:** Documented yellow under 3% boxes; expect confusion.

**Follow-up:** Metric?

**Common mistake:** Per-class AP in training report.

#### Advanced Q36. Copy signal weights after Colab?

**Answer:** Path models/trained/yolov8_signal/traffic_signals_yolov8n.pt per training report §9.

**Follow-up:** ADAS_DATA_ROOT?

**Common mistake:** Must point to parent of models/.

#### Advanced Q37. Copy sign weights after train?

**Answer:** DEFAULT_MODEL_OUTPUT in traffic_sign config.py.

**Follow-up:** Inference?

**Common mistake:** TrafficSignModule auto path via get_traffic_sign_weights_path.

#### Advanced Q38. Re-run prepare without delete?

**Answer:** Scripts may overwrite data/traffic_signal_yolo; check script argparse.

**Follow-up:** Idempotent?

**Common mistake:** Read prepare script before claiming.

#### Advanced Q39. Legal liability of ADAS demo?

**Answer:** Not a product; rule-based assist only.

**Follow-up:** Interview?

**Common mistake:** Emphasize research/education scope.

#### Advanced Q40. ISO 26262 compliance?

**Answer:** Not in repository scope.

**Follow-up:** Safety?

**Common mistake:** Explain rules as heuristic demo.

#### Advanced Q41. Open loop vs closed loop?

**Answer:** Open loop recommendations only; no vehicle CAN control.

**Follow-up:** Actuation?

**Common mistake:** Not implemented.

#### Advanced Q42. Map fusion?

**Answer:** Not in repository.

**Follow-up:** GPS?

**Common mistake:** Not implemented.

#### Advanced Q43. SLAM integration?

**Answer:** Not in repository.

**Follow-up:** Lane?

**Common mistake:** YOLOP vision only.

#### Advanced Q44. Depth estimation?

**Answer:** Not in repository.

**Follow-up:** Proximity?

**Common mistake:** BBox area heuristics R08 only.

#### Advanced Q45. Radar fusion?

**Answer:** Not in repository.

**Follow-up:** VRU?

**Common mistake:** Camera YOLOv8 only.

#### Advanced Q46. OCR on speed signs?

**Answer:** Speed from class label speed_limit_XX not OCR.

**Follow-up:** Generalization?

**Common mistake:** Only trained limits 30/60.

#### Advanced Q47. Stop sign vs stop line?

**Answer:** Sign detector only; no stop line segmentation.

**Follow-up:** R02?

**Common mistake:** Sign bbox spatial rule.

#### Advanced Q48. Flashing yellow signal?

**Answer:** Not a class; would map poorly to yellow_light.

**Follow-up:** Gap?

**Common mistake:** Acknowledge limitation.

#### Advanced Q49. Arrow green signal?

**Answer:** Not separate class; green_light only.

**Follow-up:** Intersection?

**Common mistake:** Controlling signal heuristic upper 60%.

#### Advanced Q50. Pedestrian on sidewalk?

**Answer:** R03 requires drivable overlap; sidewalk excluded.

**Follow-up:** R06?

**Common mistake:** Lower third VRU without drivable requirement.

#### Advanced Q51. Cyclist in bike lane off drivable?

**Answer:** May trigger R06 not R03 depending on position.

**Follow-up:** Rules?

**Common mistake:** Different spatial logic.

#### Advanced Q52. Double yellow lane departure?

**Answer:** YOLOP lane_departure from offset threshold not paint color.

**Follow-up:** R09?

**Common mistake:** Boolean lane_departure flag.

#### Advanced Q53. HOV lane?

**Answer:** Not modeled.

**Follow-up:** Drivable mask?

**Common mistake:** YOLOP drivable area generic.

#### Advanced Q54. Construction zone signs?

**Answer:** Only 7 classes; unmapped signs ignored.

**Follow-up:** GTSRB train?

**Common mistake:** Only mapped GTSRB IDs exported.

#### Advanced Q55. Adversarial patch attack?

**Answer:** Not evaluated in repo.

**Follow-up:** Security?

**Common mistake:** Out of scope.

#### Advanced Q56. PII in uploaded video?

**Answer:** Local processing; user responsibility.

**Follow-up:** Streamlit cloud?

**Common mistake:** Deployment dependent.

#### Advanced Q57. License of YOLOP vendor?

**Answer:** Check vendor and docs; hustvl/YOLOP origin.

**Follow-up:** Commercial?

**Common mistake:** Verify licenses before product.

#### Advanced Q58. License Ultralytics?

**Answer:** YOLOv8 via ultralytics package requirements.txt.

**Follow-up:**  AGPL?

**Common mistake:** Note license in interview if asked.

#### Advanced Q59. Why dataclasses over pydantic?

**Answer:** stdlib dataclasses for SceneState/DecisionResult.

**Follow-up:** Validation?

**Common mistake:** Manual in module predict paths.

#### Advanced Q60. Why OpenCV not PIL in pipeline?

**Answer:** cv2 imdecode/imwrite throughout app and modules.

**Follow-up:** Display?

**Common mistake:** bgr_to_rgb for Streamlit only.

#### Advanced Q61. Why separate visualization package?

**Answer:** Decouple drawing from inference — src/visualization/.

**Follow-up:** Test?

**Common mistake:** Visual not always unit tested.

#### Advanced Q62. Why inject engines in LaneDetectionModule?

**Answer:** Constructor accepts preprocessor, loader, engine, parser, geometry for testing.

**Follow-up:** Pattern?

**Common mistake:** Same across vehicle/sign/signal.

#### Advanced Q63. Why lru_cache on config load?

**Answer:** _load_config cached in model_paths.py.

**Follow-up:** Hot reload?

**Common mistake:** Restart process to pick yaml changes.

#### Advanced Q64. Why mkdir on weight paths?

**Answer:** _ensure_directory creates parent dirs not weight files.

**Follow-up:** Missing file?

**Common mistake:** is_file() false until user places weights.

#### Advanced Q65. Compare mAP50 0.443 to human?

**Answer:** Assistive hint not replacement; false negatives possible.

**Follow-up:** R01 threshold?

**Common mistake:** 0.70 confidence mitigates some FP.

#### Advanced Q66. Ablation: remove lane module?

**Answer:** R03/R09/R10/R03 drivable disabled; pipeline still runs.

**Follow-up:** Config?

**Common mistake:** run_lane false in PipelineConfig.

#### Advanced Q67. Ablation: remove signals?

**Answer:** R01/R04/R11 skip; signs/vehicles remain.

**Follow-up:** Orchestrator?

**Common mistake:** run_signals toggle.

#### Advanced Q68. Hypothesis: merge R05 and R07?

**Answer:** Separate priorities 65 vs 55; both can fire.

**Follow-up:** Arbitration?

**Common mistake:** Higher priority wins single recommendation.

#### Advanced Q69. Unit vs integration test ratio?

**Answer:** Most tests unit/stub; verify_pipeline integration gate.

**Follow-up:** CI time?

**Common mistake:** 75 tests ~22s local.

#### Advanced Q70. Flaky tests?

**Answer:** None observed at 75 passed audit run.

**Follow-up:** GPU?

**Common mistake:** CPU default tests.

#### Advanced Q71. Coverage tool?

**Answer:** Not configured in repo audit.

**Follow-up:** pytest only?

**Common mistake:** Yes.

#### Advanced Q72. MyPy types?

**Answer:** Partial typing with TYPE_CHECKING; not strict enforced.

**Follow-up:** Frame alias?

**Common mistake:** Typed as ndarray.

#### Advanced Q73. Logging level production?

**Answer:** INFO in orchestrator decision STOP; modules log info/debug.

**Follow-up:** Structured?

**Common mistake:** Plain text logging.

#### Advanced Q74. Metrics export Prometheus?

**Answer:** Not implemented.

**Follow-up:** Observability?

**Common mistake:** Logs and Streamlit metrics only.

#### Advanced Q75. Docker deployment?

**Answer:** Not in repository.

**Follow-up:** Run?

**Common mistake:** pip install -r requirements.txt manual.

#### Advanced Q76. Makefile?

**Answer:** Not in repository.

**Follow-up:** Scripts?

**Common mistake:** python scripts/*.py directly.

#### Advanced Q77. Git LFS for weights?

**Answer:** Check .gitignore; pretrained present locally.

**Follow-up:** Remote clone?

**Common mistake:** May lack weights; download_weights.py.

#### Advanced Q78. Branch strategy?

**Answer:** Not documented in audit.

**Follow-up:** Commits?

**Common mistake:** User manages git.

#### Advanced Q79. Colab vs local train signal?

**Answer:** docs/traffic_signal_colab_training.md + full data local.

**Follow-up:** Sample zip?

**Common mistake:** For limited Colab storage.

#### Advanced Q80. Align sign training with Colab doc?

**Answer:** docs/traffic_sign_colab_training.md if present mirrors scripts.

**Follow-up:** GTSRB path?

**Common mistake:** Set ADAS_GTSRB_ROOT.

#### Advanced Q81. Future: end-to-end differentiable?

**Answer:** Not in repo; explicit rule layer.

**Follow-up:** Research?

**Common mistake:** Contrast with current design choice.

#### Advanced Q82. Future: transformer detector?

**Answer:** Not in repo; YOLOv8 chosen.

**Follow-up:** Why YOLO?

**Common mistake:** Ecosystem + separate fine-tune heads.

#### Advanced Q83. Explain _register side effect?

**Answer:** Decorator appends to RULE_REGISTRY at import time.

**Follow-up:** Order?

**Common mistake:** Evaluation order equals registration order not priority.

#### Advanced Q84. Explain max() stability on tie?

**Answer:** Python max stable; equal priority uses confidence then first encountered.

**Follow-up:** Tie rare?

**Common mistake:** Document tie-break explicitly.

#### Advanced Q85. RuntimeError mask shape message?

**Answer:** Tells to sync lane_detection.py and restart kernel.

**Follow-up:** Colab?

**Common mistake:** Kernel stale import issue documented in message.

#### Advanced Q86. Hypothesis test for yellow recall?

**Answer:** Would need evaluate per-class AP from training report.

**Follow-up:** Improve?

**Common mistake:** Weighted loss not in current scripts audit.

#### Advanced Q87. Cross-module latency budget?

**Answer:** total_time_ms logged; no per-module in orchestrator except module logs.

**Follow-up:** Optimize?

**Common mistake:** Parallel inference not implemented.

#### Advanced Q88. Energy on CPU four models?

**Answer:** Sequential; not profiled in docs.

**Follow-up:** Edge?

**Common mistake:** Would need model fusion or smaller variants.

#### Advanced Q89. Privacy retention of outputs?

**Answer:** outputs/streamlit/ persists until user deletes.

**Follow-up:** Temp video?

**Common mistake:** Unlinked after process in app finally.

#### Advanced Q90. Accessibility of Streamlit UI?

**Answer:** Standard Streamlit; not audited for a11y.

**Follow-up:** HUD?

**Common mistake:** Color-coded recommendations.

#### Advanced Q91. Internationalization?

**Answer:** English messages only in rules and UI.

**Follow-up:** Signs?

**Common mistake:** European GTSRB mapping subset.

#### Advanced Q92. Can R05 fire without speed_limit label?

**Answer:** No — requires `active_speed_limit_kmh` from sign summary (`rule_r05_active_speed_limit`).

**Follow-up:** turn_left sign?

**Common mistake:** Any sign triggers slow down.

#### Advanced Q93. Can R07 fire at low confidence?

**Answer:** Yes — no extra threshold beyond sign detection confidence in rule; uses det.confidence in hit.

**Follow-up:** vs R02?

**Common mistake:** Applying stop_sign_confidence to R07.

#### Advanced Q94. What happens if lane and vehicle both fail?

**Answer:** R03/R06/R08/R09/R10 skip; signal/sign rules may still fire; else R12.

**Follow-up:** SceneState flags?

**Common mistake:** Pipeline raises exception.

#### Advanced Q95. Why frozenset for ALLOWED_COCO_CLASS_IDS?

**Answer:** Immutable set for fast membership tests in `yolov8/output_parser.py`.

**Follow-up:** Sign IDs?

**Common mistake:** Mutating allowed IDs at runtime.

#### Advanced Q96. Why STATE_PRIORITY for signals?

**Answer:** Conservative merge when multiple states detected — `yolov8_signal/class_map.py`.

**Follow-up:** red vs green?

**Common mistake:** Averaging confidences across states.

#### Advanced Q97. What is REGULATORY_LABELS for signs?

**Answer:** frozenset stop, speed limits, keep_right — `yolov8_sign/class_map.py`.

**Follow-up:** is_regulatory_label()?

**Common mistake:** Treating turn signs as regulatory speed rules.

#### Advanced Q98. What is gtsrb_id_to_adas_label None behavior?

**Answer:** Converter skips unmapped GTSRB classes — not exported to YOLO dataset.

**Follow-up:** 43 GTSRB classes?

**Common mistake:** All GTSRB classes in 7-class head.

#### Advanced Q99. What is validate_frame in orchestrator vs modules?

**Answer:** Orchestrator `_validate_frame` duplicates module checks for early fail — `orchestrator.py`.

**Follow-up:** Double validation?

**Common mistake:** Only orchestrator validates.

#### Advanced Q100. What is the definitive completion percentage?

**Answer:** Section 20 — ~74% product / ~92% core software with module table.

**Follow-up:** Biggest gap?

**Common mistake:** Claiming 100% without missing sign/signal weights.


---

# SECTION 18 — Viva Preparation Guide

1. Draw pipeline: four modules → SceneState → DecisionEngine.  
2. Explain YOLOP outputs and mask resize (v2).  
3. Walk one rule with priority (R01 vs R11).  
4. Sign vs signal modules and class counts.  
5. BDD100K stats and class imbalance.  
6. Implemented vs stub (segmentation, Gradio, prepare_datasets).  
7. Demo `streamlit run app.py` or stub `verify_pipeline.py`.  
8. Test strategy and stub engines.  
9. Limitations of mAP50 0.443.  
10. Sign training pipeline scripts vs missing weights.

---

# SECTION 19 — Ownership Narrative

I structured the system around `BaseModule` and `PipelineOrchestrator` so perception and decision layers stay separable. I integrated vendored YOLOP, implemented frame-space mask resize (pipeline version 2), and mirrored the YOLOv8 loader/inference/parser pattern for vehicles, signs, and signals.

I built the BDD100K → YOLO signal training stack with class alignment to `yolov8_signal/class_map.py`, documented validation metrics, and added the GTSRB sign training package with prepare/train/evaluate/export scripts. I implemented the 12-rule `DecisionEngine` with explicit priorities and `SceneState` health flags for graceful degradation.

I delivered the Streamlit UI (`app.py`) with real/stub weight modes and 75 pytest tests covering rules, geometry, converters, and pipeline orchestration.

---

# SECTION 20 — Final Completion Report

## Module completion

| Module | Software | Weights/Training | Overall |
|--------|----------|------------------|---------|
| Lane | 100% | Pretrained present | 95% |
| Vehicle | 100% | Pretrained present | 95% |
| Traffic sign | 100% | Weight missing; training pipeline implemented | 75% |
| Traffic signal | 100% | Metrics documented; .pt missing locally | 85% |
| Decision engine | 100% | N/A | 100% |
| Orchestrator | 100% | N/A | 100% |
| Visualization | 100% | N/A | 100% |
| Streamlit UI | 95% | N/A | 95% |
| Segmentation | 5% stub | N/A | 5% |
| Tests | 95% | 75 pass | 95% |

**Weighted completion:** ~74% product / ~92% core software.

---

*End of Final Project Master Report.*
