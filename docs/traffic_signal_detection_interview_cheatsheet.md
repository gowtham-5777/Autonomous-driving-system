# Traffic Signal Detection — Interview Cheatsheet

High-value Q&A from the **Autonomous Driving Car** traffic signal detection module.  
Full 75-question bank: `docs/traffic_signal_detection_audit_report.md` Section 12.

---

## Elevator pitch (30 seconds)

Built `TrafficSignalModule` with Ultralytics **YOLOv8n** fine-tuned for **three traffic-light states** (`red_light`, `yellow_light`, `green_light`). Frame-space bounding boxes, `controlling_signal` and `dominant_state` summaries, loader/inference/parser architecture matching vehicle and sign modules. Seven pytest tests + gate script; fine-tuned weights and orchestrator not wired yet.

---

## Architecture (must know)

**Q: Walk through `TrafficSignalModule.predict()`.**  
Auto-init if needed → `_validate_input()` → `YOLOv8SignalInferenceEngine.run()` → `YOLOv8SignalOutputParser.parse()` → `build_summary(frame_shape=...)` → `TrafficSignalDetectionResult` → `_assert_detections_within_frame()`.

**Q: Where is YOLOv8 signal code?**  
`src/modules/yolov8_signal/` — loader, inference, parser, schema, class_map. Orchestrator: `traffic_signal.py`.

**Q: What replaced the original plan?**  
CNN classifier stub (two-stage ROI + classify, never implemented) → **YOLOv8n** single-pass detect + classify via `ultralytics`.

**Q: Signs vs signals?**  
**Signs** = static boards (`TrafficSignModule`, 7 classes). **Signals** = illuminated light **state** (`TrafficSignalModule`, 3 classes). Separate modules by design.

**Q: Does COCO traffic light class suffice?**  
**No.** COCO id 9 is generic `traffic light` with **no R/Y/G state** — cannot drive stop/go rules.

---

## Three classes (memorize)

| `class_id` | Label | Stop | Caution | Proceed |
|------------|-------|------|---------|---------|
| 0 | `red_light` | Yes | No | No |
| 1 | `yellow_light` | No | Yes | No |
| 2 | `green_light` | No | No | Yes |

Filter: `ALLOWED_SIGNAL_CLASS_IDS` = `{0, 1, 2}` in `output_parser.py`.

---

## Detection logic (viva-critical)

| Concept | Rule | Code |
|---------|------|------|
| **`controlling_signal`** | Among lights in **upper 60%** of frame, pick max `center_y`, tie-break by confidence; else `nearest_signal` | `_select_controlling_signal()` |
| **`dominant_state`** | Label of `controlling_signal`; if none, max `state_priority` among all detections | `_select_dominant_state()` |
| **`state_priority`** | red=3, yellow=2, green=1 (conservative fallback only) | `class_map.STATE_PRIORITY` |
| **`nearest_signal`** | Max `center_y`, then `area` | `_select_nearest_signal()` |
| **Red + green conflict** | Warning logged; `dominant_state` follows **controlling** pick, not global red-wins | `build_summary()` |

**Important:** When both red and green are in the upper region, **higher confidence / position** picks controlling — test `test_dominant_state_priority` expects `red_light` when red has higher conf (0.92 vs 0.88).

---

## Data structures (quick reference)

| Type | Key fields |
|------|------------|
| `TrafficSignalDetectionResult` | `detections`, `summary`, `raw_status`, `inference_time_ms` |
| `DetectedSignal` | `signal_label`, `confidence`, `bbox`, `is_stop_state`, `is_caution_state`, `is_proceed_state` |
| `TrafficSignalSummary` | `controlling_signal`, `dominant_state`, `has_stop_state`, `has_proceed_state` |

`to_prediction_dict()` keys include `dominant_state`, `controlling_signal`, `has_stop_state`, `has_proceed_state`.

`TRAFFIC_SIGNAL_OUTPUT_KEYS` — stable orchestrator contract in `output_schema.py`.

---

## Config & weights

| Setting | Value |
|---------|-------|
| Weights file | `yolov8_signal/traffic_signals_yolov8n.pt` |
| Location | `{data_root}/models/trained/` |
| Variant | `n` |
| Confidence | `0.5` (`thresholds.signal_confidence`) |
| IoU NMS | `0.45` (`thresholds.signal_iou`) |
| Input size | `640` |
| Max detections | `20` |

Path: `get_traffic_signal_weights_path()` in `model_paths.py`.

**No COCO fallback** — missing weights → `WeightsNotFoundError`.

---

## Error handling (quick)

| Scenario | Behavior |
|----------|----------|
| Weights missing | `WeightsNotFoundError` on init |
| `predict()` before init | Auto-init; failure → `empty(raw_status="init_failed")` |
| Bad frame | `ValueError` from `_validate_input()` |
| Inference fail | `empty(raw_status="pipeline_error")` |
| Red + green both detected | Warning log; `has_stop_state` and `has_proceed_state` both true |

---

## Testing & verification

| Artifact | Purpose |
|----------|---------|
| `tests/test_traffic_signal_pipeline.py` | 7 integration tests |
| `tests/conftest.py` | `stub_yolov8_signal_*` fixtures |
| `scripts/verify_traffic_signal_detection.py` | Gate (stub default, `--real` optional) |

Stub returns one `red_light` at upper-center, conf=0.90, `class_id=0`.

---

## What's NOT built (honest answers)

| Item | Status |
|------|--------|
| Pipeline orchestrator | **Stub** — comments only |
| `SceneState.signals` / decision rules | **Not implemented** |
| Fine-tuned weights in repo | **Pending** |
| Temporal smoothing (flicker) | **v2 — not in code** |
| Evaluation script | **Missing** |
| Unit tests for parser/class_map | **Missing** |

---

## Red flags to avoid in interviews

1. Claiming CNN classifier was implemented — **it was a stub**
2. Saying `dominant_state` always picks red when red+green present — **follows controlling heuristic first**
3. Confusing traffic signs with traffic signals
4. Claiming COCO pretrained works for R/Y/G state
5. Saying weights auto-download like vehicle module — **they do not**

---

## One-liner resume bullet

Implemented YOLOv8n traffic signal detection (red/yellow/green) with controlling-signal heuristics, dominant-state summarization, modular loader/inference/parser architecture, and stub-based CI — Python, Ultralytics, OpenCV.

---

*Cheatsheet derived from repository inspection — June 2026.*
