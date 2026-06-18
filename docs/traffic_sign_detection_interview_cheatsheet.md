# Traffic Sign Detection — Interview Cheatsheet

High-value Q&A from the **Autonomous Driving Car** traffic sign detection module.  
Full 75-question bank: `docs/traffic_sign_detection_audit_report.md` Section 12.

---

## Elevator pitch (30 seconds)

Built `TrafficSignModule` with Ultralytics **YOLOv8n** fine-tuned for **seven ADAS sign classes** (stop, speed limits, turns, keep right, pedestrian crossing). Frame-space bounding boxes, `active_speed_limit_kmh` summary, loader/inference/parser architecture matching vehicle detection. Six pytest tests + gate script; fine-tuned weights and orchestrator not wired yet.

---

## Architecture (must know)

**Q: Walk through `TrafficSignModule.predict()`.**  
Auto-init if needed → `_validate_input()` → `YOLOv8SignInferenceEngine.run()` (Ultralytics `predict`) → `YOLOv8SignOutputParser.parse(frame_shape=...)` → `build_summary()` → `TrafficSignDetectionResult` → `_assert_detections_within_frame()`.

**Q: Where is YOLOv8 sign code?**  
`src/modules/yolov8_sign/` — `model_loader.py`, `inference.py`, `output_parser.py`, `output_schema.py`, `class_map.py`. Orchestrator: `traffic_sign.py`.

**Q: What replaced the original plan?**  
YOLOv5 stub (`predict()` returned `{}`) → **YOLOv8n** fine-tuned via `ultralytics` (same stack as vehicle module).

**Q: Does this use the vehicle YOLOv8 weights?**  
**No.** Separate fine-tuned head with 7 sign classes. Vehicle filters COCO road users only.

**Q: Does COCO `stop sign` suffice?**  
**No.** COCO has generic stop sign only — misses speed limits, turns, EU/GTSRB-specific classes.

---

## Seven classes (memorize)

| `class_id` | Label | Regulatory |
|------------|-------|------------|
| 0 | `stop` | Yes |
| 1 | `speed_limit_30` | Yes |
| 2 | `speed_limit_60` | Yes |
| 3 | `turn_left` | No |
| 4 | `turn_right` | No |
| 5 | `keep_right` | Yes |
| 6 | `pedestrian_crossing` | Warning |

Filter: `ALLOWED_SIGN_CLASS_IDS` in `output_parser.py` — IDs 0–6 only.

---

## Data structures (quick reference)

| Type | Key fields |
|------|------------|
| `TrafficSignDetectionResult` | `detections`, `summary`, `raw_status`, `inference_time_ms` |
| `DetectedSign` | `sign_label`, `confidence`, `bbox`, `speed_limit_kmh`, `is_regulatory` |
| `TrafficSignDetectionSummary` | `count_by_label`, `nearest_sign`, `active_speed_limit_kmh` |

`to_prediction_dict()` keys: `detections`, `count_by_label`, `total_count`, `nearest_sign`, `active_speed_limit_kmh`, `raw_status`.

`TRAFFIC_SIGN_OUTPUT_KEYS` — stable orchestrator contract tuple in `output_schema.py`.

---

## GTSRB mapping (viva)

| GTSRB ID | ADAS label |
|----------|------------|
| 14 | `stop` |
| 1 | `speed_limit_30` |
| 5 | `speed_limit_60` |
| 38 | `turn_left` |
| 34 | `turn_right` |
| 36 | `keep_right` |
| 12 | `pedestrian_crossing` |

Defined in `class_map.py` → `GTSRB_CLASS_ID_TO_ADAS_LABEL`.

---

## Design decisions (why questions)

| Decision | Why |
|----------|-----|
| YOLOv8 over YOLOv5 | Same Ultralytics API as vehicle; active v8 maintenance |
| YOLOv8n over yolov8s | Lower latency when chained with YOLOP + vehicle + signal |
| Fine-tune required | COCO cannot detect speed-limit / turn signs |
| No COCO fallback | Sign head is custom — meaningless without trained weights |
| Separate `yolov8_sign/` package | Avoid overwriting vehicle `yolov8/` package |
| `active_speed_limit_kmh = min(...)` | Most restrictive visible limit for ADAS safety |
| `nearest_sign` = max `center_y` | Lower in image ≈ closer to ego (same as vehicle) |

---

## Config & weights

| Setting | Value |
|---------|-------|
| Weights file | `yolov8_sign/traffic_signs_yolov8n.pt` |
| Location | `{data_root}/models/trained/` |
| Variant | `n` (`yolov8_sign.model_variant`) |
| Confidence | `0.5` (`thresholds.sign_confidence`) |
| IoU NMS | `0.45` (`thresholds.sign_iou`) |
| Input size | `640` |
| Max detections | `50` |

Path resolver: `get_traffic_sign_weights_path()` in `model_paths.py`.

---

## Error handling (quick)

| Scenario | Behavior |
|----------|----------|
| Weights missing | `WeightsNotFoundError` on init — **no download fallback** |
| `predict()` before init | Auto-init; failure → `empty(raw_status="init_failed")` |
| Bad frame | `ValueError` from `_validate_input()` |
| Inference fail | `empty(raw_status="pipeline_error")` |
| Bbox out of frame | `RuntimeError` from `_assert_detections_within_frame()` |

---

## Testing & verification

| Artifact | Purpose |
|----------|---------|
| `tests/test_traffic_sign_pipeline.py` | 6 integration tests |
| `tests/conftest.py` | `stub_yolov8_sign_*` fixtures |
| `scripts/verify_traffic_sign_detection.py` | Gate script (stub default, `--real` optional) |

Stub returns one `stop` box at upper-center, conf=0.92.

---

## What's NOT built (honest answers)

| Item | Status |
|------|--------|
| Pipeline orchestrator | **Stub** — `orchestrator.py` comments only |
| Decision engine / `SceneState.signs` | **Not implemented** |
| Fine-tuned weights in repo | **Pending** — `traffic_signs_yolov8n.pt` |
| Evaluation script | **Missing** |
| Unit tests for parser/class_map | **Missing** |
| README | Still lists YOLOv5 |

---

## Red flags to avoid in interviews

1. Claiming orchestrator uses sign output — **stubs only**
2. Saying COCO pretrained YOLOv8 works for speed limits — **wrong**
3. Confusing traffic **signs** with traffic **signals** (separate module)
4. Claiming GTSRB crops alone teach full-frame localization without fine-tune
5. Saying weights auto-download like vehicle module — **they do not**

---

## One-liner resume bullet

Implemented YOLOv8n traffic sign detection (7-class ADAS) with modular loader/inference/parser architecture, frame-space bbox validation, speed-limit summary heuristics, and stub-based CI — Python, Ultralytics, OpenCV.

---

*Cheatsheet derived from repository inspection — June 2026.*
