# Vehicle Detection — Interview Cheatsheet

High-value Q&A from the **Autonomous Driving Car** vehicle detection module.  
Full 75-question bank: `docs/vehicle_detection_audit_report.md` Section 12.

---

## Elevator pitch (30 seconds)

Built `VehicleDetectionModule` with Ultralytics **YOLOv8s** — six ADAS road-user classes (person, bicycle, car, motorcycle, bus, truck), frame-space bounding boxes, loader/inference/parser architecture matching YOLOP lane detection. Five pytest tests + gate script; orchestrator and decision engine not wired yet.

---

## Architecture (must know)

**Q: Walk through `VehicleDetectionModule.predict()`.**  
Auto-init if needed → `_validate_input()` → `YOLOv8InferenceEngine.run()` (Ultralytics `predict`) → `YOLOv8OutputParser.parse(frame_shape=...)` → `build_summary()` → `VehicleDetectionResult` → `_assert_detections_within_frame()`.

**Q: Where is YOLOv8 code?**  
`src/modules/yolov8/` — `model_loader.py`, `inference.py`, `output_parser.py`, `output_schema.py`. Orchestrator: `vehicle_detection.py`.

**Q: What replaced the original plan?**  
SSD MobileNetV2 stub → **YOLOv8** via `ultralytics` (PyTorch-native, COCO pretrained).

**Q: Does this use YOLOP's detection head?**  
**No.** YOLOP `det_out` exists but is unparsed. Separate YOLOv8 forward pass.

---

## YOLOv8 fundamentals

**Q: What is NMS?**  
Non-Maximum Suppression — removes overlapping duplicate boxes. IoU threshold **0.45** (`thresholds.object_iou`).

**Q: Default model and input size?**  
`yolov8s.pt`, `imgsz=640`, confidence **0.5**.

**Q: How are boxes represented?**  
`BoundingBoxData`: inclusive `(x1, y1, x2, y2)` in **original frame pixels** + derived width, height, center, area.

**Q: What is confidence?**  
Ultralytics `boxes.conf` — filtered at inference (`conf=`) and parser (`confidence_threshold`).

---

## Six classes (memorize)

| Label | COCO ID |
|-------|---------|
| person | 0 |
| bicycle | 1 |
| car | 2 |
| motorcycle | 3 |
| bus | 5 |
| truck | 7 |

Filter: `ALLOWED_COCO_CLASS_IDS` in `output_parser.py` — drops other 74 COCO classes.

---

## Data structures (quick reference)

| Type | Key fields |
|------|------------|
| `VehicleDetectionResult` | `detections`, `summary`, `raw_status`, `inference_time_ms` |
| `DetectedObject` | `label`, `confidence`, `bbox`, `coco_class_id` |
| `VehicleDetectionSummary` | `count_by_label`, `nearest_object`, `highest_confidence` |

`to_prediction_dict()` keys: `detections`, `count_by_label`, `total_count`, `nearest_object`, `raw_status`.

---

## Design decisions (why questions)

| Decision | Why |
|----------|-----|
| YOLOv8 over SSD | PyTorch stack, Ultralytics API, no TF `.pb` |
| yolov8s default | Balance recall (person/bicycle) vs dual-model GPU with YOLOP |
| Separate parser | Testable COCO filter, decoupled from Ultralytics |
| Frame-space boxes | Same principle as lane mask resize — geometry must match input frame |
| Injectable stubs | CI passes without downloading 21.5 MB weights |

---

## Auto-rickshaw / real-world gaps

**Q: Why aren't auto-rickshaws detected?**  
**Not a COCO class.** May be misclassified as motorcycle/car with low conf, or missed. Fix: fine-tune with local labels + extend `COCO_CLASS_ID_TO_LABEL`.

**Q: Why 0 detections on test road image with real weights?**  
`road_sample.jpg` is synthetic lane lines — no real vehicles. Stub pipeline injects fake car for CI.

---

## Error handling

| Scenario | Behavior |
|----------|----------|
| Init fails | `empty(raw_status="init_failed")` on auto-init |
| Bad frame | `ValueError` from `_validate_input()` |
| Inference error | `empty(raw_status="pipeline_error")` |
| Box out of frame | `RuntimeError` from `_assert_detections_within_frame()` |
| Missing Drive weights | Warning + Ultralytics downloads `yolov8s.pt` |

---

## Performance numbers

| Setup | Approximate |
|-------|-------------|
| CPU real inference | ~700 ms/frame (measured) |
| GPU T4 (estimated) | 15–40 ms |
| Variants | n (fast) / **s** (default) / m (accurate) |
| Dual YOLOP + YOLOv8 | Call `cleanup()` between modules |

---

## Testing & verification

```bash
python -m pytest tests/test_vehicle_detection_pipeline.py -v   # 5 tests
python scripts/verify_vehicle_detection.py                      # stub gate
python scripts/verify_vehicle_detection.py --real               # live weights
```

**5 tests prove:** init, predict, auto-init, visualize, class filter.  
**Do not prove:** real-video mAP, GPU path, parser unit isolation.

---

## Integration status (honest)

| Component | Status |
|-----------|--------|
| `VehicleDetectionModule` | **Done** (~85%) |
| Orchestrator | **Stub** |
| Decision engine | **Stub** |
| Lane + vehicle fusion | **Not coded** |
| Tracking (`track_id`) | **None** |

---

## Code locations (rapid fire)

| Concern | File |
|---------|------|
| Orchestrator | `src/modules/vehicle_detection.py` |
| Load weights | `src/modules/yolov8/model_loader.py` |
| Forward pass | `src/modules/yolov8/inference.py` |
| COCO filter | `src/modules/yolov8/output_parser.py` |
| Schemas | `src/modules/yolov8/output_schema.py` |
| Config | `config/default.yaml`, `get_yolov8_config()` |
| Visualization | `draw_vehicle_detections()` in `overlays.py` |
| Tests | `tests/test_vehicle_detection_pipeline.py` |

---

## Advanced / system design

**Q: How fuse with lane detection?**  
Future: pedestrian bbox overlap with `LaneDetectionResult.drivable_mask` → slow-down rule in `decision/rules.py`.

**Q: How add a new class?**  
Fine-tune YOLOv8 → add ID to `COCO_CLASS_ID_TO_LABEL` and `ADAS_VEHICLE_LABELS`.

**Q: Production blockers?**  
No orchestrator, no SceneState, no mAP eval, no tracking, README still mentions SSD.

---

## Red flags to avoid in interviews

1. Claiming orchestrator/decision engine use vehicle output — **stubs only**.
2. Saying YOLOP detection head is used — **separate YOLOv8**.
3. Claiming real-video accuracy is tested — **stub inference in CI**.
4. Expecting auto-rickshaw detection out of the box — **not in COCO**.
5. Using README module 2 line literally — still says SSD MobileNetV2 in README line 8.

---

## One-liner resume bullet

*Implemented YOLOv8 vehicle detection with six-class ADAS filtering, frame-validated bounding boxes, and pytest-verified pipeline architecture aligned with the YOLOP lane module.*

---

*Source: `docs/vehicle_detection_audit_report.md` — generated from repository audit, June 2026.*
