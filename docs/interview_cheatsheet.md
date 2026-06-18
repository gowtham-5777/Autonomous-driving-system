# ADAS Lane Detection — Interview Cheatsheet

High-value questions and answers from the **Autonomous Driving Car** codebase.  
Full 75-question bank: `docs/project_implementation_report.md` Section 9.

---

## Elevator pitch (30 seconds)

Built a modular Python ADAS stack with **YOLOP MCnet** lane detection: vendored network, PyTorch inference, segmentation parsing, mask resize to frame space, lane center + vehicle offset. Eleven pytest tests; four other perception modules are stubs. Pipeline version 2 fixes a documented 640×640 vs full-frame coordinate bug.

---

## Architecture (must know)

**Q: Walk through `LaneDetectionModule.predict()`.**  
Validate frame → `LanePreprocessor` (Canny edges, stored in result) → `YOLOPInferenceEngine.run()` (640×640 MCnet forward) → `YOLOPOutputParser.parse(frame_shape=...)` → optional `postprocess_lane_mask` → `_resize_masks_to_frame_shape` → `LaneGeometryExtractor` → `LaneDetectionResult` → `_assert_result_masks_match_frame`.

**Q: Where is MCnet defined?**  
`src/modules/yolop/vendor/models/YOLOP.py` — `get_net()`, `MCnet` class. Wired in `inference.attach_model()` and `_execute_forward()`.

**Q: What are the three YOLOP heads used?**  
Forward returns `(det_out, drivable_head, lane_head)`. Lane module uses indices **1** (drivable) and **2** (lane). Detection head not consumed yet.

**Q: What is `LANE_DETECTION_PIPELINE_VERSION = 2`?**  
Marks mask-resize-before-geometry semantics. Colab must load `>= 2` or masks stay 640×640 and offset is wrong.

---

## YOLOP fundamentals

**Q: What is YOLOP?**  
Multi-task driving perception network (lane seg + drivable seg + object detection). This repo vendors hustvl/YOLOP MCnet under `src/modules/yolop/vendor/`.

**Q: Detection vs segmentation for lanes?**  
Detection = boxes + classes. Segmentation = per-pixel lane mask. This project uses **segmentation**; polylines from mask are not implemented (`left_lane`/`right_lane` always `None`).

**Q: Why 640×640?**  
`DEFAULT_INPUT_SIZE = (640, 640)` in `inference.py` — YOLOP training input.

---

## Geometry (high interview value)

**Q: How is lane center computed?**  
`mean(x)` of all pixels where `lane_mask > 0` (`lane_geometry.compute_lane_center`). Full-mask mean, not bottom row only.

**Q: What is vehicle offset?**  
`lane_center_x - vehicle_center_x`, where `vehicle_center_x = image_width / 2` (image center as ego proxy).

**Q: Why resize masks back to frame resolution?**  
YOLOP outputs 640×640 masks; frames may be 1024×2048. Geometry on 640-space center with 2048-width offset gave **~−800 px error**. Fix: `cv2.resize(..., INTER_NEAREST)` in `_resize_masks_to_frame_shape` before geometry.

**Q: `parsed.lane_lines.lane_mask` vs `result.lane_mask`?**  
Parser mask = **model resolution** (640×640). Result mask = **frame resolution** after resize. **Always use `result.lane_mask` for geometry/display.**

---

## Data structures (quick reference)

| Type | Key fields |
|------|------------|
| `LaneDetectionResult` | `lane_center_x`, `vehicle_offset`, `lane_mask`, `drivable_mask`, `lane_departure`, `raw_status` |
| `ParsedYOLOPOutput` | Intermediate parser output; includes parse-time geometry |
| `LaneLineData` | `lane_mask`, `left_lane`, `right_lane` (latter two always `None`) |
| `VehicleOffsetData` | `offset_pixels`, `vehicle_x`, `lane_center_x` |

---

## Implementation status (honest answers)

**Q: Is the project complete?**  
**No.** README says "scaffold only" but **lane detection ~75%** done. Vehicle/sign/signal/segmentation/decision/app are **stubs** (`predict()` returns `{}`).

**Q: What's stubbed in lane detection?**  
- `visualize()` → `frame.copy()` only  
- Left/right lane polylines  
- `output_parser.detect_lane_departure()` → always `False` (module uses simple threshold instead)  
- `utils.py` legacy placeholders  

**Q: Do tests use real YOLOP weights?**  
Usually **stub checkpoint** + `_StubYOLOPInferenceEngine` in `conftest.py`. Real `End-to-end.pth` path configured but optional locally.

---

## Error handling & deployment

**Q: What if weights are missing?**  
`CheckpointNotFoundError` on init; `predict()` may return `empty(raw_status="init_failed")`.

**Q: What if masks aren't resized (Colab bug)?**  
Pipeline v2: `_assert_result_masks_match_frame` raises `RuntimeError`. Pre-v2: silent wrong offset (~−806 on 2048-wide frame).

**Q: How to verify before demo?**  
```bash
python scripts/verify_mask_resize.py
```
```python
from src.modules.lane_detection import LANE_DETECTION_PIPELINE_VERSION
assert LANE_DETECTION_PIPELINE_VERSION >= 2
assert result.lane_mask.shape == frame.shape[:2]
```

---

## Technical decisions (why questions)

| Decision | Why | Tradeoff |
|----------|-----|----------|
| YOLOP | Multi-task, open weights, real-time 640 input | Vendor maintenance, GPU needed |
| Segmentation lanes | Robust to markings vs pure edges | Needs trained weights |
| Image center = ego | No camera calibration in repo | Wrong if camera offset |
| INTER_NEAREST resize | Preserve binary masks | No perspective homography |
| Vendoring vs pip | Colab portability, pinned commit | Manual upstream merges |

---

## Code locations (rapid fire)

| Concern | File |
|---------|------|
| Pipeline orchestration | `src/modules/lane_detection.py` |
| MCnet forward | `src/modules/yolop/inference.py` |
| Weights | `src/modules/yolop/model_loader.py` |
| Mask argmax | `src/modules/yolop/output_parser.py` |
| Geometry | `src/modules/yolop/lane_geometry.py` |
| Morphology | `src/modules/yolop/postprocess.py` |
| Resize helper | `src/modules/yolop/mask_resize.py` |
| Schemas | `src/modules/yolop/output_schema.py` |
| Config paths | `config/default.yaml`, `src/utils/model_paths.py` |
| Tests | `tests/test_lane_detection_pipeline.py`, `test_mask_resize_geometry.py` |

---

## Advanced / system design

**Q: How would you add vehicle detection?**  
Implement `VehicleDetectionModule.predict()` loading `ssd_mobilenet_v2_coco.pb` from config; return bboxes dict; wire in `orchestrator.py` (currently comments only).

**Q: How would you improve lane center?**  
Sample bottom 20% of mask only; fit polynomial; temporal Kalman filter; populate `left_lane`/`right_lane` polylines.

**Q: Production readiness blockers?**  
No IoU evaluation (`evaluation_lane_detection.py` empty), no camera calibration, no temporal smoothing, four modules stubbed, no orchestrator.

**Q: 11 tests — what do they prove?**  
Pipeline init/predict works; parser needs `frame_shape` for offset; frame-sized masks on 1024×2048 with pipeline v2. **Do not prove** real MCnet accuracy.

---

## Numbers to remember

- **Pipeline version:** 2  
- **YOLOP input:** 640×640  
- **Departure threshold:** 50 px (`ParserConfig.departure_threshold_px`)  
- **Min component area:** 400 px (`postprocess.py`)  
- **Tests:** 11 passed  
- **Lane module completion:** ~75%  
- **Stub modules:** 4 of 5 perception modules  

---

## Red flags to avoid in interviews

1. Claiming all six README modules are implemented — **only lane detection is real**.  
2. Saying README status is accurate — it says "scaffold only".  
3. Using `parsed.lane_lines.lane_mask` for display — it's 640×640.  
4. Claiming real-weight accuracy is tested — tests use **stub inference**.  
5. Saying lane departure is fully implemented — parser stub; module uses basic threshold only.

---

## One-liner resume bullet

*Integrated YOLOP MCnet into a modular ADAS pipeline with frame-aligned segmentation masks, geometry correction, and pytest-verified end-to-end lane detection.*

---

*Source: `docs/project_implementation_report.md` — generated from repository audit, June 2026.*
