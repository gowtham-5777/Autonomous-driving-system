# Autonomous Driving Car — Final Interview Master Guide

**Companion:** `docs/final_project_master_report.md`  
**Tests:** 75 passed (`python -m pytest tests/ -q`)  
**UI:** `streamlit run app.py`

Answers map to real files only.

---

## 1. Elevator Pitches

**10s:** Modular ADAS — YOLOP lanes + YOLOv8 perception + 12-rule decisions + Streamlit.

**30s:** Dashcam frames through lane/vehicle/sign/signal modules → `SceneState` → explainable STOP/SLOW_DOWN/PROCEED. 75 tests, signal training mAP50 0.443, sign training scripts ready.

---

## 2. Architecture (draw from memory)

```
BGR Frame → PipelineOrchestrator.run_frame()
  → LaneDetectionModule (YOLOP)
  → VehicleDetectionModule (YOLOv8s, COCO 0,1,2,3,5,7)
  → TrafficSignModule (YOLOv8n, 7 classes)
  → TrafficSignalModule (YOLOv8n, 3 classes)
  → SceneState.from_perception()
  → DecisionEngine.evaluate() → DecisionResult
  → visualize (overlays + HUD)
```

---

## 3. High-Value Q&A

### Pipeline

**Q:** Entry point?  
**A:** `PipelineOrchestrator.run_frame()` — `src/pipeline/orchestrator.py`.  
**Follow-up:** What validates the frame? — `_validate_frame`: ndarray (H,W,3).  
**Red flag:** Claiming segmentation runs (it does not).  
**Defense:** `run_segmentation: false` in `config/default.yaml`.

### Lane

**Q:** Why YOLOP?  
**A:** Lane + drivable masks in one pass for geometry and R03.  
**Follow-up:** Pipeline version 2? — Mask resize before geometry in `lane_detection.py`.  
**Red flag:** Offset computed at 640×640.  
**Defense:** `_resize_masks_to_frame_shape` + `test_mask_resize_geometry.py`.

### Vehicle

**Q:** Which classes?  
**A:** Six road users — COCO IDs 0,1,2,3,5,7 in `yolov8/output_parser.py`.  
**Red flag:** README says SSD — stale.  
**Defense:** Implemented module is YOLOv8s with weights on disk.

### Traffic sign

**Q:** Training status?  
**A:** Code + scripts implemented; weights and GTSRB data not on disk at audit.  
**Follow-up:** Scripts? — prepare/train/evaluate/export_traffic_sign.py + `training/traffic_sign/`.  
**Red flag:** "No training pipeline."  
**Defense:** Point to `gtsrb_converter.py` and `test_traffic_sign_training.py`.

### Traffic signal

**Q:** Dataset and metrics?  
**A:** 42,048 images, 147,550 boxes; P=0.58, R=0.443, mAP50=0.443 — training report.  
**Follow-up:** Class order? — Matches `SIGNAL_CLASS_ID_TO_LABEL` 0/1/2.  
**Red flag:** Perfect traffic light detector.  
**Defense:** Yellow 2.7% of boxes; mAP50 0.443 documented.

### Decision engine

**Q:** How is winner chosen?  
**A:** Highest `(priority, confidence)` in `DecisionEngine.arbitrate`.  
**Follow-up:** R01 vs R12? — R01 priority 100 vs R12 priority 1.  
**Red flag:** ML policy network.  
**Defense:** `evaluate_all` over `RULE_REGISTRY` in `rules.py`.

### UI

**Q:** Real vs demo mode?  
**A:** `real_weights_available()` in `app.py`; else stub orchestrator.  
**Red flag:** Always uses real weights.  
**Defense:** Sign/signal .pt missing → stub unless user supplies weights.

---

## 4. Model & Dataset Cheat Sheet

| Item | Value | File |
|------|-------|------|
| YOLOP weights | End-to-end.pth | models/pretrained/yolop/ |
| YOLOv8s | yolov8s.pt | models/pretrained/yolov8/ |
| Sign classes | 7 | yolov8_sign/class_map.py |
| Signal classes | 3 | yolov8_signal/class_map.py |
| Signal dataset | 42048 img | data/traffic_signal_yolo/ |
| GTSRB | External | training/traffic_sign/config.py |

---

## 5. Design Decisions

1. **Rule engine over learned policy** — explainable `RuleHit.rule_id`.  
2. **Separate YOLOv8 subpackages** — different heads; no sign COCO fallback.  
3. **SceneState health flags** — rules skip when module not ok.  
4. **Stub orchestrator** — demo without all weights (`verify_pipeline.py`).  
5. **Pipeline v2 mask resize** — correct lane offset at full resolution.

---

## 6. Red Flags (do not claim)

- End-to-end autonomous driving (assistive rules only)  
- Gradio UI production-ready (`src/app.py` stub)  
- U-Net segmentation in pipeline  
- GTSRB bundled in repo  
- Trained sign/signal .pt present on audit machine  
- README module list without noting staleness  

---

## 7. Best Defense Answers

**Missing weights:** "Inference code is complete; trained artifacts are configured paths under `models/trained/`. Streamlit falls back to stubs. Signal metrics come from Colab/local training documented in `traffic_signal_training_report.md`."

**Low mAP50:** "Documented 0.443 on BDD100K val with severe yellow imbalance (2.7%). Architecture prioritizes explainable integration over SOTA detection."

**Your contribution:** "Orchestrator, decision engine, YOLOv8 module pattern, YOLOP v2 geometry fix, signal training stack, sign training scripts, Streamlit UI, 75 tests."

---

## 10. Follow-Up Chains (interviewer drills)

**Chain A — Red light:** Signal module → dominant_state red_light → R01 confidence ≥ 0.70 → STOP beats R11.  
**Chain B — Missing weights:** sign .pt absent → init_failed → signs_ok false → no R02/R05/R07 → may R12 PROCEED.  
**Chain C — Lane v2:** YOLOP 640 mask → resize → offset px → R10 if |offset|>35 and not departure.  
**Chain D — Training alignment:** bdd100k_converter class 0=red → SIGNAL_CLASS_ID_TO_LABEL → parser no change.

---

## 11. Model Explanations (30-second each)

**YOLOP:** Single MCnet forward pass outputs drivable + lane segmentation masks; geometry derives center/offset for KEEP_LANE and departure rules. Weights: `End-to-end.pth`.

**YOLOv8s vehicle:** COCO-pretrained detector filtered to six road-user classes; supports R03/R06/R08 via labels person/bicycle/truck/bus.

**YOLOv8n sign (7-class):** Fine-tuned head required; maps GTSRB subset to ADAS labels; drives R02/R05/R07. Training via `training/traffic_sign/` + scripts; weights missing locally at audit.

**YOLOv8n signal (3-class):** BDD100K state detector; mAP50 0.443 documented; class imbalance on yellow; drives R01/R04/R11.

---

## 12. Dataset Explanations

| Dataset | Role | On disk |
|---------|------|---------|
| BDD100K → traffic_signal_yolo | Signal train/val | Yes (42048 img) |
| GTSRB → traffic_sign_yolo | Sign train | External GTSRB; YOLO dir not prepared |
| COCO/Cityscapes | Config placeholders | No; prepare_datasets stub |

---

## 13. Top 25 Rapid-Fire Questions

1. Frame? BGR uint8 ndarray  
2. Tests? 75  
3. Rules? 12 (R01–R12)  
4. Highest priority? R01 (100)  
5. Default rule? R12  
6. Sign classes? 7  
7. Signal classes? 3  
8. COCO car ID? 2  
9. Env var? ADAS_DATA_ROOT  
10. Lane pipeline version? 2  
11. Yellow box share? ~2.7%  
12. Signal mAP50? 0.443  
13. Segmentation? Stub  
14. UI? Streamlit app.py  
15. Sign train script? train_traffic_sign.py  
16. R03 mask? drivable_mask  
17. R02 spatial? lower 40% frame  
18. Stub builder? verify_pipeline.build_stub_orchestrator  
19. YOLOP vendor? src/modules/yolop/vendor/  
20. prepare_datasets? Stub  
21. Signal converter? bdd100k_converter.py  
22. Sign converter? gtsrb_converter.py  
23. Production? Demo/research  
24. Vehicle model? YOLOv8s  
25. Arbitration key? (priority, confidence)  

---

## 14. Viva — Answer in 3 Sentences Each

**Architecture:** Four `BaseModule` detectors run in `PipelineOrchestrator`, fuse to `SceneState`, then twelve priority rules yield one `ADASRecommendation`. Visualization uses OpenCV overlays and decision HUD.

**Contribution:** Integrated YOLOP with frame-aligned masks, YOLOv8 subpackages for three tasks, rule engine + orchestrator, BDD100K signal training, GTSRB sign training tooling, Streamlit + 75 tests.

**Challenge:** Lane offset wrong until masks resized to frame space — pipeline v2 with regression tests.

**Gap:** Trained sign/signal `.pt` not on audit disk; GTSRB/traffic_sign_yolo not prepared locally.

---

*Full 20-section audit and 300 Q&A: `docs/final_project_master_report.md`*
