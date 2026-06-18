# Lane Detection Requirements Verification Report

**Date:** 2026-06-17  
**Method:** Static trace of `LaneDetectionModule.predict()` → `_run_pipeline()` plus local gate script and Colab runtime evidence supplied by user.

## Colab runtime evidence (authoritative for deployment)

| Observation | Value |
|-------------|-------|
| `frame.shape` | `(1024, 2048, 3)` |
| `result.lane_mask.shape` | `(640, 640)` |
| `vehicle_offset` | `-806.96` |

`vehicle_center_x ≈ 2048 / 2 = 1024`. With `offset ≈ lane_center_x - 1024`, implied `lane_center_x ≈ 217` — consistent with **640×640 mask geometry**, not frame-space geometry.

**Conclusion:** Colab is **not** executing pipeline version 2 (mask resize + frame-space geometry). A silent `(640, 640)` result with offset ≈ −806 cannot occur when `_assert_result_masks_match_frame` (committed) is active; that guard raises `RuntimeError` before return.

Local gate (same commit `7eaab8e` on `origin/main`):

```
python scripts/verify_mask_resize.py
→ lane_mask.shape == frame.shape[:2] == (1024, 2048)  PASS
```

---

## Requirements checklist

| # | Requirement | Status | Repository code | Colab runtime |
|---|-------------|--------|-----------------|---------------|
| 1 | YOLOP MCnet integrated; real weights loaded | **PASS** (code) / **PASS**† (Colab) | See §1 | Inference produces 640×640 masks → MCnet forward path active |
| 2 | `predict()` uses real YOLOP inference | **PASS** | See §2 | `inference_engine.run()` used (masks present) |
| 3 | Lane mask resized before geometry | **FAIL** | Implemented — not active on Colab | `(640, 640)` returned |
| 4 | Drivable mask resized before geometry | **FAIL** | Implemented — not active on Colab | Same stale pipeline |
| 5 | `lane_center_x` in frame coordinates | **FAIL** | Correct only after §3 | Implied ~217 px (640-space) |
| 6 | `vehicle_offset` in frame coordinates | **FAIL** | Mixed if §3 skipped | −806.96 (640-space center vs 2048 width) |
| 7 | `LaneDetectionResult` returns resized masks | **FAIL** | Implemented — not active on Colab | `lane_mask` still `(640, 640)` |
| 8 | Runtime: `result.*.shape == frame.shape[:2]` | **FAIL** (Colab) / **PASS** (local gate) | Guard + tests | Shapes differ |

† Req 1 “real weights”: assumes Colab has `End-to-end.pth` at the path from `get_yolop_weights_path()` and `initialize()` succeeded. Loader permits `strict=False` fallback (`inference.py` 155–160), so verify checkpoint key count (~542 MCnet keys) on Colab.

**Overall:** Requirements **1–2 PASS** on Colab. Requirements **3–8 FAIL** on Colab until the synced pipeline version 2 is loaded (kernel restart + `git pull`).

---

## §1 — YOLOP MCnet integrated and real weights loaded

### Status: **PASS** (repository)

**MCnet vendored and attached**

```7:7:src/modules/yolop/vendor/__init__.py
from .models import MCnet, get_net
```

```19:19:src/modules/yolop/inference.py
from .vendor import get_net
```

```139:176:src/modules/yolop/inference.py
        device = self._resolve_device()
        logger.info("Instantiating vendored MCnet via get_net()")
        ...
            model = get_net(cfg=None)
        ...
            load_result = model.load_state_dict(state_dict, strict=True)
        ...
        model = model.to(device)
        model.eval()
        self._model = model
        self._architecture_ready = True
```

**Weights loaded in `initialize()`**

```122:125:src/modules/lane_detection.py
            metadata = self.model_loader.load_model()
            self._model_package = self.model_loader.get_model()
            self.inference_engine.attach_model(self._model_package)
```

**Real forward pass**

```385:386:src/modules/yolop/inference.py
            with torch.no_grad():
                outputs = self._model(input_tensor)
```

### Colab note

Colab produces segmentation masks → MCnet + checkpoint path are working. Confirm `module.model_loader.metadata.num_tensor_keys` is ≫ stub size and weights path points to `End-to-end.pth`.

---

## §2 — `predict()` uses real YOLOP inference

### Status: **PASS**

```179:180:src/modules/lane_detection.py
        try:
            return self._run_pipeline(frame)
```

```218:219:src/modules/lane_detection.py
        try:
            raw_outputs = self.inference_engine.run(frame)
```

```253:279:src/modules/yolop/inference.py
    def run(self, frame: Frame) -> YOLOPRawOutput:
        ...
        preprocessed = self.preprocess(frame)
        raw_forward = self._execute_forward(preprocessed)
        results = self.postprocess(raw_forward, preprocessed)
```

No stub bypass in the default `LaneDetectionModule` constructor (`inference_engine or YOLOPInferenceEngine()` at line 95).

---

## §3 — Lane mask resized before geometry

### Status: **FAIL on Colab** / **PASS in repository (when version 2 loaded)**

**Intended path (version 2)**

```238:243:src/modules/lane_detection.py
        lane_mask, drivable_mask = self._resize_masks_to_frame_shape(
            lane_mask,
            drivable_mask,
            frame,
        )
```

```328:335:src/modules/lane_detection.py
            return cv2.resize(
                np.asarray(mask, dtype=np.uint8),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )
```

**Geometry runs after resize**

```252:253:src/modules/lane_detection.py
        if lane_mask is not None:
            lane_center_x = self.geometry_extractor.compute_lane_center(lane_mask)
```

### Root cause (Colab FAIL)

Colab kernel is running **pre-version-2** `lane_detection.py` (no `_resize_masks_to_frame_shape`, or resize never reached). Evidence: `(640, 640)` mask returned without `RuntimeError` from `_assert_result_masks_match_frame`.

### Fix for Colab

1. `git pull origin main` (commit `7eaab8e` or later).
2. **Runtime → Restart session** (clear cached modules).
3. Verify before `predict()`:

```python
from src.modules.lane_detection import LANE_DETECTION_PIPELINE_VERSION, LaneDetectionModule
assert LANE_DETECTION_PIPELINE_VERSION >= 2
assert hasattr(LaneDetectionModule, "_resize_masks_to_frame_shape")
```

4. Run `!python scripts/verify_mask_resize.py`.

**Do not read** `parsed.lane_lines.lane_mask` — parser intentionally keeps model-resolution masks; only `result.lane_mask` is frame-sized after step 5.

---

## §4 — Drivable mask resized before geometry

### Status: **FAIL on Colab** / **PASS in repository**

Same call as §3 resizes both masks:

```337:338:src/modules/lane_detection.py
        lane_resized = _resize(lane_mask)
        drivable_resized = _resize(drivable_mask)
```

```272:273:src/modules/lane_detection.py
            lane_mask=lane_mask,
            drivable_mask=drivable_mask,
```

### Root cause / fix

Identical to §3 — stale Colab module cache.

---

## §5 — `lane_center_x` in original frame coordinates

### Status: **FAIL on Colab** / **PASS in repository (conditional on §3)**

```83:98:src/modules/yolop/lane_geometry.py
    def compute_lane_center(self, lane_mask: LaneMask) -> float | None:
        ...
        lane_center_x = float(np.mean(lane_pixels.x_coords))
```

Mean x is in **mask pixel space**. Frame coordinates require resized mask input (§3).

### Root cause (Colab)

640×640 mask → `lane_center_x ≈ 217`. Expected frame-space center for a centered lane on 2048 px width ≈ 1024.

### Fix

Activate §3 on Colab; no separate geometry change needed.

---

## §6 — `vehicle_offset` in original frame coordinates

### Status: **FAIL on Colab** / **PASS in repository (conditional on §3)**

```250:257:src/modules/lane_detection.py
        frame_width = int(frame.shape[1])
        ...
                offset_result = self.geometry_extractor.compute_vehicle_offset(
                    lane_center_x=lane_center_x,
                    image_width=frame_width,
                )
```

```131:132:src/modules/yolop/lane_geometry.py
        vehicle_center_x = image_width / 2.0
        offset_pixels = lane_center_x - vehicle_center_x
```

Vehicle center uses **frame width**; lane center must also be frame-space. Without §3, offset is wrong (−806.96 matches 217 − 1024).

### Root cause / fix

Same as §3 — resize masks before geometry.

---

## §7 — `LaneDetectionResult` returns resized masks

### Status: **FAIL on Colab** / **PASS in repository**

```266:277:src/modules/lane_detection.py
        result = LaneDetectionResult(
            ...
            lane_mask=lane_mask,
            drivable_mask=drivable_mask,
            ...
        )
```

`lane_mask` / `drivable_mask` variables hold post-resize arrays from step 5.

### Root cause (Colab)

Stale pipeline returns parser-resolution `(640, 640)` masks in `result`.

### Fix

§3 Colab sync + use `result.lane_mask`, not `parsed.lane_lines.lane_mask`.

---

## §8 — Runtime shape verification

### Status: **FAIL on Colab** / **PASS locally**

**Hard guard in pipeline**

```284:285:src/modules/lane_detection.py
        self._assert_result_masks_match_frame(result, frame)
        return result
```

```295:301:src/modules/lane_detection.py
        if result.lane_mask is not None and result.lane_mask.shape[:2] != target_shape:
            raise RuntimeError(
                f"Lane mask shape {result.lane_mask.shape[:2]} != frame {target_shape}. "
                ...
            )
```

**Automated gate**

- `scripts/verify_mask_resize.py` — asserts `(1024, 2048)` for test frame.
- `tests/test_mask_resize_geometry.py` — `test_geometry_valid_when_frame_and_mask_shapes_differ`
- `tests/test_lane_detection_pipeline.py` — `result.lane_mask.shape == road_frame.shape[:2]`

### Colab evidence

```
result.lane_mask.shape (640, 640) != frame.shape[:2] (1024, 2048)  → FAIL
```

With version 2 code, this would **raise** instead of returning silently.

### Fix for Colab

```python
result = module.predict(frame)
assert result.lane_mask.shape == frame.shape[:2]
assert result.drivable_mask.shape == frame.shape[:2]
```

Or run `scripts/verify_mask_resize.py` before manual testing.

---

## Predict() call graph (reference)

```
LaneDetectionModule.predict(frame)                    lane_detection.py:143
└─ _run_pipeline(frame)                               lane_detection.py:204
   ├─ preprocessor.preprocess(frame)                  lane_detection.py:207
   ├─ inference_engine.run(frame)                    lane_detection.py:219
   │  └─ YOLOPInferenceEngine._execute_forward        inference.py:350
   │     └─ self._model(input_tensor)  [MCnet]      inference.py:386
   ├─ output_parser.parse(..., frame_shape=...)      lane_detection.py:229
   │  └─ lane_mask @ 640×640 in parsed.lane_lines    output_parser.py:196
   ├─ postprocess_lane_mask [optional]               lane_detection.py:234
   ├─ _resize_masks_to_frame_shape  ★ version 2      lane_detection.py:238
   ├─ geometry_extractor.compute_lane_center         lane_detection.py:253
   ├─ geometry_extractor.compute_vehicle_offset      lane_detection.py:255
   ├─ LaneDetectionResult(lane_mask=resized, ...)   lane_detection.py:266
   └─ _assert_result_masks_match_frame  ★ version 2  lane_detection.py:284
```

★ Steps absent on Colab per current evidence.

---

## Summary action items

| Priority | Action |
|----------|--------|
| P0 | Colab: `git pull` + restart runtime |
| P0 | Colab: assert `LANE_DETECTION_PIPELINE_VERSION >= 2` |
| P0 | Colab: run `python scripts/verify_mask_resize.py` |
| P1 | Colab: read `result.lane_mask`, not `parsed.lane_lines.lane_mask` |
| P1 | Colab: confirm `End-to-end.pth` loaded (tensor key count in logs) |

**Do not treat lane detection geometry requirements as satisfied on Colab until `result.lane_mask.shape == frame.shape[:2]` is proven after kernel restart.**
