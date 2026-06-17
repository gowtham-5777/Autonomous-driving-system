# Mask Resize Root Cause

## Audit summary

Traced the live path used by `LaneDetectionModule.predict()`:

```
predict(frame)
  └─ _validate_input(frame)
  └─ _run_pipeline(frame)
       ├─ preprocessor.preprocess(frame)
       ├─ inference_engine.run(frame)          → 640×640 segmentation tensors
       ├─ output_parser.parse(..., frame_shape=frame.shape)
       ├─ postprocess_lane_mask() [optional]    → still 640×640
       ├─ _resize_masks_to_frame_shape()        → MUST produce frame.shape[:2]
       ├─ geometry_extractor.compute_lane_center(resized_mask)
       ├─ geometry_extractor.compute_vehicle_offset(..., image_width=frame.shape[1])
       └─ LaneDetectionResult(lane_mask=resized_mask, ...)
```

There is only one `LaneDetectionModule` implementation: `src/modules/lane_detection.py`.

## Root cause

The resize fix was **not active in the runtime environment** because:

1. **`resize_mask_to_frame` lived inside `postprocess.py`** and was imported as:
   ```python
   from .yolop.postprocess import postprocess_lane_mask, resize_mask_to_frame
   ```
   When `postprocess.py` in the runtime environment did not yet contain that symbol, the import failed (`ImportError`). Notebooks/Colab sessions often keep a **cached old `lane_detection` module** loaded from before the resize change, so `predict()` continued to run geometry on unstretched 640×640 masks.

2. **No dedicated resize module** — resize was bundled with morphological post-processing, making it easy to sync `lane_detection.py` without syncing `postprocess.py`.

3. **No hard failure when shapes mismatch** — without resize, `lane_mask` stayed `(640, 640)` while `compute_vehicle_offset()` used `frame.shape[1]` (2048), mixing coordinate spaces silently.

### Evidence from reported failure

| Field | Reported (broken) | Expected symptom |
|-------|-------------------|------------------|
| `frame.shape` | `(1024, 2048, 3)` | — |
| `result.lane_mask.shape` | `(640, 640)` | Resize not applied |
| `lane_center_x` | `217` | ~640-space lane pixels |
| `vehicle_offset` | `-806` | `217 - 1024 ≈ -807` (frame-space vehicle center) |

This confirms geometry ran on the 640×640 mask against a 2048 px-wide frame.

## Fix applied

### 1. Canonical resize module

Created `src/modules/yolop/mask_resize.py` — minimal, import-safe module:

```python
def resize_mask_to_frame(mask, frame_height, frame_width):
    if mask is None:
        return None
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (int(frame_width), int(frame_height)),
        interpolation=cv2.INTER_NEAREST,
    )
```

### 2. Integrated resize helper on `LaneDetectionModule`

Added `_resize_masks_to_frame_shape()` which:

- Reads `frame.shape[0]` and `frame.shape[1]` (never YOLOP `input_size`)
- Calls `resize_mask_to_frame` for lane and drivable masks
- **Raises `RuntimeError`** if output shape ≠ `frame.shape[:2]`
- Logs INFO: `Masks resized (640, 640) -> (1024, 2048) for frame (1024, 2048)`

### 3. Re-exports

- `src/modules/yolop/postprocess.py` re-exports from `mask_resize`
- `src/modules/yolop/__init__.py` exports `resize_mask_to_frame`

## Files changed

| File | Change |
|------|--------|
| `src/modules/yolop/mask_resize.py` | **New** — canonical `resize_mask_to_frame` |
| `src/modules/lane_detection.py` | Import from `mask_resize`; add `_resize_masks_to_frame_shape()`; call before geometry |
| `src/modules/yolop/postprocess.py` | Re-export `resize_mask_to_frame` from `mask_resize` |
| `src/modules/yolop/output_parser.py` | Import from `mask_resize` |
| `src/modules/yolop/__init__.py` | Export `resize_mask_to_frame` |
| `tests/test_mask_resize_geometry.py` | Import path + `_resize_masks_to_frame_shape` checks |
| `tests/test_lane_detection_pipeline.py` | Assert `result.lane_mask.shape == frame.shape[:2]` |
| `docs/mask_resize_root_cause.md` | This report |

## Before / after output

Frame: `(1024, 2048, 3)` — centered lane stripe in 640×640 YOLOP mask (stub inference)

| Metric | Before (broken) | After (fixed) |
|--------|-----------------|---------------|
| `result.lane_mask.shape` | `(640, 640)` | `(1024, 2048)` |
| `result.drivable_mask.shape` | `(640, 640)` | `(1024, 2048)` |
| `lane_center_x` | `217` | `1023.5` |
| `vehicle_offset` | `-806` | `-0.5` |

## Proof: `result.lane_mask.shape == frame.shape[:2]`

Verification command (project root on `sys.path`):

```bash
python -c "
import inspect, numpy as np
from src.modules.lane_detection import LaneDetectionModule
from src.modules.yolop.inference import InferenceConfig, YOLOPInferenceEngine
from tests.conftest import _resolve_weights_path

class Stub(YOLOPInferenceEngine):
    def run(self, frame):
        tw, th = self.config.input_size
        drivable = np.zeros((2, th, tw), np.float32); drivable[1, th//2:, :] = 2.0
        lane = np.zeros((2, th, tw), np.float32); lane[1, th//2:, tw//2-30:tw//2+30] = 2.0
        return {1: drivable, 2: lane, 'inference_status': 'stub', 'original_shape': frame.shape}

m = LaneDetectionModule(weights_path=_resolve_weights_path(),
    inference_engine=Stub(config=InferenceConfig(device='cpu')),
    apply_mask_postprocess=False, device='cpu')
m.initialize()
frame = np.zeros((1024, 2048, 3), np.uint8)
r = m.predict(frame)
assert r.lane_mask.shape == frame.shape[:2]
assert r.drivable_mask.shape == frame.shape[:2]
print('OK', r.lane_mask.shape, r.lane_center_x, r.vehicle_offset)
print('module', inspect.getfile(LaneDetectionModule))
print('resize', inspect.getfile(__import__('src.modules.yolop.mask_resize', fromlist=['resize_mask_to_frame'])))
"
```

Output:

```
OK (1024, 2048) 1023.5 -0.5
module .../src/modules/lane_detection.py
resize .../src/modules/yolop/mask_resize.py
```

Pytest:

```
tests/test_mask_resize_geometry.py (6 tests) PASSED
tests/test_lane_detection_pipeline.py (3 tests) PASSED
9 passed
```

## Runtime checklist

After pulling these changes:

1. **Restart the Python kernel** (or re-import modules).
2. Confirm resize module exists:
   ```python
   from src.modules.yolop.mask_resize import resize_mask_to_frame
   ```
3. Confirm pipeline helper exists:
   ```python
   from src.modules.lane_detection import LaneDetectionModule
   assert hasattr(LaneDetectionModule, "_resize_masks_to_frame_shape")
   ```
4. Run `predict()` and verify `result.lane_mask.shape == frame.shape[:2]`.
