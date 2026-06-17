# Mask Resize Verification

## Summary

Verified and hardened mask resize integration so `LaneDetectionModule.predict()` returns frame-sized masks and frame-space geometry when YOLOP outputs 640×640 segmentation heads on larger input frames (e.g. 1024×2048).

## Root cause of reported failure

Two issues explained the observed symptoms:

1. **`resize_mask_to_frame` was not exported** from `src/modules/yolop/__init__.py`, so `from src.modules.yolop.postprocess import resize_mask_to_frame` could fail in environments loading a stale or partial package tree.
2. **Early-return optimization** in the previous implementation skipped `cv2.resize` when target dimensions matched mask dimensions. If resize was accidentally called with YOLOP `input_size` `(640, 640)` instead of `frame.shape[:2]`, masks stayed `(640, 640)` with no error.

## Files modified

| File | Change |
|------|--------|
| `src/modules/yolop/postprocess.py` | Moved `resize_mask_to_frame` to module top; always calls `cv2.resize` with `astype(np.uint8)`; added `__all__` export |
| `src/modules/yolop/__init__.py` | Re-export `resize_mask_to_frame` |
| `src/modules/lane_detection.py` | Keyword resize args from `frame.shape[0]` / `frame.shape[1]`; post-resize shape validation; debug logging |
| `tests/test_mask_resize_geometry.py` | Added package export test; asserts exact `(1024, 2048)` output masks |
| `docs/mask_resize_verification.md` | This report |

## Reference: `resize_mask_to_frame`

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

Import paths that work after this change:

```python
from src.modules.yolop.postprocess import resize_mask_to_frame
from src.modules.yolop import resize_mask_to_frame
```

## Before / after mask shapes

Test frame: `(1024, 2048, 3)`  
YOLOP model mask: `(640, 640)`

| Stage | Before fix | After fix |
|-------|------------|-----------|
| Parsed lane mask | `(640, 640)` | `(640, 640)` |
| After `_run_pipeline` resize | `(640, 640)` | `(1024, 2048)` |
| `LaneDetectionResult.lane_mask` | `(640, 640)` | `(1024, 2048)` |
| `LaneDetectionResult.drivable_mask` | `(640, 640)` | `(1024, 2048)` |

## Before / after geometry (centered lane stripe)

Synthetic centered lane stripe on 640×640 mask, frame width 2048 px:

| Metric | Before (640-space geometry) | After (resized mask geometry) |
|--------|----------------------------|-------------------------------|
| `lane_center_x` | 319.5 px | 1023.5 px |
| `vehicle_center_x` | 1024.0 px | 1024.0 px |
| `vehicle_offset` | **−704.5 px** | **−0.5 px** |

`LaneDetectionModule.predict()` on `(1024, 2048, 3)` frame (stub inference):

| Field | Value |
|-------|-------|
| `lane_mask.shape` | `(1024, 2048)` |
| `lane_center_x` | 1023.5 |
| `vehicle_offset` | −0.5 |

## Pipeline integration (`_run_pipeline`)

Resize runs **after** optional morphological post-processing and **before** geometry:

1. `output_parser.parse(..., frame_shape=frame.shape)`
2. `postprocess_lane_mask()` (optional, at 640×640)
3. `resize_mask_to_frame(..., frame_height=frame.shape[0], frame_width=frame.shape[1])`
4. `compute_lane_center()` / `compute_vehicle_offset()` / `lane_departure`

Target dimensions come from `frame.shape`, not `InferenceConfig.input_size`.

## Test evidence

```
tests/test_mask_resize_geometry.py::TestMaskResizeGeometry::test_resize_mask_to_frame_is_exported_from_yolop_package PASSED
tests/test_mask_resize_geometry.py::TestMaskResizeGeometry::test_resize_mask_to_frame_scales_dimensions PASSED
tests/test_mask_resize_geometry.py::TestMaskResizeGeometry::test_geometry_valid_when_frame_and_mask_shapes_differ PASSED
tests/test_mask_resize_geometry.py::TestMaskResizeGeometry::test_parser_geometry_uses_resized_masks PASSED
tests/test_lane_detection_pipeline.py (3 tests) PASSED

7 passed in 4.32s
```

Command:

```bash
python -m pytest tests/test_mask_resize_geometry.py tests/test_lane_detection_pipeline.py -v
```

Quick manual check:

```python
from src.modules.yolop.postprocess import resize_mask_to_frame
import numpy as np

mask = np.zeros((640, 640), dtype=np.uint8)
resized = resize_mask_to_frame(mask, frame_height=1024, frame_width=2048)
assert resized.shape == (1024, 2048)
```
