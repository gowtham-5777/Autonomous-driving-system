# Mask Resize Root Cause

## Colab gate (read this first)

**Local verification PASSED** — `result.lane_mask.shape == frame.shape[:2]` for `frame.shape == (1024, 2048, 3)`.

Colab will keep showing `(640, 640)` until **all** of the following are true:

1. **Sync code** — `git pull` the commit that includes:
   - `src/modules/yolop/mask_resize.py`
   - `src/modules/lane_detection.py` with `LANE_DETECTION_PIPELINE_VERSION = 2`
   - `_resize_masks_to_frame_shape` + `_assert_result_masks_match_frame`
2. **Restart runtime** — Factory reset or Runtime → Restart session (cached `lane_detection` without resize is the usual cause).
3. **Run gate on Colab** before `predict()` on real frames:

```python
import sys
from pathlib import Path
PROJECT_ROOT = Path("/content/...")  # your clone root
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.lane_detection import LANE_DETECTION_PIPELINE_VERSION, LaneDetectionModule
assert LANE_DETECTION_PIPELINE_VERSION >= 2
assert hasattr(LaneDetectionModule, "_resize_masks_to_frame_shape")

# or: !python scripts/verify_mask_resize.py
```

If version is `< 2` or `_resize_masks_to_frame_shape` is missing, **stop** — you are not running the fixed pipeline.

With version 2 active, `predict()` **raises `RuntimeError`** if masks are still 640×640 (no more silent wrong offsets).

---

## Executive summary

`LaneDetectionModule.predict()` was returning `(640, 640)` lane masks on `(1024, 2048, 3)` frames because the resize step either **never executed** (stale/cached module without `_resize_masks_to_frame_shape`) or **could not import** its helper (`resize_mask_to_frame` missing from `postprocess.py` in the runtime tree). Geometry then ran in 640-space while `vehicle_offset` used frame width 2048, producing offsets like **−806**.

---

## `resize_mask_to_frame` — exact definition

**Canonical definition:** `src/modules/yolop/mask_resize.py`

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

**Re-exported from:**

| File | Mechanism |
|------|-----------|
| `src/modules/yolop/postprocess.py` | `from .mask_resize import resize_mask_to_frame` + `__all__` |
| `src/modules/yolop/__init__.py` | `from .mask_resize import resize_mask_to_frame` |

**Pipeline resize (production path):** `LaneDetectionModule._resize_masks_to_frame_shape()` in `src/modules/lane_detection.py` calls **`cv2.resize` directly** (no import dependency) so mask scaling cannot be skipped by a missing helper module.

---

## All references to `resize_mask_to_frame`

| File | Usage |
|------|-------|
| `src/modules/yolop/mask_resize.py` | **Definition** |
| `src/modules/yolop/postprocess.py` | Re-export |
| `src/modules/yolop/__init__.py` | Package export |
| `src/modules/yolop/output_parser.py` | `_align_lane_lines_to_frame()` — parser-internal geometry copy |
| `tests/test_mask_resize_geometry.py` | Import + unit tests |
| `src/modules/lane_detection.py` | **Does not import** — uses inline `cv2.resize` in `_resize_masks_to_frame_shape()` |

---

## Full code trace: `predict()` → `LaneDetectionResult`

```
LaneDetectionModule.predict(frame)                          # lane_detection.py:140
│
├─ _validate_input(frame)                                   # shape (H, W, 3)
│
└─ _run_pipeline(frame)                                      # lane_detection.py:201
    │
    ├─ Step 1: preprocessor.preprocess(frame)
    │           → preprocessed_edges (full frame size)
    │
    ├─ Step 2: inference_engine.run(frame)
    │           → YOLOPInferenceEngine.preprocess: resize frame → 640×640 tensor
    │           → MCnet forward → lane/drivable heads at 640×640
    │           → postprocess dict with original_shape=frame.shape
    │
    ├─ Step 3: output_parser.parse(raw_outputs, frame_shape=frame.shape)
    │           → extract_lane_information: binary lane_mask (640, 640)
    │           → extract_drivable_area: binary drivable_mask (640, 640)
    │           → parser geometry uses resized copy (frame-space) when frame_shape set
    │           → parsed.lane_lines.lane_mask still model resolution (640, 640)
    │
    ├─ Step 4: postprocess_lane_mask(lane_mask) [optional]
    │           → morphological cleanup, still (640, 640)
    │
    ├─ Step 5: _resize_masks_to_frame_shape(lane_mask, drivable_mask, frame)  ★ FIX
    │           → cv2.resize(..., (frame.shape[1], frame.shape[0]))
    │           → lane_mask, drivable_mask become (1024, 2048)
    │           → RuntimeError if shape mismatch
    │
    ├─ Step 6: geometry_extractor.compute_lane_center(lane_mask)   # frame-space
    │           geometry_extractor.compute_vehicle_offset(..., image_width=frame.shape[1])
    │           lane_departure from |offset| vs threshold
    │
    └─ LaneDetectionResult(
           lane_mask=lane_mask,        # resized (1024, 2048)
           drivable_mask=drivable_mask,
           lane_center_x, vehicle_offset, ...
       )
```

### Why `lane_mask` stayed `(640, 640)` before fix

| Check | Broken runtime | Fixed runtime |
|-------|----------------|---------------|
| `_resize_masks_to_frame_shape` exists | Missing (old cached module) | Present |
| Resize called in `_run_pipeline` | No / failed import | Step 5 always runs |
| Return value assigned | N/A | `lane_mask, drivable_mask = ...` |
| Geometry uses resized mask | No — 640-space | Yes — frame-space |
| `LaneDetectionResult.lane_mask` | `(640, 640)` | `(1024, 2048)` |

### Symptom validation (reported failure)

```
frame.shape           = (1024, 2048, 3)
result.lane_mask.shape = (640, 640)
lane_center_x         = 217
vehicle_offset        = -806
```

`vehicle_center_x = 2048 / 2 = 1024`  
`217 - 1024 ≈ -807` → confirms 640-space `lane_center_x` mixed with frame-space vehicle center.

---

## Root cause

1. **Helper lived in the wrong place** — early versions defined `resize_mask_to_frame` only inside `postprocess.py`. Runtimes that synced `lane_detection.py` but not `postprocess.py` (or lacked `mask_resize.py`) hit `ImportError` on `from src.modules.yolop.postprocess import resize_mask_to_frame`.

2. **Cached Python modules** — notebooks/REPL sessions kept an old `LaneDetectionModule` without Step 5 resize, so `predict()` continued returning 640×640 masks.

3. **No hard failure on shape mismatch** — without resize, nothing raised when `lane_mask.shape != frame.shape[:2]`.

---

## Fix applied

| File | Change |
|------|--------|
| `src/modules/yolop/mask_resize.py` | Canonical `resize_mask_to_frame` |
| `src/modules/lane_detection.py` | `LANE_DETECTION_PIPELINE_VERSION=2`; inline `cv2.resize`; `_assert_result_masks_match_frame` |
| `scripts/verify_mask_resize.py` | **New** — local/Colab gate script (exit 0 only when shapes match) |
| `src/modules/yolop/postprocess.py` | Re-exports from `mask_resize` |
| `src/modules/yolop/output_parser.py` | Imports from `mask_resize` for parser geometry |
| `src/modules/yolop/__init__.py` | Public export |
| `tests/test_mask_resize_geometry.py` | Shape + import regression tests |
| `tests/test_lane_detection_pipeline.py` | `result.lane_mask.shape == frame.shape[:2]` |

---

## Before / after output

Frame: `(1024, 2048, 3)`, YOLOP mask `(640, 640)` (centered lane stripe, stub inference)

| Field | Before (broken) | After (fixed) |
|-------|-----------------|---------------|
| `result.lane_mask.shape` | `(640, 640)` | `(1024, 2048)` |
| `result.drivable_mask.shape` | `(640, 640)` | `(1024, 2048)` |
| `lane_center_x` | `217` | `1023.5` |
| `vehicle_offset` | `-806` | `-0.5` |

---

## Verification proof

```bash
python -m pytest tests/test_mask_resize_geometry.py tests/test_lane_detection_pipeline.py -v
# 9 passed
```

```python
import numpy as np
from src.modules.lane_detection import LaneDetectionModule

# after initialize + predict on (1024, 2048, 3) frame:
assert result.lane_mask.shape == frame.shape[:2]      # (1024, 2048)
assert result.drivable_mask.shape == frame.shape[:2]   # (1024, 2048)
```

Verified locally:

```
PASS lane (1024, 2048) drivable (1024, 2048) offset -0.5
postprocess import: True   # from src.modules.yolop.postprocess import resize_mask_to_frame
```

### Import paths (all valid after fix)

```python
from src.modules.yolop.mask_resize import resize_mask_to_frame   # canonical
from src.modules.yolop.postprocess import resize_mask_to_frame   # re-export
from src.modules.yolop import resize_mask_to_frame               # package
```

### Runtime checklist

1. **Restart kernel** / reload modules (`importlib.reload` insufficient if `.py` files changed on disk).
2. Confirm helper exists: `hasattr(LaneDetectionModule, "_resize_masks_to_frame_shape")` → `True`.
3. Confirm file on disk: `src/modules/yolop/mask_resize.py` present.
4. Run `predict()` — log should show: `Masks resized (640, 640) -> (1024, 2048) for frame (1024, 2048)`.
