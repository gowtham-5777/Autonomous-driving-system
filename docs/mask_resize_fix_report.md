# Mask Resize Geometry Fix

## Summary

YOLOP segmentation masks are produced at model input resolution (640×640). Geometry extraction previously operated on those masks while vehicle center used the original frame width (e.g. 2048 px), mixing coordinate spaces and producing incorrect `vehicle_offset` values.

## Files modified

| File | Change |
|------|--------|
| `src/modules/yolop/postprocess.py` | Added `resize_mask_to_frame()` using `cv2.INTER_NEAREST` |
| `src/modules/lane_detection.py` | Resize `lane_mask` and `drivable_mask` to frame size before geometry; compute `lane_departure` from frame-space offset |
| `src/modules/yolop/output_parser.py` | Resize a geometry copy of the lane mask to frame dimensions before `compute_lane_center()` / `compute_vehicle_offset()` |
| `tests/test_mask_resize_geometry.py` | **New** — regression tests for mismatched frame/mask resolution |
| `docs/mask_resize_fix_report.md` | This report |

## Before behavior

Pipeline order:

1. Parse 640×640 masks from YOLOP heads
2. Optional morphological post-processing at 640×640
3. `compute_lane_center()` → mean x of lane pixels in **640-space** (e.g. ~320 px for a centered stripe)
4. `compute_vehicle_offset(image_width=frame.shape[1])` → vehicle center in **original-frame space** (e.g. 1024 px for width 2048)

Example with a centered lane stripe on a 2048 px-wide frame:

| Quantity | Value |
|----------|-------|
| `lane_center_x` | ~320 (640-space) |
| `vehicle_center_x` | 1024 (frame-space) |
| `vehicle_offset` | ~−704 px (incorrect) |

`drivable_mask` and result `lane_mask` also remained at 640×640, misaligned with the input frame for visualization.

## After behavior

Pipeline order:

1. Parse 640×640 masks from YOLOP heads
2. Optional morphological post-processing at model resolution
3. **Resize** `lane_mask` and `drivable_mask` to `(frame_height, frame_width)` via nearest-neighbor interpolation
4. Compute `lane_center_x`, `vehicle_offset`, and `lane_departure` in **frame-space**

Same example after fix:

| Quantity | Value |
|----------|-------|
| `lane_center_x` | ~1024 (frame-space) |
| `vehicle_center_x` | 1024 |
| `vehicle_offset` | ~0 px (correct) |

`YOLOPOutputParser.parse()` applies the same resize internally for geometry when `frame_shape` is supplied, while returned raw masks stay at model resolution until the pipeline resizes them for output.

## Resize implementation

```python
cv2.resize(
    mask,
    (frame_width, frame_height),
    interpolation=cv2.INTER_NEAREST,
)
```

Nearest-neighbor preserves hard binary class boundaries after upscaling.

## Test evidence

```
tests/test_mask_resize_geometry.py::TestMaskResizeGeometry::test_resize_mask_to_frame_scales_dimensions PASSED
tests/test_mask_resize_geometry.py::TestMaskResizeGeometry::test_geometry_valid_when_frame_and_mask_shapes_differ PASSED
tests/test_mask_resize_geometry.py::TestMaskResizeGeometry::test_parser_geometry_uses_resized_masks PASSED
tests/test_yolop_output_parser.py::TestYOLOPOutputParserFrameShape::test_parse_without_frame_shape_leaves_vehicle_offset_none PASSED
tests/test_yolop_output_parser.py::TestYOLOPOutputParserFrameShape::test_parse_with_frame_shape_computes_lane_center_and_vehicle_offset PASSED
tests/test_lane_detection_pipeline.py::TestLaneDetectionPipeline::test_module_initializes_with_weights PASSED
tests/test_lane_detection_pipeline.py::TestLaneDetectionPipeline::test_end_to_end_predict_pipeline PASSED
tests/test_lane_detection_pipeline.py::TestLaneDetectionPipeline::test_predict_from_uninitialized_module_auto_inits PASSED

8 passed in 2.44s
```

Key regression (`test_geometry_valid_when_frame_and_mask_shapes_differ`):

- Input frame: `(720, 2048, 3)`
- Stub masks: `(640, 640)` with centered lane stripe
- Asserts `frame.shape != mask.shape` before resize
- Asserts output masks match frame dimensions
- Asserts `|vehicle_offset| < 80 px` and lane center near image center

Command:

```bash
python -m pytest tests/test_mask_resize_geometry.py tests/test_yolop_output_parser.py tests/test_lane_detection_pipeline.py -v
```
