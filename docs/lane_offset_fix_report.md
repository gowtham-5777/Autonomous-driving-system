# Lane Offset Parser Integration Fix

## Summary

`YOLOPOutputParser.parse()` needs the original image dimensions to compute `vehicle_offset`. Lane center is derived only from the lane mask, so it can be non-`None` even when frame width resolves to zero. This fix confirms the production caller passes `frame_shape`, adds regression tests, and tightens pipeline assertions.

## Files changed

| File | Change |
|------|--------|
| `src/modules/lane_detection.py` | **No edit required** — already calls `parse(raw_outputs, frame_shape=frame.shape)` at line 225 |
| `tests/test_yolop_output_parser.py` | **New** — regression tests for `parse()` with and without `frame_shape` |
| `tests/test_lane_detection_pipeline.py` | Assert `lane_center_x` and `vehicle_offset` are non-`None` in end-to-end test |
| `docs/lane_offset_fix_report.md` | This report |

## Call-site audit

Grep for `YOLOPOutputParser.parse(` / `output_parser.parse(` found **one** production caller:

```225:225:src/modules/lane_detection.py
        parsed = self.output_parser.parse(raw_outputs, frame_shape=frame.shape)
```

No notebooks, tests, or other modules invoked `parse()` before this change.

## Previous behavior

When `parse()` was called without `frame_shape` on **list-style** MCnet outputs (indices `1` = drivable, `2` = lane):

1. `_resolve_frame_shape()` found no explicit argument and no `original_shape` key (lists are not mappings).
2. Fallback shape became `(0, 0, 3)`.
3. `compute_lane_center()` still succeeded from the lane mask → `lane_center.center_x_at_bottom != None`.
4. `compute_vehicle_offset()` received `frame_width=0` → `vehicle_offset.offset_pixels = None`.

Dict-style outputs from `YOLOPInferenceEngine.postprocess()` include `original_shape`, so offset could work without the explicit argument — but list-style raw tensors (direct MCnet forward output) do not.

## New behavior

- **Production path:** `LaneDetectionModule._run_pipeline()` passes `frame_shape=frame.shape`, so `resolved_shape` matches the input BGR frame and `vehicle_offset` is computed when lane pixels exist.
- **Tests:** Regression coverage documents the failure mode (no `frame_shape` on sequence outputs) and the fix (supply `frame_shape=(H, W, C)`).
- **Pipeline test:** End-to-end `predict()` now asserts both `lane_center_x` and `vehicle_offset` are populated.

## Root cause (parser)

```479:492:src/modules/yolop/output_parser.py
    def _resolve_frame_shape(
        raw_outputs: YOLOPRawOutput,
        frame_shape: FrameShape | None,
    ) -> FrameShape:
        """Resolve original frame shape from args or raw output metadata."""
        if frame_shape is not None:
            return frame_shape

        if isinstance(raw_outputs, Mapping):
            original_shape = raw_outputs.get("original_shape")
            if original_shape is not None:
                return tuple(original_shape)

        return (0, 0, 3)
```

`compute_vehicle_offset()` rejects `frame_width <= 0`, which is why offset stays `None` when shape resolution fails.

## Test results

```
tests/test_yolop_output_parser.py::TestYOLOPOutputParserFrameShape::test_parse_without_frame_shape_leaves_vehicle_offset_none PASSED
tests/test_yolop_output_parser.py::TestYOLOPOutputParserFrameShape::test_parse_with_frame_shape_computes_lane_center_and_vehicle_offset PASSED
tests/test_lane_detection_pipeline.py::TestLaneDetectionPipeline::test_module_initializes_with_weights PASSED
tests/test_lane_detection_pipeline.py::TestLaneDetectionPipeline::test_end_to_end_predict_pipeline PASSED
tests/test_lane_detection_pipeline.py::TestLaneDetectionPipeline::test_predict_from_uninitialized_module_auto_inits PASSED

5 passed in 4.41s
```

Command:

```bash
python -m pytest tests/test_yolop_output_parser.py tests/test_lane_detection_pipeline.py -v
```

## Notes

- `LaneDetectionModule` still recomputes geometry in step 5 (after optional mask post-processing) using `frame.shape[1]` directly. That path was already correct; the gap was isolated to direct `parse()` usage without `frame_shape` on sequence outputs.
- For defense in depth, callers should always pass `frame_shape=frame.shape` even when `raw_outputs` is a dict that may carry `original_shape`.
