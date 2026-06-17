"""Regression tests for mask resize before lane geometry extraction."""

from __future__ import annotations

import numpy as np
import pytest

from src.modules.lane_detection import LaneDetectionModule
from src.modules.yolop.inference import InferenceConfig, YOLOPInferenceEngine
from src.modules.yolop.postprocess import resize_mask_to_frame


def _synthetic_mcnet_sequence_outputs(
    *,
    mask_height: int = 640,
    mask_width: int = 640,
) -> list:
    """YOLOP list outputs with a centered lane stripe at model resolution."""
    drivable = np.zeros((2, mask_height, mask_width), dtype=np.float32)
    drivable[1, mask_height // 2 :, :] = 2.0

    lane = np.zeros((2, mask_height, mask_width), dtype=np.float32)
    lane[1, mask_height // 2 :, mask_width // 2 - 30 : mask_width // 2 + 30] = 2.0

    return [None, drivable, lane]


class _WideFrameStubInferenceEngine(YOLOPInferenceEngine):
    """Returns 640x640 masks regardless of input frame size."""

    def run(self, frame: np.ndarray) -> dict:
        if not self.is_ready:
            return self.empty_output()

        self._validate_frame(frame)
        target_width, target_height = self.config.input_size
        outputs = _synthetic_mcnet_sequence_outputs(
            mask_height=target_height,
            mask_width=target_width,
        )

        return {
            1: outputs[1],
            2: outputs[2],
            "inference_status": "stub_segmentation",
            "original_shape": frame.shape,
            "input_size": self.config.input_size,
            "confidence_threshold": self.config.confidence_threshold,
        }


@pytest.fixture
def wide_frame() -> np.ndarray:
    """Wide frame whose width differs from the 640px YOLOP mask."""
    return np.zeros((720, 2048, 3), dtype=np.uint8)


@pytest.fixture
def wide_lane_module(
    yolop_weights_path,
    wide_frame: np.ndarray,
) -> LaneDetectionModule:
    engine = _WideFrameStubInferenceEngine(config=InferenceConfig(device="cpu"))
    module = LaneDetectionModule(
        weights_path=yolop_weights_path,
        inference_engine=engine,
        device="cpu",
        apply_mask_postprocess=False,
    )
    module.initialize()
    return module


class TestMaskResizeGeometry:
    """Geometry must use frame-space coordinates, not model-resolution masks."""

    def test_resize_mask_to_frame_scales_dimensions(self) -> None:
        mask = np.zeros((640, 640), dtype=np.uint8)
        mask[320:, 310:330] = 255

        resized = resize_mask_to_frame(mask, frame_height=720, frame_width=2048)

        assert resized is not None
        assert resized.shape == (720, 2048)

    def test_geometry_valid_when_frame_and_mask_shapes_differ(
        self,
        wide_frame: np.ndarray,
        wide_lane_module: LaneDetectionModule,
    ) -> None:
        """Centered 640px lane stripe should map near frame center after resize."""
        assert wide_frame.shape[:2] != (640, 640)

        result = wide_lane_module.predict(wide_frame)

        frame_width = wide_frame.shape[1]
        expected_vehicle_center = frame_width / 2.0

        assert result.lane_mask is not None
        assert result.lane_mask.shape[:2] == wide_frame.shape[:2]
        assert result.lane_center_x is not None
        assert result.vehicle_offset is not None
        assert result.vehicle_center_x == expected_vehicle_center

        # Without resize, lane center stays near 320 while vehicle center is 1024.
        assert abs(result.lane_center_x - expected_vehicle_center) < 80.0
        assert abs(result.vehicle_offset) < 80.0

    def test_parser_geometry_uses_resized_masks(self) -> None:
        """Parser offset must be near zero when lane stripe is centered in 640 space."""
        from src.modules.yolop.output_parser import YOLOPOutputParser

        parser = YOLOPOutputParser()
        frame_shape = (720, 2048, 3)
        parsed = parser.parse(
            _synthetic_mcnet_sequence_outputs(),
            frame_shape=frame_shape,
        )

        assert parsed.lane_lines.lane_mask.shape == (640, 640)
        assert parsed.lane_center.center_x_at_bottom is not None
        assert parsed.vehicle_offset.offset_pixels is not None
        assert abs(parsed.vehicle_offset.offset_pixels) < 80.0
