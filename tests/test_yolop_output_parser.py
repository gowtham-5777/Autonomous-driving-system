"""Regression tests for :class:`YOLOPOutputParser` frame-shape integration."""

from __future__ import annotations

import numpy as np
import pytest

from src.modules.yolop.output_parser import YOLOPOutputParser


def _synthetic_mcnet_sequence_outputs(
    *,
    mask_height: int = 640,
    mask_width: int = 640,
) -> list:
    """Build list-style YOLOP outputs (no ``original_shape`` metadata)."""
    drivable = np.zeros((2, mask_height, mask_width), dtype=np.float32)
    drivable[1, mask_height // 2 :, :] = 2.0

    lane = np.zeros((2, mask_height, mask_width), dtype=np.float32)
    lane[1, mask_height // 2 :, mask_width // 2 - 30 : mask_width // 2 + 30] = 2.0

    return [None, drivable, lane]


@pytest.fixture
def parser() -> YOLOPOutputParser:
    return YOLOPOutputParser()


@pytest.fixture
def frame_shape() -> tuple[int, int, int]:
    return (720, 1280, 3)


@pytest.fixture
def mcnet_outputs() -> list:
    return _synthetic_mcnet_sequence_outputs()


class TestYOLOPOutputParserFrameShape:
    """Verify vehicle offset depends on a valid original frame shape."""

    def test_parse_without_frame_shape_leaves_vehicle_offset_none(
        self,
        parser: YOLOPOutputParser,
        mcnet_outputs: list,
    ) -> None:
        """Sequence outputs lack metadata; omitting frame_shape breaks offset."""
        parsed = parser.parse(mcnet_outputs)

        assert parsed.lane_center.center_x_at_bottom is not None
        assert parsed.vehicle_offset.offset_pixels is None

    def test_parse_with_frame_shape_computes_lane_center_and_vehicle_offset(
        self,
        parser: YOLOPOutputParser,
        mcnet_outputs: list,
        frame_shape: tuple[int, int, int],
    ) -> None:
        """Supplying frame_shape enables lateral offset from mask geometry."""
        parsed = parser.parse(mcnet_outputs, frame_shape=frame_shape)

        assert parsed.lane_center.center_x_at_bottom is not None
        assert parsed.vehicle_offset.offset_pixels is not None
        assert parsed.vehicle_offset.vehicle_x is not None
        assert parsed.vehicle_offset.lane_center_x is not None
