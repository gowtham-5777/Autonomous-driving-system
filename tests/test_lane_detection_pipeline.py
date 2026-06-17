"""End-to-end integration tests for :class:`LaneDetectionModule`.

Validates pipeline wiring (preprocess → inference stub → parse → geometry)
before real YOLOP forward-pass integration is complete.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from src.modules.lane_detection import LaneDetectionModule
from src.modules.yolop.output_schema import LaneDetectionResult

logger = logging.getLogger(__name__)


def _log_pipeline_summary(
    *,
    weights_path: Path,
    frame_shape: tuple[int, ...],
    result: LaneDetectionResult,
    initialized: bool,
) -> None:
    """Emit a detailed summary of the lane detection pipeline run."""
    lane_center = result.lane_center_x
    prediction = result.to_prediction_dict()

    logger.info("=" * 72)
    logger.info("Lane Detection Pipeline — Integration Test Summary")
    logger.info("=" * 72)
    logger.info("Weights path       : %s", weights_path)
    logger.info("Weights exist      : %s", weights_path.is_file())
    logger.info("Module initialized : %s", initialized)
    logger.info("Input frame shape  : %s", frame_shape)
    logger.info("-" * 72)
    logger.info("raw_status         : %s", result.raw_status)
    logger.info("lane_center        : %s", lane_center)
    logger.info("vehicle_offset     : %s", result.vehicle_offset)
    logger.info("lane_departure     : %s", result.lane_departure)
    logger.info("-" * 72)
    logger.info("left_lane          : %s", result.left_lane)
    logger.info("right_lane         : %s", result.right_lane)
    logger.info("vehicle_center_x   : %s", result.vehicle_center_x)
    logger.info("lane_mask present  : %s", result.lane_mask is not None)
    logger.info("drivable_mask      : %s", result.drivable_mask is not None)
    logger.info("preprocessed_edges : %s", result.preprocessed_edges is not None)
    logger.info("prediction dict    : %s", prediction)
    logger.info("=" * 72)

    print("\n--- Lane Detection Pipeline Output ---")
    print(f"raw_status      : {result.raw_status}")
    print(f"lane_center     : {lane_center}")
    print(f"vehicle_offset  : {result.vehicle_offset}")
    print(f"lane_departure  : {result.lane_departure}")
    print("--------------------------------------\n")


class TestLaneDetectionPipeline:
    """End-to-end integration coverage for the lane detection module."""

    def test_module_initializes_with_weights(
        self,
        yolop_weights_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``initialize()`` loads weights and marks the module ready."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = LaneDetectionModule(weights_path=yolop_weights_path, device="cpu")
        logger.info("Created LaneDetectionModule with weights=%s", yolop_weights_path)

        module.initialize()

        assert module.is_initialized is True
        assert module.model_loader.is_loaded is True
        assert module.inference_engine.is_ready is True
        logger.info("Module initialization verified")

    def test_end_to_end_predict_pipeline(
        self,
        road_frame: np.ndarray,
        lane_detection_module: LaneDetectionModule,
        yolop_weights_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Full pipeline: load image → initialize → predict → validate result."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = lane_detection_module
        assert module.is_initialized is True

        logger.info("Running predict() on road frame shape=%s", road_frame.shape)
        result = module.predict(road_frame)

        _log_pipeline_summary(
            weights_path=yolop_weights_path,
            frame_shape=road_frame.shape,
            result=result,
            initialized=module.is_initialized,
        )

        assert isinstance(result, LaneDetectionResult)
        assert isinstance(result.raw_status, str)
        assert result.lane_departure is False
        assert result.preprocessed_edges is not None
        assert result.preprocessed_edges.shape[:2] == road_frame.shape[:2]
        assert result.lane_mask is not None
        assert result.drivable_mask is not None
        assert result.lane_center_x is not None
        assert result.vehicle_offset is not None

        prediction = result.to_prediction_dict()
        assert prediction["lane_center"] == result.lane_center_x
        assert prediction["vehicle_offset"] == result.vehicle_offset
        assert prediction["lane_departure"] is False

        logger.info("End-to-end predict() completed without exception")

    def test_predict_from_uninitialized_module_auto_inits(
        self,
        road_frame: np.ndarray,
        yolop_weights_path: Path,
        stub_inference_engine,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``predict()`` auto-initializes when weights are available."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = LaneDetectionModule(
            weights_path=yolop_weights_path,
            inference_engine=stub_inference_engine,
            device="cpu",
        )
        assert module.is_initialized is False

        result = module.predict(road_frame)

        assert module.is_initialized is True
        assert isinstance(result, LaneDetectionResult)
        assert result.raw_status in {"stub_segmentation", "parsed", "stub"}
        logger.info(
            "Auto-init predict succeeded — raw_status=%s",
            result.raw_status,
        )
