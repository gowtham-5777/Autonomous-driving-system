"""End-to-end integration tests for :class:`VehicleDetectionModule`."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.modules.vehicle_detection import VehicleDetectionModule
from src.modules.yolov8.output_schema import VehicleDetectionResult

logger = logging.getLogger(__name__)


class TestVehicleDetectionPipeline:
    """End-to-end integration coverage for the vehicle detection module."""

    def test_module_initializes_with_stub_loader(
        self,
        stub_yolov8_model_loader,
        stub_yolov8_inference_engine,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``initialize()`` attaches stub weights and marks the module ready."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = VehicleDetectionModule(
            model_loader=stub_yolov8_model_loader,
            inference_engine=stub_yolov8_inference_engine,
            device="cpu",
        )
        module.initialize()

        assert module.is_initialized is True
        assert module.model_loader.is_loaded is True
        assert module.inference_engine.is_ready is True

    def test_end_to_end_predict_pipeline(
        self,
        road_frame: np.ndarray,
        vehicle_detection_module: VehicleDetectionModule,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Full pipeline: load image → predict → validate detections."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = vehicle_detection_module
        assert module.is_initialized is True

        result = module.predict(road_frame)

        assert isinstance(result, VehicleDetectionResult)
        assert result.raw_status == "stub"
        assert result.frame_shape == (road_frame.shape[0], road_frame.shape[1])
        assert result.summary.total_count >= 1
        assert "car" in result.summary.count_by_label

        det = result.detections[0]
        assert det.label == "car"
        assert det.confidence >= 0.5
        assert det.bbox.x2 <= road_frame.shape[1]
        assert det.bbox.y2 <= road_frame.shape[0]

        prediction = result.to_prediction_dict()
        assert prediction["total_count"] == result.summary.total_count
        assert len(prediction["detections"]) >= 1

    def test_predict_from_uninitialized_module_auto_inits(
        self,
        road_frame: np.ndarray,
        stub_yolov8_model_loader,
        stub_yolov8_inference_engine,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``predict()`` auto-initializes when stub loader is available."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = VehicleDetectionModule(
            model_loader=stub_yolov8_model_loader,
            inference_engine=stub_yolov8_inference_engine,
            device="cpu",
        )
        assert module.is_initialized is False

        result = module.predict(road_frame)

        assert module.is_initialized is True
        assert isinstance(result, VehicleDetectionResult)
        assert result.raw_status == "stub"

    def test_visualize_returns_annotated_frame(
        self,
        road_frame: np.ndarray,
        vehicle_detection_module: VehicleDetectionModule,
    ) -> None:
        """``visualize()`` returns a uint8 BGR frame with unchanged shape."""
        result = vehicle_detection_module.predict(road_frame)
        annotated = vehicle_detection_module.visualize(road_frame, result)

        assert annotated.shape == road_frame.shape
        assert annotated.dtype == np.uint8
        assert not np.array_equal(annotated, road_frame)

    def test_only_allowed_classes_in_output(
        self,
        road_frame: np.ndarray,
        vehicle_detection_module: VehicleDetectionModule,
    ) -> None:
        """Detections are limited to the six ADAS road-user classes."""
        from src.modules.yolov8.output_schema import ADAS_VEHICLE_LABELS

        result = vehicle_detection_module.predict(road_frame)
        for det in result.detections:
            assert det.label in ADAS_VEHICLE_LABELS
