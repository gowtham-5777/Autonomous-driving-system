"""End-to-end integration tests for :class:`TrafficSignalModule`."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.modules.traffic_signal import TrafficSignalModule
from src.modules.yolov8_signal.output_schema import TrafficSignalDetectionResult

logger = logging.getLogger(__name__)


class TestTrafficSignalPipeline:
    """End-to-end integration coverage for the traffic signal module."""

    def test_module_initializes_with_stub_loader(
        self,
        stub_yolov8_signal_model_loader,
        stub_yolov8_signal_inference_engine,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``initialize()`` attaches stub weights and marks the module ready."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = TrafficSignalModule(
            model_loader=stub_yolov8_signal_model_loader,
            inference_engine=stub_yolov8_signal_inference_engine,
            device="cpu",
        )
        module.initialize()

        assert module.is_initialized is True
        assert module.model_loader.is_loaded is True
        assert module.inference_engine.is_ready is True

    def test_end_to_end_predict_pipeline(
        self,
        road_frame: np.ndarray,
        traffic_signal_module: TrafficSignalModule,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Full pipeline: load image → predict → validate red light detection."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = traffic_signal_module
        assert module.is_initialized is True

        result = module.predict(road_frame)

        assert isinstance(result, TrafficSignalDetectionResult)
        assert result.raw_status == "stub"
        assert result.frame_shape == (road_frame.shape[0], road_frame.shape[1])
        assert result.summary.total_count >= 1
        assert "red_light" in result.summary.count_by_label
        assert result.summary.dominant_state == "red_light"
        assert result.summary.has_stop_state is True

        det = result.detections[0]
        assert det.signal_label == "red_light"
        assert det.confidence >= 0.5
        assert det.is_stop_state is True
        assert det.bbox.x2 <= road_frame.shape[1]
        assert det.bbox.y2 <= road_frame.shape[0]

        prediction = result.to_prediction_dict()
        assert prediction["total_count"] == result.summary.total_count
        assert len(prediction["detections"]) >= 1

    def test_predict_auto_inits(
        self,
        road_frame: np.ndarray,
        stub_yolov8_signal_model_loader,
        stub_yolov8_signal_inference_engine,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``predict()`` auto-initializes when stub loader is available."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = TrafficSignalModule(
            model_loader=stub_yolov8_signal_model_loader,
            inference_engine=stub_yolov8_signal_inference_engine,
            device="cpu",
        )
        assert module.is_initialized is False

        result = module.predict(road_frame)

        assert module.is_initialized is True
        assert isinstance(result, TrafficSignalDetectionResult)
        assert result.raw_status == "stub"

    def test_visualize_returns_annotated_frame(
        self,
        road_frame: np.ndarray,
        traffic_signal_module: TrafficSignalModule,
    ) -> None:
        """``visualize()`` returns a uint8 BGR frame with unchanged shape."""
        result = traffic_signal_module.predict(road_frame)
        annotated = traffic_signal_module.visualize(road_frame, result)

        assert annotated.shape == road_frame.shape
        assert annotated.dtype == np.uint8
        assert not np.array_equal(annotated, road_frame)

    def test_only_allowed_signal_classes(
        self,
        road_frame: np.ndarray,
        traffic_signal_module: TrafficSignalModule,
    ) -> None:
        """Detections are limited to the three ADAS signal classes."""
        from src.modules.yolov8_signal.output_schema import ADAS_SIGNAL_LABELS

        result = traffic_signal_module.predict(road_frame)
        for det in result.detections:
            assert det.signal_label in ADAS_SIGNAL_LABELS

    def test_controlling_signal_in_summary(
        self,
        road_frame: np.ndarray,
        stub_yolov8_signal_model_loader,
    ) -> None:
        """Summary selects controlling signal from upper-frame candidates."""
        from src.modules.yolov8_signal.inference import (
            YOLOv8SignalInferenceConfig,
            YOLOv8SignalInferenceEngine,
        )

        class _MultiLightStubEngine(YOLOv8SignalInferenceEngine):
            def attach_model(self, model_package: dict) -> None:
                self.model_package = model_package
                self._model = model_package.get("model") or object()

            def run(self, frame: np.ndarray) -> dict:
                self._validate_frame(frame)
                height, width = frame.shape[:2]
                upper_y = int(height * 0.15)
                lower_y = int(height * 0.75)
                boxes = np.array(
                    [
                        [width // 2 - 20, upper_y - 30, width // 2 + 20, upper_y + 30],
                        [width // 2 - 20, lower_y - 30, width // 2 + 20, lower_y + 30],
                    ],
                    dtype=np.float32,
                )
                confidences = np.array([0.85, 0.95], dtype=np.float32)
                class_ids = np.array([2, 0], dtype=np.int32)  # green upper, red lower

                return {
                    "boxes_xyxy": boxes,
                    "confidences": confidences,
                    "class_ids": class_ids,
                    "inference_status": "stub",
                    "original_shape": frame.shape,
                    "inference_time_ms": 2.0,
                    "imgsz": self.config.imgsz,
                }

        module = TrafficSignalModule(
            model_loader=stub_yolov8_signal_model_loader,
            inference_engine=_MultiLightStubEngine(
                config=YOLOv8SignalInferenceConfig(device="cpu")
            ),
            device="cpu",
        )
        module.initialize()
        result = module.predict(road_frame)

        assert result.summary.controlling_signal is not None
        assert result.summary.controlling_signal.signal_label == "green_light"

    def test_dominant_state_priority(
        self,
        road_frame: np.ndarray,
        stub_yolov8_signal_model_loader,
    ) -> None:
        """Red wins dominant state when red and green are both detected."""
        from src.modules.yolov8_signal.inference import (
            YOLOv8SignalInferenceConfig,
            YOLOv8SignalInferenceEngine,
        )

        class _ConflictStubEngine(YOLOv8SignalInferenceEngine):
            def attach_model(self, model_package: dict) -> None:
                self.model_package = model_package
                self._model = model_package.get("model") or object()

            def run(self, frame: np.ndarray) -> dict:
                self._validate_frame(frame)
                height, width = frame.shape[:2]
                cy = int(height * 0.2)
                boxes = np.array(
                    [
                        [width // 3 - 20, cy - 30, width // 3 + 20, cy + 30],
                        [2 * width // 3 - 20, cy - 30, 2 * width // 3 + 20, cy + 30],
                    ],
                    dtype=np.float32,
                )
                confidences = np.array([0.88, 0.92], dtype=np.float32)
                class_ids = np.array([2, 0], dtype=np.int32)  # green, red

                return {
                    "boxes_xyxy": boxes,
                    "confidences": confidences,
                    "class_ids": class_ids,
                    "inference_status": "stub",
                    "original_shape": frame.shape,
                    "inference_time_ms": 2.0,
                    "imgsz": self.config.imgsz,
                }

        module = TrafficSignalModule(
            model_loader=stub_yolov8_signal_model_loader,
            inference_engine=_ConflictStubEngine(
                config=YOLOv8SignalInferenceConfig(device="cpu")
            ),
            device="cpu",
        )
        module.initialize()
        result = module.predict(road_frame)

        assert result.summary.has_stop_state is True
        assert result.summary.has_proceed_state is True
        assert result.summary.dominant_state == "red_light"
