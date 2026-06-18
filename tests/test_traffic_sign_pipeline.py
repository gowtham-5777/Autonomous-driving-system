"""End-to-end integration tests for :class:`TrafficSignModule`."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.modules.traffic_sign import TrafficSignModule
from src.modules.yolov8_sign.output_schema import TrafficSignDetectionResult

logger = logging.getLogger(__name__)


class TestTrafficSignPipeline:
    """End-to-end integration coverage for the traffic sign module."""

    def test_module_initializes_with_stub_loader(
        self,
        stub_yolov8_sign_model_loader,
        stub_yolov8_sign_inference_engine,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``initialize()`` attaches stub weights and marks the module ready."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = TrafficSignModule(
            model_loader=stub_yolov8_sign_model_loader,
            inference_engine=stub_yolov8_sign_inference_engine,
            device="cpu",
        )
        module.initialize()

        assert module.is_initialized is True
        assert module.model_loader.is_loaded is True
        assert module.inference_engine.is_ready is True

    def test_end_to_end_predict_pipeline(
        self,
        road_frame: np.ndarray,
        traffic_sign_module: TrafficSignModule,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Full pipeline: load image → predict → validate stop sign detection."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = traffic_sign_module
        assert module.is_initialized is True

        result = module.predict(road_frame)

        assert isinstance(result, TrafficSignDetectionResult)
        assert result.raw_status == "stub"
        assert result.frame_shape == (road_frame.shape[0], road_frame.shape[1])
        assert result.summary.total_count >= 1
        assert "stop" in result.summary.count_by_label

        det = result.detections[0]
        assert det.sign_label == "stop"
        assert det.confidence >= 0.5
        assert det.is_regulatory is True
        assert det.bbox.x2 <= road_frame.shape[1]
        assert det.bbox.y2 <= road_frame.shape[0]

        prediction = result.to_prediction_dict()
        assert prediction["total_count"] == result.summary.total_count
        assert len(prediction["detections"]) >= 1

    def test_predict_auto_inits(
        self,
        road_frame: np.ndarray,
        stub_yolov8_sign_model_loader,
        stub_yolov8_sign_inference_engine,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``predict()`` auto-initializes when stub loader is available."""
        caplog.set_level(logging.DEBUG, logger="adas")

        module = TrafficSignModule(
            model_loader=stub_yolov8_sign_model_loader,
            inference_engine=stub_yolov8_sign_inference_engine,
            device="cpu",
        )
        assert module.is_initialized is False

        result = module.predict(road_frame)

        assert module.is_initialized is True
        assert isinstance(result, TrafficSignDetectionResult)
        assert result.raw_status == "stub"

    def test_visualize_returns_annotated_frame(
        self,
        road_frame: np.ndarray,
        traffic_sign_module: TrafficSignModule,
    ) -> None:
        """``visualize()`` returns a uint8 BGR frame with unchanged shape."""
        result = traffic_sign_module.predict(road_frame)
        annotated = traffic_sign_module.visualize(road_frame, result)

        assert annotated.shape == road_frame.shape
        assert annotated.dtype == np.uint8
        assert not np.array_equal(annotated, road_frame)

    def test_only_allowed_sign_classes(
        self,
        road_frame: np.ndarray,
        traffic_sign_module: TrafficSignModule,
    ) -> None:
        """Detections are limited to the seven ADAS sign classes."""
        from src.modules.yolov8_sign.output_schema import ADAS_SIGN_LABELS

        result = traffic_sign_module.predict(road_frame)
        for det in result.detections:
            assert det.sign_label in ADAS_SIGN_LABELS

    def test_active_speed_limit_in_summary(
        self,
        road_frame: np.ndarray,
        stub_yolov8_sign_model_loader,
    ) -> None:
        """Summary picks the minimum visible speed limit (most restrictive)."""
        from src.modules.yolov8_sign.inference import (
            YOLOv8SignInferenceConfig,
            YOLOv8SignInferenceEngine,
        )

        class _SpeedLimitStubEngine(YOLOv8SignInferenceEngine):
            def attach_model(self, model_package: dict) -> None:
                self.model_package = model_package
                self._model = model_package.get("model") or object()

            def run(self, frame: np.ndarray) -> dict:
                self._validate_frame(frame)
                height, width = frame.shape[:2]
                cy = height // 3
                boxes = np.array(
                    [
                        [width // 4 - 30, cy - 30, width // 4 + 30, cy + 30],
                        [width // 2 - 30, cy - 30, width // 2 + 30, cy + 30],
                    ],
                    dtype=np.float32,
                )
                confidences = np.array([0.88, 0.91], dtype=np.float32)
                class_ids = np.array([1, 2], dtype=np.int32)  # 30 and 60 km/h

                return {
                    "boxes_xyxy": boxes,
                    "confidences": confidences,
                    "class_ids": class_ids,
                    "inference_status": "stub",
                    "original_shape": frame.shape,
                    "inference_time_ms": 2.0,
                    "imgsz": self.config.imgsz,
                }

        module = TrafficSignModule(
            model_loader=stub_yolov8_sign_model_loader,
            inference_engine=_SpeedLimitStubEngine(
                config=YOLOv8SignInferenceConfig(device="cpu")
            ),
            device="cpu",
        )
        module.initialize()
        result = module.predict(road_frame)

        assert result.summary.active_speed_limit_kmh == 30
        assert result.summary.count_by_label.get("speed_limit_30") == 1
        assert result.summary.count_by_label.get("speed_limit_60") == 1
