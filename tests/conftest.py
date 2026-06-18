"""Shared pytest fixtures for ADAS integration tests."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ROAD_IMAGE_PATH = FIXTURES_DIR / "road_sample.jpg"
STUB_WEIGHTS_PATH = FIXTURES_DIR / "stub_yolop_weights.pth"

# Ensure ``import src...`` resolves when running pytest from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def configure_test_logging() -> None:
    """Enable detailed logging for ADAS modules during test runs."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    for logger_name in (
        "adas",
        "adas.modules",
        "adas.modules.lane_detection",
        "adas.modules.yolop",
        "adas.preprocessing",
        "src",
    ):
        logging.getLogger(logger_name).setLevel(logging.DEBUG)


def _create_road_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """Build a synthetic forward-facing road scene (BGR, uint8)."""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    horizon = height // 2
    for row in range(horizon):
        shade = max(90, 210 - row // 4)
        image[row, :] = (shade, shade - 10, shade - 30)

    image[horizon:, :] = (70, 70, 70)

    left_bottom = (width // 2 - 120, height - 1)
    left_top = (width // 2 - 40, horizon + 40)
    right_bottom = (width // 2 + 120, height - 1)
    right_top = (width // 2 + 40, horizon + 40)

    cv2.fillPoly(
        image,
        [np.array([left_bottom, right_bottom, right_top, left_top], dtype=np.int32)],
        (55, 55, 55),
    )
    cv2.line(image, left_bottom, left_top, (255, 255, 255), thickness=10)
    cv2.line(image, right_bottom, right_top, (255, 255, 255), thickness=10)
    cv2.line(
        image,
        (width // 2, height - 1),
        (width // 2, horizon + 60),
        (220, 220, 0),
        thickness=4,
    )

    return image


def _ensure_road_image() -> Path:
    """Load or create the road sample image used by integration tests."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    if ROAD_IMAGE_PATH.is_file():
        return ROAD_IMAGE_PATH

    road_image = _create_road_image()
    if not cv2.imwrite(str(ROAD_IMAGE_PATH), road_image):
        raise RuntimeError(f"Failed to write road fixture to {ROAD_IMAGE_PATH}")

    return ROAD_IMAGE_PATH


def _ensure_stub_weights() -> Path:
    """Create a minimal valid YOLOP checkpoint when real weights are unavailable."""
    torch = pytest.importorskip("torch")

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    if STUB_WEIGHTS_PATH.is_file():
        return STUB_WEIGHTS_PATH

    checkpoint = {
        "state_dict": {
            "stub.conv.weight": torch.zeros(3, 3, 3, 3),
            "stub.conv.bias": torch.zeros(3),
        },
        "epoch": 0,
    }
    torch.save(checkpoint, STUB_WEIGHTS_PATH)
    return STUB_WEIGHTS_PATH


def _resolve_weights_path() -> Path:
    """Prefer real YOLOP weights; fall back to a minimal stub checkpoint."""
    from src.utils.model_paths import get_yolop_weights_path

    configured = get_yolop_weights_path()
    if configured.is_file():
        return configured

    return _ensure_stub_weights()


@pytest.fixture(scope="session")
def road_image_path() -> Path:
    """Filesystem path to a road sample image."""
    return _ensure_road_image()


@pytest.fixture
def road_frame(road_image_path: Path) -> np.ndarray:
    """BGR road image loaded from the test fixture."""
    frame = cv2.imread(str(road_image_path), cv2.IMREAD_COLOR)
    if frame is None:
        pytest.fail(f"Could not load road image: {road_image_path}")

    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    return frame


@pytest.fixture
def yolop_weights_path() -> Path:
    """Resolved YOLOP checkpoint path (real weights or stub)."""
    return _resolve_weights_path()


@pytest.fixture
def stub_inference_engine():
    """Inference engine that returns synthetic segmentation heads for pipeline tests."""
    from src.modules.yolop.inference import InferenceConfig, YOLOPInferenceEngine

    class _StubYOLOPInferenceEngine(YOLOPInferenceEngine):
        """Stub forward pass with fake drivable/lane tensors until YOLOP is integrated."""

        def run(self, frame: np.ndarray) -> dict:
            if not self.is_ready:
                return self.empty_output()

            self._validate_frame(frame)
            target_width, target_height = self.config.input_size

            drivable = np.zeros((2, target_height, target_width), dtype=np.float32)
            drivable[1, target_height // 2 :, :] = 2.0

            lane = np.zeros((2, target_height, target_width), dtype=np.float32)
            lane[1, target_height // 2 :, target_width // 2 - 30 : target_width // 2 + 30] = 2.0

            logger = logging.getLogger("tests.stub_inference")
            logger.info(
                "Stub inference returning synthetic masks — shape=(2, %d, %d)",
                target_height,
                target_width,
            )

            return {
                1: drivable,
                2: lane,
                "inference_status": "stub_segmentation",
                "original_shape": frame.shape,
                "input_size": self.config.input_size,
                "confidence_threshold": self.config.confidence_threshold,
            }

    return _StubYOLOPInferenceEngine(config=InferenceConfig(device="cpu"))


@pytest.fixture
def lane_detection_module(yolop_weights_path: Path, stub_inference_engine):
    """Initialized :class:`LaneDetectionModule` ready for inference."""
    from src.modules.lane_detection import LaneDetectionModule

    module = LaneDetectionModule(
        weights_path=yolop_weights_path,
        inference_engine=stub_inference_engine,
        device="cpu",
    )
    module.initialize()
    return module


@pytest.fixture
def stub_yolov8_inference_engine():
    """Inference engine that returns synthetic detections without Ultralytics."""
    from src.modules.yolov8.inference import YOLOv8InferenceConfig, YOLOv8InferenceEngine

    class _StubYOLOv8InferenceEngine(YOLOv8InferenceEngine):
        """Stub forward pass with fake boxes for pipeline tests."""

        def attach_model(self, model_package: dict) -> None:
            self.model_package = model_package
            self._model = model_package.get("model") or object()

        def run(self, frame: np.ndarray) -> dict:
            if not self.is_ready:
                return self.empty_output(original_shape=frame.shape)

            self._validate_frame(frame)
            height, width = frame.shape[:2]
            cx = width // 2
            cy = height - 80

            boxes = np.array(
                [[cx - 70, cy - 50, cx + 70, cy + 30]],
                dtype=np.float32,
            )
            confidences = np.array([0.91], dtype=np.float32)
            class_ids = np.array([2], dtype=np.int32)  # car

            return {
                "boxes_xyxy": boxes,
                "confidences": confidences,
                "class_ids": class_ids,
                "inference_status": "stub",
                "original_shape": frame.shape,
                "inference_time_ms": 2.5,
                "imgsz": self.config.imgsz,
            }

    return _StubYOLOv8InferenceEngine(config=YOLOv8InferenceConfig(device="cpu"))


@pytest.fixture
def stub_yolov8_model_loader():
    """Model loader that skips Ultralytics for unit/integration tests."""
    from src.modules.yolov8.model_loader import WeightsMetadata, YOLOv8ModelLoader

    class _StubYOLOv8ModelLoader(YOLOv8ModelLoader):
        def load_model(self) -> WeightsMetadata:
            self._resolved_source = "stub://yolov8s.pt"
            self._model = object()
            self._metadata = WeightsMetadata(
                weights_path=self._resolved_source,
                model_variant=self.model_variant,
                file_size_bytes=None,
                device=self.device,
                loaded_at="stub",
                ultralytics_version="stub",
            )
            return self._metadata

    return _StubYOLOv8ModelLoader(device="cpu", model_variant="s")


@pytest.fixture
def vehicle_detection_module(stub_yolov8_model_loader, stub_yolov8_inference_engine):
    """Initialized :class:`VehicleDetectionModule` ready for inference."""
    from src.modules.vehicle_detection import VehicleDetectionModule

    module = VehicleDetectionModule(
        model_loader=stub_yolov8_model_loader,
        inference_engine=stub_yolov8_inference_engine,
        device="cpu",
    )
    module.initialize()
    return module


@pytest.fixture
def stub_yolov8_sign_inference_engine():
    """Inference engine that returns synthetic sign detections without Ultralytics."""
    from src.modules.yolov8_sign.inference import (
        YOLOv8SignInferenceConfig,
        YOLOv8SignInferenceEngine,
    )

    class _StubYOLOv8SignInferenceEngine(YOLOv8SignInferenceEngine):
        """Stub forward pass with a fake stop sign at upper-center."""

        def attach_model(self, model_package: dict) -> None:
            self.model_package = model_package
            self._model = model_package.get("model") or object()

        def run(self, frame: np.ndarray) -> dict:
            if not self.is_ready:
                return self.empty_output(original_shape=frame.shape)

            self._validate_frame(frame)
            height, width = frame.shape[:2]
            cx = width // 2
            cy = height // 4

            boxes = np.array(
                [[cx - 40, cy - 40, cx + 40, cy + 40]],
                dtype=np.float32,
            )
            confidences = np.array([0.92], dtype=np.float32)
            class_ids = np.array([0], dtype=np.int32)  # stop

            return {
                "boxes_xyxy": boxes,
                "confidences": confidences,
                "class_ids": class_ids,
                "inference_status": "stub",
                "original_shape": frame.shape,
                "inference_time_ms": 2.0,
                "imgsz": self.config.imgsz,
            }

    return _StubYOLOv8SignInferenceEngine(config=YOLOv8SignInferenceConfig(device="cpu"))


@pytest.fixture
def stub_yolov8_sign_model_loader():
    """Model loader that skips Ultralytics for traffic sign tests."""
    from src.modules.yolov8_sign.model_loader import (
        WeightsMetadata,
        YOLOv8SignModelLoader,
    )

    class _StubYOLOv8SignModelLoader(YOLOv8SignModelLoader):
        def load_model(self) -> WeightsMetadata:
            self._resolved_source = "stub://traffic_signs_yolov8n.pt"
            self._model = object()
            self._metadata = WeightsMetadata(
                weights_path=self._resolved_source,
                model_variant=self.model_variant,
                file_size_bytes=None,
                device=self.device,
                loaded_at="stub",
                ultralytics_version="stub",
            )
            return self._metadata

    return _StubYOLOv8SignModelLoader(device="cpu", model_variant="n")


@pytest.fixture
def traffic_sign_module(stub_yolov8_sign_model_loader, stub_yolov8_sign_inference_engine):
    """Initialized :class:`TrafficSignModule` ready for inference."""
    from src.modules.traffic_sign import TrafficSignModule

    module = TrafficSignModule(
        model_loader=stub_yolov8_sign_model_loader,
        inference_engine=stub_yolov8_sign_inference_engine,
        device="cpu",
    )
    module.initialize()
    return module


@pytest.fixture
def stub_yolov8_signal_inference_engine():
    """Inference engine that returns synthetic signal detections without Ultralytics."""
    from src.modules.yolov8_signal.inference import (
        YOLOv8SignalInferenceConfig,
        YOLOv8SignalInferenceEngine,
    )

    class _StubYOLOv8SignalInferenceEngine(YOLOv8SignalInferenceEngine):
        """Stub forward pass with a fake red light at upper-center."""

        def attach_model(self, model_package: dict) -> None:
            self.model_package = model_package
            self._model = model_package.get("model") or object()

        def run(self, frame: np.ndarray) -> dict:
            if not self.is_ready:
                return self.empty_output(original_shape=frame.shape)

            self._validate_frame(frame)
            height, width = frame.shape[:2]
            cx = width // 2
            cy = height // 5

            boxes = np.array(
                [[cx - 25, cy - 35, cx + 25, cy + 35]],
                dtype=np.float32,
            )
            confidences = np.array([0.90], dtype=np.float32)
            class_ids = np.array([0], dtype=np.int32)  # red_light

            return {
                "boxes_xyxy": boxes,
                "confidences": confidences,
                "class_ids": class_ids,
                "inference_status": "stub",
                "original_shape": frame.shape,
                "inference_time_ms": 2.0,
                "imgsz": self.config.imgsz,
            }

    return _StubYOLOv8SignalInferenceEngine(config=YOLOv8SignalInferenceConfig(device="cpu"))


@pytest.fixture
def stub_yolov8_signal_model_loader():
    """Model loader that skips Ultralytics for traffic signal tests."""
    from src.modules.yolov8_signal.model_loader import (
        WeightsMetadata,
        YOLOv8SignalModelLoader,
    )

    class _StubYOLOv8SignalModelLoader(YOLOv8SignalModelLoader):
        def load_model(self) -> WeightsMetadata:
            self._resolved_source = "stub://traffic_signals_yolov8n.pt"
            self._model = object()
            self._metadata = WeightsMetadata(
                weights_path=self._resolved_source,
                model_variant=self.model_variant,
                file_size_bytes=None,
                device=self.device,
                loaded_at="stub",
                ultralytics_version="stub",
            )
            return self._metadata

    return _StubYOLOv8SignalModelLoader(device="cpu", model_variant="n")


@pytest.fixture
def traffic_signal_module(stub_yolov8_signal_model_loader, stub_yolov8_signal_inference_engine):
    """Initialized :class:`TrafficSignalModule` ready for inference."""
    from src.modules.traffic_signal import TrafficSignalModule

    module = TrafficSignalModule(
        model_loader=stub_yolov8_signal_model_loader,
        inference_engine=stub_yolov8_signal_inference_engine,
        device="cpu",
    )
    module.initialize()
    return module


@pytest.fixture
def pipeline_orchestrator(
    lane_detection_module,
    vehicle_detection_module,
    traffic_sign_module,
    traffic_signal_module,
):
    """Pipeline orchestrator wired to stub perception module fixtures."""
    from src.pipeline.orchestrator import PipelineConfig, PipelineOrchestrator

    return PipelineOrchestrator(
        lane_module=lane_detection_module,
        vehicle_module=vehicle_detection_module,
        sign_module=traffic_sign_module,
        signal_module=traffic_signal_module,
        config=PipelineConfig(auto_initialize=False, collect_timing=True),
    )
