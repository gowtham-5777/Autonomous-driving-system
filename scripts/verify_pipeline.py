#!/usr/bin/env python3
"""Gate script: verify end-to-end ADAS pipeline orchestration and decision output.

Run from project root:

    python scripts/verify_pipeline.py
    python scripts/verify_pipeline.py --image path/to/image.jpg
    python scripts/verify_pipeline.py --video path/to/video.mp4

Uses stub perception engines by default (no model weights required).
Pass ``--real`` to use default module weights from config when available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "pipeline_verify_output.jpg"
ROAD_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "road_sample.jpg"


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def _pass(message: str) -> None:
    print(f"PASS: {message}")


def _create_synthetic_frame(width: int = 1280, height: int = 720) -> np.ndarray:
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


def _ensure_stub_yolop_weights() -> Path:
    """Return a minimal stub YOLOP checkpoint (always stub for gate script)."""
    fixtures = PROJECT_ROOT / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    stub_path = fixtures / "stub_yolop_weights.pth"
    if stub_path.is_file():
        return stub_path

    torch = __import__("torch")
    checkpoint = {
        "state_dict": {
            "stub.conv.weight": torch.zeros(3, 3, 3, 3),
            "stub.conv.bias": torch.zeros(3),
        },
        "epoch": 0,
    }
    torch.save(checkpoint, stub_path)
    return stub_path


def _build_stub_yolop_engine():
    from src.modules.yolop.inference import InferenceConfig, YOLOPInferenceEngine

    class _StubYOLOPInferenceEngine(YOLOPInferenceEngine):
        def run(self, frame: np.ndarray) -> dict:
            if not self.is_ready:
                return self.empty_output()

            self._validate_frame(frame)
            target_width, target_height = self.config.input_size

            drivable = np.zeros((2, target_height, target_width), dtype=np.float32)
            drivable[1, target_height // 2 :, :] = 2.0

            lane = np.zeros((2, target_height, target_width), dtype=np.float32)
            lane[1, target_height // 2 :, target_width // 2 - 30 : target_width // 2 + 30] = 2.0

            return {
                1: drivable,
                2: lane,
                "inference_status": "stub_segmentation",
                "original_shape": frame.shape,
                "input_size": self.config.input_size,
                "confidence_threshold": self.config.confidence_threshold,
            }

    return _StubYOLOPInferenceEngine(config=InferenceConfig(device="cpu"))


def _build_stub_yolov8_engine():
    from src.modules.yolov8.inference import YOLOv8InferenceConfig, YOLOv8InferenceEngine

    class _StubEngine(YOLOv8InferenceEngine):
        def attach_model(self, model_package: dict) -> None:
            self.model_package = model_package
            self._model = model_package.get("model") or object()

        def run(self, frame: np.ndarray) -> dict:
            self._validate_frame(frame)
            height, width = frame.shape[:2]
            cx, cy = width // 2, height - 80
            return {
                "boxes_xyxy": np.array(
                    [[cx - 70, cy - 50, cx + 70, cy + 30]],
                    dtype=np.float32,
                ),
                "confidences": np.array([0.91], dtype=np.float32),
                "class_ids": np.array([2], dtype=np.int32),
                "inference_status": "stub",
                "original_shape": frame.shape,
                "inference_time_ms": 2.5,
                "imgsz": self.config.imgsz,
            }

    return _StubEngine(config=YOLOv8InferenceConfig(device="cpu"))


def _build_stub_yolov8_loader(variant: str = "s"):
    from src.modules.yolov8.model_loader import WeightsMetadata, YOLOv8ModelLoader

    class _StubLoader(YOLOv8ModelLoader):
        def load_model(self) -> WeightsMetadata:
            self._resolved_source = f"stub://yolov8{variant}.pt"
            self._model = object()
            self._metadata = WeightsMetadata(
                weights_path=self._resolved_source,
                model_variant=self.model_variant,
                file_size_bytes=None,
                device=self.device,
                loaded_at="verify",
                ultralytics_version="stub",
            )
            return self._metadata

    return _StubLoader(device="cpu", model_variant=variant)


def _build_stub_sign_engine():
    from src.modules.yolov8_sign.inference import (
        YOLOv8SignInferenceConfig,
        YOLOv8SignInferenceEngine,
    )

    class _StubEngine(YOLOv8SignInferenceEngine):
        def attach_model(self, model_package: dict) -> None:
            self.model_package = model_package
            self._model = model_package.get("model") or object()

        def run(self, frame: np.ndarray) -> dict:
            self._validate_frame(frame)
            height, width = frame.shape[:2]
            cx, cy = width // 2, height // 4
            return {
                "boxes_xyxy": np.array(
                    [[cx - 40, cy - 40, cx + 40, cy + 40]],
                    dtype=np.float32,
                ),
                "confidences": np.array([0.92], dtype=np.float32),
                "class_ids": np.array([0], dtype=np.int32),
                "inference_status": "stub",
                "original_shape": frame.shape,
                "inference_time_ms": 2.0,
                "imgsz": self.config.imgsz,
            }

    return _StubEngine(config=YOLOv8SignInferenceConfig(device="cpu"))


def _build_stub_sign_loader():
    from src.modules.yolov8_sign.model_loader import (
        WeightsMetadata,
        YOLOv8SignModelLoader,
    )

    class _StubLoader(YOLOv8SignModelLoader):
        def load_model(self) -> WeightsMetadata:
            self._resolved_source = "stub://traffic_signs_yolov8n.pt"
            self._model = object()
            self._metadata = WeightsMetadata(
                weights_path=self._resolved_source,
                model_variant=self.model_variant,
                file_size_bytes=None,
                device=self.device,
                loaded_at="verify",
                ultralytics_version="stub",
            )
            return self._metadata

    return _StubLoader(device="cpu", model_variant="n")


def _build_stub_signal_engine():
    from src.modules.yolov8_signal.inference import (
        YOLOv8SignalInferenceConfig,
        YOLOv8SignalInferenceEngine,
    )

    class _StubEngine(YOLOv8SignalInferenceEngine):
        def attach_model(self, model_package: dict) -> None:
            self.model_package = model_package
            self._model = model_package.get("model") or object()

        def run(self, frame: np.ndarray) -> dict:
            self._validate_frame(frame)
            height, width = frame.shape[:2]
            cx, cy = width // 2, height // 5
            return {
                "boxes_xyxy": np.array(
                    [[cx - 25, cy - 35, cx + 25, cy + 35]],
                    dtype=np.float32,
                ),
                "confidences": np.array([0.90], dtype=np.float32),
                "class_ids": np.array([0], dtype=np.int32),
                "inference_status": "stub",
                "original_shape": frame.shape,
                "inference_time_ms": 2.0,
                "imgsz": self.config.imgsz,
            }

    return _StubEngine(config=YOLOv8SignalInferenceConfig(device="cpu"))


def _build_stub_signal_loader():
    from src.modules.yolov8_signal.model_loader import (
        WeightsMetadata,
        YOLOv8SignalModelLoader,
    )

    class _StubLoader(YOLOv8SignalModelLoader):
        def load_model(self) -> WeightsMetadata:
            self._resolved_source = "stub://traffic_signals_yolov8n.pt"
            self._model = object()
            self._metadata = WeightsMetadata(
                weights_path=self._resolved_source,
                model_variant=self.model_variant,
                file_size_bytes=None,
                device=self.device,
                loaded_at="verify",
                ultralytics_version="stub",
            )
            return self._metadata

    return _StubLoader(device="cpu", model_variant="n")


def build_stub_orchestrator():
    """Construct a :class:`PipelineOrchestrator` with stub perception modules."""
    from src.decision import DecisionEngine
    from src.modules.lane_detection import LaneDetectionModule
    from src.modules.traffic_sign import TrafficSignModule
    from src.modules.traffic_signal import TrafficSignalModule
    from src.modules.vehicle_detection import VehicleDetectionModule
    from src.pipeline import PipelineConfig, PipelineOrchestrator

    weights_path = _ensure_stub_yolop_weights()
    return PipelineOrchestrator(
        lane_module=LaneDetectionModule(
            weights_path=weights_path,
            inference_engine=_build_stub_yolop_engine(),
            device="cpu",
        ),
        vehicle_module=VehicleDetectionModule(
            model_loader=_build_stub_yolov8_loader("s"),
            inference_engine=_build_stub_yolov8_engine(),
            device="cpu",
        ),
        sign_module=TrafficSignModule(
            model_loader=_build_stub_sign_loader(),
            inference_engine=_build_stub_sign_engine(),
            device="cpu",
        ),
        signal_module=TrafficSignalModule(
            model_loader=_build_stub_signal_loader(),
            inference_engine=_build_stub_signal_engine(),
            device="cpu",
        ),
        decision_engine=DecisionEngine(),
        config=PipelineConfig(auto_initialize=False, collect_timing=True),
    )


def load_input_frame(
    *,
    image_path: Path | None = None,
    video_path: Path | None = None,
) -> tuple[np.ndarray, str]:
    """Load a BGR frame from default synthetic, image file, or first video frame."""
    if image_path is not None:
        if not image_path.is_file():
            _fail(f"image not found: {image_path}")
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            _fail(f"could not read image: {image_path}")
        return frame, f"image:{image_path.name}"

    if video_path is not None:
        if not video_path.is_file():
            _fail(f"video not found: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            _fail(f"could not open video: {video_path}")
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            _fail(f"could not read first frame from video: {video_path}")
        return frame, f"video:{video_path.name}:frame0"

    if ROAD_FIXTURE.is_file():
        frame = cv2.imread(str(ROAD_FIXTURE), cv2.IMREAD_COLOR)
        if frame is not None:
            return frame, "fixture:road_sample.jpg"

    return _create_synthetic_frame(), "synthetic:default"


def print_verification_report(result, *, source: str, output_path: Path) -> None:
    """Print decision, rule, module status, and timing details."""
    scene = result.scene_state
    decision = result.decision
    winning_rule = decision.rule_hits[0] if decision.rule_hits else None

    print()
    print("--- Decision ---")
    print(f"  recommendation   = {decision.recommendation.value}")
    print(f"  priority         = {decision.priority}")
    print(f"  primary_message  = {decision.primary_message}")

    print()
    print("--- Rule ---")
    if winning_rule is not None:
        print(f"  rule_id          = {winning_rule.rule_id}")
        print(f"  source_module    = {winning_rule.source_module}")
        print(f"  confidence       = {winning_rule.confidence:.4f}")
    else:
        print("  (no rule hits)")

    print()
    print("--- Module statuses ---")
    for status in scene.module_statuses:
        print(
            f"  {status.module_name:20s} ok={status.ok} "
            f"raw_status={status.raw_status} "
            f"inference_ms={status.inference_time_ms}"
        )

    print()
    print("--- Inference timing ---")
    print(f"  pipeline_total_ms = {result.total_time_ms}")
    if scene.lane is not None:
        print(f"  lane_raw_status   = {scene.lane.raw_status}")
    if scene.vehicles is not None:
        print(f"  vehicle_ms        = {scene.vehicles.inference_time_ms}")
    if scene.signs is not None:
        print(f"  sign_ms           = {scene.signs.inference_time_ms}")
    if scene.signals is not None:
        print(f"  signal_ms         = {scene.signals.inference_time_ms}")

    print()
    print("=" * 60)
    print("PIPELINE GATE: ALL CHECKS PASSED")
    print(f"  source           = {source}")
    print(f"  frame_shape      = {scene.frame_shape}")
    print(f"  output_image     = {output_path}")
    print("=" * 60)


def run_verification(
    orchestrator,
    frame: np.ndarray,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> object:
    """Run gate checks; return :class:`PipelineResult` on success."""
    from src.decision import DecisionEngine, SceneState
    from src.decision.types import DecisionResult
    from src.pipeline import PipelineOrchestrator, PipelineResult

    if not isinstance(orchestrator, PipelineOrchestrator):
        _fail(f"expected PipelineOrchestrator, got {type(orchestrator)}")
    _pass("orchestrator initializes")

    orchestrator.initialize()
    modules = (
        ("lane_detection", orchestrator.lane_module),
        ("vehicle_detection", orchestrator.vehicle_module),
        ("traffic_sign", orchestrator.sign_module),
        ("traffic_signal", orchestrator.signal_module),
    )
    for name, module in modules:
        if not module.is_initialized:
            _fail(f"{name} failed to initialize")
    _pass("modules initialize")

    if not isinstance(orchestrator.decision_engine, DecisionEngine):
        _fail("orchestrator decision_engine is not DecisionEngine")

    result = orchestrator.run_frame(frame, frame_index=0)
    if not isinstance(result, PipelineResult):
        _fail(f"run_frame returned {type(result)}, expected PipelineResult")
    _pass("pipeline runs")

    if not isinstance(result.scene_state, SceneState):
        _fail(f"scene_state is {type(result.scene_state)}, expected SceneState")
    _pass("SceneState created")

    if not isinstance(result.decision, DecisionResult):
        _fail(f"decision is {type(result.decision)}, expected DecisionResult")
    _pass("DecisionResult generated")

    annotated = orchestrator.visualize(frame, result)
    if annotated.shape != frame.shape or annotated.dtype != np.uint8:
        _fail(f"visualize returned invalid frame: shape={annotated.shape} dtype={annotated.dtype}")
    if np.array_equal(annotated, frame):
        _fail("visualize did not modify frame")
    _pass("visualization generated")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        _fail(f"failed to write output image: {output_path}")

    orchestrator.cleanup()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ADAS pipeline orchestration gate")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image", type=Path, help="Path to input image (BGR)")
    source_group.add_argument("--video", type=Path, help="Path to input video (first frame)")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Annotated output image path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use create_default_orchestrator() with real module weights when available",
    )
    args = parser.parse_args()

    from src.decision import DecisionEngine, SceneState
    from src.pipeline import PipelineOrchestrator, create_default_orchestrator

    _ = (DecisionEngine, SceneState, PipelineOrchestrator, create_default_orchestrator)

    frame, source = load_input_frame(image_path=args.image, video_path=args.video)

    if args.real:
        orchestrator = create_default_orchestrator(
            device="cpu",
        )
        mode = "real"
    else:
        orchestrator = build_stub_orchestrator()
        mode = "stub"

    result = run_verification(orchestrator, frame, output_path=args.output)
    print_verification_report(result, source=f"{mode}:{source}", output_path=args.output)


if __name__ == "__main__":
    main()
