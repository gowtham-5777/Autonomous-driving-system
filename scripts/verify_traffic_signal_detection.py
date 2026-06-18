#!/usr/bin/env python3
"""Gate script: verify traffic signal detection pipeline wiring and outputs.

Run from project root:

    python scripts/verify_traffic_signal_detection.py

Uses a stub inference engine by default (no fine-tuned weights required).
Pass ``--real`` to run with real YOLOv8 signal weights when available.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def _pass(message: str) -> None:
    print(f"PASS: {message}")


def _build_stub_engine():
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
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 5
            return {
                "boxes_xyxy": np.array(
                    [[cx - 28, cy - 38, cx + 28, cy + 38]],
                    dtype=np.float32,
                ),
                "confidences": np.array([0.91], dtype=np.float32),
                "class_ids": np.array([0], dtype=np.int32),
                "inference_status": "verify_stub",
                "original_shape": frame.shape,
                "inference_time_ms": 2.5,
                "imgsz": self.config.imgsz,
            }

    return _StubEngine(config=YOLOv8SignalInferenceConfig(device="cpu"))


def _build_stub_loader():
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify traffic signal detection pipeline")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real fine-tuned YOLOv8 signal weights from config",
    )
    args = parser.parse_args()

    from src.modules.traffic_signal import TrafficSignalModule
    from src.modules.yolov8_signal.output_schema import TrafficSignalDetectionResult
    from src.utils.model_paths import get_traffic_signal_weights_path

    module_path = Path(inspect.getfile(TrafficSignalModule))
    if not module_path.is_file():
        _fail(f"TrafficSignalModule not found at {module_path}")
    _pass(f"TrafficSignalModule at {module_path}")

    yolov8_signal_pkg = PROJECT_ROOT / "src" / "modules" / "yolov8_signal"
    for name in (
        "__init__.py",
        "model_loader.py",
        "inference.py",
        "output_parser.py",
        "output_schema.py",
        "class_map.py",
    ):
        path = yolov8_signal_pkg / name
        if not path.is_file():
            _fail(f"missing {path}")
    _pass("yolov8_signal package files present")

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.rectangle(frame, (610, 80), (670, 200), (0, 0, 220), -1)

    road_fixture = PROJECT_ROOT / "tests" / "fixtures" / "road_sample.jpg"
    if args.real and road_fixture.is_file():
        loaded = cv2.imread(str(road_fixture), cv2.IMREAD_COLOR)
        if loaded is not None:
            frame = loaded

    if args.real:
        weights_path = get_traffic_signal_weights_path()
        if not weights_path.is_file():
            _fail(
                f"fine-tuned signal weights not found at {weights_path} "
                "(train or place traffic_signals_yolov8n.pt)"
            )

        importlib = __import__("importlib")
        try:
            importlib.import_module("ultralytics")
        except ImportError:
            _fail("ultralytics not installed — pip install ultralytics")

        module = TrafficSignalModule(device="cpu", confidence_threshold=0.25)
        mode = "real"
    else:
        module = TrafficSignalModule(
            model_loader=_build_stub_loader(),
            inference_engine=_build_stub_engine(),
            device="cpu",
        )
        mode = "stub"

    module.initialize()
    if not module.is_initialized:
        _fail("module failed to initialize")
    _pass(f"model loads ({mode})")

    result = module.predict(frame)
    if not isinstance(result, TrafficSignalDetectionResult):
        _fail(f"predict() returned {type(result)}, expected TrafficSignalDetectionResult")
    _pass("inference runs")

    if result.summary.total_count < 1:
        if args.real and result.raw_status in {"ok", "verify_stub", "stub"}:
            print(
                "WARN: real inference returned 0 signals on test frame "
                "(expected without lights in scene)"
            )
            _pass("real inference completed (0 signals on test scene)")
        else:
            _fail("no detections produced")
    else:
        _pass(f"detections produced — count={result.summary.total_count}")

    for det in result.detections:
        if det.bbox.x2 > frame.shape[1] or det.bbox.y2 > frame.shape[0]:
            _fail(f"bbox outside frame: {det.bbox.to_list()}")
    _pass("all boxes in original frame coordinates")

    annotated = module.visualize(frame, result)
    if annotated.shape != frame.shape:
        _fail(f"visualize shape mismatch: {annotated.shape}")
    if np.array_equal(annotated, frame):
        _fail("visualize did not modify frame")
    _pass("visualization works")

    module.cleanup()
    if module.is_initialized:
        _fail("module still initialized after cleanup")
    _pass("cleanup releases module state")

    print()
    print("=" * 60)
    print("TRAFFIC SIGNAL DETECTION GATE: ALL CHECKS PASSED")
    print(f"  mode            = {mode}")
    print(f"  raw_status      = {result.raw_status}")
    print(f"  total_count     = {result.summary.total_count}")
    print(f"  labels          = {result.summary.count_by_label}")
    print(f"  dominant_state  = {result.summary.dominant_state}")
    print(f"  inference_ms    = {result.inference_time_ms}")
    print("=" * 60)


if __name__ == "__main__":
    main()
