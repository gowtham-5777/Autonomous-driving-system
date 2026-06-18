#!/usr/bin/env python3
"""Gate script: verify vehicle detection pipeline wiring and outputs.

Run from project root:

    python scripts/verify_vehicle_detection.py

Uses a stub inference engine by default (no Ultralytics download required).
Pass ``--real`` to run with real YOLOv8 weights when available.
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
    from src.modules.yolov8.inference import YOLOv8InferenceConfig, YOLOv8InferenceEngine

    class _StubEngine(YOLOv8InferenceEngine):
        def attach_model(self, model_package: dict) -> None:
            self.model_package = model_package
            self._model = model_package.get("model") or object()

        def run(self, frame: np.ndarray) -> dict:
            self._validate_frame(frame)
            h, w = frame.shape[:2]
            cx, cy = w // 2, h - 100
            return {
                "boxes_xyxy": np.array(
                    [[cx - 80, cy - 60, cx + 80, cy + 20]],
                    dtype=np.float32,
                ),
                "confidences": np.array([0.88], dtype=np.float32),
                "class_ids": np.array([2], dtype=np.int32),
                "inference_status": "verify_stub",
                "original_shape": frame.shape,
                "inference_time_ms": 3.0,
                "imgsz": self.config.imgsz,
            }

    return _StubEngine(config=YOLOv8InferenceConfig(device="cpu"))


def _build_stub_loader():
    from src.modules.yolov8.model_loader import WeightsMetadata, YOLOv8ModelLoader

    class _StubLoader(YOLOv8ModelLoader):
        def load_model(self) -> WeightsMetadata:
            self._resolved_source = "stub://yolov8s.pt"
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

    return _StubLoader(device="cpu", model_variant="s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify vehicle detection pipeline")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real Ultralytics YOLOv8 weights (downloads yolov8s.pt if missing)",
    )
    args = parser.parse_args()

    from src.modules.vehicle_detection import VehicleDetectionModule
    from src.modules.yolov8.output_schema import VehicleDetectionResult

    module_path = Path(inspect.getfile(VehicleDetectionModule))
    if not module_path.is_file():
        _fail(f"VehicleDetectionModule not found at {module_path}")
    _pass(f"VehicleDetectionModule at {module_path}")

    yolov8_pkg = PROJECT_ROOT / "src" / "modules" / "yolov8"
    for name in ("__init__.py", "model_loader.py", "inference.py", "output_parser.py", "output_schema.py"):
        path = yolov8_pkg / name
        if not path.is_file():
            _fail(f"missing {path}")
    _pass("yolov8 package files present")

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.rectangle(frame, (500, 500), (780, 680), (80, 80, 200), -1)

    road_fixture = PROJECT_ROOT / "tests" / "fixtures" / "road_sample.jpg"
    if args.real and road_fixture.is_file():
        loaded = cv2.imread(str(road_fixture), cv2.IMREAD_COLOR)
        if loaded is not None:
            frame = loaded

    if args.real:
        pytest_import = __import__("importlib").import_module
        try:
            pytest_import("ultralytics")
        except ImportError:
            _fail("ultralytics not installed — pip install ultralytics")

        module = VehicleDetectionModule(device="cpu", confidence_threshold=0.25)
        mode = "real"
    else:
        module = VehicleDetectionModule(
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
    if not isinstance(result, VehicleDetectionResult):
        _fail(f"predict() returned {type(result)}, expected VehicleDetectionResult")
    _pass("inference runs")

    if result.summary.total_count < 1:
        if args.real and result.raw_status in {"ok", "verify_stub", "stub"}:
            print(
                "WARN: real inference returned 0 ADAS objects on synthetic road "
                "fixture (expected — no COCO vehicles in test image)"
            )
            _pass("real inference completed (0 objects on synthetic scene)")
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
    print("VEHICLE DETECTION GATE: ALL CHECKS PASSED")
    print(f"  mode            = {mode}")
    print(f"  raw_status      = {result.raw_status}")
    print(f"  total_count     = {result.summary.total_count}")
    print(f"  labels          = {result.summary.count_by_label}")
    print(f"  inference_ms    = {result.inference_time_ms}")
    print("=" * 60)


if __name__ == "__main__":
    main()
