#!/usr/bin/env python3
"""Gate script: exit 0 only when lane masks match input frame shape.

Run before Colab validation:

    python scripts/verify_mask_resize.py

Requires project root on sys.path (script adds it automatically).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FRAME_SHAPE = (1024, 2048, 3)


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def _pass(message: str) -> None:
    print(f"PASS: {message}")


def main() -> None:
    from src.modules import lane_detection as lane_detection_module
    from src.modules.lane_detection import (
        LANE_DETECTION_PIPELINE_VERSION,
        LaneDetectionModule,
    )
    from src.modules.yolop.inference import InferenceConfig, YOLOPInferenceEngine
    from tests.conftest import _resolve_weights_path

    module_path = Path(inspect.getfile(LaneDetectionModule))
    if not module_path.is_file():
        _fail(f"LaneDetectionModule not found at {module_path}")

    if LANE_DETECTION_PIPELINE_VERSION < 2:
        _fail(
            f"LANE_DETECTION_PIPELINE_VERSION={LANE_DETECTION_PIPELINE_VERSION} "
            "(need >= 2 for frame-sized masks)"
        )
    _pass(f"pipeline version {LANE_DETECTION_PIPELINE_VERSION}")

    if not hasattr(LaneDetectionModule, "_resize_masks_to_frame_shape"):
        _fail("LaneDetectionModule missing _resize_masks_to_frame_shape")
    _pass("_resize_masks_to_frame_shape present")

    if not hasattr(LaneDetectionModule, "_assert_result_masks_match_frame"):
        _fail("LaneDetectionModule missing _assert_result_masks_match_frame")
    _pass("_assert_result_masks_match_frame present")

    mask_resize_path = PROJECT_ROOT / "src" / "modules" / "yolop" / "mask_resize.py"
    if not mask_resize_path.is_file():
        _fail(f"missing {mask_resize_path}")
    _pass(f"mask_resize.py at {mask_resize_path}")

    class _StubEngine(YOLOPInferenceEngine):
        def run(self, frame: np.ndarray) -> dict:
            self._validate_frame(frame)
            tw, th = self.config.input_size
            drivable = np.zeros((2, th, tw), dtype=np.float32)
            drivable[1, th // 2 :, :] = 2.0
            lane = np.zeros((2, th, tw), dtype=np.float32)
            lane[1, th // 2 :, tw // 2 - 30 : tw // 2 + 30] = 2.0
            return {
                1: drivable,
                2: lane,
                "inference_status": "verify_stub",
                "original_shape": frame.shape,
            }

    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    module = LaneDetectionModule(
        weights_path=_resolve_weights_path(),
        inference_engine=_StubEngine(config=InferenceConfig(device="cpu")),
        device="cpu",
        apply_mask_postprocess=False,
    )
    module.initialize()
    result = module.predict(frame)

    if result.lane_mask is None:
        _fail("lane_mask is None")
    if result.drivable_mask is None:
        _fail("drivable_mask is None")

    if result.lane_mask.shape != frame.shape[:2]:
        _fail(
            f"lane_mask.shape {result.lane_mask.shape} != frame.shape[:2] "
            f"{frame.shape[:2]}"
        )
    _pass(f"lane_mask.shape == frame.shape[:2] == {result.lane_mask.shape}")

    if result.drivable_mask.shape != frame.shape[:2]:
        _fail(
            f"drivable_mask.shape {result.drivable_mask.shape} != frame.shape[:2] "
            f"{frame.shape[:2]}"
        )
    _pass(f"drivable_mask.shape == frame.shape[:2] == {result.drivable_mask.shape}")

    print()
    print("=" * 60)
    print("MASK RESIZE GATE: ALL CHECKS PASSED")
    print(f"  lane_detection: {module_path}")
    print(f"  frame.shape     = {frame.shape}")
    print(f"  lane_mask.shape = {result.lane_mask.shape}")
    print(f"  vehicle_offset  = {result.vehicle_offset}")
    print("=" * 60)
    print(f"Safe to validate on Colab after syncing commit with version >= 2.")
    print(f"Loaded from: {lane_detection_module.__file__}")


if __name__ == "__main__":
    main()
