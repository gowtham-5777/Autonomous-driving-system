"""Tests for scripts/verify_pipeline.py gate script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify_pipeline.py"
OUTPUT_IMAGE = PROJECT_ROOT / "outputs" / "pipeline_verify_output.jpg"


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    from scripts.verify_pipeline import _create_synthetic_frame

    return _create_synthetic_frame(width=640, height=480)


class TestVerifyPipelineModule:
    """Unit tests for verify_pipeline helpers."""

    def test_build_stub_orchestrator(self) -> None:
        from scripts.verify_pipeline import build_stub_orchestrator
        from src.pipeline import PipelineOrchestrator

        orch = build_stub_orchestrator()
        assert isinstance(orch, PipelineOrchestrator)

    def test_run_verification_stub(
        self,
        synthetic_frame: np.ndarray,
        tmp_path: Path,
    ) -> None:
        from scripts.verify_pipeline import build_stub_orchestrator, run_verification
        from src.decision import SceneState
        from src.decision.types import DecisionResult

        output_path = tmp_path / "pipeline_out.jpg"
        orch = build_stub_orchestrator()
        result = run_verification(orch, synthetic_frame, output_path=output_path)

        assert isinstance(result.scene_state, SceneState)
        assert isinstance(result.decision, DecisionResult)
        assert output_path.is_file()

    def test_load_input_frame_synthetic(self) -> None:
        from scripts.verify_pipeline import load_input_frame

        frame, source = load_input_frame()
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert source  # non-empty label


class TestVerifyPipelineScript:
    """Subprocess tests for the gate script CLI."""

    def test_cli_default_exit_zero(self, tmp_path: Path) -> None:
        output = tmp_path / "gate_default.jpg"
        proc = subprocess.run(
            [
                sys.executable,
                str(VERIFY_SCRIPT),
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "PASS: orchestrator initializes" in proc.stdout
        assert "PASS: modules initialize" in proc.stdout
        assert "PASS: pipeline runs" in proc.stdout
        assert "PASS: SceneState created" in proc.stdout
        assert "PASS: DecisionResult generated" in proc.stdout
        assert "PASS: visualization generated" in proc.stdout
        assert "PIPELINE GATE: ALL CHECKS PASSED" in proc.stdout
        assert output.is_file()

    def test_cli_with_image(self, synthetic_frame: np.ndarray, tmp_path: Path) -> None:
        image_path = tmp_path / "input.jpg"
        output_path = tmp_path / "gate_image.jpg"
        assert cv2.imwrite(str(image_path), synthetic_frame)

        proc = subprocess.run(
            [
                sys.executable,
                str(VERIFY_SCRIPT),
                "--image",
                str(image_path),
                "--output",
                str(output_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "image:input.jpg" in proc.stdout
        assert output_path.is_file()

    def test_cli_with_video(self, synthetic_frame: np.ndarray, tmp_path: Path) -> None:
        video_path = tmp_path / "input.avi"
        output_path = tmp_path / "gate_video.jpg"
        height, width = synthetic_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            1.0,
            (width, height),
        )
        assert writer.isOpened()
        writer.write(synthetic_frame)
        writer.release()

        proc = subprocess.run(
            [
                sys.executable,
                str(VERIFY_SCRIPT),
                "--video",
                str(video_path),
                "--output",
                str(output_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "video:input.avi:frame0" in proc.stdout
        assert output_path.is_file()

    def test_stub_pipeline_emits_stop(self, synthetic_frame: np.ndarray, tmp_path: Path) -> None:
        from scripts.verify_pipeline import build_stub_orchestrator, run_verification
        from src.decision.types import ADASRecommendation

        result = run_verification(
            build_stub_orchestrator(),
            synthetic_frame,
            output_path=tmp_path / "stop_check.jpg",
        )
        assert result.decision.recommendation == ADASRecommendation.STOP
