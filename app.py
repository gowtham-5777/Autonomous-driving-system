#!/usr/bin/env python3
"""Streamlit presentation layer for the Autonomous Driving Car ADAS pipeline.

Run from project root::

    streamlit run app.py

This module does not implement perception or decision logic. It wraps
:class:`~src.pipeline.orchestrator.PipelineOrchestrator` for image and video
inference with annotated output and summary panels.
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.decision.types import ADASRecommendation, DecisionResult
from src.pipeline import PipelineConfig, PipelineOrchestrator, PipelineResult, create_default_orchestrator
from src.utils.model_paths import (
    get_traffic_signal_weights_path,
    get_traffic_sign_weights_path,
    get_yolop_weights_path,
    get_yolov8_weights_path,
)
from verify_pipeline import build_stub_orchestrator

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "streamlit"
IMAGE_TYPES = ("jpg", "jpeg", "png", "bmp", "webp")
VIDEO_TYPES = ("mp4", "avi", "mov", "mkv", "webm")

RECOMMENDATION_COLORS = {
    ADASRecommendation.STOP: "#dc3545",
    ADASRecommendation.PROCEED: "#28a745",
    ADASRecommendation.SLOW_DOWN: "#fd7e14",
    ADASRecommendation.WARNING: "#ffc107",
    ADASRecommendation.KEEP_LANE: "#0d6efd",
}


@dataclass
class VideoSummary:
    """Aggregated statistics from a processed video."""

    frame_count: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration_sec: float = 0.0
    avg_pipeline_ms: float = 0.0
    min_pipeline_ms: float = 0.0
    max_pipeline_ms: float = 0.0
    recommendation_counts: dict[str, int] = field(default_factory=dict)
    dominant_recommendation: str = ""
    output_path: Path | None = None
    pipeline_times_ms: list[float] = field(default_factory=list)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame for Streamlit display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def real_weights_available() -> bool:
    """Return True when all default model weight files exist on disk."""
    weight_paths = (
        get_yolop_weights_path(),
        get_yolov8_weights_path(),
        get_traffic_sign_weights_path(),
        get_traffic_signal_weights_path(),
    )
    return all(path.is_file() for path in weight_paths)


@st.cache_resource(show_spinner="Loading ADAS pipeline…")
def load_orchestrator(*, use_real_weights: bool, device: str) -> PipelineOrchestrator:
    """Construct and initialize a cached pipeline orchestrator."""
    config = PipelineConfig(auto_initialize=False, collect_timing=True)
    if use_real_weights:
        orchestrator = create_default_orchestrator(device=device, config=config)
    else:
        orchestrator = build_stub_orchestrator()
    orchestrator.initialize()
    return orchestrator


def read_uploaded_image(uploaded_file: Any) -> np.ndarray:
    """Decode an uploaded image file into a BGR uint8 frame."""
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the uploaded image.")
    return frame


def save_uploaded_video(uploaded_file: Any) -> Path:
    """Persist an uploaded video to a temporary file for OpenCV capture."""
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    return Path(temp_file.name)


def run_pipeline_on_frame(
    orchestrator: PipelineOrchestrator,
    frame: np.ndarray,
    *,
    frame_index: int = 0,
    timestamp_ms: float | None = None,
    show_hud: bool = True,
) -> tuple[PipelineResult, np.ndarray]:
    """Run one frame through the orchestrator and return results plus annotation."""
    result = orchestrator.run_frame(
        frame,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
    )
    annotated = orchestrator.visualize(frame, result, show_hud=show_hud)
    return result, annotated


def render_lane_panel(scene) -> None:
    """Display lane detection summary in the Streamlit sidebar panel."""
    lane = scene.lane
    if lane is None:
        st.warning("Lane module did not run.")
        return

    status_ok = "✅" if scene.lane_ok else "⚠️"
    st.markdown(f"**Status:** {status_ok} `{lane.raw_status}`")

    col1, col2 = st.columns(2)
    col1.metric("Lane center (px)", f"{lane.lane_center_x:.1f}" if lane.lane_center_x is not None else "—")
    col2.metric("Vehicle offset (px)", f"{lane.vehicle_offset:.1f}" if lane.vehicle_offset is not None else "—")

    st.write(f"**Lane departure:** {'Yes' if lane.lane_departure else 'No'}")
    left_pts = len(lane.left_lane) if lane.left_lane else 0
    right_pts = len(lane.right_lane) if lane.right_lane else 0
    st.caption(f"Lane polylines: left={left_pts} pts, right={right_pts} pts")


def render_vehicle_panel(scene) -> None:
    """Display vehicle detection summary."""
    vehicles = scene.vehicles
    if vehicles is None:
        st.warning("Vehicle module did not run.")
        return

    status_ok = "✅" if scene.vehicles_ok else "⚠️"
    st.markdown(f"**Status:** {status_ok} `{vehicles.raw_status}`")

    summary = vehicles.summary
    st.metric("Total detections", summary.total_count)
    if summary.count_by_label:
        st.write("**Count by label:**")
        st.json(summary.count_by_label)
    else:
        st.caption("No road users detected.")

    if summary.nearest_object is not None:
        nearest = summary.nearest_object
        st.write(
            f"**Nearest:** {nearest.label} "
            f"({nearest.confidence:.2f}) bbox={nearest.bbox.to_list()}"
        )
    if vehicles.inference_time_ms is not None:
        st.caption(f"Inference: {vehicles.inference_time_ms:.1f} ms")


def render_sign_panel(scene) -> None:
    """Display traffic sign detection summary."""
    signs = scene.signs
    if signs is None:
        st.warning("Traffic sign module did not run.")
        return

    status_ok = "✅" if scene.signs_ok else "⚠️"
    st.markdown(f"**Status:** {status_ok} `{signs.raw_status}`")

    summary = signs.summary
    st.metric("Total signs", summary.total_count)
    if summary.active_speed_limit_kmh is not None:
        st.metric("Active speed limit", f"{summary.active_speed_limit_kmh} km/h")
    if summary.count_by_label:
        st.write("**Count by label:**")
        st.json(summary.count_by_label)
    else:
        st.caption("No traffic signs detected.")

    if summary.nearest_sign is not None:
        sign = summary.nearest_sign
        st.write(
            f"**Nearest:** {sign.sign_label} "
            f"({sign.confidence:.2f}) bbox={sign.bbox.to_list()}"
        )
    if signs.inference_time_ms is not None:
        st.caption(f"Inference: {signs.inference_time_ms:.1f} ms")


def render_signal_panel(scene) -> None:
    """Display traffic signal detection summary."""
    signals = scene.signals
    if signals is None:
        st.warning("Traffic signal module did not run.")
        return

    status_ok = "✅" if scene.signals_ok else "⚠️"
    st.markdown(f"**Status:** {status_ok} `{signals.raw_status}`")

    summary = signals.summary
    st.metric("Total signals", summary.total_count)
    if summary.dominant_state:
        st.write(f"**Dominant state:** `{summary.dominant_state}`")
    st.write(
        f"**Stop / proceed flags:** "
        f"stop={summary.has_stop_state}, proceed={summary.has_proceed_state}"
    )
    if summary.count_by_label:
        st.write("**Count by label:**")
        st.json(summary.count_by_label)
    else:
        st.caption("No traffic signals detected.")

    if summary.controlling_signal is not None:
        ctrl = summary.controlling_signal
        st.write(
            f"**Controlling signal:** {ctrl.signal_label} "
            f"({ctrl.confidence:.2f})"
        )
    if signals.inference_time_ms is not None:
        st.caption(f"Inference: {signals.inference_time_ms:.1f} ms")


def render_decision_panel(result: PipelineResult) -> None:
    """Display final ADAS decision and fired rules."""
    decision = result.decision
    color = RECOMMENDATION_COLORS.get(decision.recommendation, "#6c757d")
    st.markdown(
        f"<div style='padding:0.75rem 1rem;border-radius:0.5rem;"
        f"background:{color}22;border-left:4px solid {color};'>"
        f"<strong style='font-size:1.25rem;color:{color};'>"
        f"{decision.recommendation.value}</strong><br>"
        f"{decision.primary_message}</div>",
        unsafe_allow_html=True,
    )

    if result.total_time_ms is not None:
        st.metric("Pipeline time", f"{result.total_time_ms:.1f} ms")

    if decision.rule_hits:
        st.write("**Fired rules (priority order):**")
        for hit in decision.rule_hits:
            st.write(
                f"- `{hit.rule_id}` → **{hit.recommendation.value}** "
                f"(p={hit.priority}, conf={hit.confidence:.2f}): {hit.message}"
            )
    else:
        st.caption("No rules fired.")

    with st.expander("Full decision JSON"):
        st.json(decision.to_dict())


def render_image_tab(orchestrator: PipelineOrchestrator, *, show_hud: bool) -> None:
    """Image upload tab: run full pipeline and show annotated output."""
    uploaded = st.file_uploader(
        "Upload an image",
        type=list(IMAGE_TYPES),
        key="image_uploader",
    )
    if uploaded is None:
        st.info("Upload a forward-facing road image to run the ADAS pipeline.")
        return

    if st.button("Run ADAS pipeline", type="primary", key="run_image"):
        try:
            frame = read_uploaded_image(uploaded)
        except ValueError as exc:
            st.error(str(exc))
            return

        with st.spinner("Running perception modules and decision engine…"):
            result, annotated = run_pipeline_on_frame(
                orchestrator,
                frame,
                show_hud=show_hud,
            )

        st.session_state["image_result"] = result
        st.session_state["image_annotated"] = annotated
        st.session_state["image_source_name"] = uploaded.name

    if "image_annotated" not in st.session_state:
        return

    annotated = st.session_state["image_annotated"]
    result: PipelineResult = st.session_state["image_result"]
    scene = result.scene_state

    st.subheader("Annotated output")
    st.image(
        bgr_to_rgb(annotated),
        caption=st.session_state.get("image_source_name", "annotated"),
        use_container_width=True,
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Perception")
        with st.expander("Lane detection", expanded=True):
            render_lane_panel(scene)
        with st.expander("Vehicle detection", expanded=True):
            render_vehicle_panel(scene)
        with st.expander("Traffic sign detection", expanded=True):
            render_sign_panel(scene)
        with st.expander("Traffic signal detection", expanded=True):
            render_signal_panel(scene)

    with col_right:
        st.subheader("Final decision")
        render_decision_panel(result)

        with st.expander("Module health"):
            for status in scene.module_statuses:
                icon = "✅" if status.ok else "⚠️"
                ms = (
                    f"{status.inference_time_ms:.1f} ms"
                    if status.inference_time_ms is not None
                    else "—"
                )
                st.write(f"{icon} **{status.module_name}** — `{status.raw_status}` ({ms})")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = OUTPUT_DIR / f"image_{stamp}.jpg"
        cv2.imwrite(str(save_path), annotated)
        st.caption(f"Saved locally: `{save_path.relative_to(PROJECT_ROOT)}`")


def process_video(
    orchestrator: PipelineOrchestrator,
    video_path: Path,
    *,
    show_hud: bool,
    progress_bar: Any,
    status_text: Any,
) -> VideoSummary:
    """Process a video frame-by-frame and write an annotated output file."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"video_{stamp}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError("Could not create output video writer.")

    summary = VideoSummary(fps=fps, width=width, height=height, output_path=output_path)
    recommendation_counter: Counter[str] = Counter()
    frame_index = 0

    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            timestamp_ms = frame_index * (1000.0 / fps) if fps > 0 else None
            result, annotated = run_pipeline_on_frame(
                orchestrator,
                frame,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                show_hud=show_hud,
            )
            writer.write(annotated)

            if result.total_time_ms is not None:
                summary.pipeline_times_ms.append(result.total_time_ms)
            recommendation_counter[result.decision.recommendation.value] += 1
            summary.frame_count += 1
            frame_index += 1

            if total_frames > 0:
                progress_bar.progress(min(frame_index / total_frames, 1.0))
            status_text.text(f"Processed frame {frame_index}" + (f" / {total_frames}" if total_frames else ""))
    finally:
        capture.release()
        writer.release()

    summary.duration_sec = summary.frame_count / fps if fps > 0 else 0.0
    summary.recommendation_counts = dict(recommendation_counter)
    if summary.recommendation_counts:
        summary.dominant_recommendation = recommendation_counter.most_common(1)[0][0]
    if summary.pipeline_times_ms:
        summary.avg_pipeline_ms = float(np.mean(summary.pipeline_times_ms))
        summary.min_pipeline_ms = float(np.min(summary.pipeline_times_ms))
        summary.max_pipeline_ms = float(np.max(summary.pipeline_times_ms))

    return summary


def render_video_summary(summary: VideoSummary) -> None:
    """Display aggregated video processing statistics."""
    st.subheader("Summary statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames processed", summary.frame_count)
    c2.metric("Duration", f"{summary.duration_sec:.1f} s")
    c3.metric("FPS", f"{summary.fps:.1f}")
    c4.metric("Resolution", f"{summary.width}×{summary.height}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Avg pipeline time", f"{summary.avg_pipeline_ms:.1f} ms")
    c6.metric("Min pipeline time", f"{summary.min_pipeline_ms:.1f} ms")
    c7.metric("Max pipeline time", f"{summary.max_pipeline_ms:.1f} ms")

    if summary.dominant_recommendation:
        st.metric("Dominant recommendation", summary.dominant_recommendation)

    if summary.recommendation_counts:
        st.write("**Recommendation distribution:**")
        st.bar_chart(summary.recommendation_counts)

    st.write("**Counts by recommendation:**")
    st.json(summary.recommendation_counts)


def render_video_tab(orchestrator: PipelineOrchestrator, *, show_hud: bool) -> None:
    """Video upload tab: frame-by-frame processing with saved output."""
    uploaded = st.file_uploader(
        "Upload a video",
        type=list(VIDEO_TYPES),
        key="video_uploader",
    )
    if uploaded is None:
        st.info("Upload a driving video to process frame-by-frame through the ADAS pipeline.")
        return

    if st.button("Process video", type="primary", key="run_video"):
        temp_video_path: Path | None = None
        try:
            temp_video_path = save_uploaded_video(uploaded)
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            with st.spinner("Processing video frames…"):
                summary = process_video(
                    orchestrator,
                    temp_video_path,
                    show_hud=show_hud,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )

            st.session_state["video_summary"] = summary
            status_text.success(f"Finished — {summary.frame_count} frames processed.")
        except ValueError as exc:
            st.error(str(exc))
        finally:
            if temp_video_path is not None and temp_video_path.is_file():
                temp_video_path.unlink(missing_ok=True)

    summary: VideoSummary | None = st.session_state.get("video_summary")
    if summary is None or summary.output_path is None:
        return

    if not summary.output_path.is_file():
        st.warning("Annotated video file is missing.")
        return

    render_video_summary(summary)

    st.subheader("Annotated output video")
    st.video(str(summary.output_path))
    st.caption(f"Saved locally: `{summary.output_path.relative_to(PROJECT_ROOT)}`")

    with summary.output_path.open("rb") as video_file:
        st.download_button(
            label="Download annotated video",
            data=video_file.read(),
            file_name=summary.output_path.name,
            mime="video/mp4",
        )


def render_sidebar() -> tuple[bool, str, bool]:
    """Render sidebar controls and return pipeline options."""
    st.sidebar.header("Pipeline settings")

    weights_on_disk = real_weights_available()
    if weights_on_disk:
        mode = st.sidebar.radio(
            "Weight mode",
            options=["Real weights", "Demo (stub)"],
            index=0,
            help="Real weights load from config paths (set ADAS_DATA_ROOT if needed).",
        )
        use_real = mode == "Real weights"
    else:
        st.sidebar.info(
            "Model weights not found on disk. Running in **demo (stub)** mode. "
            "Set `ADAS_DATA_ROOT` to your models directory for real inference."
        )
        use_real = False

    device = st.sidebar.selectbox("Device", options=["cpu", "cuda"], index=0)
    show_hud = st.sidebar.checkbox("Show decision HUD on frames", value=True)

    st.sidebar.divider()
    st.sidebar.caption(
        "Pipeline order: Lane → Vehicle → Sign → Signal → Decision. "
        "Outputs are written to `outputs/streamlit/`."
    )
    return use_real, device, show_hud


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="ADAS Demo",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Autonomous Driving Assistance System")
    st.caption(
        "Upload an image or video to run the full perception pipeline "
        "(lane, vehicle, sign, signal) and rule-based decision engine."
    )

    use_real, device, show_hud = render_sidebar()

    try:
        orchestrator = load_orchestrator(use_real_weights=use_real, device=device)
    except Exception as exc:
        st.error(f"Failed to load pipeline: {exc}")
        st.stop()

    image_tab, video_tab = st.tabs(["Image", "Video"])
    with image_tab:
        render_image_tab(orchestrator, show_hud=show_hud)
    with video_tab:
        render_video_tab(orchestrator, show_hud=show_hud)


if __name__ == "__main__":
    main()
