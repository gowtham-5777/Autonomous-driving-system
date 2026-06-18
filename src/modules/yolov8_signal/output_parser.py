"""YOLOv8 signal output parser — class filtering and frame-space boxes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .class_map import (
    ALLOWED_SIGNAL_CLASS_IDS,
    CONTROLLING_SIGNAL_UPPER_FRACTION,
    SIGNAL_CLASS_ID_TO_LABEL,
    enrich_state_flags,
    state_priority,
)
from .output_schema import (
    DetectedSignal,
    ParsedYOLOv8SignalOutput,
    SignalBoundingBoxData,
    TrafficSignalSummary,
)

logger = logging.getLogger("adas.modules.yolov8_signal.output_parser")

YOLOv8SignalRawOutput = dict[str, Any]


@dataclass(frozen=True)
class ParserConfig:
    """Configuration for YOLOv8 signal output parsing."""

    confidence_threshold: float = 0.5


class YOLOv8SignalOutputParser:
    """Parse raw YOLOv8 signal inference outputs into ADAS detections."""

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()
        logger.info(
            "YOLOv8SignalOutputParser created — confidence_threshold=%.2f",
            self.config.confidence_threshold,
        )

    def parse(
        self,
        raw_outputs: YOLOv8SignalRawOutput,
        frame_shape: tuple[int, ...] | None = None,
    ) -> ParsedYOLOv8SignalOutput:
        """Parse raw inference output into filtered frame-space detections."""
        logger.info("Parsing YOLOv8 signal raw outputs")

        if not raw_outputs:
            return ParsedYOLOv8SignalOutput(raw_status="empty")

        status = str(raw_outputs.get("inference_status", "unknown"))
        shape = frame_shape or raw_outputs.get("original_shape")
        if shape is None or len(shape) < 2:
            logger.warning("Frame shape unavailable — cannot validate boxes")
            return ParsedYOLOv8SignalOutput(raw_status="parser_error")

        frame_height = int(shape[0])
        frame_width = int(shape[1])

        boxes = np.asarray(raw_outputs.get("boxes_xyxy", []), dtype=np.float32)
        confidences = np.asarray(raw_outputs.get("confidences", []), dtype=np.float32)
        class_ids = np.asarray(raw_outputs.get("class_ids", []), dtype=np.int32)

        if boxes.size == 0:
            return ParsedYOLOv8SignalOutput(
                detections=[],
                raw_status=status if status != "unknown" else "ok",
                metadata={"num_raw_boxes": 0, "num_filtered": 0},
            )

        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        detections: list[DetectedSignal] = []
        dropped_invalid = 0
        dropped_class = 0
        dropped_conf = 0

        for idx in range(len(boxes)):
            class_id = int(class_ids[idx]) if idx < len(class_ids) else -1
            confidence = float(confidences[idx]) if idx < len(confidences) else 0.0

            if class_id not in ALLOWED_SIGNAL_CLASS_IDS:
                dropped_class += 1
                continue

            if confidence < self.config.confidence_threshold:
                dropped_conf += 1
                continue

            x1, y1, x2, y2 = boxes[idx]
            bbox = self._clip_bbox_to_frame(
                int(round(float(x1))),
                int(round(float(y1))),
                int(round(float(x2))),
                int(round(float(y2))),
                frame_width,
                frame_height,
            )
            if bbox is None:
                dropped_invalid += 1
                logger.warning(
                    "Dropped invalid signal bbox after clipping: [%s, %s, %s, %s]",
                    x1,
                    y1,
                    x2,
                    y2,
                )
                continue

            signal_label = SIGNAL_CLASS_ID_TO_LABEL[class_id]
            stop_state, caution_state, proceed_state = enrich_state_flags(signal_label)
            detections.append(
                DetectedSignal(
                    signal_label=signal_label,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=bbox,
                    is_stop_state=stop_state,
                    is_caution_state=caution_state,
                    is_proceed_state=proceed_state,
                )
            )

        parsed_status = status if status not in {"unknown", "empty"} else "ok"
        metadata = {
            "num_raw_boxes": int(len(boxes)),
            "num_filtered": len(detections),
            "dropped_class": dropped_class,
            "dropped_confidence": dropped_conf,
            "dropped_invalid": dropped_invalid,
            "frame_shape": (frame_height, frame_width),
        }

        logger.info(
            "YOLOv8 signal parse complete — status=%s, detections=%d",
            parsed_status,
            len(detections),
        )

        return ParsedYOLOv8SignalOutput(
            detections=detections,
            raw_status=parsed_status,
            metadata=metadata,
        )

    def build_summary(
        self,
        detections: list[DetectedSignal],
        frame_shape: tuple[int, ...] | None = None,
    ) -> TrafficSignalSummary:
        """Build aggregate statistics from a list of signal detections."""
        count_by_label: dict[str, int] = {}
        for det in detections:
            count_by_label[det.signal_label] = count_by_label.get(det.signal_label, 0) + 1

        nearest = self._select_nearest_signal(detections)
        highest = max(detections, key=lambda d: d.confidence) if detections else None
        controlling = self._select_controlling_signal(detections, frame_shape)
        dominant = self._select_dominant_state(detections, controlling)

        has_stop = any(det.is_stop_state for det in detections)
        has_proceed = any(det.is_proceed_state for det in detections)
        if has_stop and has_proceed:
            logger.warning(
                "Conflicting signal states detected (red + green) — dominant=%s",
                dominant,
            )

        return TrafficSignalSummary(
            count_by_label=count_by_label,
            total_count=len(detections),
            nearest_signal=nearest,
            highest_confidence=highest,
            controlling_signal=controlling,
            dominant_state=dominant,
            has_stop_state=has_stop,
            has_proceed_state=has_proceed,
        )

    @staticmethod
    def _select_nearest_signal(detections: list[DetectedSignal]) -> DetectedSignal | None:
        """Pick the signal closest to the ego (highest center_y, then largest area)."""
        if not detections:
            return None

        return max(
            detections,
            key=lambda det: (det.bbox.center_y, det.bbox.area),
        )

    @staticmethod
    def _select_controlling_signal(
        detections: list[DetectedSignal],
        frame_shape: tuple[int, ...] | None,
    ) -> DetectedSignal | None:
        """Pick the signal governing ego in the upper road region."""
        if not detections:
            return None

        if frame_shape is not None and len(frame_shape) >= 1:
            upper_bound = int(frame_shape[0]) * CONTROLLING_SIGNAL_UPPER_FRACTION
            upper_candidates = [
                det for det in detections if det.bbox.center_y <= upper_bound
            ]
            if upper_candidates:
                return max(
                    upper_candidates,
                    key=lambda det: (det.bbox.center_y, det.confidence),
                )

        return YOLOv8SignalOutputParser._select_nearest_signal(detections)

    @staticmethod
    def _select_dominant_state(
        detections: list[DetectedSignal],
        controlling: DetectedSignal | None,
    ) -> str | None:
        """Return the conservative dominant signal state."""
        if controlling is not None:
            return controlling.signal_label

        if not detections:
            return None

        return max(detections, key=lambda det: state_priority(det.signal_label)).signal_label

    @staticmethod
    def _clip_bbox_to_frame(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_width: int,
        frame_height: int,
    ) -> SignalBoundingBoxData | None:
        """Clip box to frame bounds; return None if invalid after clipping."""
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            return None

        return SignalBoundingBoxData.from_xyxy(x1, y1, x2, y2)
