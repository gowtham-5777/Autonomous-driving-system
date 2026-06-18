"""YOLOv8 sign output parser — class filtering and frame-space boxes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .class_map import (
    ALLOWED_SIGN_CLASS_IDS,
    SIGN_CLASS_ID_TO_LABEL,
    extract_speed_limit_kmh,
    is_regulatory_label,
)
from .output_schema import (
    DetectedSign,
    ParsedYOLOv8SignOutput,
    SignBoundingBoxData,
    TrafficSignDetectionSummary,
)

logger = logging.getLogger("adas.modules.yolov8_sign.output_parser")

YOLOv8SignRawOutput = dict[str, Any]


@dataclass(frozen=True)
class ParserConfig:
    """Configuration for YOLOv8 sign output parsing."""

    confidence_threshold: float = 0.5


class YOLOv8SignOutputParser:
    """Parse raw YOLOv8 sign inference outputs into ADAS detections."""

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()
        logger.info(
            "YOLOv8SignOutputParser created — confidence_threshold=%.2f",
            self.config.confidence_threshold,
        )

    def parse(
        self,
        raw_outputs: YOLOv8SignRawOutput,
        frame_shape: tuple[int, ...] | None = None,
    ) -> ParsedYOLOv8SignOutput:
        """Parse raw inference output into filtered frame-space detections."""
        logger.info("Parsing YOLOv8 sign raw outputs")

        if not raw_outputs:
            return ParsedYOLOv8SignOutput(raw_status="empty")

        status = str(raw_outputs.get("inference_status", "unknown"))
        shape = frame_shape or raw_outputs.get("original_shape")
        if shape is None or len(shape) < 2:
            logger.warning("Frame shape unavailable — cannot validate boxes")
            return ParsedYOLOv8SignOutput(raw_status="parser_error")

        frame_height = int(shape[0])
        frame_width = int(shape[1])

        boxes = np.asarray(raw_outputs.get("boxes_xyxy", []), dtype=np.float32)
        confidences = np.asarray(raw_outputs.get("confidences", []), dtype=np.float32)
        class_ids = np.asarray(raw_outputs.get("class_ids", []), dtype=np.int32)

        if boxes.size == 0:
            return ParsedYOLOv8SignOutput(
                detections=[],
                raw_status=status if status != "unknown" else "ok",
                metadata={"num_raw_boxes": 0, "num_filtered": 0},
            )

        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        detections: list[DetectedSign] = []
        dropped_invalid = 0
        dropped_class = 0
        dropped_conf = 0

        for idx in range(len(boxes)):
            class_id = int(class_ids[idx]) if idx < len(class_ids) else -1
            confidence = float(confidences[idx]) if idx < len(confidences) else 0.0

            if class_id not in ALLOWED_SIGN_CLASS_IDS:
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
                    "Dropped invalid sign bbox after clipping: [%s, %s, %s, %s]",
                    x1,
                    y1,
                    x2,
                    y2,
                )
                continue

            sign_label = SIGN_CLASS_ID_TO_LABEL[class_id]
            detections.append(
                DetectedSign(
                    sign_label=sign_label,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=bbox,
                    speed_limit_kmh=extract_speed_limit_kmh(sign_label),
                    is_regulatory=is_regulatory_label(sign_label),
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
            "YOLOv8 sign parse complete — status=%s, detections=%d",
            parsed_status,
            len(detections),
        )

        return ParsedYOLOv8SignOutput(
            detections=detections,
            raw_status=parsed_status,
            metadata=metadata,
        )

    def build_summary(self, detections: list[DetectedSign]) -> TrafficSignDetectionSummary:
        """Build aggregate statistics from a list of sign detections."""
        count_by_label: dict[str, int] = {}
        for det in detections:
            count_by_label[det.sign_label] = count_by_label.get(det.sign_label, 0) + 1

        nearest = self._select_nearest_sign(detections)
        highest = max(detections, key=lambda d: d.confidence) if detections else None
        active_speed_limit = self._select_active_speed_limit(detections)

        return TrafficSignDetectionSummary(
            count_by_label=count_by_label,
            total_count=len(detections),
            nearest_sign=nearest,
            highest_confidence=highest,
            active_speed_limit_kmh=active_speed_limit,
        )

    @staticmethod
    def _select_nearest_sign(detections: list[DetectedSign]) -> DetectedSign | None:
        """Pick the sign closest to the ego (highest center_y, then largest area)."""
        if not detections:
            return None

        return max(
            detections,
            key=lambda det: (det.bbox.center_y, det.bbox.area),
        )

    @staticmethod
    def _select_active_speed_limit(detections: list[DetectedSign]) -> int | None:
        """Return the most restrictive visible speed limit (minimum km/h)."""
        limits = [
            det.speed_limit_kmh
            for det in detections
            if det.speed_limit_kmh is not None
        ]
        return min(limits) if limits else None

    @staticmethod
    def _clip_bbox_to_frame(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_width: int,
        frame_height: int,
    ) -> SignBoundingBoxData | None:
        """Clip box to frame bounds; return None if invalid after clipping."""
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            return None

        return SignBoundingBoxData.from_xyxy(x1, y1, x2, y2)
