"""Unit tests for GTSRB → YOLO traffic-sign conversion."""

from __future__ import annotations

from src.modules.yolov8_sign.class_map import (
    GTSRB_CLASS_ID_TO_ADAS_LABEL,
    SIGN_CLASS_ID_TO_LABEL,
)
from training.traffic_sign.gtsrb_converter import (
    GTSRB_TO_YOLO_CLASS,
    YOLO_CLASS_NAMES,
    YOLO_TO_ADAS_LABEL,
    gtsrb_class_id_to_yolo_class,
    gtsrb_test_row_to_yolo_line,
    train_crop_to_yolo_line,
)


def test_class_mapping_matches_adas_runtime():
    expected_names = [SIGN_CLASS_ID_TO_LABEL[i] for i in range(7)]
    assert YOLO_CLASS_NAMES == expected_names
    assert YOLO_TO_ADAS_LABEL == {i: name for i, name in enumerate(expected_names)}

    for gtsrb_id, adas_label in GTSRB_CLASS_ID_TO_ADAS_LABEL.items():
        yolo_id = gtsrb_class_id_to_yolo_class(gtsrb_id)
        assert yolo_id is not None
        assert YOLO_CLASS_NAMES[yolo_id] == adas_label
        assert GTSRB_TO_YOLO_CLASS[gtsrb_id] == yolo_id


def test_train_crop_uses_full_frame_bbox():
    line = train_crop_to_yolo_line(0)
    class_id, cx, cy, w, h = map(float, line.split())
    assert int(class_id) == 0
    assert cx == 0.5 and cy == 0.5 and w == 1.0 and h == 1.0


def test_unmapped_gtsrb_class_returns_none():
    assert gtsrb_class_id_to_yolo_class(0) is None
    assert gtsrb_class_id_to_yolo_class(99) is None


def test_test_csv_row_converts_roi_to_yolo():
    row = {
        "Filename": "00000.ppm",
        "Width": "100",
        "Height": "100",
        "Roi.X1": "10",
        "Roi.Y1": "20",
        "Roi.X2": "90",
        "Roi.Y2": "80",
        "ClassId": "14",
    }
    parsed = gtsrb_test_row_to_yolo_line(row)
    assert parsed is not None
    line, width, height = parsed
    assert width == 100 and height == 100
    class_id, cx, cy, w, h = map(float, line.split())
    assert int(class_id) == 0  # stop
    assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0
    assert 0.0 < w < 1.0 and 0.0 < h < 1.0


def test_unmapped_test_row_is_skipped():
    row = {
        "Filename": "00000.ppm",
        "Width": "100",
        "Height": "100",
        "Roi.X1": "10",
        "Roi.Y1": "20",
        "Roi.X2": "90",
        "Roi.Y2": "80",
        "ClassId": "2",
    }
    assert gtsrb_test_row_to_yolo_line(row) is None
