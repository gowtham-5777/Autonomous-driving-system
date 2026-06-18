"""Unit tests for BDD100K → YOLO traffic-signal conversion."""

from __future__ import annotations

from training.traffic_signal.bdd100k_converter import (
    BDD100K_TO_YOLO_CLASS,
    YOLO_CLASS_NAMES,
    YOLO_TO_ADAS_LABEL,
    convert_bdd100k_annotation,
    is_valid_traffic_light_box,
)


def test_class_mapping_matches_adas_runtime():
    assert BDD100K_TO_YOLO_CLASS == {"red": 0, "yellow": 1, "green": 2}
    assert YOLO_CLASS_NAMES == ["red", "yellow", "green"]
    assert YOLO_TO_ADAS_LABEL == {
        0: "red_light",
        1: "yellow_light",
        2: "green_light",
    }


def test_ignores_none_color_and_invalid_boxes():
    annotation = {
        "frames": [
            {
                "objects": [
                    {
                        "category": "traffic light",
                        "attributes": {"trafficLightColor": "none"},
                        "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 40},
                    },
                    {
                        "category": "traffic light",
                        "attributes": {"trafficLightColor": "green"},
                        "box2d": {"x1": 30, "y1": 30, "x2": 20, "y2": 40},
                    },
                    {
                        "category": "traffic light",
                        "attributes": {"trafficLightColor": "red"},
                        "box2d": {"x1": 100, "y1": 100, "x2": 120, "y2": 140},
                    },
                ]
            }
        ]
    }
    lines, stats = convert_bdd100k_annotation(annotation, 1280, 720)
    assert len(lines) == 1
    assert lines[0].startswith("0 ")
    assert stats["skipped_none_color"] == 1
    assert stats["skipped_invalid_box"] == 1
    assert stats["kept"] == 1


def test_yolo_line_is_normalized():
    annotation = {
        "frames": [
            {
                "objects": [
                    {
                        "category": "traffic light",
                        "attributes": {"trafficLightColor": "yellow"},
                        "box2d": {"x1": 0, "y1": 0, "x2": 128, "y2": 72},
                    }
                ]
            }
        ]
    }
    lines, _ = convert_bdd100k_annotation(annotation, 1280, 720)
    class_id, cx, cy, w, h = map(float, lines[0].split())
    assert int(class_id) == 1
    assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0
    assert 0.0 < w < 1.0 and 0.0 < h < 1.0


def test_green_maps_to_class_two():
    annotation = {
        "frames": [
            {
                "objects": [
                    {
                        "category": "traffic light",
                        "attributes": {"trafficLightColor": "green"},
                        "box2d": {"x1": 100, "y1": 100, "x2": 120, "y2": 140},
                    }
                ]
            }
        ]
    }
    lines, _ = convert_bdd100k_annotation(annotation, 1280, 720)
    assert lines[0].startswith("2 ")


def test_is_valid_traffic_light_box_rejects_oob():
    assert not is_valid_traffic_light_box(
        {"x1": -1, "y1": 0, "x2": 10, "y2": 10},
        1280,
        720,
    )
