"""Default paths and hyperparameters for traffic-sign YOLOv8 training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# GTSRB defaults (overridable via CLI / environment).
DEFAULT_GTSRB_ROOT = Path(
    os.environ.get(
        "ADAS_GTSRB_ROOT",
        r"C:\Users\gauth\Desktop\ADAS_DATASETS\gtsrb",
    )
)

DEFAULT_DATASET_DIR = _PROJECT_ROOT / "data" / "traffic_sign_yolo"
DEFAULT_MODEL_OUTPUT = (
    _PROJECT_ROOT / "models" / "trained" / "yolov8_sign" / "traffic_signs_yolov8n.pt"
)
DEFAULT_STATS_PATH = _PROJECT_ROOT / "data" / "traffic_sign_yolo" / "dataset_stats.json"


@dataclass
class TrafficSignTrainingConfig:
    """Training pipeline configuration."""

    gtsrb_root: Path = field(default_factory=lambda: DEFAULT_GTSRB_ROOT)
    dataset_dir: Path = field(default_factory=lambda: DEFAULT_DATASET_DIR)
    model_output: Path = field(default_factory=lambda: DEFAULT_MODEL_OUTPUT)
    stats_path: Path = field(default_factory=lambda: DEFAULT_STATS_PATH)

    # YOLOv8n training defaults.
    model_variant: str = "n"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    patience: int = 20
    workers: int = 4
    device: str = ""
    seed: int = 42

    # Augmentation tuned for sign crops and small objects.
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0

    # Hold out 20% of mapped GTSRB train crops for val when official test is absent.
    val_fraction: float = 0.2

    @property
    def dataset_yaml(self) -> Path:
        return self.dataset_dir / "dataset.yaml"

    @property
    def gtsrb_train_dir(self) -> Path:
        return self.gtsrb_root / "Train"

    @property
    def gtsrb_test_dir(self) -> Path:
        return self.gtsrb_root / "Test"

    @property
    def gtsrb_test_csv(self) -> Path:
        return self.gtsrb_root / "Test.csv"

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT
