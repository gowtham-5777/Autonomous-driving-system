"""Default paths and hyperparameters for traffic-signal YOLOv8 training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# BDD100K defaults (overridable via CLI / environment).
DEFAULT_BDD100K_IMAGES = Path(
    r"C:\Users\gauth\Desktop\ADAS_DATASETS\bdd100k_images"
)
DEFAULT_BDD100K_LABELS = Path(
    r"C:\Users\gauth\Desktop\ADAS_DATASETS\bdd100k_labels"
)

DEFAULT_DATASET_DIR = _PROJECT_ROOT / "data" / "traffic_signal_yolo"
DEFAULT_MODEL_OUTPUT = (
    _PROJECT_ROOT / "models" / "trained" / "yolov8_signal" / "traffic_signals_yolov8n.pt"
)
DEFAULT_STATS_PATH = _PROJECT_ROOT / "data" / "traffic_signal_yolo" / "dataset_stats.json"


@dataclass
class TrafficSignalTrainingConfig:
    """Training pipeline configuration."""

    bdd100k_images: Path = field(default_factory=lambda: DEFAULT_BDD100K_IMAGES)
    bdd100k_labels: Path = field(default_factory=lambda: DEFAULT_BDD100K_LABELS)
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

    # Augmentation tuned for small traffic-light boxes.
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0

    @property
    def dataset_yaml(self) -> Path:
        return self.dataset_dir / "dataset.yaml"

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT
