# Package Portability Fix Report

**Date:** 2026-06-17  
**Scope:** `src/modules/**` and `src/modules/yolop/**`  
**Goal:** Replace absolute `from src.modules...` imports with package-relative imports inside module code.

---

## Summary

All absolute `from src.*` imports inside `src/modules/` and `src/modules/yolop/` have been removed. The `modules` and `yolop` packages now use relative imports exclusively. Cross-package imports from `lane_detection.py` to `preprocessing` and `utils` use two-dot sibling imports (`from ..preprocessing`, `from ..utils`).

A minimal `src/__init__.py` was added because it is required for `src` to act as a regular parent package when resolving relative imports from `src.modules.*`.

---

## 1. Files Modified

| File | Change |
|------|--------|
| `src/__init__.py` | **Created** — marks `src` as a regular package (required for relative import resolution) |
| `src/modules/__init__.py` | Absolute → relative imports |
| `src/modules/lane_detection.py` | Absolute → relative imports |
| `src/modules/vehicle_detection.py` | Absolute → relative import |
| `src/modules/segmentation.py` | Absolute → relative import |
| `src/modules/traffic_sign.py` | Absolute → relative import |
| `src/modules/traffic_signal.py` | Absolute → relative import |
| `src/modules/yolop/__init__.py` | Absolute → relative imports |
| `src/modules/yolop/inference.py` | Absolute → relative import |
| `src/modules/yolop/output_parser.py` | Absolute → relative imports |

**Total: 10 files** (9 edited + 1 created).

No changes to `vendor/` (already relative), `model_loader.py`, `postprocess.py`, `lane_geometry.py`, `output_schema.py`, or `utils.py`.

---

## 2. Before / After Imports

### `src/modules/__init__.py`

| Before | After |
|--------|-------|
| `from src.modules.base import BaseModule, Frame, PredictionResult` | `from .base import BaseModule, Frame, PredictionResult` |
| `from src.modules.lane_detection import LaneDetectionModule, LANE_OUTPUT_KEYS` | `from .lane_detection import LaneDetectionModule, LANE_OUTPUT_KEYS` |
| `from src.modules.yolop.output_schema import LaneDetectionResult` | `from .yolop.output_schema import LaneDetectionResult` |
| `from src.modules.segmentation import SegmentationModule` | `from .segmentation import SegmentationModule` |
| `from src.modules.traffic_sign import TrafficSignModule` | `from .traffic_sign import TrafficSignModule` |
| `from src.modules.traffic_signal import TrafficSignalModule` | `from .traffic_signal import TrafficSignalModule` |
| `from src.modules.vehicle_detection import VehicleDetectionModule` | `from .vehicle_detection import VehicleDetectionModule` |

### `src/modules/lane_detection.py`

| Before | After |
|--------|-------|
| `from src.modules.base import BaseModule, Frame` | `from .base import BaseModule, Frame` |
| `from src.modules.yolop.inference import (...)` | `from .yolop.inference import (...)` |
| `from src.modules.yolop.lane_geometry import LaneGeometryExtractor` | `from .yolop.lane_geometry import LaneGeometryExtractor` |
| `from src.modules.yolop.model_loader import (...)` | `from .yolop.model_loader import (...)` |
| `from src.modules.yolop.output_parser import YOLOPOutputParser` | `from .yolop.output_parser import YOLOPOutputParser` |
| `from src.modules.yolop.output_schema import LaneDetectionResult` | `from .yolop.output_schema import LaneDetectionResult` |
| `from src.modules.yolop.postprocess import postprocess_lane_mask` | `from .yolop.postprocess import postprocess_lane_mask` |
| `from src.preprocessing.lane_preprocess import LanePreprocessor` | `from ..preprocessing.lane_preprocess import LanePreprocessor` |
| `from src.utils.model_paths import get_yolop_weights_path` | `from ..utils.model_paths import get_yolop_weights_path` |

### Module stubs (`vehicle_detection.py`, `segmentation.py`, `traffic_sign.py`, `traffic_signal.py`)

| Before | After |
|--------|-------|
| `from src.modules.base import BaseModule, Frame, PredictionResult` | `from .base import BaseModule, Frame, PredictionResult` |

### `src/modules/yolop/__init__.py`

| Before | After |
|--------|-------|
| `from src.modules.yolop.inference import (...)` | `from .inference import (...)` |
| `from src.modules.yolop.model_loader import (...)` | `from .model_loader import (...)` |
| `from src.modules.yolop.lane_geometry import (...)` | `from .lane_geometry import (...)` |
| `from src.modules.yolop.postprocess import (...)` | `from .postprocess import (...)` |
| `from src.modules.yolop.output_parser import ParserConfig, YOLOPOutputParser` | `from .output_parser import ParserConfig, YOLOPOutputParser` |
| `from src.modules.yolop.output_schema import (...)` | `from .output_schema import (...)` |
| `from src.modules.yolop.utils import (...)` | `from .utils import (...)` |

### `src/modules/yolop/inference.py`

| Before | After |
|--------|-------|
| `from src.modules.yolop.vendor import get_net` | `from .vendor import get_net` |

### `src/modules/yolop/output_parser.py`

| Before | After |
|--------|-------|
| `from src.modules.yolop.lane_geometry import LaneGeometryExtractor` | `from .lane_geometry import LaneGeometryExtractor` |
| `from src.modules.yolop.output_schema import (...)` | `from .output_schema import (...)` |

---

## 3. Remaining Absolute Imports and Justification

### Inside `src/modules/**` and `src/modules/yolop/**`

**None.** Grep for `from src.` / `import src` in `src/modules/**/*.py` returns zero matches.

### Inside `src/modules/yolop/vendor/**`

**None.** Vendor subtree already uses relative imports only (prior fix).

### Outside `src/modules/` (not modified — documented for completeness)

| File | Import | Justification |
|------|--------|---------------|
| `src/preprocessing/__init__.py` | `from src.preprocessing.lane_preprocess import ...` | Outside scope; self-referential absolute import — candidate for future `from .lane_preprocess` fix |
| `src/visualization/__init__.py` | `from src.visualization.overlays import ...` | Outside scope; same pattern |
| `tests/*.py` | `from src.modules...` | **Intentional** — entry points after `sys.path` bootstrap |
| `scripts/verify_environment.py` | `from src.utils...` | **Intentional** — script entry point |

### Supporting file added (not under `modules/`)

| File | Reason |
|------|--------|
| `src/__init__.py` | Required so `src.modules.lane_detection` can resolve `from ..preprocessing` and `from ..utils` without `ImportError: attempted relative import beyond top-level package` |

---

## 4. Validation Steps Performed

### 4.1 Static audit

```text
grep "from src." src/modules/**/*.py  → 0 matches
grep "from src." src/modules/yolop/vendor/**/*.py  → 0 matches
```

### 4.2 Import smoke test (project root on `sys.path`)

```python
from src.modules import LaneDetectionModule, BaseModule, VehicleDetectionModule
from src.modules.yolop import YOLOPInferenceEngine
from src.modules.yolop.vendor import get_net
```

**Result:** Pass

### 4.3 MCnet forward pass

```python
model = get_net(cfg=None)
out = model(torch.zeros(1, 3, 640, 640))  # 3 outputs
```

**Result:** Pass

### 4.4 Regression tests

```bash
python -m pytest tests/test_lane_detection_pipeline.py -q
```

**Result:** 3/3 passed

---

## 5. Relative Import Reference

From a module in `src.modules` (e.g. `lane_detection.py`):

| Target | Import |
|--------|--------|
| Same package (`base`, `yolop.*`) | `from .base import ...` / `from .yolop.inference import ...` |
| Sibling under `src` (`preprocessing`, `utils`) | `from ..preprocessing...` / `from ..utils...` |

From a module in `src.modules.yolop` (e.g. `inference.py`):

| Target | Import |
|--------|--------|
| Same subpackage | `from .vendor import get_net` |
| Same subpackage | `from .output_schema import ...` |

**Note:** Use `..` (two dots) for siblings under `src`, not `...` (three dots). Three dots would resolve above `src` and raise `attempted relative import beyond top-level package`.

---

## 6. Colab Usage (unchanged entry-point pattern)

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/adas-project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.lane_detection import LaneDetectionModule
from src.modules.yolop.vendor import get_net
```

Internal module code no longer depends on the `src.` prefix; only external callers need the project root on `sys.path`.
