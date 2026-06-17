# Package Import Fix — Audit & Recommendations

**Date:** 2026-06-17  
**Scope:** Import strategy across `src/` and Colab portability  
**Status:** Report only — no code changes applied

---

## Executive Summary

The repository uses **two incompatible import styles** inside `src/`:

| Style | Where used | Count |
|-------|------------|-------|
| Absolute `from src.*` | ADAS integration layer | **11 Python files** |
| Relative `from .` / `from ..` | `src/utils/__init__.py`, `src/modules/yolop/vendor/**` | **6 Python files** |
| Absolute `from modules.*` | Nowhere in `src/` | **0** |

Colab failures are caused primarily by **missing `sys.path` setup** and a **missing `src/__init__.py`**, not by `vendor` alone. The vendor package was already fixed to use relative imports; the rest of `src/` still hard-codes `from src.modules...`, which requires the **project root** (not `src/`) on `PYTHONPATH`.

---

## 1. Import Audit — All Files Under `src/`

### 1.1 Files using absolute `from src.*` (portability risk)

| File | Import lines |
|------|--------------|
| `src/modules/__init__.py` | 7 × `from src.modules.*` |
| `src/modules/lane_detection.py` | 8 × `from src.modules.*`, `from src.preprocessing.*`, `from src.utils.*` |
| `src/modules/yolop/__init__.py` | 8 × `from src.modules.yolop.*` |
| `src/modules/yolop/inference.py` | `from src.modules.yolop.vendor import get_net` |
| `src/modules/yolop/output_parser.py` | `from src.modules.yolop.lane_geometry`, `output_schema` |
| `src/modules/vehicle_detection.py` | `from src.modules.base import ...` |
| `src/modules/segmentation.py` | `from src.modules.base import ...` |
| `src/modules/traffic_sign.py` | `from src.modules.base import ...` |
| `src/modules/traffic_signal.py` | `from src.modules.base import ...` |
| `src/preprocessing/__init__.py` | `from src.preprocessing.lane_preprocess import ...` |
| `src/visualization/__init__.py` | `from src.visualization.overlays import ...` |

**Total: 11 files, ~35 import statements.**

### 1.2 Files using relative imports (portable within package)

| File | Pattern |
|------|---------|
| `src/utils/__init__.py` | `from .model_paths import ...` |
| `src/modules/yolop/vendor/__init__.py` | `from .models import MCnet, get_net` |
| `src/modules/yolop/vendor/models/__init__.py` | `from .YOLOP import ...` |
| `src/modules/yolop/vendor/models/YOLOP.py` | `from ..utils import ...`, `from .common import ...` |
| `src/modules/yolop/vendor/utils/__init__.py` | `from .autoanchor`, `from .utils` |
| `src/modules/yolop/vendor/utils/autoanchor.py` | `from .utils import is_parallel` |

### 1.3 Files with no internal `src` imports (stdlib / third-party only)

These files are import-neutral and do not contribute to the mixed-style problem:

| File | Notes |
|------|-------|
| `src/modules/base.py` | No package imports |
| `src/modules/yolop/model_loader.py` | No package imports |
| `src/modules/yolop/lane_geometry.py` | No package imports |
| `src/modules/yolop/postprocess.py` | No package imports |
| `src/modules/yolop/output_schema.py` | No package imports |
| `src/modules/yolop/utils.py` | No package imports |
| `src/preprocessing/lane_preprocess.py` | No package imports |
| `src/preprocessing/image_ops.py` | No package imports |
| `src/utils/model_paths.py` | No package imports |
| `src/visualization/overlays.py` | No package imports |
| `src/visualization/hud.py` | No package imports |
| `src/decision/scene_state.py` | No package imports |
| `src/decision/rules.py` | No package imports |
| `src/pipeline/orchestrator.py` | No package imports |
| `src/app.py` | No package imports |
| `src/modules/yolop/vendor/models/common.py` | torch / numpy only |
| `src/modules/yolop/vendor/utils/utils.py` | torch / numpy only |

### 1.4 Files using `from modules.*` (without `src.`)

**None found under `src/`.**  
`from modules...` is a common Colab mistake when users add `src/` itself to `sys.path` instead of the project root. Python then expects `modules` as a top-level package, which does not match the repository layout.

### 1.5 Package `__init__.py` inventory

| Path | Exists? |
|------|---------|
| `src/__init__.py` | **No** |
| `src/modules/__init__.py` | Yes |
| `src/modules/yolop/__init__.py` | Yes |
| `src/modules/yolop/vendor/__init__.py` | Yes |
| `src/utils/__init__.py` | Yes |
| `src/preprocessing/__init__.py` | Yes |
| `src/visualization/__init__.py` | Yes |
| `src/decision/__init__.py` | **No** |
| `src/pipeline/__init__.py` | **No** |

There is **no `pyproject.toml` / `setup.py`** — the project is not installable as a package. All runtime imports depend on manually adding the project root to `sys.path`.

---

## 2. Recommended Canonical Import Strategy

### Strategy: **Relative imports inside `src/`, absolute `from src.*` only at entry points**

This matches Python packaging best practice for a `src/` layout without an editable install.

#### Rule 1 — Within the same subpackage, use single-dot relative imports

```python
# Inside src/modules/
from .base import BaseModule
from .yolop.inference import YOLOPInferenceEngine
```

#### Rule 2 — Cross subpackage under `src/`, use triple-dot sibling imports

```python
# Inside src/modules/lane_detection.py
from ...utils.model_paths import get_yolop_weights_path
from ...preprocessing.lane_preprocess import LanePreprocessor
```

(`...` resolves to the `src` package, then descends into `utils` or `preprocessing`.)

#### Rule 3 — Within `vendor/`, keep relative imports only (already done)

```python
from .models import get_net
from ..utils import initialize_weights
```

#### Rule 4 — Entry points outside `src/` (tests, scripts, notebooks, Colab cells)

Always bootstrap path once, then use absolute imports:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/adas-project")  # or Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.lane_detection import LaneDetectionModule
from src.modules.yolop.vendor import get_net
```

#### Rule 5 — Add missing package markers

Create empty `src/__init__.py` (required). Optionally add `src/decision/__init__.py` and `src/pipeline/__init__.py`.

#### Rule 6 — Long-term: add `pyproject.toml` with editable install

```bash
pip install -e .
```

Then Colab and local dev can import `src.*` without manual `sys.path` hacks. This is optional but eliminates an entire class of failures.

### What NOT to do in Colab

| Anti-pattern | Why it fails |
|--------------|--------------|
| `sys.path.append(".../src")` | Makes `modules` top-level; `from src.modules...` breaks |
| `from modules.yolop...` | No top-level `modules` package at project root |
| `%cd src` then `import src` | Python looks for `src/src/` on path |
| Import submodule before path bootstrap | Parent package `src` never registered |

---

## 3. Files That Break Package Portability

Any file listed in **§1.1** breaks portability when:

1. Project root is not on `sys.path`, or
2. `src/` directory (instead of project root) is added to `sys.path`, or
3. `src/__init__.py` is missing and the import context expects a regular package.

### Severity ranking

| Priority | File | Issue |
|----------|------|-------|
| **P0** | *(missing)* `src/__init__.py` | `src` is not a proper regular package |
| **P1** | `src/modules/yolop/inference.py` | Absolute vendor import in hot path |
| **P1** | `src/modules/lane_detection.py` | Hub module; 8 absolute cross-package imports |
| **P1** | `src/modules/yolop/__init__.py` | Barrel re-exports all use absolute paths |
| **P2** | `src/modules/__init__.py` | Package entry barrel |
| **P2** | `src/modules/yolop/output_parser.py` | Sibling imports should be relative |
| **P3** | `src/preprocessing/__init__.py` | Self-referential absolute import |
| **P3** | `src/visualization/__init__.py` | Self-referential absolute import |
| **P3** | Four `src/modules/*_detection.py` stubs | Single absolute base import each |

The **vendor** subtree is no longer a portability problem (fixed in prior work).

---

## 4. Proposed Exact Code Changes

### 4.1 Create `src/__init__.py` (new file)

```python
"""Autonomous Driving Assistance System — source package."""
```

### 4.2 `src/modules/lane_detection.py`

```python
# Before
from src.modules.base import BaseModule, Frame
from src.modules.yolop.inference import (...)
from src.modules.yolop.lane_geometry import LaneGeometryExtractor
from src.modules.yolop.model_loader import (...)
from src.modules.yolop.output_parser import YOLOPOutputParser
from src.modules.yolop.output_schema import LaneDetectionResult
from src.modules.yolop.postprocess import postprocess_lane_mask
from src.preprocessing.lane_preprocess import LanePreprocessor
from src.utils.model_paths import get_yolop_weights_path

# After
from .base import BaseModule, Frame
from .yolop.inference import (...)
from .yolop.lane_geometry import LaneGeometryExtractor
from .yolop.model_loader import (...)
from .yolop.output_parser import YOLOPOutputParser
from .yolop.output_schema import LaneDetectionResult
from .yolop.postprocess import postprocess_lane_mask
from ...preprocessing.lane_preprocess import LanePreprocessor
from ...utils.model_paths import get_yolop_weights_path
```

### 4.3 `src/modules/yolop/inference.py`

```python
# Before
from src.modules.yolop.vendor import get_net

# After
from .vendor import get_net
```

### 4.4 `src/modules/yolop/output_parser.py`

```python
# Before
from src.modules.yolop.lane_geometry import LaneGeometryExtractor
from src.modules.yolop.output_schema import (...)

# After
from .lane_geometry import LaneGeometryExtractor
from .output_schema import (...)
```

### 4.5 `src/modules/yolop/__init__.py`

Replace every `from src.modules.yolop.X` with `from .X` (8 blocks):

```python
from .inference import (...)
from .model_loader import (...)
from .lane_geometry import (...)
from .postprocess import (...)
from .output_parser import ParserConfig, YOLOPOutputParser
from .output_schema import (...)
from .utils import (...)
```

### 4.6 `src/modules/__init__.py`

```python
# Before → After
from src.modules.base import ...           → from .base import ...
from src.modules.lane_detection import ... → from .lane_detection import ...
from src.modules.yolop.output_schema import ... → from .yolop.output_schema import ...
from src.modules.segmentation import ...   → from .segmentation import ...
from src.modules.traffic_sign import ...   → from .traffic_sign import ...
from src.modules.traffic_signal import ... → from .traffic_signal import ...
from src.modules.vehicle_detection import ... → from .vehicle_detection import ...
```

### 4.7 Module stubs (`vehicle_detection.py`, `segmentation.py`, `traffic_sign.py`, `traffic_signal.py`)

```python
# Before
from src.modules.base import BaseModule, Frame, PredictionResult

# After
from .base import BaseModule, Frame, PredictionResult
```

### 4.8 `src/preprocessing/__init__.py`

```python
# Before
from src.preprocessing.lane_preprocess import LanePreprocessor

# After
from .lane_preprocess import LanePreprocessor
```

### 4.9 `src/visualization/__init__.py`

```python
# Before
from src.visualization.overlays import (...)

# After
from .overlays import (...)
```

### 4.10 No changes required

| File | Reason |
|------|--------|
| `src/utils/__init__.py` | Already uses `from .model_paths` |
| `src/modules/yolop/vendor/**` | Already relative |
| Leaf modules with no internal imports | N/A |

### 4.11 External entry points (outside `src/`, for Colab documentation)

These should **keep** `from src.*` after path bootstrap:

- `tests/conftest.py` — already inserts `PROJECT_ROOT` into `sys.path` ✓
- `tests/test_lane_detection_pipeline.py` — uses `from src.modules...` ✓
- `scripts/verify_environment.py` — inserts project root ✓

---

## 5. Why `import src` / `from src.modules.yolop.vendor import get_net` Fails in Colab

### 5.1 `src/__init__.py` does not exist

Despite common assumptions, **`src/__init__.py` is not present in this repository** (verified via filesystem search). Python 3.3+ can treat `src/` as a **namespace package** when the project root is on `sys.path`, but:

- Namespace packages are fragile in notebooks (reload order, partial imports).
- Some tooling and IDEs expect `src/__init__.py`.
- Subpackages like `src.modules` have `__init__.py`, but the parent `src` does not — an inconsistent hierarchy.

**Fix:** Add `src/__init__.py`.

### 5.2 Project root not on `sys.path`

`from src.modules...` requires Python to find a top-level name `src`. That only works when the directory **containing** `src/` (the repo root) is on `sys.path`:

```
/content/drive/MyDrive/adas-project/   ← must be on sys.path
    src/
        modules/
        utils/
```

If Colab does instead:

```python
sys.path.append("/content/drive/MyDrive/adas-project/src")
```

then Python sees `modules/` as top-level, and `import src` fails with `ModuleNotFoundError: No module named 'src'`.

### 5.3 Import chain failure propagates

`from src.modules.yolop.vendor import get_net` triggers a deep import chain:

```
src
 └── modules          (src/modules/__init__.py — absolute imports)
      └── yolop       (src/modules/yolop/__init__.py — absolute imports)
           └── vendor (relative ✓)
                └── models
                     └── YOLOP (relative ✓)
```

Any broken link in `modules` or `yolop` `__init__.py` (e.g. failed absolute import) prevents `get_net` from loading even if vendor itself is correct.

### 5.4 Working directory vs package root

In Colab:

```python
%cd /content/drive/MyDrive/adas-project/src
import src  # FAIL — looks for src/src/
```

The cwd does not define the package root; only `sys.path` entries do.

### 5.5 No installable package metadata

Without `pyproject.toml` / `pip install -e .`, every Colab session must manually configure `sys.path`. Forgetting this step produces errors that look like import-style bugs but are environment setup bugs.

### 5.6 Summary diagram

```
Colab cell
    │
    ├─ sys.path has PROJECT ROOT? ──No──► ModuleNotFoundError: src
    │
    ├─ sys.path has src/ folder? ──Yes──► from modules... works, from src... FAILS
    │
    └─ sys.path has PROJECT ROOT? ──Yes──► from src.modules... works IF
            │                              src/__init__.py exists OR
            │                              namespace package resolves
            │
            └─ internal absolute imports in __init__.py chain fail? ──► partial ImportError
```

---

## 6. Recommended Colab Bootstrap Snippet

Until relative-import refactor is complete, use this at the top of every Colab notebook:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/adas-project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Verify
import src  # should succeed after adding src/__init__.py
from src.modules.yolop.vendor import get_net
```

---

## 7. Validation Plan (after code changes)

1. Grep `src/` for remaining `from src.` inside `src/` (target: zero).
2. `python -c "from src.modules.yolop.vendor import get_net"` from project root.
3. `python -m pytest tests/test_lane_detection_pipeline.py` (entry points unchanged).
4. Colab smoke test with PROJECT_ROOT bootstrap only (no `pip install -e`).

---

## 8. Files Modified (when implemented)

| Action | File |
|--------|------|
| **Create** | `src/__init__.py` |
| **Edit** | `src/modules/lane_detection.py` |
| **Edit** | `src/modules/__init__.py` |
| **Edit** | `src/modules/yolop/__init__.py` |
| **Edit** | `src/modules/yolop/inference.py` |
| **Edit** | `src/modules/yolop/output_parser.py` |
| **Edit** | `src/modules/vehicle_detection.py` |
| **Edit** | `src/modules/segmentation.py` |
| **Edit** | `src/modules/traffic_sign.py` |
| **Edit** | `src/modules/traffic_signal.py` |
| **Edit** | `src/preprocessing/__init__.py` |
| **Edit** | `src/visualization/__init__.py` |
| **Optional create** | `src/decision/__init__.py`, `src/pipeline/__init__.py` |

**Total: 1 new file + 11 edits** (vendor subtree already correct).

---

## 9. Remaining Absolute Imports After Refactor

After implementing §4, the only `from src.*` imports should live **outside** `src/`:

| Location | Acceptable? |
|----------|-------------|
| `tests/*.py` | Yes — entry points |
| `scripts/*.py` | Yes — entry points |
| `notebooks/*.ipynb` | Yes — with path bootstrap |
| Inside `src/**` | **No** — should be zero |

External callers and documentation may continue to show `from src.modules.yolop.vendor import get_net` as the public API.
