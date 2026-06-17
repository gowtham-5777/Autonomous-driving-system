# YOLOP Vendor Import Fix Report

**Date:** 2026-06-17  
**Scope:** `src/modules/yolop/vendor/` only  
**Problem:** `vendor/__init__.py` used `from src.modules.yolop.vendor.models import ...`, which fails when `src` is not registered as a top-level package (e.g. Colab with partial `sys.path` setup).

---

## Summary

Replaced the one remaining absolute import inside the vendor package with a relative import. All other vendor Python modules already used relative imports from the Phase 3 vendoring work.

---

## Files Modified

| File | Change |
|------|--------|
| `src/modules/yolop/vendor/__init__.py` | Absolute import → relative import |
| `src/modules/yolop/vendor/README.md` | Documented internal vs external import patterns |

No files outside `src/modules/yolop/vendor/` were modified.

---

## Imports Changed

### `src/modules/yolop/vendor/__init__.py`

| Before | After |
|--------|-------|
| `from src.modules.yolop.vendor.models import MCnet, get_net` | `from .models import MCnet, get_net` |

---

## Audit: All Vendor Python Files

| File | Import style | Status |
|------|--------------|--------|
| `vendor/__init__.py` | `from .models import MCnet, get_net` | **Fixed** |
| `vendor/models/__init__.py` | `from .YOLOP import MCnet, get_net` | Already relative |
| `vendor/models/YOLOP.py` | `from ..utils import ...`, `from .common import ...` | Already relative |
| `vendor/models/common.py` | stdlib + `torch` only | No package imports |
| `vendor/utils/__init__.py` | `from .autoanchor import ...`, `from .utils import ...` | Already relative |
| `vendor/utils/autoanchor.py` | `from .utils import is_parallel` | Already relative |
| `vendor/utils/utils.py` | stdlib + `torch` only | No package imports |

### Patterns searched (in `*.py`)

- `from src.`
- `from modules.`
- `from lib.`
- `import src.`
- `import modules.`
- `import lib.`

**Result:** Zero matching import statements remain in vendor Python files.

---

## Remaining Absolute References (non-import)

These are documentation strings only — not executed imports:

| File | Line | Content |
|------|------|---------|
| `vendor/__init__.py` | docstring | References ``src.modules.yolop.inference`` as documentation |
| `vendor/README.md` | example block | Shows external ADAS import `from src.modules.yolop.vendor import ...` for callers outside the vendor tree |

External integration code (`inference.py`, tests, etc.) correctly continues to use:

```python
from src.modules.yolop.vendor import get_net
```

when the project root is on `sys.path`. That is outside the vendor package and was not changed per requirements.

---

## Validation Performed

### 1. Import audit (automated grep)

Scanned all `src/modules/yolop/vendor/**/*.py` for `from`/`import` lines starting with `src.`, `modules.`, or `lib.`.

**Result:** Pass — no bad import lines.

### 2. Project-root import + forward pass

```python
from src.modules.yolop.vendor import get_net, MCnet
model = get_net(cfg=None)
out = model(torch.zeros(1, 3, 640, 640))
```

**Result:** Pass — 3 outputs; drivable/lane shapes `(1, 2, 640, 640)`.

### 3. Direct `vendor/__init__.py` load

Loaded `vendor/__init__.py` via `importlib` with package search path set to the vendor directory (simulates environments where only the package subtree is wired).

**Result:** Pass — `get_net` and `MCnet` exported successfully via relative import chain.

### 4. Regression tests

```bash
python -m pytest tests/test_lane_detection_pipeline.py -q
```

**Result:** 3/3 passed (unchanged; tests use stub engine).

---

## Conclusion

The vendor package is now self-contained with relative imports throughout. The Colab failure mode caused by `vendor/__init__.py` re-importing through `src.modules.yolop.vendor.models` is resolved.
