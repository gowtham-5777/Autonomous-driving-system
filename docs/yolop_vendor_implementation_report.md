# YOLOP Vendor Implementation Report

**Date:** 2026-06-17  
**Scope:** Phases 1–4 of `docs/yolop_vendor_plan.md`  
**Upstream:** [hustvl/YOLOP](https://github.com/hustvl/YOLOP) @ `8d8f68df318c71f01d6f813c024df646c7d1978f`

---

## Summary

Phases 1–4 are complete. The official YOLOP MCnet architecture is vendored under `src/modules/yolop/vendor/`, imports are rewritten to use relative package paths, and isolated validation confirms `get_net()` instantiation and a dummy forward pass both succeed.

**Not in scope (unchanged):** `inference.py`, `attach_model()`, stub forward pass.

---

## 1. Files Copied

Copied verbatim from the official repository (then modified where noted):

| Official source | Destination |
|-----------------|-------------|
| `lib/models/YOLOP.py` | `src/modules/yolop/vendor/models/YOLOP.py` |
| `lib/models/common.py` | `src/modules/yolop/vendor/models/common.py` |
| `lib/utils/utils.py` | `src/modules/yolop/vendor/utils/utils.py` |
| `lib/utils/autoanchor.py` | `src/modules/yolop/vendor/utils/autoanchor.py` |
| `LICENSE` | `src/modules/yolop/vendor/LICENSE` |

---

## 2. Files Created

| File | Purpose |
|------|---------|
| `src/modules/yolop/vendor/__init__.py` | Public API: `get_net`, `MCnet` |
| `src/modules/yolop/vendor/models/__init__.py` | Re-exports from `YOLOP.py` |
| `src/modules/yolop/vendor/utils/__init__.py` | Inference-minimal utility exports |
| `src/modules/yolop/vendor/README.md` | Upstream pin, modifications log |

---

## 3. Files Modified

| File | Changes |
|------|---------|
| `vendor/models/YOLOP.py` | Import rewrite; removed `sys.path` hacks; removed `__main__` block |
| `vendor/utils/autoanchor.py` | `lib.utils` → relative import; lazy training deps |
| `vendor/utils/utils.py` | Guarded `prefetch_generator` import |

`vendor/models/common.py` was copied as-is (no edits).

---

## 4. Import Rewrites Performed

### `vendor/models/YOLOP.py`

| Removed | Added |
|---------|-------|
| `import sys, os` + `sys.path.append(os.getcwd())` | — |
| `from lib.utils import initialize_weights` | `from ..utils import check_anchor_order, initialize_weights` |
| `from lib.models.common import Conv, SPP, ...` | `from .common import Bottleneck, BottleneckCSP, Concat, Conv, Detect, Focus, SPP, SharpenConv` |
| `from lib.utils import check_anchor_order` | *(merged into `..utils` import above)* |
| `from lib.core.evaluate import SegmentationMetric` | — |
| `from lib.utils.utils import time_synchronized` | — |
| `from torch import tensor` (unused) | — |

### `vendor/utils/autoanchor.py`

| Before | After |
|--------|-------|
| `from lib.utils import is_parallel` | `from .utils import is_parallel` |

### `vendor/utils/__init__.py` (new, trimmed barrel)

```python
from .autoanchor import check_anchor_order
from .utils import initialize_weights, is_parallel
```

Upstream exports for `augmentations`, `plot`, `run_anchor`, etc. were intentionally omitted.

---

## 5. Compatibility Fixes Required

| Issue | Fix applied |
|-------|-------------|
| `prefetch_generator` not installed | Guarded import in `utils/utils.py`; `DataLoaderX` falls back to standard `DataLoader` iteration |
| `tqdm` / `scipy` imported at module level in `autoanchor.py` | Moved to lazy imports inside `kmean_anchors()` (training-only; not needed for `check_anchor_order`) |
| Unused `yaml` import in `autoanchor.py` | Removed |
| `torch.meshgrid` deprecation warning (PyTorch 2.x) | **Not fixed** — warning only, forward pass succeeds; optional fix: add `indexing="ij"` in `common.py` `Detect._make_grid()` during a future sync |
| `End-to-end.pth` not present locally | Validation skipped weight load; path resolves to Colab Drive default (`exists=False` on Windows dev machine) |

No changes were made to `inference.py` or the ADAS integration stub.

---

## 6. Validation Results

Validation command (project root):

```python
from src.modules.yolop.vendor import get_net, MCnet
import torch

model = get_net(cfg=None)
dummy = torch.zeros(1, 3, 640, 640)
model.eval()
with torch.no_grad():
    out = model(dummy)
```

### Results

| Check | Result |
|-------|--------|
| Import `get_net`, `MCnet` | **PASS** |
| `get_net(cfg=None)` returns `MCnet` | **PASS** |
| Forward pass output count | **PASS** — 3 outputs |
| Drivable segmentation shape | **PASS** — `(1, 2, 640, 640)` |
| Lane segmentation shape | **PASS** — `(1, 2, 640, 640)` |
| Detection head output | **PASS** — `tuple` (inference-mode `Detect` output) |
| `load_state_dict` with `End-to-end.pth` | **SKIP** — weights file not on local disk |
| Existing pipeline tests | **PASS** — `tests/test_lane_detection_pipeline.py` (3/3) |

### Environment

- Python 3.13
- PyTorch (system install with CUDA build)
- Project root on `sys.path`

---

## 7. MCnet Instantiation

**Yes — MCnet can be instantiated successfully.**

```text
get_net(cfg=None) -> MCnet
```

`get_net()` ignores `cfg` (same as upstream) and builds `MCnet` with the hardcoded `YOLOP` block configuration. Constructor runs a probe forward pass at 128×128 to set detector strides and anchor scales.

---

## 8. Dummy Forward Pass

**Yes — a dummy forward pass succeeds.**

Input: `torch.zeros(1, 3, 640, 640)`  
Output:

```text
[
  det_out,           # tuple from Detect head (inference mode)
  drivable_seg,      # (1, 2, 640, 640) sigmoid activations
  lane_seg,          # (1, 2, 640, 640) sigmoid activations
]
```

This matches the contract expected by `YOLOPOutputParser` (drivable index 1, lane index 2 in list form).

---

## Final Vendor Tree

```
src/modules/yolop/vendor/
├── __init__.py
├── LICENSE
├── README.md
├── models/
│   ├── __init__.py
│   ├── YOLOP.py
│   └── common.py
└── utils/
    ├── __init__.py
    ├── utils.py
    └── autoanchor.py
```

---

## Next Steps (Phase 5 — not executed)

1. Wire `get_net()` + `load_state_dict()` into `YOLOPInferenceEngine.attach_model()`.
2. Replace `_execute_forward()` stub with real `model(tensor)` call.
3. Align preprocessing normalization with official demo if needed (`/255` vs ImageNet mean/std).
4. Add integration test with real `End-to-end.pth` weights.
5. Remove `_StubYOLOPInferenceEngine` fallback in `tests/conftest.py` when weights are available.
