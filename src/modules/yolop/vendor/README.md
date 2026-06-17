# Vendored YOLOP (MCnet)

Third-party code from the official YOLOP repository, isolated under this
directory for lane detection inference.

## Upstream

| Field | Value |
|-------|-------|
| Repository | https://github.com/hustvl/YOLOP |
| Pinned commit | `8d8f68df318c71f01d6f813c024df646c7d1978f` |
| License | MIT (see `LICENSE`) |
| Vendored | 2026-06-17 |

## Local modifications

1. **Import paths** — `lib.models.*` / `lib.utils.*` replaced with relative
   imports inside `vendor/`.
2. **`YOLOP.py`** — removed `sys.path.append(os.getcwd())` and `__main__` demo
   block (depended on `SegmentationMetric`, TensorBoard).
3. **`utils/__init__.py`** — trimmed to inference-only exports
   (`initialize_weights`, `is_parallel`, `check_anchor_order`).
4. **`utils/utils.py`** — `prefetch_generator` import guarded (training-only).
5. **`utils/autoanchor.py`** — `scipy` / `tqdm` lazy-imported inside `kmean_anchors()` (training-only).

## Not vendored

`lib/config/`, `lib/core/`, `lib/dataset/`, `tools/`, `common2.py`, `light.py`.
Mask post-processing is handled by `src/modules/yolop/postprocess.py`.

## Public API

```python
from src.modules.yolop.vendor import get_net, MCnet
```

## Name collision

Do not merge with `src/modules/yolop/utils.py` (ADAS placeholder helpers).
