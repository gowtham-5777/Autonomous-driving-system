# YOLOP Vendor Integration Plan

**Goal:** Integrate the official [hustvl/YOLOP](https://github.com/hustvl/YOLOP) MCnet architecture into this repository under a vendored subtree, without disturbing the existing ADAS integration layer.

**Status:** Planning document only — no code changes or file copies performed.

**Related docs:** `docs/yolop_architecture_audit.md`, `docs/lane_detection_review.md`

---

## 1. Official YOLOP Repository Analysis

### Source repository

| Property | Value |
|----------|-------|
| Repository | [hustvl/YOLOP](https://github.com/hustvl/YOLOP) |
| License | MIT |
| Default branch | `main` |
| Architecture entry point | `lib/models/YOLOP.py` → `class MCnet`, `get_net()` |
| Default weights | `weights/End-to-end.pth` (~91 MB) |
| Checkpoint key | `checkpoint['state_dict']` |

### How Colab uses YOLOP today

This project's Colab setup (see `config/default.yaml`, `YOLOPModelLoader`) aligns with the official inference pattern:

1. Clone or mount the official YOLOP repo (or rely on `sys.path` to `lib/`).
2. Build the network: `model = get_net(cfg)` — `cfg` is accepted but **not used**; `get_net()` always instantiates `MCnet` with the hardcoded `YOLOP` block config.
3. Load weights: `checkpoint = torch.load("End-to-end.pth")` then `model.load_state_dict(checkpoint['state_dict'])`.
4. Run inference: `det_out, da_seg_out, ll_seg_out = model(img_tensor)` under `torch.no_grad()`.

Official reference scripts:

| Script | Role |
|--------|------|
| `hubconf.py` | `yolop(pretrained=True)` — minimal load path |
| `tools/demo.py` | Full image/video demo with NMS, mask resize, visualization |
| `tools/train.py` | Training (not needed for ADAS inference) |

### MCnet forward-pass contract

`MCnet.forward(x)` returns a **3-element list**:

```python
[detections, drivable_area_seg, lane_line_seg]
```

- `detections` — tuple `(inf_out, raw_feature_maps)` from the `Detect` head.
- `drivable_area_seg` — sigmoid-activated tensor (2-class segmentation).
- `lane_line_seg` — sigmoid-activated tensor (2-class segmentation).

This matches the index convention already used in `src/modules/yolop/output_parser.py`:

- Index `1` → drivable area
- Index `2` → lane segmentation

### Official `lib/` layout (relevant subset)

```
lib/
├── __init__.py
├── config/
│   ├── __init__.py          # exports cfg, update_config
│   └── default.py           # training/inference hyperparameters
├── core/
│   ├── general.py           # NMS, scale_coords (demo only)
│   ├── postprocess.py       # morphological_process, connect_lane
│   ├── function.py          # training loops
│   ├── evaluate.py          # SegmentationMetric
│   └── loss.py              # training losses
├── dataset/                 # LoadImages, BDD datasets
├── models/
│   ├── __init__.py          # exports get_net
│   ├── YOLOP.py             # MCnet, get_net, YOLOP block config
│   ├── common.py            # Conv, Detect, BottleneckCSP, ...
│   ├── common2.py           # alternate conv (unused by default)
│   └── light.py             # lightweight variant
└── utils/
    ├── __init__.py
    ├── utils.py             # initialize_weights, is_parallel, ...
    ├── autoanchor.py        # check_anchor_order
    ├── augmentations.py     # letterbox, training aug
    └── plot.py              # visualization
```

### Dependency graph (inference-minimal path)

```
YOLOP.py
  ├── lib.models.common  (Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv)
  ├── lib.utils          (initialize_weights)
  ├── lib.utils          (check_anchor_order)
  ├── lib.utils.utils    (time_synchronized — __main__ only)
  └── lib.core.evaluate  (SegmentationMetric — __main__ only)

common.py
  └── (stdlib + torch only — no lib.* imports)

autoanchor.py
  └── lib.utils          (is_parallel — training helper run_anchor only)

utils/__init__.py
  ├── .utils             (initialize_weights, is_parallel, ...)
  └── .autoanchor        (check_anchor_order, ...)
```

---

## 2. Proposed Destination Structure

Vendored code lives under `src/modules/yolop/vendor/` to keep third-party code isolated from the ADAS integration layer (`inference.py`, `model_loader.py`, `output_parser.py`, etc.).

```
src/modules/yolop/
├── __init__.py                    # existing ADAS exports (unchanged)
├── inference.py                   # future: import get_net from vendor
├── model_loader.py                # existing checkpoint loader
├── output_parser.py               # existing parser
├── postprocess.py                 # existing (already reimplements lib/core/postprocess)
├── utils.py                       # existing ADAS placeholders (name collision — see §4)
└── vendor/                        # NEW — official YOLOP code only
    ├── __init__.py                # re-export get_net, MCnet for integration layer
    ├── LICENSE                    # copy MIT license from upstream
    ├── README.md                  # upstream version pin + source URL + modification notes
    ├── models/
    │   ├── __init__.py
    │   ├── YOLOP.py
    │   └── common.py
    └── utils/
        ├── __init__.py
        ├── utils.py
        └── autoanchor.py
```

### Design rationale

| Decision | Reason |
|----------|--------|
| `vendor/` subtree | Clear boundary between upstream and ADAS code; simplifies license tracking and future updates |
| No top-level `lib/` | Avoids `sys.path` hacks and collision with Python's import semantics |
| No `vendor/config/` | `get_net(cfg)` ignores `cfg`; ADAS config lives in `config/default.yaml` |
| Keep `postprocess.py` at integration layer | Already adapted from `lib/core/postprocess.py`; no need to duplicate |
| Rename path only, not files | Minimizes diff when syncing with upstream |

---

## 3. Official → Destination File Mapping

### Files to copy (8 files)

| # | Official source | Destination | Action |
|---|-----------------|-------------|--------|
| 1 | `lib/models/YOLOP.py` | `src/modules/yolop/vendor/models/YOLOP.py` | Copy + rewrite imports |
| 2 | `lib/models/common.py` | `src/modules/yolop/vendor/models/common.py` | Copy as-is |
| 3 | `lib/models/__init__.py` | `src/modules/yolop/vendor/models/__init__.py` | Copy + adjust relative import |
| 4 | `lib/utils/utils.py` | `src/modules/yolop/vendor/utils/utils.py` | Copy as-is (training helpers harmless) |
| 5 | `lib/utils/autoanchor.py` | `src/modules/yolop/vendor/utils/autoanchor.py` | Copy + rewrite imports |
| 6 | `lib/utils/__init__.py` | `src/modules/yolop/vendor/utils/__init__.py` | Copy + trim to inference-only exports |
| 7 | *(new)* | `src/modules/yolop/vendor/__init__.py` | Create — re-export `get_net`, `MCnet` |
| 8 | `LICENSE` | `src/modules/yolop/vendor/LICENSE` | Copy MIT license text |

### New files to author (not from upstream)

| File | Purpose |
|------|---------|
| `src/modules/yolop/vendor/README.md` | Record upstream commit SHA, vendoring date, local modifications |
| `src/modules/yolop/vendor/__init__.py` | Stable public API for integration layer |

Suggested `vendor/__init__.py` public API:

```python
from src.modules.yolop.vendor.models import get_net
from src.modules.yolop.vendor.models.YOLOP import MCnet

__all__ = ["get_net", "MCnet"]
```

Suggested `vendor/models/__init__.py`:

```python
from src.modules.yolop.vendor.models.YOLOP import get_net, MCnet

__all__ = ["get_net", "MCnet"]
```

---

## 4. Import Path Changes Required

### 4.1 In `vendor/models/YOLOP.py`

| Official import | Replacement |
|-----------------|-------------|
| `import sys, os` + `sys.path.append(os.getcwd())` | **Remove** — not needed with package layout |
| `from lib.utils import initialize_weights` | `from src.modules.yolop.vendor.utils import initialize_weights` |
| `from lib.models.common import Conv, SPP, ...` | `from src.modules.yolop.vendor.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv` |
| `from lib.utils import check_anchor_order` | `from src.modules.yolop.vendor.utils import check_anchor_order` |
| `from lib.core.evaluate import SegmentationMetric` | **Remove** — only used in `if __name__ == "__main__"` block |
| `from lib.utils.utils import time_synchronized` | **Remove** — only used in `__main__` block |

**Preferred alternative (less brittle):** use relative imports inside the vendor package:

```python
from ..utils import initialize_weights, check_anchor_order
from .common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
```

Relative imports are recommended — they survive package moves and do not hardcode `src.`.

### 4.2 In `vendor/utils/autoanchor.py`

| Official import | Replacement |
|-----------------|-------------|
| `from lib.utils import is_parallel` | `from src.modules.yolop.vendor.utils.utils import is_parallel` |

Or relative: `from .utils import is_parallel`

### 4.3 In `vendor/utils/__init__.py`

Trim the upstream barrel export to inference dependencies only:

```python
# Official (exports everything):
from .utils import initialize_weights, xyxy2xywh, is_parallel, DataLoaderX, torch_distributed_zero_first, clean_str
from .autoanchor import check_anchor_order, run_anchor, kmean_anchors
from .augmentations import augment_hsv, random_perspective, cutout, letterbox, letterbox_for_img
from .plot import plot_img_and_mask, plot_one_box, show_seg_result

# Vendored (inference-minimal):
from .utils import initialize_weights, is_parallel
from .autoanchor import check_anchor_order

__all__ = ["initialize_weights", "is_parallel", "check_anchor_order"]
```

### 4.4 In ADAS integration layer (future wiring — not part of vendor copy)

These files will eventually import from vendor instead of stubbing:

| File | Future import |
|------|---------------|
| `src/modules/yolop/inference.py` | `from src.modules.yolop.vendor import get_net` |
| `src/modules/yolop/model_loader.py` | No vendor import needed (checkpoint loading stays here) |

### 4.5 Naming collision: `utils.py`

| Path | Role |
|------|------|
| `src/modules/yolop/utils.py` | ADAS placeholder parsing helpers |
| `src/modules/yolop/vendor/utils/` | Official YOLOP utilities package |

No conflict at import time — different module paths. Document clearly in `vendor/README.md` to prevent accidental merges.

### 4.6 `lib.models.*` → vendor mapping summary

| Official | Vendored |
|----------|----------|
| `from lib.models import get_net` | `from src.modules.yolop.vendor import get_net` |
| `from lib.models.YOLOP import MCnet` | `from src.modules.yolop.vendor import MCnet` |
| `from lib.models.common import Detect` | `from src.modules.yolop.vendor.models.common import Detect` |

### 4.7 `lib.utils.*` → vendor mapping summary

| Official | Vendored |
|----------|----------|
| `from lib.utils import initialize_weights` | `from src.modules.yolop.vendor.utils import initialize_weights` |
| `from lib.utils import check_anchor_order` | `from src.modules.yolop.vendor.utils import check_anchor_order` |
| `from lib.utils.utils import select_device` | Not vendored — ADAS uses `InferenceConfig.device` |
| `from lib.utils import plot_one_box` | Not vendored — ADAS uses `src/visualization/overlays.py` |

---

## 5. Files That Can Be Omitted

### Omit entirely (not needed for lane/drivable inference)

| Official path | Reason |
|---------------|--------|
| `lib/__init__.py` | Empty; replaced by `vendor/__init__.py` |
| `lib/config/` | `get_net(cfg)` ignores `cfg`; ADAS has its own `config/default.yaml` |
| `lib/core/` | Training, NMS demo, evaluation — ADAS layer handles post-inference |
| `lib/dataset/` | Training and demo dataloaders only |
| `lib/models/common2.py` | Commented out in `YOLOP.py`; not used by `get_net()` |
| `lib/models/light.py` | Alternate architecture; not used by `End-to-end.pth` |
| `lib/utils/augmentations.py` | Training/demo letterbox; ADAS preprocesses in `inference.py` |
| `lib/utils/plot.py` | Demo visualization |
| `lib/utils/split_dataset.py` | Dataset tooling |
| `tools/` | CLI train/test/demo scripts |
| `hubconf.py` | PyTorch Hub entry; replaced by `vendor/__init__.py` API |
| `export_onnx.py`, `test_onnx.py` | ONNX export/testing |
| `toolkits/` | C++ TensorRT deployment |
| `inference/` | Sample images/videos |
| `pictures/` | Documentation assets |
| `weights/` | Weights live in `models/pretrained/yolop/` per ADAS config |

### Already covered elsewhere in ADAS

| Official path | ADAS equivalent |
|---------------|-----------------|
| `lib/core/postprocess.py` | `src/modules/yolop/postprocess.py` (morphology, `connect_lane`) |
| `lib/core/general.py` (`non_max_suppression`) | Not needed for lane-only MVP; add later if object head is used |
| `lib/utils/utils.py` (`select_device`) | `InferenceConfig.device` + `model_loader.device` |

### Optional third-party dependency note

Official `lib/utils/utils.py` imports `prefetch_generator` (training DataLoader only). Not required for `initialize_weights()`. If import errors occur, either:

- Keep the import (add `prefetch-generator` to `requirements.txt`), or
- Guard/remove `DataLoaderX` class (training-only) in the vendored copy.

---

## 6. Step-by-Step Migration Plan

### Phase 0 — Preparation

1. Read `docs/yolop_architecture_audit.md` and confirm scope: inference-only vendoring.
2. Pin upstream version in `vendor/README.md` (record git commit SHA from `hustvl/YOLOP`).
3. Copy `LICENSE` from upstream into `vendor/LICENSE`.
4. Verify `End-to-end.pth` is available at the path resolved by `get_yolop_weights_path()` (`models/pretrained/yolop/End-to-end.pth` locally or on Colab Drive).
5. Confirm `torch>=2.0` in `requirements.txt` is compatible (official repo targets older PyTorch but MCnet is standard `nn.Module`).

### Phase 1 — Create vendor directory skeleton

1. Create `src/modules/yolop/vendor/`.
2. Create `src/modules/yolop/vendor/models/` and `src/modules/yolop/vendor/utils/`.
3. Add empty `__init__.py` files at each package level.

### Phase 2 — Copy upstream files

1. Copy `lib/models/common.py` → `vendor/models/common.py` (no edits).
2. Copy `lib/models/YOLOP.py` → `vendor/models/YOLOP.py`.
3. Copy `lib/utils/utils.py` → `vendor/utils/utils.py`.
4. Copy `lib/utils/autoanchor.py` → `vendor/utils/autoanchor.py`.
5. Do **not** copy `common2.py`, `light.py`, `config/`, `core/`, `dataset/`, or demo scripts.

### Phase 3 — Rewrite imports in vendored files

1. **`vendor/models/YOLOP.py`:**
   - Remove `sys.path.append(os.getcwd())`.
   - Replace all `lib.*` imports with relative imports (`from ..utils import ...`, `from .common import ...`).
   - Remove `SegmentationMetric` and `time_synchronized` imports.
   - Remove or guard the `if __name__ == "__main__"` block (references removed deps).

2. **`vendor/utils/autoanchor.py`:**
   - Change `from lib.utils import is_parallel` → `from .utils import is_parallel`.

3. **`vendor/utils/__init__.py`:**
   - Export only `initialize_weights`, `is_parallel`, `check_anchor_order`.

4. **`vendor/models/__init__.py`:**
   - `from .YOLOP import get_net, MCnet`.

5. **`vendor/__init__.py`:**
   - Re-export `get_net` and `MCnet` as the stable public API.

### Phase 4 — Validate vendor package in isolation

Run from project root (no ADAS wiring yet):

```python
from src.modules.yolop.vendor import get_net
import torch

model = get_net(cfg=None)
dummy = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    out = model(dummy)
assert len(out) == 3  # [det, drivable, lane]
```

Load real weights:

```python
from src.utils.model_paths import get_yolop_weights_path

ckpt = torch.load(get_yolop_weights_path(), map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

**Checkpoint troubleshooting:** If `load_state_dict` raises key mismatch, inspect `ckpt` top-level keys (`state_dict` vs `model`) — `YOLOPModelLoader._extract_state_dict()` already normalizes this for the integration layer.

### Phase 5 — Wire ADAS integration layer (separate PR)

*Not part of vendor copy; documented here for end-to-end completeness.*

1. **`inference.py` → `attach_model()`:**
   - Call `get_net(cfg=None)`.
   - `model.load_state_dict(model_package["state_dict"])`.
   - `model.to(self.config.device).eval()`.
   - Store `nn.Module` on `self._model`; set `_architecture_ready = True`.

2. **`inference.py` → `_execute_forward()`:**
   - Convert `preprocessed["input_tensor"]` to `torch.Tensor`.
   - Run `det_out, da_seg_out, ll_seg_out = self._model(tensor)`.
   - Map outputs to format expected by `YOLOPOutputParser` (list indices 1 and 2).

3. **`inference.py` → `preprocess()`:**
   - Align normalization with official demo if needed (official uses ImageNet mean/std; current ADAS code uses `/255` only — **document and test** before merging).

4. **Remove test stub dependency:**
   - Update `tests/conftest.py` to use real `YOLOPInferenceEngine` when weights are present; keep stub as fallback.

### Phase 6 — Integration tests

1. Add test: vendor `get_net()` + `End-to-end.pth` produces non-empty lane mask on `tests/fixtures/road_sample.jpg`.
2. Run existing pipeline test without `_StubYOLOPInferenceEngine`:
   ```bash
   python -m pytest tests/test_lane_detection_pipeline.py -v -s --log-cli-level=INFO
   ```
3. Assert `lane_mask` is non-`None` and `lane_center_x` is plausible.

### Phase 7 — Documentation and maintenance

1. Update `docs/yolop_architecture_audit.md` — mark architecture as integrated.
2. Update `src/modules/yolop/__init__.py` docstring — remove "No YOLOP code" statement.
3. Add vendoring policy to `vendor/README.md`:
   - Upstream URL and pinned commit
   - List of local modifications (import paths, trimmed `__init__.py`, removed `__main__`)
   - Procedure for syncing with upstream

---

## 7. Risk Register

| Risk | Mitigation |
|------|------------|
| Preprocessing mismatch (`/255` vs ImageNet normalize) | Compare outputs against official `tools/demo.py` on a fixed frame before enabling real inference |
| `load_state_dict` key prefix mismatch | Reuse `YOLOPModelLoader._extract_state_dict()`; log `sample_tensor_keys` on failure |
| `prefetch_generator` import in `utils.py` | Lazy-import or strip `DataLoaderX` from vendored copy |
| Name collision `yolop/utils.py` vs `yolop/vendor/utils/` | Document in vendor README; never merge the two |
| GPU device split (`model_loader.device` vs `InferenceConfig.device`) | Single source of truth in `attach_model()` |
| Official code uses `eval(block)` for layer names | Keep vendored `YOLOP.py` intact — `eval()` resolves `Conv`, `Detect`, etc. from imports |

---

## 8. Success Criteria

- [ ] `from src.modules.yolop.vendor import get_net, MCnet` succeeds
- [ ] `get_net(None)` builds MCnet without `sys.path` manipulation
- [ ] `End-to-end.pth` loads with zero missing/unexpected keys
- [ ] Forward pass on `(1, 3, 640, 640)` returns 3 outputs with expected shapes
- [ ] `YOLOPInferenceEngine` produces non-empty masks (after Phase 5 wiring)
- [ ] Integration tests pass with real weights on CPU
- [ ] No files copied from `lib/config`, `lib/core`, `lib/dataset`, or `tools/`

---

## 9. Quick Reference

### Minimum vendored tree (final state)

```
src/modules/yolop/vendor/
├── __init__.py          # export get_net, MCnet
├── LICENSE
├── README.md
├── models/
│   ├── __init__.py
│   ├── YOLOP.py         # MCnet + get_net + YOLOP config
│   └── common.py        # layer building blocks
└── utils/
    ├── __init__.py      # initialize_weights, check_anchor_order
    ├── utils.py
    └── autoanchor.py
```

### Integration entry point (after Phase 5)

```python
from src.modules.yolop.vendor import get_net

model = get_net(cfg=None)
model.load_state_dict(state_dict)
model.to(device).eval()

with torch.no_grad():
    det_out, da_seg_out, ll_seg_out = model(input_tensor)
```

### Checkpoint path (this project)

```
models/pretrained/yolop/End-to-end.pth
# Colab: /content/drive/MyDrive/adas-project/models/pretrained/yolop/End-to-end.pth
```
