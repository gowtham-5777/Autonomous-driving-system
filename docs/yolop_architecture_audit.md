# YOLOP Architecture Audit

**Repository:** `Autonomous Driving Car`  
**Audit date:** 2026-06-17  
**Official reference:** [hustvl/YOLOP](https://github.com/hustvl/YOLOP)

---

## Executive Summary

The official YOLOP **MCnet** neural network architecture is **not present** anywhere in this repository. The project contains an integration scaffold (checkpoint loading, inference wrapper, output parsing) but no `torch.nn.Module` definitions, no `get_net()`, and no `lib/models/` tree. Inference is explicitly stubbed pending architecture integration.

---

## 1. Is the official YOLOP MCnet architecture present anywhere in this repository?

**No.**

A repository-wide search found:

| Pattern | Matches |
|---------|---------|
| `class MCnet` | **0** |
| `MCnet` (any reference) | **0** |
| `get_net(` | **0** |
| `lib.models` / `lib/models` | **0** |
| `torch.nn` / `nn.Module` definitions | **0** (only mentioned in docs and error strings) |

The only `torch` usage under `src/modules/yolop/` is in `model_loader.py` for `torch.load()` — it deserializes checkpoint dictionaries but never constructs a network.

---

## 2. Search Results

### `class MCnet`

```
No matches found
```

### `get_net(`

```
No matches found
```

### `YOLOP`

**17 files** reference the string `YOLOP`, but none define the neural network:

| File | Nature of reference |
|------|---------------------|
| `src/modules/yolop/inference.py` | Inference wrapper; forward pass stubbed |
| `src/modules/yolop/model_loader.py` | Checkpoint loading only |
| `src/modules/yolop/output_parser.py` | Output parsing (expects tensors) |
| `src/modules/yolop/output_schema.py` | Dataclass schemas |
| `src/modules/yolop/postprocess.py` | Mask post-processing (OpenCV/NumPy) |
| `src/modules/yolop/lane_geometry.py` | Geometry from binary masks |
| `src/modules/yolop/utils.py` | Placeholder parsing helpers |
| `src/modules/yolop/__init__.py` | Package exports |
| `src/modules/lane_detection.py` | Pipeline orchestration |
| `src/preprocessing/lane_preprocess.py` | Classical preprocessing (no YOLOP logic) |
| `src/utils/model_paths.py` | Weight path resolution |
| `src/visualization/overlays.py` | Placeholder overlays |
| `config/default.yaml` | Model name / weight filename |
| `tests/conftest.py` | Test fixtures + stub inference engine |
| `tests/test_lane_detection_pipeline.py` | Integration tests |
| `scripts/verify_environment.py` | Environment check |
| `scripts/download_weights.py` | TODO stub |
| `README.md` | Module list |
| `docs/lane_detection_review.md` | Internal review |

### `lib.models`

```
No matches found
```

No `lib/` directory exists in this project. The official YOLOP repo structure (`lib/models/YOLOP.py`, `lib/models/common.py`, etc.) has not been vendored.

---

## 3. Files Containing YOLOP Model Architecture Code

**None.**

There are zero files in this repository that contain YOLOP model architecture code (`MCnet`, `get_net`, layer blocks, or `nn.Module` building blocks like `Conv`, `BottleneckCSP`, `Detect` from the official repo).

### Related files that are **not** architecture

These files are YOLOP-related but operate **downstream** of or **around** the missing network:

| File | What it contains |
|------|------------------|
| `src/modules/yolop/inference.py` | Preprocessing + stub `_execute_forward()` |
| `src/modules/yolop/model_loader.py` | `torch.load` + `state_dict` extraction |
| `src/modules/yolop/postprocess.py` | Mask morphology adapted from official `lib/core/postprocess.py` |
| `src/modules/yolop/output_parser.py` | Parses raw head outputs (given real tensors) |
| `src/modules/yolop/lane_geometry.py` | Pixel geometry from binary masks |

`postprocess.py` references the official repo's `lib/core/postprocess.py`, which is **post-inference mask refinement**, not model architecture.

---

## 4. Minimum Files to Copy from the Official YOLOP Repository

Since no architecture exists, these are the **minimum files** required to instantiate `MCnet` and load `End-to-end.pth` for inference:

### Required — model definition

| Official path | Purpose |
|---------------|---------|
| `lib/models/YOLOP.py` | `MCnet` class, `YOLOP` block config, `get_net()` factory |
| `lib/models/common.py` | Building blocks: `Conv`, `SPP`, `Bottleneck`, `BottleneckCSP`, `Focus`, `Concat`, `Detect`, `SharpenConv` |
| `lib/models/__init__.py` | Package init |

### Required — model construction dependencies

| Official path | Purpose |
|---------------|---------|
| `lib/utils/__init__.py` | Re-exports `initialize_weights`, `check_anchor_order` |
| `lib/utils/utils.py` | `initialize_weights()` (called in `MCnet.__init__`) |
| `lib/utils/autoanchor.py` | `check_anchor_order()` (called during detector stride setup) |
| `lib/__init__.py` | Package init (empty) |

### Not required for basic lane/drivable inference

| Official path | Reason to skip |
|---------------|----------------|
| `lib/models/common2.py` | Alternate conv impl; commented out in `YOLOP.py` |
| `lib/models/light.py` | Lightweight variant; not used by `get_net()` default |
| `lib/config/` | Training config; inference uses hardcoded `YOLOP` block in `YOLOP.py` |
| `lib/core/postprocess.py` | Already partially reimplemented in `src/modules/yolop/postprocess.py` |
| `lib/core/evaluate.py` | Only used in `YOLOP.py` `__main__` block |
| `lib/dataset/` | Training only |
| `lib/utils/augmentations.py` | Training / demo preprocessing |
| `lib/utils/plot.py` | Visualization only |

### Minimum copy set (8 files)

```
lib/__init__.py
lib/models/__init__.py
lib/models/YOLOP.py
lib/models/common.py
lib/utils/__init__.py
lib/utils/utils.py
lib/utils/autoanchor.py
```

### Integration notes

After copying, `attach_model()` in `inference.py` would need to:

1. Call `get_net(cfg)` to build `MCnet`
2. Call `model.load_state_dict(state_dict)` (may require key prefix stripping depending on checkpoint format)
3. Move model to `self.config.device` and call `model.eval()`
4. Replace `_execute_forward()` stub with `model(input_tensor)` under `torch.no_grad()`

Official `MCnet.forward()` returns:

```python
[detections, drivable_area_seg, lane_line_seg]
```

where segmentation heads are sigmoid-activated. This maps to indices used by `YOLOPOutputParser` (drivable = index 1, lane = index 2 in raw multi-head convention).

`YOLOP.py` uses `sys.path.append(os.getcwd())` and `from lib.models.common import ...` — import paths will need adjustment when vendoring into `src/`.

---

## 5. If Architecture Existed — Why Would `attach_model()` Still Stub?

**Not applicable.** Architecture does not exist in this repository.

Current `attach_model()` behavior (by design, pending integration):

```python
self.model_package = model_package
self._architecture_ready = False

# TODO: Instantiate YOLOP network architecture and load state_dict.
# TODO: Move instantiated model to self.config.device and set eval mode.
```

`_execute_forward()` similarly returns placeholder `None` heads with `status: "stub"`. The `_architecture_ready` flag is never set to `True`.

`src/modules/yolop/__init__.py` states explicitly:

> *"No YOLOP code or model downloads are performed at this stage."*

Tests work around this via `_StubYOLOPInferenceEngine` in `tests/conftest.py`, which injects synthetic segmentation tensors.

---

## Current vs. Target State

```
CURRENT (this repo)                    TARGET (after integration)
─────────────────────                  ────────────────────────────
YOLOPModelLoader.load_model()          YOLOPModelLoader.load_model()
  └─ torch.load → state_dict dict        └─ torch.load → state_dict dict
YOLOPInferenceEngine.attach_model()    YOLOPInferenceEngine.attach_model()
  └─ stores dict only                    └─ get_net() → MCnet
                                           └─ load_state_dict()
                                           └─ .to(device).eval()
YOLOPInferenceEngine._execute_forward()  YOLOPInferenceEngine._execute_forward()
  └─ returns None masks (stub)           └─ model(tensor) → real heads
```

---

## Conclusion

| Question | Answer |
|----------|--------|
| MCnet architecture present? | **No** |
| `class MCnet` / `get_net(` / `lib.models`? | **Not found** |
| Files with architecture code? | **None** |
| Minimum official files to copy? | **8 files** under `lib/models/` and `lib/utils/` |
| Why is `attach_model()` stubbed? | **N/A** — architecture missing; stub is intentional scaffolding |

**Production lane detection is blocked** until `lib/models/YOLOP.py` (and dependencies) are integrated and wired into `attach_model()` / `_execute_forward()`.
