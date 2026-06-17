# YOLOP Phase 5 Implementation Report

**Date:** 2026-06-17  
**Scope:** Wire vendored MCnet into `src/modules/yolop/inference.py` (Phase 5 of `docs/yolop_vendor_plan.md`)  
**Modified file:** `src/modules/yolop/inference.py` only

---

## Summary

Phase 5 replaces the stub forward pass with a real vendored MCnet inference path. `attach_model()` now builds the network, loads weights, moves to device, and sets eval mode. `_execute_forward()` runs `self._model(input_tensor)` under `torch.no_grad()`.

---

## 1. Changes Made

### `attach_model()`

- Imports `get_net` from `src.modules.yolop.vendor`.
- Validates `state_dict` is present (`StateDictMissingError` if missing/empty).
- Instantiates MCnet via `get_net(cfg=None)`.
- Loads weights with `load_state_dict(strict=True)`, falling back to `strict=False` on mismatch (logged).
- Moves model to `self._resolve_device()` and calls `model.eval()`.
- Stores module on `self._model` and sets `self._architecture_ready = True`.

### `_execute_forward()`

- Raises `InferenceNotReadyError` if `self._model` is unset or architecture not ready.
- Converts NumPy preprocessed input to `torch.float32` tensor on model device.
- Runs `outputs = self._model(input_tensor)` inside `torch.no_grad()`.
- Unpacks `[det_out, drivable_head, lane_head]` and returns dict with real tensors and `status: "ok"`.

### `run()`

- Raises `InferenceNotReadyError` instead of silently returning empty output when not ready.

### `is_ready`

- Now requires `model_package`, `self._model`, and `_architecture_ready` all true.

### `detach_model()`

- Clears `self._model` in addition to `model_package`.

### New exception

- `StateDictMissingError` — raised when `state_dict` key is missing or empty.

### New helper

- `_resolve_device()` — maps config device string to `torch.device`, with CPU fallback if CUDA unavailable.

### Logging added

| Event | Level | Message pattern |
|-------|-------|-----------------|
| MCnet instantiation | INFO | `Instantiating vendored MCnet via get_net()` |
| state_dict load | INFO | `Loading YOLOP state_dict — N tensor(s)` |
| strict load success | INFO | `state_dict loaded successfully with strict=True` |
| strict load fallback | WARNING | `Strict state_dict load failed — retrying with strict=False` |
| key mismatch | WARNING | `state_dict load — missing=... unexpected=...` |
| device placement | INFO | `Moving MCnet to device ...` |
| attach complete | INFO | `YOLOP model attached — architecture_ready=True, device=...` |
| forward pass start | INFO | `Executing YOLOP forward pass — input_shape=..., device=...` |
| forward pass done | INFO | `YOLOP forward pass complete — det=..., drivable=..., lane=...` |

---

## 2. Compatibility Issues

| Issue | Impact | Handling |
|-------|--------|----------|
| **Preprocessing normalization** | ADAS uses `/255` RGB; official YOLOP demo uses ImageNet mean/std | Not changed in Phase 5. Untrained/black-frame inference runs but produces empty lane masks. Real weights on road frames need validation on Colab. |
| **Stub test weights** | `tests/fixtures/stub_yolop_weights.pth` has 2 keys vs 542 MCnet keys | `strict=False` fallback loads with 542 missing keys; forward still runs but output is meaningless. Tests bypass this via `_StubYOLOPInferenceEngine` in `conftest.py`. |
| **`torch.meshgrid` deprecation** | UserWarning from vendored `Detect._make_grid` | Non-blocking; fix in future vendor sync. |
| **`End-to-end.pth` path** | Resolves to Colab Drive path on this machine | File not present locally (`exists=False`). Weight-load verification used MCnet's own `state_dict` (542 keys, `strict=True`). |
| **`postprocess()` still thin** | Returns raw torch tensors in `lane_mask`/`drivable_mask` keys | `output_parser.py` converts to binary masks downstream; no change required in Phase 5. |
| **CUDA device string** | `InferenceConfig.device="cuda"` maps to `torch.device("cuda")` | Falls back to CPU with warning if CUDA unavailable. |

---

## 3. Whether `End-to-end.pth` Loads Successfully

| Environment | Result |
|-------------|--------|
| **Local Windows dev** | **Not tested** — file not found at `\content\drive\MyDrive\adas-project\models\pretrained\yolop\End-to-end.pth` |
| **Compatible checkpoint (542 keys)** | **Yes** — MCnet `state_dict` loads with `strict=True`, zero missing/unexpected keys |
| **Stub checkpoint (2 keys)** | **Partial** — `strict=False` only; 542 missing keys, 2 unexpected keys; attach still completes |

**Expected on Colab** with real `End-to-end.pth`: load should succeed with `strict=True` (matches official `tools/demo.py` pattern).

---

## 4. Whether Real Inference Executes

**Yes.**

Validated with `YOLOPInferenceEngine` using a full MCnet `state_dict`:

```text
inference_status = "ok"
forward pass complete — det=tuple, drivable=(1, 2, 640, 640), lane=(1, 2, 640, 640)
```

Full pipeline (`LaneDetectionModule` + real engine, untrained weights, black frame):

```text
raw_status = ok
lane_mask shape = (640, 640)   # binary mask after output_parser
lane_center_x = None           # expected on uniform black input
```

Existing integration tests (stub engine injection): **3/3 passed**.

---

## 5. Output Tensor Shapes

### Raw MCnet forward (`_execute_forward` internals)

| Output | Shape | Dtype |
|--------|-------|-------|
| `detection_head` | tuple `(tensor, list)` | float |
| `drivable_head` | `(1, 2, 640, 640)` | float (sigmoid) |
| `lane_head` | `(1, 2, 640, 640)` | float (sigmoid) |

### After `run()` → `postprocess()` (engine return dict)

| Key | Shape | Notes |
|-----|-------|-------|
| `lane_mask` | `(1, 2, 640, 640)` | torch tensor passthrough |
| `drivable_mask` | `(1, 2, 640, 640)` | torch tensor passthrough |
| `detections` | tuple | detection head output |
| `inference_status` | `"ok"` | |

### After `output_parser` (full pipeline)

| Field | Shape (untrained, black frame) |
|-------|-------------------------------|
| `lane_mask` | `(640, 640)` uint8 binary |
| `drivable_mask` | `(640, 640)` uint8 binary |

---

## 6. Remaining Blockers

| Blocker | Priority | Notes |
|---------|----------|-------|
| **`End-to-end.pth` not on local disk** | High for local QA | Place weights at `models/pretrained/yolop/End-to-end.pth` or run validation on Colab |
| **Preprocessing mismatch** | High for accuracy | Align `preprocess()` with official ImageNet normalization before expecting meaningful lane output |
| **Tests still use stub engine** | Medium | `conftest.py` injects `_StubYOLOPInferenceEngine`; add real-weights integration test when file available |
| **Stub weights silent degradation** | Medium | `strict=False` on stub checkpoint produces random MCnet weights; consider failing attach when too many keys missing |
| **`postprocess()` mask resize** | Medium | Masks are model resolution (640×640), not resized to `original_shape` |
| **Detection head unused** | Low | `detection_head` returned but not decoded (NMS, etc.) |
| **Lane departure / left-right lanes** | Low | Downstream parser/geometry gaps unchanged |
| **`torch.meshgrid` warning** | Low | Vendor `common.py` update |

---

## Error Handling Verification

| Condition | Exception |
|-----------|-----------|
| `run()` without attach | `InferenceNotReadyError` |
| `_execute_forward()` without model | `InferenceNotReadyError` |
| Package missing `state_dict` | `StateDictMissingError` |
| MCnet build failure | `InferenceExecutionError` |
| Forward pass exception | `InferenceExecutionError` |

---

## Files Touched

| File | Action |
|------|--------|
| `src/modules/yolop/inference.py` | Modified |
| `docs/yolop_phase5_report.md` | Created |

No changes to `output_parser.py`, `lane_geometry.py`, `postprocess.py`, or tests.
