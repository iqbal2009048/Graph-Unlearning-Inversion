# Python 3.13 Compatibility Patch Notes

## Summary

This document describes the changes made to ensure the repository is compatible with Python 3.13 and modern versions of PyTorch and PyTorch Geometric.

---

## Changes Made

### 1. Removed `from cgi import test`

**Files modified:**
- `exp/exp_GIF.py`
- `exp/exp_retrain.py`
- `exp/exp_attack.py`
- `exp/exp_unlearn_inv.py`

**Why:** The `cgi` module was deprecated in Python 3.11 and **fully removed in Python 3.13**. This import caused an `ImportError` on Python 3.13+. The `test` symbol from `cgi` was never actually used in any of these files, so the import was safely removed.

---

### 2. Fixed deprecated `torch_geometric.data.NeighborSampler` import

**Files modified:**
- `exp/exp_GIF.py`
- `exp/exp_retrain.py`
- `exp/exp_attack.py`
- `exp/exp_unlearn_inv.py`

**Change:**
```python
# Before (deprecated/removed)
from torch_geometric.data import NeighborSampler

# After (correct location in modern PyG)
from torch_geometric.loader import NeighborSampler
```

**Why:** In newer versions of PyTorch Geometric, data loaders (including `NeighborSampler`) were moved from `torch_geometric.data` to `torch_geometric.loader`. The old import path raises an `ImportError` in current releases.

---

### 3. Added `weights_only=False` to `torch.load()` calls

**Files modified:**
- `lib_dataset/data_store.py` (4 call sites)
- `lib_gnn_model/gnn_base.py` (1 call site)

**Change:**
```python
# Before (raises FutureWarning / TypeError in PyTorch 2.x+)
torch.load(path)
torch.load(path, map_location=device)

# After (explicit and warning-free)
torch.load(path, weights_only=False)
torch.load(path, map_location=device, weights_only=False)
```

**Why:** Starting with PyTorch 2.0, `torch.load()` emits a `FutureWarning` when `weights_only` is not specified, because the default will change to `True` in a future release. Setting `weights_only=False` preserves the original behaviour (full pickle deserialization), which is required here since the saved files contain non-tensor objects (dicts with mixed types).

---

## Compatibility Matrix

| Component          | Required Version |
|--------------------|-----------------|
| Python             | 3.13+           |
| PyTorch            | 2.0+ (latest stable recommended) |
| torch-geometric    | 2.3+ (latest stable recommended) |
| NumPy              | 1.24+ (latest stable recommended) |

---

## Files Modified

| File | Changes |
|------|---------|
| `exp/exp_GIF.py` | Removed `from cgi import test`; fixed `NeighborSampler` import |
| `exp/exp_retrain.py` | Removed `from cgi import test`; fixed `NeighborSampler` import |
| `exp/exp_attack.py` | Removed `from cgi import test`; fixed `NeighborSampler` import |
| `exp/exp_unlearn_inv.py` | Removed `from cgi import test`; fixed `NeighborSampler` import |
| `lib_dataset/data_store.py` | Added `weights_only=False` to all `torch.load()` calls |
| `lib_gnn_model/gnn_base.py` | Added `weights_only=False` to `torch.load()` call |
