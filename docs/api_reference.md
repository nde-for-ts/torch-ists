# torch-ists API Reference

> **Package Version**: 0.1.7
> **Status**: Package reference
> **Scope**: Exported runtime APIs and packaged scripts shipped in `0.1.7`
> **Note**: This document describes the `0.1.7` package release. It covers the model and task lists shipped in this version.

---

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [Model Reference](model_reference.md) · [Scripts Reference](scripts_reference.md) · [New Data Onboarding](new_data_onboarding.md) · [New Model Onboarding](new_model_onboarding.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [RNG Management API](#2-rng-management-api)
3. [Data Loading API](#3-data-loading-api)
4. [Preprocessing API](#4-preprocessing-api)
5. [Split Utils API](#5-split-utils-api)
6. [Model API](#6-model-api)
7. [Config API](#7-config-api)
8. [Related References](#8-related-references)

---

## 1. Quick Start

### 1.1 Installation

```bash
# editable install from the package checkout
cd torch-ists
PYTHONNOUSERSITE=1 python -m pip install -e .
```

For the packaged command-line pipeline, including `scripts/param_search.py`:

```bash
cd torch-ists
PYTHONNOUSERSITE=1 python -m pip install -e .[tune]
```

Environment:

- `PYTHONNOUSERSITE=1` is required for commands
- `Python >= 3.10`
- `.[tune]` is required for Ray Tune-backed parameter search
- `.[data]` installs `pandas` and `sktime` for `torch_ists.benchmark_loader`
- `.[ode]` installs the packaged ODE backend requirements (`torchdiffeq`, `torchdyn`)
- `.[cde]` installs the packaged NeuralCDE backend requirements (`torchcde`, `torchdiffeq`)
- `.[sde]` installs the packaged SDE backend requirements (`torchsde`, `torchcde`)
- `.[all]` installs the optional data and model backend dependencies for version `0.1.7`
- `TORCH_ISTS_RUNTIME_ROOT` is optional when running the bundled scripts from outside the package directory
- Base-install CLI discovery is supported: `scripts/* --help` must not import optional tuning/model dependencies before the selected runtime path needs them.

### 1.2 Basic Usage (5-line example)

```python
from torch_ists import set_global_seed, ists_classifier

# Set seed for reproducibility
set_global_seed(seed=0)

# Create classifier (supported package models)
model = ists_classifier(model_name='cnn', num_class=2, input_dim=8, seq_len=64, device='cpu')

# Forward pass (seq: [N,3,L,D], coeffs: [N,C] or None for basic models)
import torch
seq = torch.randn(16, 3, 64, 8)  # [batch, channels, length, features]
output = model(seq, coeffs=None)  # → [16, 2]
```

Import package as `torch_ists`. This document covers the `0.1.7` package surface.

---

## 2. RNG Management API

### Overview

The RNG Management API provides utilities for reproducible runs by controlling random number generators across:
- Python `random`
- NumPy `np.random`
- PyTorch `torch.manual_seed()`
- PyTorch CUDA `torch.cuda.manual_seed_all()`
- Deterministic backend flags when temporary seeding is used via `SeedContext`

**Location**: `torch_ists.seed_utils`

**Exported**: ✅ Yes (`__all__` includes `set_global_seed`, `get_rng_state`, `set_rng_state`, `SeedContext`)

---

### 2.1 `set_global_seed()`

Set all random seeds for reproducibility.

#### Signature

```python
def set_global_seed(
    seed: int,
    deterministic_mode: Optional[str] = None,
    include_cuda: bool = True,
    deterministic: Optional[str] = None,  # Legacy alias
    warn_only: bool = True,
    set_torch: bool = True,
    set_np: bool = True,
    set_random: bool = True,
) -> Dict[str, Any]:
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | (required) | Random seed value (0-based integer) |
| `deterministic_mode` | `Optional[str]` | `None` | Deterministic mode: `"off"`, `"warn"`, `"strict"` |
| `include_cuda` | `bool` | `True` | Apply seed to CUDA devices (if available) |
| `deterministic` | `Optional[str]` | `None` | **Legacy alias** for `deterministic_mode` |
| `warn_only` | `bool` | `True` | If True, only warn on non-deterministic ops |
| `set_torch` | `bool` | `True` | Set PyTorch seeds |
| `set_np` | `bool` | `True` | Set NumPy seed |
| `set_random` | `bool` | `True` | Set Python random seed |

#### Returns

`Dict[str, Any]` - Seed application metadata (`seed`, `mode`, `deterministic_mode`, ...)

#### Example

```python
from torch_ists import set_global_seed

# Basic usage
set_global_seed(seed=42)

# Strict determinism
set_global_seed(seed=42, deterministic_mode='strict', warn_only=False)
```

#### Notes

- Call once at the start of your run
- `"strict"` mode may be slower but guarantees bitwise reproducibility
- CUDA seeding applied when available

---

### 2.2 `get_rng_state()`

Get current RNG states from all sources.

#### Signature

```python
def get_rng_state() -> Dict[str, Any]:
```

#### Returns

`Dict[str, Any]` with keys:
- `'python'`: Python random state (tuple)
- `'numpy'`: NumPy RNG state (dict with 'MT19937' generator)
- `'torch'`: PyTorch CPU RNG state (list of uint8)
- `'torch_cuda'`: PyTorch CUDA RNG states (list of lists, one per device)

#### Example

```python
from torch_ists import get_rng_state, set_rng_state

# Checkpoint RNG state
state = get_rng_state()

# ... do some random operations ...

# Restore state
set_rng_state(state)
```

---

### 2.3 `set_rng_state()`

Restore RNG states to all sources.

#### Signature

```python
def set_rng_state(state: Dict[str, Any]) -> None:
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `Dict[str, Any]` | RNG state dictionary (from `get_rng_state()`) |

#### Returns

`None` - Side-effect API (RNG states are restored globally)

---

### 2.4 `SeedContext`

Context manager for temporary seed changes.

#### Signature

```python
class SeedContext:
    def __init__(
        self,
        seed: int,
        deterministic_mode: Optional[str] = None,
        include_cuda: bool = True,
        deterministic: Optional[str] = None,
    ):
```

#### Example

```python
from torch_ists import SeedContext

# Temporary seed for augmentation
with SeedContext(seed=42):
    augmented_data = augment(batch)  # Always same augmentation

# Original RNG state restored here
```

On exit, `SeedContext` restores the saved RNG states and deterministic backend flags (`torch.backends.cudnn.deterministic`, `torch.backends.cudnn.benchmark`, deterministic algorithms mode).

---

## 3. Data Loading API

### Overview

The Data Loading API provides:
1. **Custom Data Loading**: `DataLoader` factory (from_numpy, from_dataframe, from_file)
2. **Built-in Benchmark Loader**: `benchmark_loader.download()` / `benchmark_loader.get_data()`

**Exported**: ✅ `DataLoader`, `DatasetBundle`, `benchmark_loader`

`DataLoader` is for user-provided arrays, data frames, and files. The `benchmark_loader` submodule is the built-in archive loader used by the bundled benchmark scripts. For the end-to-end custom dataset workflow, see [New Data Onboarding](new_data_onboarding.md).

```python
from torch_ists import benchmark_loader

X, y = benchmark_loader.get_data("ECG200")
```

Install `.[data]` for the optional `pandas` and `sktime` dependencies. The loader stores and searches UCR/UEA archives under `TORCH_ISTS_DATA_ROOT` when set, otherwise under `<package checkout>/data`. Fallback search checks `Univariate_ts/<dataset>/<dataset>_{TRAIN,TEST}.ts` and `Multivariate_ts/<dataset>/<dataset>_{TRAIN,TEST}.ts`.

Benchmark loader behavior: packaged scripts pool the original UCR/UEA train and test partitions and then create project split files. Variable-length resampling therefore uses the pooled train+test maximum length before the pooled data is split again. Labels are encoded from the pooled train+test label set by default, and ragged tensor padding uses `NaN` so downstream missingness masks can identify padded positions. Use `get_data(..., label_policy="train", padding="last")` only to reproduce older benchmark preprocessing.

#### Legacy-mode caveats

`label_policy="train"` fails loudly when `y_test` contains a class not seen in `y_train`: the loader raises `ValueError` listing the offending labels and recommending the default `"pooled"` mode. `padding="last"` reproduces the legacy forward-fill of the last observed value; in this mode the loader returns no companion mask, so downstream models cannot distinguish padded positions from real observations. Use both only for legacy reproducibility and keep the runtime defaults otherwise.

#### Train/test split behavior

The loader concatenates the original UCR/UEA train and test partitions before the project split is created. The project split (`set_splits.py`, stratified) is applied to the pooled data, so the train/val/test split that downstream scripts use does not correspond to the original UCR/UEA partition. Version `0.1.7` does not preserve the original partition in the bundled scripts; benchmarks that need the original split should load `X_train` / `y_train` / `X_test` / `y_test` directly via `sktime.datasets.load_UCR_UEA_dataset` or their own loader and run training/evaluation outside the bundled scripts.

In version `0.1.7`, the benchmark loader is included in the installed package and package checkout. Packaged split generation through `--dataset-source default` depends on this submodule.

---

### 3.1 `DataLoader.from_numpy()`

Load custom data from NumPy arrays.

#### Signature

```python
@staticmethod
def from_numpy(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    task: str = "classification",
    target_shape: Optional[Sequence[int]] = None,
    target_dtype: Optional[str] = None,
) -> DatasetBundle:
```

#### Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `X` | `np.ndarray` | `[N, D, L]` | Time series data |
| `y` | `np.ndarray` | `[N]` | Labels (0-indexed integers) |
| `timestamps` | `Optional[np.ndarray]` | `[N, L]` | Time stamps (optional) |
| `task` | `str` | scalar | Metadata field; default `"classification"` |
| `target_shape` | `Optional[Sequence[int]]` | metadata | Optional target metadata stored in the bundle |
| `target_dtype` | `Optional[str]` | metadata | Optional target dtype metadata stored in the bundle |

#### Returns

`DatasetBundle` with fields:
- `X`: `np.ndarray [N, D, L]`
- `y`: `np.ndarray [N]`
- `timestamps`: `Optional[np.ndarray [N, L]]`
- `metadata`: `Dict[str, Any]`

#### Example

```python
from torch_ists import DataLoader
import numpy as np

# Create synthetic data
N, D, L = 100, 8, 64
X = np.random.randn(N, D, L).astype(np.float32)
y = np.random.randint(0, 3, size=(N,))

# Load into DatasetBundle
bundle = DataLoader.from_numpy(X, y)
print(bundle.X.shape)  # (100, 8, 64)
```

---

### 3.2 `DataLoader.from_dataframe()`

Load custom data from pandas DataFrame (long format).

#### Signature

```python
@staticmethod
def from_dataframe(
    df: Any,  # pandas.DataFrame
    time_col: str,
    feature_cols: Sequence[str],
    label_col: str,
    id_col: Optional[str] = "series_id",
    task: str = "classification",
    target_shape: Optional[Sequence[int]] = None,
    target_dtype: Optional[str] = None,
) -> DatasetBundle:
```

#### Example

```python
import pandas as pd
from torch_ists import DataLoader

df = pd.DataFrame({
    'series_id': [0, 0, 0, 1, 1, 1],
    'time': [0, 1, 2, 3, 4, 5],
    'feature1': [0.8, 1.4, 1.1, -0.2, 0.3, 0.0],
    'feature2': [2.0, 1.7, 1.9, 4.1, 4.4, 4.2],
    'label': [0, 0, 0, 1, 1, 1],
})

bundle = DataLoader.from_dataframe(
    df,
    time_col='time',
    feature_cols=['feature1', 'feature2'],
    label_col='label',
    id_col='series_id',
)
print(bundle.X.shape)  # (2, 2, 3)
```

---

### 3.3 `DataLoader.from_file()`

Load custom data from file (npz, csv, parquet). The Python API accepts CSV and parquet through `DataLoader.from_file`; the bundled scripts accept built-in benchmark data and local NPZ files only.

#### Signature

```python
@staticmethod
def from_file(
    path: str,
    format: Optional[str] = None,
    **kwargs: Any,
) -> DatasetBundle:
```

#### Example

```python
from torch_ists import DataLoader

# From NPZ
bundle = DataLoader.from_file('data.npz')

# From CSV
bundle = DataLoader.from_file(
    'data.csv',
    time_col='time',
    feature_cols=['feat1', 'feat2'],
    label_col='label',
)
```

---

## 4. Preprocessing API

### Overview

The Preprocessing API provides:
1. **Missingness**: `apply_missingness()` - Synthetic missing data (MCAR/MAR/MNAR/BLOCK/BURST)
2. **Preprocessing**: `preprocess()` - Canonical preprocessing path (interpolation + intensity + delta)

**Exported**: ✅ `preprocess`, `apply_missingness`, `MISSINGNESS_EXPERIMENTAL`

---

### 4.1 `preprocess()`

Canonical preprocessing function (interpolation + intensity + delta).

#### Signature

```python
def preprocess(
    X: Any,
    missing_rate: Optional[float] = None,
    interpolate: Optional[str] = None,
    use_intensity: Optional[bool] = None,
    seed: int = 0,
    channel_sync: bool = False,
    mechanism: str = "MCAR",
    model_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` or `torch.Tensor` | (required) | Input data `[N, D, L]` |
| `missing_rate` | `Optional[float]` | `None` | Missing rate (0-1, None=skip missingness) |
| `interpolate` | `Optional[str]` | `None` | Interpolation mode (`natural`/`linear`/`rectilinear`/`cubic`/`hermite`). If omitted and `model_name` is not set, defaults to `cubic`. |
| `use_intensity` | `Optional[bool]` | `None` | Whether to include intensity channels. If omitted and `model_name` is not set, defaults to `True` for backward compatibility. |
| `seed` | `int` | `0` | RNG seed used when synthetic missingness is injected. |
| `channel_sync` | `bool` | `False` | If `True`, synchronize injected missing positions across channels. |
| `mechanism` | `str` | `"MCAR"` | Missingness mechanism (`MCAR`, `MAR`, `MNAR`, `BLOCK`, `BURST`). `MAR`, `MNAR`, `BLOCK`, and `BURST` are experimental. |
| `model_name` | `Optional[str]` | `None` | Optional model key. When set, `interpolate` and `use_intensity` default to `get_data_config(model_name)`. Explicit kwargs still override. |

#### Returns

`Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`:
1. `X_missing`: `[N, L, D]` - Data with NaN for missing values
2. `X_mask`: `[N, L, D]` - Observation mask (1.0=observed, 0.0=missing)
3. `X_delta`: `[N, L, D]` - Time-since-last-observation (index-based)
4. `coeffs`: `torch.Tensor` interpolation coefficients. The trailing channel dimension depends on the interpolation mode and whether intensity channels are included.

#### Example

```python
from torch_ists import preprocess
import numpy as np

# Recommended: let the model decide interpolation mode and intensity usage
X = np.random.randn(100, 8, 64).astype(np.float32)
X_missing, X_mask, X_delta, coeffs = preprocess(
    X,
    missing_rate=0.3,
    model_name="neural_lsde",
)
```

When `model_name` is set, `interpolate` and `use_intensity` default to `get_data_config(model_name)`. Explicit values for those kwargs still override the model-derived defaults. If `model_name` is omitted, the legacy defaults remain `interpolate="cubic"` and `use_intensity=True`.

#### Interpolation Modes

| Mode | Method | Use Case |
|------|--------|----------|
| `'natural'` | Natural cubic spline | Source-natural control-path families such as `neuralcde`, `gru-d*`, and advanced CDE rows |
| `'linear'` | Linear interpolation | Fast, piecewise linear |
| `'rectilinear'` | Rectilinear linear interpolation | Stepwise observation paths |
| `'cubic'` | Cubic spline | Global fallback and coeff scaffolding for basic/attention rows |
| `'hermite'` | Hermite cubic spline | Hermite-selected continuous-time rows in the packaged 0.1.7 model set |

---

### 4.2 `apply_missingness()`

Apply synthetic missingness to data (MCAR/MAR/MNAR/BLOCK/BURST).

#### Signature

```python
def apply_missingness(
    X: np.ndarray,
    rate: float,
    mechanism: str = "MCAR",
    seed: int = 0,
    channel_sync: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | (required) | Input data `[N, D, L]` |
| `rate` | `float` | (required) | Missing rate (0-1) |
| `mechanism` | `str` | `"MCAR"` | Mechanism: `"MCAR"`, `"MAR"`, `"MNAR"`, `"BLOCK"`, `"BURST"`. Default `"MCAR"`. |
| `seed` | `int` | `0` | Random seed |
| `channel_sync` | `bool` | `False` | Synchronize missingness across channels |

#### Returns

`Tuple[np.ndarray, np.ndarray]`:
1. `X_missing`: `[N, D, L]` - Data with NaN for missing values
2. `mask_observed`: `[N, D, L]` - Observation mask (1.0=observed, 0.0=missing)

#### Example

```python
from torch_ists import apply_missingness

X = np.random.randn(100, 8, 64).astype(np.float32)

# Apply MCAR missingness (30%)
X_missing, mask = apply_missingness(X, rate=0.3, mechanism="MCAR", seed=42)
print(np.isnan(X_missing).mean())  # ~0.3
```

#### Mechanisms

| Mechanism | Description | Status |
|-----------|-------------|--------|
| **MCAR** | Missing completely at random | ✅ Production |
| **MAR** | Missing at random (feature-dependent) | ⚠️ Experimental |
| **MNAR** | Missing not at random (value-dependent) | ⚠️ Experimental |
| **BLOCK** | One contiguous temporal gap per sample/channel | ⚠️ Experimental |
| **BURST** | Several short contiguous temporal gaps per sample/channel | ⚠️ Experimental |

**Feature Flag**: `MISSINGNESS_EXPERIMENTAL = True` keeps MAR/MNAR/BLOCK/BURST available in the packaged API.

**Behavior**: missingness is controlled by `rate`, `mechanism`, `seed`, and `channel_sync`.

> **Missingness caveats**
>
> - **Rate realization**: `MCAR`, `BLOCK`, and `BURST` hit the target `rate` exactly (per channel; per sample when `channel_sync=True`). `MAR` and `MNAR` are probability-driven and realize only an approximate rate (e.g., target `0.30` → actual ~`0.28`).
> - **BURST short-sequence degeneracy**: for very short sequences (length &lt; ~10), `BURST` degenerates toward scattered single-step drops (≈ `MCAR`) because the maximum burst length floors at 1.
> - **`channel_sync` semantics**: when `channel_sync=False` (default), missing indices are sampled independently per channel. When `channel_sync=True`, the same missing time indices are shared across all channels.
> - **Experimental surface**: `BLOCK`, `BURST`, and the robustness result schema are experimental and may change without semver guarantee. `MISSINGNESS_EXPERIMENTAL = True` signals that this entire synthetic-missingness type-slice is not under the stable API contract.

---

## 5. Split Utils API

### Overview

The Split Utils API provides:
1. **Stratified Splitting**: `make_stratified_splits()` - Generate train/valid/test splits
2. **I/O**: `save_splits()`, `load_splits()` - Pickle-based persistence
3. **Path utilities**: `get_split_path()` - Canonical split file paths

**Exported**: ✅ All (`make_stratified_splits`, `save_splits`, `load_splits`, `get_split_path`)

---

### 5.1 `make_stratified_splits()`

Generate reproducible stratified train/valid/test splits.

#### Signature

```python
def make_stratified_splits(
    y: np.ndarray,
    seed: int,
    train: float = 0.7,
    valid: float = 0.15,
    test: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
```

#### Example

```python
from torch_ists import make_stratified_splits
import numpy as np

y = np.random.randint(0, 3, size=(100,))

train_idx, valid_idx, test_idx = make_stratified_splits(y, seed=42)
print(len(train_idx), len(valid_idx), len(test_idx))  # 70, 15, 15
```

---

### 5.2 `save_splits()` / `load_splits()`

Save/load split indices to/from pickle file.

#### Signatures

```python
def save_splits(
    path: Path,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:

def load_splits(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
```

#### Example

```python
from torch_ists import save_splits, load_splits
from pathlib import Path

# Save
save_splits(Path("split/BasicMotions/BasicMotions_42"), train_idx, valid_idx, test_idx)

# Load
train_idx, valid_idx, test_idx = load_splits(Path("split/BasicMotions/BasicMotions_42"))
```

---

### 5.3 `get_split_path()`

Return canonical split path: `split_root/{dataset}/{dataset}_{seed}`.

#### Signature

```python
def get_split_path(
    split_root: Path,
    dataset_name: str,
    seed: int,
) -> Path:
```

#### Example

```python
from torch_ists import get_split_path
from pathlib import Path

path = get_split_path(Path("split"), "BasicMotions", seed=42)
print(path)  # split/BasicMotions/BasicMotions_42
```

---

## 6. Model API

### Overview

The package model API is model-centric:
1. Primary entry point: `ists_classifier`
2. Advanced entry point: `ists_model`
3. Bundled command-line workflow: split creation, parameter search, and evaluation scripts

Canonical model construction lives in `torch_ists.model`.

Task support tiers:
- `classification`: primary supported path in version `0.1.7`
- `regression`, `forecasting`, `imputation`, `outlier_detection`, `change_detection`: experimental

Readiness note:
- use `ists_classifier` when you want the most stable path in version `0.1.7`.
- non-classification tasks are available through `ists_model` for advanced use in version `0.1.7`.

Use `torch_ists.model` or the top-level `torch_ists` exports; `torch_ists.classifier` is not provided in version `0.1.7`.

### 6.1 `ists_classifier` (Stable)

Convenience function that constructs `ists_model(task="classification")`.

```python
def ists_classifier(
    model_name: str = "cnn",
    num_class: int = 2,
    input_dim: int = 3,
    seq_len: int = 128,
    hidden_dim: int = 32,
    hidden_hidden_dim: int = 32,
    num_layers: int = 1,
    num_hidden_layers: int = 1,
    head_depth: Optional[int] = None,
    flow_depth: Optional[int] = None,
    flow_num_layers: Optional[int] = None,
    bidirectional: bool = False,
    dropout: float = 0.1,
    use_intensity: Optional[bool] = None,
    method: Optional[str] = None,
    file: Optional[str] = None,
    device: str = "cpu",
    model_params: Optional[Dict[str, Any]] = None,
) -> ists_model:
```

Forward call:

```python
def forward(seq, coeffs) -> torch.Tensor  # [N, num_class]
```

The current `0.1.7` package scope exposes logits-only forward returns.

Behavior notes:
- If `use_intensity` is omitted (`None`), runtime resolves it from `get_data_config(model_name)`.
- `method` is an advanced solver argument for continuous-time families. The effective solver behavior after argument normalization remains family-specific; version `0.1.7` does not promise one package-wide default across ODE/CDE/SDE families.
- For sequence baselines, `transformer`, and `sand`, the packaged runtime applies `torch.nan_to_num(..., nan=0.0)` before the backbone call and does not pass a padding mask into the attention backbone.
- `transformer-mask` uses the same zero-filled values, but it also applies `src_key_padding_mask` derived from the packaged observation mask.
- `mtan` and `miam` consume zero-filled values together with the original sequence stack and normalized timestamps; they do not use interpolation coefficients.
- In the bundled script pipeline, train-only z-score normalization makes this zero-fill path behave like per-channel mean imputation in raw value space; outside that workflow it is literal zero-fill.
- Non-finite backbone outputs, hidden states, or logits raise `RuntimeError`; the package does not silently zero-fill them.

### 6.2 `ists_model` (Advanced)

```python
class ists_model(nn.Module):
    def __init__(
        self,
        model_name: str = "cnn",
        num_class: int = 2,
        input_dim: int = 3,
        seq_len: int = 128,
        hidden_dim: int = 32,
        hidden_hidden_dim: int = 32,
        num_layers: int = 1,
        num_hidden_layers: int = 1,
        head_depth: Optional[int] = None,
        flow_depth: Optional[int] = None,
        flow_num_layers: Optional[int] = None,
        bidirectional: bool = False,
        dropout: float = 0.1,
        use_intensity: Optional[bool] = None,
        method: Optional[str] = None,
        file: Optional[str] = None,
        device: str = "cpu",
        model_params: Optional[Dict[str, Any]] = None,
        task: str = "classification",  # classification|regression|forecasting|imputation|outlier_detection|change_detection
        output_dim: int = 1,
        forecast_horizon: Optional[int] = None,
    ):
```

Forward outputs by task:
- `classification` -> `Tensor[N, C]`
- `regression` -> `Tensor[N, R]`
- `forecasting` -> `Tensor[N, H, R]`
- `imputation` -> `Tensor[N, L, D]`
- `outlier_detection` -> `Tensor[N, 1]`
- `change_detection` -> `Tensor[N, L, 1]`

Unsupported task names are fail-fast with `NotImplementedError`.

Broader task surface:
- `regression/forecasting/imputation/outlier_detection/change_detection` are available for advanced use.
- `classification` remains the primary packaged path in version `0.1.7`.

Behavior notes:
- If `use_intensity` is omitted (`None`), runtime resolves it from `get_data_config(model_name)` inside `ists_layer`.
- `method` is accepted for continuous-time backbones, but effective `dt/step_size/rtol/atol/adjoint` behavior remains family-specific and should not be read as a single global runtime default.
- Sequence/attention backbone behavior (zero-fill, `transformer-mask` padding mask, `mtan`/`miam` timestamp consumption, bundled-script z-score normalization) is the same as for `ists_classifier`; see §6.1 behavior notes above.
- Some continuous-time rows honor `method` only at model creation.
- Some compatibility models accept solver-shaped inputs only for backward compatibility and may emit a runtime warning.
- Unsupported solver knobs raise `ValueError` instead of silently changing family behavior.
- `output_dim` defaults to `1`; `forecast_horizon` defaults to `seq_len` when omitted.
- `forward(..., return_aux_loss=True)` returns `(prediction, aux_loss)` for auxiliary-loss backbones.
- Non-finite predictions raise `RuntimeError`.

### 6.3 Using External Models

Version `0.1.7` documents a supported pattern for using external models alongside the packaged runtime. The packaged CLI model list is fixed for this release, and custom models should be called from user-owned training code.

Recommended pattern:
- keep custom PyTorch modules in a user-owned namespace outside `torch_ists`
- reuse the bundled data loading, split utilities, preprocessing, and task wrappers where they fit
- call the external model from user-owned training code
- treat the list of model names shipped in `0.1.7` as fixed unless a later release explicitly adds a model name

For custom models, the bundled scripts do not train names outside the packaged model list. Use user-owned training code for external model names.

See [New Model Onboarding](new_model_onboarding.md) for the wrapper pattern.

### 6.4 Depth Parameters

`head_depth` and `flow_depth` are different knobs.

- `head_depth`: controls the depth of the prediction head after the backbone. `1` applies a direct projection, and `k > 1` adds `k - 1` hidden-to-hidden layers before the final projection.
- `flow_depth`: controls the number of native flow blocks for `neuralflow`, `dualdynamics`, `flowpath`, and their explicit flow-operator variants.

Current `0.1.7` defaults:
- `cnn`, `rnn`, `lstm`, `gru`: `head_depth` is not used
- `neuralcde*` and the SDE baseline/stable-SDE models: `head_depth` defaults to the script/runtime-provided layer depth
- `neuralflow`: `flow_depth=3` by default
- `dualdynamics`: `flow_depth=2` by default
- `flowpath`: `flow_depth=1` by default
- `neuralflow` uses the native flow backend by default. `dualdynamics` and `flowpath` are CDE/control-path methodology rows that use CDE-style controls with packaged native flow-operator variants in this version. Optional `NFE`/`stribor` backends are not installed by default.
- In version `0.1.7`, model-family names use residual-flow defaults: `neuralflow`, `dualdynamics`, and `flowpath`. Explicit `-r/-g/-c` variants remain accepted variants.

Compatibility mapping:
- legacy `num_hidden_layers` is interpreted as `head_depth`.
- legacy `flow_num_layers` is interpreted as `flow_depth`.
- packaged pipeline scripts use the same runtime depth resolver.

### 6.5 Command-Line Scripts

The bundled scripts `scripts/set_splits.py`, `scripts/param_search.py`, and `scripts/model_run.py` are CLI entry points for the benchmark workflow. Helper modules under `scripts/` support those entry points and are documented in the scripts reference.

Bundled workflow behavior in version `0.1.7`:
- `scripts/param_search.py` and `scripts/model_run.py` choose solver methods from model-specific package defaults.
- Most users can rely on those defaults without setting solver arguments manually.
- Import-level solver kwargs are advanced controls, and supported options vary by model family.
- Unsupported solver knobs raise `ValueError` instead of silently changing runtime behavior.

## 7. Config API

### Overview

The Config API provides:
1. **Model metadata**: `MODEL_REGISTRY` plus model metadata constants in `torch_ists.config.models`
2. **Data config**: `get_data_config()` - Model-specific preprocessing settings
3. **Auxiliary loss**: `has_auxiliary_loss()` - Check if a supported package model returns auxiliary loss

**Location**: `torch_ists.config`

**Exported**: `MODEL_REGISTRY`, config helpers, and runtime policy helpers are available from `torch_ists.config`. Additional model metadata constants are available from `torch_ists.config.models`.

---

### 7.1 `get_data_config()`

Get model-specific preprocessing settings used by `preprocess(X, model_name=...)`.

#### Signature

```python
def get_data_config(model_name: str) -> Dict[str, Any]:
```

#### Returns

`Dict[str, Any]`:
- `'interpolation'`: `str` - Interpolation mode (`'natural'`, `'linear'`, `'rectilinear'`, `'cubic'`, `'hermite'`)
- `'use_intensity'`: `bool` - Compute intensity channel

`get_data_config()` describes the defaults that `preprocess()` applies when a `model_name` is provided. It does not mean every packaged model consumes the returned interpolation coefficients at forward time.

#### Rules

| Packaged row(s) | `interpolation` | `use_intensity` | Runtime note |
|------------|--------------|---------------|--------------|
| `cnn`, `rnn`, `lstm`, `gru`, `bilstm`, `gru-simple`, `plstm`, `tlstm`, `tglstm`, `transformer`, `sand` | `'cubic'` | `False` | `coeffs` ignored; packaged handlers use the sequence stack directly, replace `NaN` with `0.0`, and do not apply an attention padding mask |
| `transformer-mask` | `'cubic'` | `False` | `coeffs` ignored; packaged handler uses the sequence stack directly, replaces `NaN` with `0.0`, and applies `src_key_padding_mask` from the packaged observation mask |
| `mtan`, `miam` | `'cubic'` | `False` | `coeffs` ignored; packaged handler uses values, the original sequence stack, and normalized sequence timestamps |
| `neuralcde` | `'natural'` | `False` | consumes `coeffs` |
| `neuralcde-l/r/c/h` | override (`linear/rectilinear/cubic/hermite`) | `False` | consumes `coeffs` |
| `gru-d`, `gru-dt`, `ode-rnn-spline` | `'natural'` | `True` | consumes `coeffs` with intensity channels |
| `gru-ode` | `'natural'` | `False` | consumes `coeffs` |
| `ode-lstm` | `'cubic'` | `False` | packaged handler ignores `coeffs`; uses values, timestamps, and mask directly |
| `ode-rnn`, `latentode`, `gru-ode-bayes` | `'natural'` | `False` | packaged handler ignores `coeffs`; uses the sequence stack and reconstructed times directly |
| `neuralflow` and explicit NeuralFlow variants | `'hermite'` | `False` | `coeffs` ignored; packaged handler uses the direct value/time path with raw missing-value handling |
| `dualdynamics`, `flowpath`, and their explicit variants | `'hermite'` | `False` | consumes `coeffs`; CDE/control-path rows using the packaged native flow-operator backend |
| `ancde`, `exit`, `leap` | `'natural'` | `False` | consumes `coeffs`; advanced CDE control-path rows using their own modified vendored control-path solver (rk4, fixed-grid step from input times) |
| `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t`, `neural_lsde`, `neural_lnsde`, `neural_gsde` | `'hermite'` | `False` | consumes `coeffs` |

#### Example

```python
from torch_ists.config import get_data_config

# SDE model
config = get_data_config('neural_gsde')
print(config)  # {'interpolation': 'hermite', 'use_intensity': False}

# CDE model
config = get_data_config('neuralcde')
print(config)  # {'interpolation': 'natural', 'use_intensity': False}

# Sequence baseline
config = get_data_config('cnn')
print(config)  # {'interpolation': 'cubic', 'use_intensity': False}
```

If `ists_classifier()` or `ists_model()` is called with `use_intensity=None`, the runtime resolves the flag from this policy.

For sequence baselines, `transformer`, `transformer-mask`, `sand`, `mtan`, `miam`, and `neuralflow`, this policy mainly matters when you call `preprocess(X, model_name=...)`; the packaged forward path still uses `coeffs=None`. In the sequence and attention rows, `'cubic'` is a preprocessing setting rather than a control-path runtime setting. The packaged runtime zero-fills `NaN` before the backbone call. `transformer-mask` additionally applies `src_key_padding_mask`; `sand` does not. In the bundled scripts, prior train-only z-score normalization makes this behave like per-channel mean imputation in raw value space.

This reference documents the 47 models shipped in version `0.1.7`.

---

### 7.2 `has_auxiliary_loss()`

Check if model returns auxiliary loss.

#### Signature

```python
def has_auxiliary_loss(model_name: str) -> bool:
```

#### Returns

`True` if model returns `(logits, aux_loss)`, `False` otherwise.

#### Models with Auxiliary Loss

Version `0.1.7` introduces two models that return auxiliary outputs alongside the hidden state:

- `exit` returns a kinetic-energy regularization term as an auxiliary output (`(out, hn, reg)`).
- `leap` returns an auxiliary loss term (`(out, hn, loss)`).

`has_auxiliary_loss()` returns `True` for `exit` and `leap`. All other packaged models return `False`.

#### Example

```python
from torch_ists.config import has_auxiliary_loss

print(has_auxiliary_loss('cnn'))         # False
print(has_auxiliary_loss('neuralcde'))   # False
print(has_auxiliary_loss('neural_gsde')) # False
print(has_auxiliary_loss('exit'))        # True
print(has_auxiliary_loss('leap'))        # True
print(has_auxiliary_loss('ancde'))       # False
```

---

## 8. Related References

This API reference stays focused on exported entry points.

- For the packaged model roster, per-family origin notes, and package-scope notes, use [Model Reference](model_reference.md).
- For bundled CLI scripts, flags, output artifacts, and pipeline order, use [Scripts Reference](scripts_reference.md).
- For external custom-model usage, use [New Model Onboarding](new_model_onboarding.md).
- For installation and the package overview, use the [package README](../README.md).

---

**End of API Reference**
