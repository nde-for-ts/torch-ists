# torch-ists API Reference

> **Type**: Reference
> **Scope**: Exported runtime APIs and included scripts for the current package
> **Note**: Built wheels install the `torch_ists` runtime package only; top-level `docs/`, `scripts/`, and `PACKAGE_STATUS.json` are available in the repository checkout.

---

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [Model Reference](model_reference.md) · [Scripts Reference](scripts_reference.md) · [Custom Data Guide](custom_data_guide.md) · [Custom Model Guide](custom_model_guide.md) · [Attribution and Licenses](third_party_notices.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [RNG Management API](#2-rng-management-api)
3. [Data Loading API](#3-data-loading-api)
4. [Preprocessing API](#4-preprocessing-api)
5. [Split Utils API](#5-split-utils-api)
6. [Model API](#6-model-api)
7. [Config API](#7-config-api)

---

## Which path should I use?

| Your input | Use | coeffs | See |
|------------|-----|--------|-----|
| Own raw arrays / dataframe / file | `DataLoader` + `preprocess(...)` | model-specific; required by coefficient-driven rows | [Custom Data Guide](custom_data_guide.md) |
| Built-in benchmark archive | `benchmark_loader` | — | [Custom Data Guide](custom_data_guide.md) |
| Direct sequence / attention baseline | build `seq`, `coeffs=None` | none | [Model Reference](model_reference.md) |
| Continuous-time method (Neural CDE/ODE/SDE) | `preprocess(...)`, then follow the model-specific input contract | coefficient-driven rows use `coeffs`; event-driven rows may use `None` | [Model Reference](model_reference.md) |
| Benchmark scripts | `scripts/` | — | [Scripts Reference](scripts_reference.md) |
| Custom model outside `torch_ists` | wrapper pattern | model-defined | [Custom Model Guide](custom_model_guide.md) |

---

## 1. Quick Start

### 1.1 Installation

```bash
# editable install from the package checkout
cd torch-ists
PYTHONNOUSERSITE=1 python -m pip install -e .
```

For the included command-line scripts, including `scripts/param_search.py`:

```bash
cd torch-ists
PYTHONNOUSERSITE=1 python -m pip install -e .[tune]
```

Environment:

- `PYTHONNOUSERSITE=1` is recommended for isolated script commands
- `Python >= 3.10`
- `.[tune]` is required for Ray Tune-backed parameter search
- `.[data]` installs `pandas` and `sktime` for `torch_ists.benchmark_loader`
- `.[ode]` installs the packaged ODE backend requirements (`torchdiffeq`, `torchdyn`)
- `.[cde]` installs the packaged CDE backend requirements (`torchcde`, `torchdiffeq`)
- `.[sde]` installs the packaged SDE backend requirements (`torchsde`, `torchcde`)
- `.[all]` installs the optional data and model backend dependencies
- `.[signatory]` installs `signatory` from source for Neural RDE (`neuralrde-1/2/3`)
- `TORCH_ISTS_RUNTIME_ROOT` is optional when running the scripts from outside the package directory
- `signatory` is intentionally not included in `.[all]` because the PyPI wheel is torch-version-pinned

```bash
PYTHONNOUSERSITE=1 python -m pip install --no-build-isolation -e .[signatory]
```

### 1.2 Basic Usage

```python
from torch_ists import set_global_seed, ists_classifier

set_global_seed(seed=0)

model = ists_classifier(model_name='cnn', num_class=2, input_dim=8, seq_len=64, device='cpu')

import torch
seq = torch.randn(16, 3, 64, 8)  # [batch, channels, length, features]
output = model(seq, coeffs=None)  # [16, 2]
```

To bridge raw arrays into a forward call, run `preprocess` and stack the value, mask, and delta channels:

```python
from torch_ists import preprocess
# X: raw array of shape [N, D, L]
X_missing, X_mask, X_delta, coeffs = preprocess(X, model_name="neuralcde")
seq = torch.stack([X_missing, X_mask, X_delta], dim=1)  # [N, 3, L, D]
logits = model(seq, coeffs=coeffs)
```

Coefficient-driven rows consume the `coeffs` returned by `preprocess`. Direct sequence, attention, and event-driven rows such as ODE-LSTM, Neural Jump ODE (NJ-ODE), and LatentSDE accept `coeffs=None` and use the sequence stack directly.

Import package as `torch_ists`.

---

## 2. RNG Management API

### Overview

The RNG API controls random number generators across:
- Python `random`
- NumPy `np.random`
- PyTorch `torch.manual_seed()`
- PyTorch CUDA `torch.cuda.manual_seed_all()`
- Deterministic backend flags when temporary seeding is used via `SeedContext`

**Location**: `torch_ists.seed_utils`

**Exported**: `set_global_seed`, `get_rng_state`, `set_rng_state`, `SeedContext`

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
- `"strict"` mode requests deterministic PyTorch behavior and may be slower
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

**Exported**: `DataLoader`, `DatasetBundle`, `benchmark_loader`

`DataLoader` is for user-provided arrays, data frames, and files. The `benchmark_loader` submodule is the built-in archive loader used by the benchmark scripts. For the end-to-end user-provided dataset workflow, see [Custom Data Guide](custom_data_guide.md).

```python
from torch_ists import benchmark_loader

X, y = benchmark_loader.get_data("BasicMotions")
```

Install `.[data]` for the optional `pandas` and `sktime` dependencies. The loader stores and searches UCR/UEA archives under `TORCH_ISTS_DATA_ROOT` when set, otherwise under `<package checkout>/data`. Fallback search checks `Univariate_ts/<dataset>/<dataset>_{TRAIN,TEST}.ts` and `Multivariate_ts/<dataset>/<dataset>_{TRAIN,TEST}.ts`.

Benchmark loader behavior: the included scripts pool the original UCR/UEA train and test partitions before creating split files. Variable-length series are padded to the pooled train+test maximum length with `NaN` by default, and labels default to pooled train+test encoding. Use `get_data(..., label_policy="train", padding="last")` only to reproduce older benchmark preprocessing.

#### Legacy-mode caveats

`label_policy="train"` raises `ValueError` when `y_test` contains a class not seen in `y_train`. `padding="last"` uses the legacy equal-length interpolation path and forward-fills the last observed value when padding remains, returning no companion mask; downstream models cannot distinguish padded positions from real observations. Use both only for legacy reproducibility.

#### Train/test split behavior

The loader concatenates the original UCR/UEA train and test partitions before the stratified split is applied. The included scripts do not preserve the original UCR/UEA partition; to evaluate on the original split, load via `sktime.datasets.load_UCR_UEA_dataset` outside this workflow.

---

### 3.1 `DataLoader.from_numpy()`

Load user-provided data from NumPy arrays.

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

Load user-provided data from pandas DataFrame (long format).

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

Load user-provided data from file (npz, csv, parquet). The Python API accepts CSV and parquet through `DataLoader.from_file`; the included scripts accept built-in benchmark data and local NPZ files only.

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

The preprocessing API provides synthetic missingness and model-aware interpolation helpers.

**Exported**: `preprocess`, `apply_missingness`, `MISSINGNESS_EXPERIMENTAL`

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
    timestamps: Optional[Any] = None,
    timestamp_delta_scale: Optional[float] = None,
    missingness_reference: Optional[Any] = None,
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
| `timestamps` | `Optional[Any]` | `None` | Optional timestamp grid `[N, L]` used for delta-channel construction. Continuous-time solver grids remain normalized-index unless a model documents otherwise. |
| `timestamp_delta_scale` | `Optional[float]` | `None` | Positive scale for timestamp-based deltas. The scripts use the train split median positive timestamp step. |
| `missingness_reference` | `Optional[Any]` | `None` | Optional reference array used by non-MCAR missingness mechanisms such as MAR/MNAR; script workflows pass train-split reference statistics. |

#### Returns

`Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`:
1. `X_missing`: `[N, L, D]` - Data with NaN for missing values
2. `X_mask`: `[N, L, D]` - Observation mask (1.0=observed, 0.0=missing)
3. `X_delta`: `[N, L, D]` - Time-since-last-observation (index-based)
4. `coeffs`: `torch.Tensor` interpolation coefficients. The trailing channel dimension depends on the interpolation mode and whether intensity channels are included.

Keep native missing entries as `NaN` before calling `preprocess(...)`; the mask and interpolation coefficients are derived from those positions. Strict raw-missing rows such as `neuralflow-*` and `flowpath-*` require the value channel to keep `NaN` at `X_mask=0` positions.

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

When `model_name` is set, `interpolate` and `use_intensity` default to `get_data_config(model_name)` (explicit kwargs override). When omitted, defaults are `interpolate="cubic"` and `use_intensity=True`.

Timestamp kwargs affect the returned `X_delta` channel and metadata-aligned preprocessing. They do not by themselves change the solver time grid used by continuous-time backbones.

#### Interpolation Modes

| Mode | Method | Use Case |
|------|--------|----------|
| `'natural'` | Natural cubic spline | Natural-spline control-path methods such as `neuralcde`, `neuralrde-*`, `gru-d*`, `ancde`, `exit`, and `leap` |
| `'linear'` | Linear interpolation | Fast, piecewise linear |
| `'rectilinear'` | Rectilinear linear interpolation | Stepwise observation paths |
| `'cubic'` | Cubic spline | Global fallback and coeff scaffolding for basic/attention rows |
| `'hermite'` | Hermite cubic spline | Hermite-selected continuous-time rows in the packaged model set |

---

### 4.2 `apply_missingness()`

Apply synthetic missingness to data (MCAR/MAR/MNAR/BLOCK/BURST/CHANNEL_DROP).

#### Signature

```python
def apply_missingness(
    X: np.ndarray,
    rate: float,
    mechanism: str = "MCAR",
    seed: int = 0,
    channel_sync: bool = False,
    reference_X: Optional[np.ndarray] = None,
    return_stats: bool = False,
) -> tuple:
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | (required) | Input data `[N, D, L]` |
| `rate` | `float` | (required) | Missing rate (0-1) |
| `mechanism` | `str` | `"MCAR"` | Mechanism: `"MCAR"`, `"MAR"`, `"MNAR"`, `"BLOCK"`, `"BURST"`, `"CHANNEL_DROP"`. Default `"MCAR"`. |
| `seed` | `int` | `0` | Random seed |
| `channel_sync` | `bool` | `False` | Synchronize temporal missingness across channels. `CHANNEL_DROP` does not use this axis and requires `False`. |
| `reference_X` | `Optional[np.ndarray]` | `None` | Optional reference data used to derive MAR/MNAR probabilities; use training data when evaluating held-out data. |
| `return_stats` | `bool` | `False` | Return a third dictionary with observation-floor or channel-outage statistics. |

#### Returns

With `return_stats=False`, returns `Tuple[np.ndarray, np.ndarray]`:
1. `X_missing`: `[N, D, L]` - Data with NaN for missing values
2. `mask_observed`: `[N, D, L]` - Observation mask (1.0=observed, 0.0=missing)

With `return_stats=True`, a third dictionary reports mechanism-specific statistics. `CHANNEL_DROP` selects one seed-stable global channel subset and reports the requested/realized drop count, channel indices, applicability, and all-missing sequence count. A one-channel input is returned unchanged and marked non-applicable.

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
| **MCAR** | Missing completely at random | Supported |
| **MAR** | Missing at random (feature-dependent) | Experimental |
| **MNAR** | Missing not at random (value-dependent) | Experimental |
| **BLOCK** | One contiguous temporal gap per sample/channel | Experimental |
| **BURST** | Several short contiguous temporal gaps per sample/channel | Experimental |

**Missingness support**: `MISSINGNESS_EXPERIMENTAL = True` marks MAR/MNAR/BLOCK/BURST and robustness result schemas as experimental.

Missingness caveats:

- **Rate realization**: synthetic missingness is deterministic for `MCAR`, `BLOCK`, and `BURST`, and probability-driven for `MAR` and `MNAR`. Native `NaN` values remain missing, so the final observed missingness can differ from the requested rate. When a sample has at least one native observation, synthetic missingness keeps at least one observed position.
- **`channel_sync` semantics**: when `channel_sync=False` (default), missing indices are sampled independently per channel. When `channel_sync=True`, the same missing time indices are shared across all channels.

---

## 5. Split Utils API

### Overview

The split utilities provide stratified split generation, split-file I/O, and canonical split paths.

**Exported**: `make_stratified_splits`, `save_splits`, `load_splits`, `get_split_path`

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

The model API has two Python entry points and an included benchmark workflow:
1. `ists_classifier`
2. `ists_model`
3. `scripts/set_splits.py`, `scripts/param_search.py`, and `scripts/model_run.py`

Canonical model construction lives in `torch_ists.model`.

Task support tiers:
- `classification`: stable path
- `regression`, `forecasting`, `imputation`, `outlier_detection`, `change_detection`: experimental

Use `ists_classifier` for the stable classification path. Non-classification tasks are available through `ists_model`.

Use `torch_ists.model` or the top-level `torch_ists` exports; `torch_ists.classifier` is not provided.

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
    imputation_strategy: str = "zero",
) -> ists_model:
```

Forward call:

```python
def forward(seq, coeffs) -> torch.Tensor  # [N, num_class]
```

The classifier returns logits only.

`file` is an optional compatibility / pass-through parameter and should be left unset (`file=None`) for normal use; this also applies to `ists_model` and `ists_layer`.

Behavior notes:
- If `use_intensity` is omitted (`None`), runtime resolves it from `get_data_config(model_name)`.
- `method` is a solver argument for continuous-time methods. Effective solver behavior remains model-specific; the package does not promise one package-wide default across ODE/CDE/SDE methods.
- NeuralFlow (`neuralflow-*`) is flow-only and does not use a numerical solver; active solver settings raise `ValueError` instead of being ignored. The neutral shared-call setting `adjoint=False` is accepted for compatibility. Its default native transforms are identity at zero elapsed time, provide inverse transforms, and only expand a singleton initial-state time axis; other state/time length mismatches raise `ValueError`.
- For sequence baselines, `transformer`, and `sand`, the packaged runtime applies `torch.nan_to_num(..., nan=0.0)` before the backbone call and does not pass a padding mask into the attention backbone.
- `transformer-mask` uses the same zero-filled values, but it also applies `src_key_padding_mask` derived from the packaged observation mask. If a sequence is fully masked, the runtime treats that zero-filled row as an unmasked finite fallback so the Transformer encoder returns finite outputs.
- `mtan` and `miam` consume zero-filled values together with the original sequence stack and normalized timestamps; they do not use interpolation coefficients. mTAN samples its variational latent representation in both training and evaluation mode, so set the random seed for repeatable single-sample inference or aggregate multiple samples. MIAM requires `hidden_dim >= 2`.
- Under the scripts' train-only z-score normalization, the zero-fill path becomes per-channel mean imputation in raw value space.
- `imputation_strategy` selects how missing entries are filled for compatible direct value-channel backbones: `"zero"` (default), `"last_observed"`, and `"channel_mean"`. Strict raw-missing rows (`neuralflow-*`, `flowpath-*`) keep raw `NaN` until method-defined mask-aware zero filling and reject nonzero strategies; coefficient-driven rows consume the `coeffs` built by `preprocess(...)`.
- Non-finite backbone outputs, hidden states, or logits raise `RuntimeError`; the package does not silently zero-fill them.

### 6.2 `ists_model`

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
        imputation_strategy: str = "zero",
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
- `regression/forecasting/imputation/outlier_detection/change_detection` are available through `ists_model`.
- `classification` is the stable packaged path.

The five non-classification outputs are shared package task-head adapters. A model's original paper may define a narrower prediction problem; for example, the NJ-ODE forecasting adapter does not reproduce paper-native future-time filtering because the current target contract does not carry a future timestamp grid.

Behavior notes:
- If `use_intensity` is omitted (`None`), runtime resolves it from `get_data_config(model_name)` inside `ists_layer`.
- `method` is accepted for continuous-time backbones, but effective `dt/step_size/rtol/atol/adjoint` behavior remains model-specific.
- Sequence/attention backbone behavior (zero-fill, `transformer-mask` padding mask, `mtan`/`miam` timestamp consumption, script z-score normalization) is the same as for `ists_classifier`; see §6.1 behavior notes above.
- Some continuous-time rows honor `method` only at model creation.
- Some compatibility models accept solver-shaped inputs only for backward compatibility and may emit a runtime warning.
- Unsupported solver knobs raise `ValueError` instead of silently changing model behavior.
- `output_dim` defaults to `1`. Forecasting requires a positive `forecast_horizon`; omitting it or passing a non-positive value raises `ValueError`.
- `forward(..., return_aux_loss=True)` returns `(prediction, aux_loss)` for auxiliary-loss backbones.
- Non-finite predictions raise `RuntimeError`.

Source ODE model parameters:

| Model | `model_params` | Runtime behavior |
|---|---|---|
| `gru-ode-bayes` | `step_size=0.05`, `p_hidden=hidden_dim`, `prep_hidden=4`, `observation_noise_std=0.01`, `mixing=1.0`, `auxiliary_weight=1e-3` | Uses Euler integration only. `step_size` bounds every observation-interval sub-step; the auxiliary term combines pre-jump likelihood and post-jump KL per observed element. |
| `latentode` | `observation_std=0.01`, `kl_weight=1.0`, `auxiliary_weight=1e-4` | Uses the selected ODE solver for reverse recognition and generative decoding; padded positions after a sample's last observation do not affect its posterior or auxiliary term, and the decoded state is held at the last valid step. The auxiliary term combines masked reconstruction likelihood and initial-state KL. |
| `njode` | `step_size=0.05`, `loss_weight=0.5`, `auxiliary_weight=1.0`, `loss_eps=1e-10`, `max_substeps=10000` | Uses bounded Euler sub-steps only. Missing coordinates are filled from the pre-jump readout, observed events update the hidden state, and the auxiliary term is normalized by sample and observation date. |

The defaults above keep method-specific losses on a practical scale when they are added to a task objective. Applications may tune the weights with the rest of the training configuration.

LatentSDE model parameters:

| `model_params` | Default | Runtime behavior |
|---|---|---|
| `kl` | `True` | Enables initial-state and path KL. `False` uses the physical latent state only and returns a zero auxiliary term. |
| `kl_weight` | `1e-5` | Scales the combined initial-state and path KL. |
| `theta`, `mu`, `sigma` | `1.0`, `0.0`, `0.5` | Define the OU prior drift and diffusion. `theta` and `sigma` must be positive. |
| `recognition_hidden` | `hidden_dim` | Width of the reverse mask-aware posterior encoder. |
| `method`, `adjoint` | `"euler"`, `False` | Euler/direct is the default. `milstein` and direct `srk` are opt-in; `srk` with adjoint differentiation is rejected. |
| `dt` or `step_size` | inferred from the shared grid | Positive fixed solver step. The default is bounded by the smallest positive grid gap and `0.01`. |

For variable-length batches, LatentSDE integrates only through the longest valid sample. Each sample's path KL is evaluated at its own last valid step, and its returned latent path is held constant after that step. Padding-only timestamps are ignored.

### 6.3 `ists_layer`

`ists_layer` is the exported low-level backbone wrapper used inside `ists_model`. It creates the selected packaged sequence model and returns hidden/backbone features before any task head is applied.

```python
class ists_layer(nn.Module):
    def __init__(
        self,
        model_name: str = "cnn",
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
        imputation_strategy: str = "zero",
    ):
```

Forward call:

```python
def forward(seq, coeffs=None, **kwargs)
```

Returns `(out, hn)` for ordinary backbones and `(out, hn, loss)` for auxiliary-loss backbones. `seq` must be `[N, 3, L, D]`; `coeffs` should come from `preprocess(...)` for coefficient-driven models and may be `None` for direct sequence, attention, or event-driven backbones.

### 6.4 Using Custom Models

Use custom models from user-owned training code. The included scripts accept only packaged model names. Reuse the packaged data loading, split utilities, and preprocessing; treat the packaged model-name list as fixed.

See [Custom Model Guide](custom_model_guide.md) for the wrapper pattern.

### 6.5 Depth Parameters

`head_depth` and `flow_depth` are different knobs.

- `head_depth`: controls the depth of the prediction head after the backbone. `1` applies a direct projection, and `k > 1` adds `k - 1` hidden-to-hidden layers before the final projection.
- `flow_depth`: controls the number of flow blocks for `neuralflow`, `dualdynamics`, `flowpath`, and their explicit flow variants.

Current defaults:
- `cnn`, `rnn`, `lstm`, `gru`: `head_depth` is not used
- `neuralcde*`, `neuralrde-*`, and the SDE baseline/stable-SDE models: `head_depth` defaults to the script/runtime-provided layer depth
- `neuralflow`: `flow_depth=3` by default
- `dualdynamics`: `flow_depth=2` by default
- `flowpath`: `flow_depth=1` by default
- `neuralflow` uses the residual-flow variant by default as a flow-only row. `dualdynamics` and `flowpath` are CDE/control-path rows. The package-authored native operators are the default and are adaptations rather than exact reference-repository reproductions.
- `model_params={"flow_backend": "nfe"}` explicitly selects the separately installed NFE reference implementation for the flow families. The adapter accepts only the documented repository commit and `stribor==0.1.0`; missing or modified installations fail before upstream code is imported. The source-checkout benchmark scripts keep `flow_backend="native"`.
- Model names use residual-flow defaults: `neuralflow`, `dualdynamics`, and `flowpath`. Explicit `-r/-g/-c` variants remain accepted variants.

Compatibility mapping:
- legacy `num_hidden_layers` is interpreted as `head_depth`.
- legacy `flow_num_layers` is interpreted as `flow_depth`.

### 6.6 Command-Line Scripts

The scripts `scripts/set_splits.py`, `scripts/param_search.py`, and `scripts/model_run.py` are CLI entry points for the benchmark workflow. Helper modules under `scripts/` support those entry points and are documented in the scripts reference. They are not installed as wheel console commands.

## 7. Config API

### Overview

The config API provides model metadata, model-specific preprocessing settings, and auxiliary-loss lookup.

**Location**: `torch_ists.config`

**Exported**: `MODEL_REGISTRY`, config helpers, and runtime setting helpers are available from `torch_ists.config`. Additional model metadata constants are available from `torch_ists.config.models`.

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
| `cnn`, `rnn`, `lstm`, `gru`, `bilstm`, `gru-simple`, `plstm`, `tlstm`, `tglstm`, `transformer`, `sand` | `'cubic'` | `False` | `coeffs` ignored; runtime uses the sequence stack directly, replaces `NaN` with `0.0`, and does not apply an attention padding mask |
| `transformer-mask` | `'cubic'` | `False` | `coeffs` ignored; runtime uses the sequence stack directly, replaces `NaN` with `0.0`, and applies `src_key_padding_mask` from the packaged observation mask |
| `mtan`, `miam` | `'cubic'` | `False` | `coeffs` ignored; runtime uses values, the original sequence stack, and normalized sequence timestamps |
| `neuralcde` | `'natural'` | `False` | consumes `coeffs` |
| `neuralcde-l/r/c/h` | override (`linear/rectilinear/cubic/hermite`) | `False` | consumes `coeffs` |
| `neuralrde-1/2/3` | `'natural'` | `False` | consumes `coeffs`; computes log-signature control paths at the selected depth |
| `ancde`, `exit`, `leap` | `'natural'` | `False` | consumes `coeffs`; uses CDE control paths with `rk4` and a fixed-grid step from input times |
| `dualdynamics` and explicit variants | `'hermite'` | `False` | consumes `coeffs`; uses flow-based CDE/control-path dynamics |
| `flowpath` and explicit variants | `'hermite'` | `False` | consumes `coeffs` and keeps the raw missing-value path for mask-aware flow handling |
| `gru-d`, `gru-dt`, `ode-rnn-spline` | `'natural'` | `True` | consumes `coeffs` with intensity channels |
| `gru-ode` | `'natural'` | `False` | consumes `coeffs` |
| `ode-lstm` | `'cubic'` | `False` | runtime ignores `coeffs`; uses imputed values, elapsed time deltas, and the any-channel observation mask directly |
| `ode-rnn`, `latentode`, `gru-ode-bayes`, `njode` | `'natural'` | `False` | runtime ignores `coeffs`; uses the sequence stack and reconstructed times directly |
| `neuralflow` and explicit NeuralFlow variants | `'hermite'` | `False` | `coeffs` ignored; runtime uses the direct value/time path with raw missing-value handling and no numerical solver |
| `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t`, `neural_lsde`, `neural_lnsde`, `neural_gsde` | `'hermite'` | `False` | consumes `coeffs` |
| `latentsde` | `'hermite'` | `False` | runtime ignores `coeffs`; the posterior encoder consumes values, per-channel masks, deltas, and normalized times directly |

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

If `ists_classifier()` or `ists_model()` is called with `use_intensity=None`, the runtime resolves the flag from these defaults.

For sequence baselines, `transformer`, `transformer-mask`, `sand`, `mtan`, `miam`, and `neuralflow`, the `'cubic'` setting applies only when you explicitly call `preprocess(X, model_name=...)`; the packaged forward path itself uses `coeffs=None` and zero-fills `NaN`. `transformer-mask` additionally applies `src_key_padding_mask`.

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

Returns `True` for `gru-ode-bayes`, `latentode`, `njode`, `latentsde`, `exit`, and `leap`; `False` for the other packaged models. Use `forward(..., return_aux_loss=True)` to receive the auxiliary term alongside the prediction.

#### Example

```python
from torch_ists.config import has_auxiliary_loss

print(has_auxiliary_loss('cnn'))         # False
print(has_auxiliary_loss('neuralcde'))   # False
print(has_auxiliary_loss('neural_gsde')) # False
print(has_auxiliary_loss('latentsde'))   # True
print(has_auxiliary_loss('njode'))       # True
print(has_auxiliary_loss('exit'))        # True
print(has_auxiliary_loss('leap'))        # True
print(has_auxiliary_loss('ancde'))       # False
```

---

---

**End of API Reference**
