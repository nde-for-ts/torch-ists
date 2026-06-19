# torch-ists Custom Data Guide

> **Type**: Guide
> **Scope**: Guide for using benchmark datasets and bringing user-provided datasets (arrays, data frames, files, NPZ) into the Python API.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Custom Model Guide](custom_model_guide.md) · [Third-Party Notices](third_party_notices.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

Data entry paths for torch-ists: built-in benchmark loader and user-provided arrays/data frames/files.

---

## Table of Contents

1. [Supported Data Paths](#1-supported-data-paths)
2. [Input Data Format](#2-input-data-format)
3. [Choose a Data Source](#3-choose-a-data-source)
4. [Build a User-Provided NPZ Dataset](#4-build-a-user-provided-npz-dataset)
5. [Generate or Reuse Splits](#5-generate-or-reuse-splits)
6. [Preprocessing](#6-preprocessing)
7. [Validation Checklist](#7-validation-checklist)

---

## 1. Supported Data Paths

The package supports these data paths:

- built-in benchmark archive loading through `torch_ists.benchmark_loader.get_data(dataset_name)`
- user-provided arrays through `DataLoader.from_numpy`
- user-provided long-format tables through `DataLoader.from_dataframe`
- user-provided `npz`, `csv`, or `parquet` files through `DataLoader.from_file`
- externally prepared split files through `save_splits` / `load_splits`

Current limitations:

- registering a new dataset catalog entry inside the package
- preserving the original UCR/UEA train/test split in the packaged benchmark loader
- per-sample timestamp solver grids in packaged model forward calls
- bundled benchmark runner scripts; orchestration scripts are not part of the public package payload

## 2. Input Data Format

All supported data paths use this array format:

| Field | Shape | Required | Notes |
|---|---|---|---|
| `X` | `[N, D, L]` | yes | `N` samples, `D` feature channels, `L` time steps. Use `float32` where possible. |
| `y` | `[N]` | yes | One label per sample for the stable classification workflow. Integer labels are recommended. |
| `timestamps` | `[N, L]` | optional | Stored by `DataLoader`; source-checkout scripts validate and record it in NPZ runs when present. |

Missing values should be represented as `NaN`. For variable-length series, pad shorter series to a fixed `L` with `NaN`; `preprocess` builds the observed-value mask from those `NaN` positions.

## 3. Choose a Data Source

| Use case | API or script path | Expected input |
|---|---|---|
| Built-in benchmark archive dataset | `benchmark_loader.get_data("BasicMotions")` | Loads UCR/UEA TRAIN+TEST, pools them, then lets user code define train/valid/test. |
| In-memory NumPy arrays | `DataLoader.from_numpy(X, y)` | `X [N,D,L]`, `y [N]`, optional `timestamps [N,L]`. |
| Long-format pandas data frame | `DataLoader.from_dataframe(...)` | Groups by `series_id` by default, sorts each group by `time_col`, pads with `NaN`. |
| NPZ file from Python | `DataLoader.from_file("data.npz")` | Accepts `X`/`y` or `x`/`Y`; optional `timestamps`. |
| NPZ file in source-checkout scripts | `--dataset-source npz --dataset-npz data/ToyDataset.npz` | Accepts `X`/`y` or `x`/`Y`; optional `timestamps [N,L]` are validated and recorded. |

`get_data` pools UCR/UEA TRAIN+TEST before the project split is created. See [API Reference §3](api_reference.md#3-data-loading-api) for the legacy-mode caveats and full schema.

## 4. Build a User-Provided NPZ Dataset

For portable exchange, write user-provided data as an NPZ file with `X` and `y` keys:

```python
import numpy as np

from torch_ists import DataLoader

X = np.asarray(raw_values, dtype=np.float32)  # [N, D, L]
y = np.asarray(raw_labels, dtype=np.int64)    # [N]

bundle = DataLoader.from_numpy(X, y)
np.savez("data/ToyDataset.npz", X=bundle.X.astype(np.float32), y=bundle.y.astype(np.int64))
```

For long-format table input, convert through `DataLoader` first:

```python
from torch_ists import DataLoader

bundle = DataLoader.from_file(
    "data/ToyDataset.csv",
    time_col="time",
    feature_cols=["heart_rate", "oxygen"],
    label_col="label",
    id_col="series_id",
)
```

`DataLoader.from_dataframe` and CSV/parquet `from_file` return a `DatasetBundle` without writing NPZ; persist `bundle.X` and `bundle.y` when you need a portable array payload.

## 5. Generate or Reuse Splits

Use the split helpers when you already have a fixed split and want to store it in the package split format:

```python
from pathlib import Path

from torch_ists import get_split_path, save_splits

path = get_split_path(Path("split"), "ToyDataset", seed=42)
save_splits(path, train_idx, valid_idx, test_idx)
```

## 6. Preprocessing

`preprocess(X, model_name=...)` expects `X [N,D,L]` and returns:

| Output | Shape | Meaning |
|---|---|---|
| `X_missing` | `[N,L,D]` | values with `NaN` at missing positions |
| `X_mask` | `[N,L,D]` | 1.0 observed, 0.0 missing |
| `X_delta` | `[N,L,D]` | steps since last observation; source-checkout NPZ runs use timestamp-normalized deltas when `timestamps [N,L]` are present |
| `coeffs` | model-dependent | interpolation coefficients |

Continuous-time interpolation coefficients still use the package's normalized sequence grid.

When `model_name` is supplied, `interpolate` and `use_intensity` default to `get_data_config(model_name)`; explicit kwargs override.

Missingness controls:

| Argument | Default | Notes |
|---|---|---|
| `missing_rate` | `0.0` when omitted | Additional synthetic missingness rate in `[0,1)`. Existing `NaN` values remain missing. |
| `mechanism` | `"MCAR"` | Missingness mechanism: `"MCAR"`, `"MAR"`, `"MNAR"`, `"BLOCK"`, `"BURST"`. `"MAR"`, `"MNAR"`, `"BLOCK"`, and `"BURST"` are experimental. |
| `seed` | `0` | Controls synthetic missingness sampling. |
| `channel_sync` | `False` | If true, synthetic missing time indices are shared across feature channels. |

Sequence/attention baselines can use `coeffs=None`; CDE/SDE/continuous-time methods need `preprocess(X, model_name=...)`.

## 7. Validation Checklist

Before using a new dataset:

- Confirm `X.shape == (N, D, L)` and `y.shape == (N,)`.
- Use numeric labels for classification, preferably 0-based contiguous integers.
- Use `NaN` for missing or padded values; do not use a numeric sentinel such as `-999`.
- Ensure each class has enough samples for the split method used by your training code.

---

**End of Custom Data Guide**
