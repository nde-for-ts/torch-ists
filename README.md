# torch-ists

**PyTorch toolkit for irregular time series classification with neural ODE, CDE, SDE, and attention models.**

**Related docs**:
[Docs index](docs/README.md) ·
[API Reference v0.1.1](docs/api_reference_v0.1.1.md) ·
[Model Reference v0.1.1](docs/model_reference_v0.1.1.md) ·
[Scripts Reference v0.1.1](docs/scripts_reference_v0.1.1.md) ·
[Project site](https://nde-for-ts.github.io/)

## Overview

`torch-ists` is a unified interface for training and evaluating neural
differential equation models on irregularly-sampled multivariate time series.
Version `0.1.1` ships 29 packaged architectures across sequence, attention,
CDE, ODE, and SDE families. Version `0.1.1` includes 29 models across seven
families.

`ists_classifier` is the main entry point. `ists_model` is the lower-level
constructor for advanced use.

## Installation

Requirements:

- `Python >= 3.10`
- `PyTorch >= 2.0`
- `PYTHONNOUSERSITE=1` for runtime and packaged script commands

Install the base package:

```bash
PYTHONNOUSERSITE=1 python -m pip install -e .
PYTHONNOUSERSITE=1 python -c "import torch_ists; print(torch_ists.__version__)"
```

Optional dependencies:

```bash
# Hyperparameter search (Ray Tune)
PYTHONNOUSERSITE=1 python -m pip install -e .[tune]

# Built-in UCR/UEA dataset loader
PYTHONNOUSERSITE=1 python -m pip install -e .[data]

# ODE backends
PYTHONNOUSERSITE=1 python -m pip install -e .[ode]

# CDE backends
PYTHONNOUSERSITE=1 python -m pip install -e .[cde]

# SDE backends
PYTHONNOUSERSITE=1 python -m pip install -e .[sde]

# Full optional stack, including data loading and continuous-time model backends
PYTHONNOUSERSITE=1 python -m pip install -e .[all]
```

## Data Loading

Use `DataLoader` for user-provided NumPy arrays, data frames, or files. Use
`torch_ists.uea_ucr_loader` for built-in UCR/UEA archive datasets and the
bundled benchmark scripts:

```python
from torch_ists import uea_ucr_loader

X, y = uea_ucr_loader.get_data("ECG200")
```

See the API reference for archive cache-location details.

## Quick Start

```python
import torch
from torch_ists import ists_classifier, set_global_seed

set_global_seed(0)

model = ists_classifier(
    model_name="transformer",
    num_class=2,
    input_dim=8,
    seq_len=64,
    device="cpu",
)

seq = torch.randn(16, 3, 64, 8)
logits = model(seq, coeffs=None)
print(logits.shape)  # torch.Size([16, 2])
```

For the full classification benchmark pipeline:

```bash
python scripts/set_splits.py
python scripts/param_search.py
python scripts/model_run.py
```

Input preparation:

- sequence baselines, `transformer`, `transformer-mask`, `ode-lstm`,
  `ode-rnn`, `latentode`, and `gru-ode-bayes` accept the sequence stack
  directly with `coeffs=None`.
- `neuralcde*`, `gru-d*`, `gru-ode`, `ode-rnn-spline`, and the SDE family
  need `preprocess(X, model_name=...)` to build matching coefficients.

## Models Included

Version `0.1.1` includes 29 models across seven families:

| Family | Models |
|--------|--------|
| **Sequence baselines** | `cnn`, `rnn`, `lstm`, `gru`, `bilstm`, `gru-simple` |
| **Attention bridge** | `transformer`, `transformer-mask` |
| **Neural CDE** | `neuralcde`, `neuralcde-l`, `neuralcde-r`, `neuralcde-c`, `neuralcde-h` |
| **ODE runtime family** | `gru-dt`, `gru-d`, `gru-ode`, `ode-rnn-spline`, `ode-lstm` |
| **Additional ODE models** | `ode-rnn`, `latentode`, `gru-ode-bayes` |
| **Continuous-time baselines** | `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t` |
| **Stable Neural SDE** | `neural_lsde`, `neural_lnsde`, `neural_gsde` |

## Bundled Scripts

- `scripts/set_splits.py` — generate reproducible train/validation/test splits
- `scripts/param_search.py` — Ray Tune hyperparameter search
- `scripts/model_run.py` — train and evaluate a model

## Documentation

- `docs/README.md` — docs index
- `docs/api_reference_v0.1.1.md` — exported Python API
- `docs/model_reference_v0.1.1.md` — per-model origin paper, canonical repo, and implementation note
- `docs/scripts_reference_v0.1.1.md` — CLI flags, outputs, and pipeline order

## Project Status

This is a **preview** release (`v0.1.1`). Documentation and bundled scripts cover the packaged `0.1.1` workflow end-to-end.
