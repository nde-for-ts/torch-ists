# torch-ists

**PyTorch toolkit for irregular time series classification with neural ODE, CDE, SDE, and attention models.**

**Related docs**:
[Docs index](docs/README.md) ·
[API Reference v0.1.1](docs/api_reference_v0.1.1.md) ·
[Model Reference v0.1.1](docs/model_reference_v0.1.1.md) ·
[Scripts Reference v0.1.1](docs/scripts_reference_v0.1.1.md) ·
[Project site](https://nde-for-ts.github.io/)

## Overview

`torch-ists` provides a unified interface for training and evaluating neural
differential equation models on irregularly-sampled multivariate time series.
Version `0.1.1` ships 28 packaged architectures across sequence, attention,
CDE, ODE, and SDE families.

The package centers on the `ists_classifier` classification path. The
lower-level `ists_model` constructor remains available for advanced
experimentation.

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

# ODE backends
PYTHONNOUSERSITE=1 python -m pip install -e .[ode]

# CDE backends
PYTHONNOUSERSITE=1 python -m pip install -e .[cde]

# SDE backends
PYTHONNOUSERSITE=1 python -m pip install -e .[sde]

# Full optional stack for the continuous-time models included in version 0.1.1
PYTHONNOUSERSITE=1 python -m pip install -e .[all]
```

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

## Models Included

Version `0.1.1` includes 28 models across seven families:

| Family | Models |
|--------|--------|
| **Sequence baselines** | `cnn`, `rnn`, `lstm`, `gru`, `bilstm`, `gru-simple` |
| **Attention bridge** | `transformer` |
| **Neural CDE** | `neuralcde`, `neuralcde-l`, `neuralcde-r`, `neuralcde-c`, `neuralcde-h` |
| **ODE runtime family** | `gru-dt`, `gru-d`, `gru-ode`, `ode-rnn-spline`, `ode-lstm` |
| **Additional ODE models** | `ode-rnn`, `latentode`, `gru-ode-bayes` |
| **Continuous-time baselines** | `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t` |
| **Stable Neural SDE** | `neural_lsde`, `neural_lnsde`, `neural_gsde` |

## Packaged Scripts

- `scripts/set_splits.py` — generate reproducible train/validation/test splits
- `scripts/param_search.py` — Ray Tune-based parameter search for packaged models and datasets
- `scripts/model_run.py` — train and evaluate a selected packaged model

## Documentation

- `docs/README.md` — docs index for the current public package
- `docs/api_reference_v0.1.1.md` — exported runtime API and packaged script contract
- `docs/model_reference_v0.1.1.md` — per-model origin paper, canonical repo, and implementation note
- `docs/scripts_reference_v0.1.1.md` — bundled CLI scripts: flags, outputs, and pipeline order

## Project Status

This is a **preview** release (`v0.1.1`):

- Core classification pipeline is reproducible
- The current public line includes the basic/attention bridge (`bilstm`, `gru-simple`, `transformer`)
- Custom-data, custom-model, flow-family, LatentSDE, and TSDE-family backbones remain outside version `0.1.1`
- Stable/parity-verified claims remain outside the preview package scope
