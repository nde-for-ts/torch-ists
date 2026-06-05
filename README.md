# torch-ists

[![Version](https://img.shields.io/badge/version-0.1.5-blue)](pyproject.toml) [![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml) [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Package status](https://img.shields.io/badge/status-preview-orange)](PACKAGE_STATUS.json)

**PyTorch toolkit for irregular and multivariate time series modeling with sequence models, attention models, neural ODE, neural CDE, neural SDE, stable SDE, and hybrid continuous-time models.**

`torch-ists` is a research-oriented Python package for irregularly sampled multivariate time series modeling, missing-data benchmarks, and continuous-time sequence modeling. It provides a consistent PyTorch API with task heads for classification, regression, forecasting, imputation, outlier detection, and change detection. The current package pairs those runtime APIs with packaged benchmark scripts for the stable classification workflow, UCR/UEA archive data loading, and documented model lineage for neural differential equation methods including neural ODE, neural CDE, neural SDE, stable SDE, and hybrid continuous-time models.

**Related docs**: [Docs index](docs/README.md) · [API Reference](docs/api_reference.md) · [Model Reference](docs/model_reference.md) · [Scripts Reference](docs/scripts_reference.md) · [New Model Onboarding](docs/new_model_onboarding.md) · [Project site](https://nde-for-ts.github.io/)

## Overview

`torch-ists` provides a unified interface for training and evaluating continuous-time and discrete-time PyTorch models on irregularly sampled multivariate time series. Version `0.1.5` ships 44 packaged architectures across sequence, attention, flow, CDE, ODE, SDE, and hybrid method families, with the classification benchmark pipeline as the stable packaged workflow.

The package exposes `ists_classifier` for the stable classification path and the lower-level `ists_model` constructor for advanced experimentation with additional task heads, irregular sampling, missing observations, and continuous-time model variants.

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

Use `DataLoader` for user-provided NumPy arrays, data frames, or files. Use `torch_ists.uea_ucr_loader` for built-in UCR/UEA archive datasets and the bundled benchmark scripts:

```python
from torch_ists import uea_ucr_loader

X, y = uea_ucr_loader.get_data("ECG200")
```

See the API reference for archive cache-location details.

UCR/UEA package contract: the bundled scripts pool the original UCR/UEA TRAIN+TEST partitions before generating project splits. Variable-length padding/resampling is fitted on the pooled TRAIN+TEST length distribution, labels default to pooled TRAIN+TEST encoding, and ragged padding defaults to `NaN`.

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

- sequence baselines, `tlstm`, `plstm`, `tglstm`, `transformer`, `transformer-mask`, `miam`, `mtan`, `sand`, `neuralflow-*`, `ode-lstm`, `ode-rnn`, `latentode`, and `gru-ode-bayes` accept the sequence stack directly with `coeffs=None`.
- `flowpath-*`, `dualdynamics-*`, `neuralcde*`, `gru-d*`, `gru-ode`, `ode-rnn-spline`, and the SDE family need `preprocess(X, model_name=...)` to build matching coefficients.
- `flowpath-*` and `dualdynamics-*` use CDE control paths, so install `.[cde]` or `.[all]` before running those models.

## Models Included

Version `0.1.5` includes 44 models across eight families:

| Family | Models |
|--------|--------|
| **Sequence baselines** | `cnn`, `rnn`, `lstm`, `gru`, `bilstm`, `gru-simple`, `tlstm`, `plstm`, `tglstm` |
| **Attention family** | `transformer`, `transformer-mask`, `miam`, `mtan`, `sand` |
| **NeuralFlow, FlowPath, and DualDynamics** | `neuralflow-r`, `neuralflow-g`, `neuralflow-c`, `flowpath-r`, `flowpath-g`, `flowpath-c`, `dualdynamics-r`, `dualdynamics-g`, `dualdynamics-c` |
| **Neural CDE** | `neuralcde`, `neuralcde-l`, `neuralcde-r`, `neuralcde-c`, `neuralcde-h` |
| **ODE runtime family** | `gru-dt`, `gru-d`, `gru-ode`, `ode-rnn-spline`, `ode-lstm` |
| **Additional ODE models** | `ode-rnn`, `latentode`, `gru-ode-bayes` |
| **Continuous-time baselines** | `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t` |
| **Stable Neural SDE** | `neural_lsde`, `neural_lnsde`, `neural_gsde` |

## Packaged Scripts

- `scripts/set_splits.py` — generate reproducible train/validation/test splits
- `scripts/param_search.py` — parameter search for packaged models and datasets; Ray Tune is loaded only for the Ray-backed search path
- `scripts/model_run.py` — train and evaluate a selected packaged model

The public script help paths are base-install compatible:

```bash
PYTHONPATH=. python scripts/set_splits.py --help
PYTHONPATH=. python scripts/param_search.py --help
PYTHONPATH=. python scripts/model_run.py --help
```

## New Model Onboarding

Version `0.1.5` documents the public extension boundary for external model work. Custom PyTorch modules should live outside the `torch_ists` namespace and compose the packaged data, split, preprocessing, and task wrappers from user-owned code. The package does not expose a public handler-registration API in this release.

See `docs/new_model_onboarding.md` for the wrapper pattern.

## Documentation

- `docs/README.md` — docs index for the current public package
- `docs/api_reference.md` — exported runtime API and packaged script contract
- `docs/model_reference.md` — per-model origin paper, canonical repo, and implementation note
- `docs/scripts_reference.md` — bundled CLI scripts: flags, outputs, and pipeline order
- `docs/new_model_onboarding.md` — public extension boundary for external custom models

## Project Status

This is a **preview** release (`v0.1.5`):

- Core classification pipeline is reproducible
- The package includes the sequence, attention variants, NeuralFlow, FlowPath, DualDynamics, NeuralCDE, ODE, and SDE families listed above
- UCR/UEA loading and bundled script import paths are part of the `0.1.5` package contract
- New-model onboarding guidance documents the supported external-wrapper pattern
- Documentation and bundled scripts cover the packaged `0.1.5` workflow
