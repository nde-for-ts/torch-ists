# torch-ists

[![Version](https://img.shields.io/badge/version-0.1.7-blue)](pyproject.toml) [![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml) [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Package status](https://img.shields.io/badge/status-preview-orange)](#project-status)

**PyTorch toolkit for irregular and multivariate time series modeling with sequence models, attention models, neural ODE, neural CDE, neural SDE, stable SDE, and hybrid continuous-time models.**

`torch-ists` is a research-oriented Python package for irregularly sampled multivariate time series modeling, missing-data benchmarks, and continuous-time sequence modeling. It provides a consistent PyTorch API with task heads for classification, regression, forecasting, imputation, outlier detection, and change detection. The current package pairs those runtime APIs with source-checkout benchmark scripts for the stable classification workflow, benchmark archive data loading, and documented model lineage for neural differential equation methods including neural ODE, neural CDE, neural SDE, stable SDE, and hybrid continuous-time models.

**Related source docs**: [Docs index](docs/README.md) · [API Reference](docs/api_reference.md) · [Model Reference](docs/model_reference.md) · [Scripts Reference](docs/scripts_reference.md) · [New Data Onboarding](docs/new_data_onboarding.md) · [New Model Onboarding](docs/new_model_onboarding.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

## Overview

`torch-ists` provides a unified interface for training and evaluating continuous-time and discrete-time PyTorch models on irregularly sampled multivariate time series. Version `0.1.7` ships 47 models across sequence, attention, CDE/control-path, flow, ODE, SDE, and hybrid method families, with the classification benchmark pipeline as the stable, reproducible workflow. In this package line, `neuralflow` is the flow-only row, while `dualdynamics` and `flowpath` are CDE/control-path methodology rows with flow-operator variants, and `ancde`, `exit`, and `leap` are advanced CDE control-path rows.

The package exposes `ists_classifier` for the stable classification path and the lower-level `ists_model` constructor for advanced experimentation with additional task heads, irregular sampling, missing observations, and continuous-time model variants.

## Installation

Requirements:

- `Python >= 3.10`
- `PyTorch >= 2.0`
- `PYTHONNOUSERSITE=1` for runtime and source-checkout script commands

Install the base package:

```bash
PYTHONNOUSERSITE=1 python -m pip install -e .
PYTHONNOUSERSITE=1 python -c "import torch_ists; print(torch_ists.__version__)"
```

Artifact boundary for version `0.1.7`:

- built wheels install the `torch_ists` runtime package and package data only
- top-level `docs/`, `scripts/`, and `PACKAGE_STATUS.json` are source-checkout assets
- the benchmark script chain is run from a source checkout with `PYTHONPATH=.`
- no console entry points are exposed in this preview release

Optional dependencies:

```bash
# Hyperparameter search (Ray Tune)
PYTHONNOUSERSITE=1 python -m pip install -e .[tune]

# Built-in benchmark dataset loader
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

Use `DataLoader` for user-provided NumPy arrays, data frames, or files. Use `torch_ists.benchmark_loader` for built-in benchmark archive datasets and the source-checkout benchmark scripts:

```python
from torch_ists import benchmark_loader

X, y = benchmark_loader.get_data("ECG200")
```

See the API reference for archive cache-location details and `docs/new_data_onboarding.md` for custom dataset onboarding.

Benchmark loader behavior: the source-checkout scripts pool the original UCR/UEA TRAIN+TEST partitions before generating project splits. Variable-length padding/resampling is fitted on the pooled TRAIN+TEST length distribution, labels default to pooled TRAIN+TEST encoding, and ragged padding defaults to `NaN`.

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

For the full classification benchmark pipeline from a source checkout:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/set_splits.py
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/param_search.py
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/model_run.py
```

Input preparation:

- sequence baselines, `plstm`, `tlstm`, `tglstm`, `transformer`, `transformer-mask`, `sand`, `mtan`, `miam`, `neuralflow`, `neuralflow-r/g/c`, `ode-lstm`, `ode-rnn`, `latentode`, and `gru-ode-bayes` accept the sequence stack directly with `coeffs=None`.
- `dualdynamics`, `flowpath`, `neuralcde*`, `gru-d*`, `gru-ode`, `ode-rnn-spline`, and the SDE family need `preprocess(X, model_name=...)` to build matching coefficients.
- `dualdynamics`, `flowpath`, `ancde`, `exit`, and `leap` use CDE control paths, so install `.[cde]` or `.[all]` before running those models.

## Models Included

Version `0.1.7` includes 47 models across the groups below:

| Family | Models |
|--------|--------|
| **Sequence baselines** | `cnn`, `rnn`, `lstm`, `gru`, `bilstm`, `gru-simple`, `plstm`, `tlstm`, `tglstm` |
| **Attention family** | `transformer`, `transformer-mask`, `sand`, `mtan`, `miam` |
| **ODE family** | `gru-dt`, `gru-d`, `gru-ode`, `ode-rnn-spline`, `ode-lstm` |
| **Advanced ODE family** | `ode-rnn`, `latentode`, `gru-ode-bayes` |
| **NeuralFlow** | flow-only official name `neuralflow`; explicit variants `neuralflow-r`, `neuralflow-g`, `neuralflow-c` |
| **Neural CDE** | `neuralcde`, `neuralcde-l`, `neuralcde-r`, `neuralcde-c`, `neuralcde-h` |
| **Advanced Neural CDE control-path models** | official names `dualdynamics`, `flowpath`; explicit flow-operator variants `dualdynamics-r/g/c`, `flowpath-r/g/c` |
| **Advanced CDE control-path (new in 0.1.6)** | `ancde`, `exit`, `leap` |
| **Continuous-time baseline adapters** | `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t` |
| **Stable Neural SDE** | `neural_lsde`, `neural_lnsde`, `neural_gsde` |

In version `0.1.7`, model-family names use the residual-flow default: `neuralflow`, `dualdynamics`, and `flowpath`. The `-r`, `-g`, and `-c` suffixes remain accepted explicit native flow-operator variants. Methodologically, do not read all three as one pure flow family: `neuralflow` is flow-only, while `dualdynamics` and `flowpath` are CDE/control-path rows that use native flow operators. `ancde`, `exit`, and `leap` are additional CDE/control-path rows added in version `0.1.6`.

## Source Checkout Scripts

- `scripts/set_splits.py` — generate reproducible train/validation/test splits
- `scripts/param_search.py` — parameter search for packaged models and datasets; Ray Tune is loaded only for the Ray-backed search path
- `scripts/model_run.py` — train and evaluate a selected packaged model

These scripts are tracked in the source checkout and are not installed as wheel console commands. Their help paths are base-dependency compatible:

```bash
PYTHONPATH=. python scripts/set_splits.py --help
PYTHONPATH=. python scripts/param_search.py --help
PYTHONPATH=. python scripts/model_run.py --help
```

## New Model Onboarding

Version `0.1.7` documents how external models can compose with the package. Custom PyTorch modules should live outside the `torch_ists` namespace and reuse the packaged data, split, preprocessing, and task wrappers from user-owned code. Registering new model names is not provided in this release.

See `docs/new_model_onboarding.md` for the wrapper pattern.

## Documentation

The `docs/` directory is part of the source checkout. Built wheels embed this README as package metadata but do not install the top-level documentation tree.

- `docs/README.md` — docs index for the current package
- `docs/api_reference.md` — exported runtime API and source-checkout script behavior
- `docs/model_reference.md` — per-model origin paper, canonical repo, and implementation note
- `docs/scripts_reference.md` — source-checkout CLI scripts: flags, outputs, and pipeline order
- `docs/new_data_onboarding.md` — guidance for adding user-provided datasets and NPZ script inputs
- `docs/new_model_onboarding.md` — guidance for using external custom models alongside the package

### Tutorial

A companion tutorial on Neural Differential Equations for irregular time series — covering the methodological foundations behind `torch-ists` — is hosted at the **CIKM 2025 Tutorial site**: <https://nde-for-ts.github.io/>. The site bundles slides, references, and runnable notebooks aligned with the packaged model families.

## Project Status

This is a **preview** release (`v0.1.7`):

- Core classification pipeline is reproducible
- The package includes the sequence, attention, ODE, NeuralFlow, NeuralCDE, DualDynamics/FlowPath CDE/control-path, advanced CDE control-path (`ancde`, `exit`, `leap`), continuous-time baseline adapter, and SDE groups listed above
- Version `0.1.7` ships 47 models and includes the canonical `torch_ists.benchmark_loader` submodule
- benchmark loading and source-checkout script import paths are documented for the `0.1.7` workflow
- New-model onboarding guidance documents the supported external-wrapper pattern
- Source documentation and source-checkout scripts cover the packaged `0.1.7` workflow
- Built wheels install the runtime package only; source docs, scripts, and `PACKAGE_STATUS.json` remain source-checkout assets
- **Experimental**: Synthetic missingness now supports five mechanisms — `MCAR`, `MAR`, `MNAR`, `BLOCK`, and `BURST` — via `apply_missingness()`. The `BLOCK` and `BURST` mechanisms and the robustness result schema are experimental and flagged by `MISSINGNESS_EXPERIMENTAL = True`. They are not under the stable API contract and may change without semver guarantee.

## How to Cite

If you use `torch-ists` in your research, please cite the following three papers that motivated and shaped this package — the ICLR 2024 Stable Neural SDE paper, the AAAI 2025 DualDynamics paper, and the IJCAI 2025 survey on Neural Differential Equations for time series analysis:

```bibtex
@inproceedings{oh_stable_2024,
    title        = {Stable {Neural} {Stochastic} {Differential} {Equations} in {Analyzing} {Irregular} {Time} {Series} {Data}},
    author       = {Oh, YongKyung and Lim, Dongyoung and Kim, Sungil},
    year         = 2024,
    booktitle    = {The {Twelfth} {International} {Conference} on {Learning} {Representations}, {ICLR} 2024, {Vienna}, {Austria}, {May} 7-11, 2024},
    publisher    = {OpenReview.net}
}

@inproceedings{oh_dualdynamics_2025,
    title        = {{DualDynamics}: {Synergizing} {Implicit} and {Explicit} {Methods} for {Robust} {Irregular} {Time} {Series} {Analysis}},
    shorttitle   = {{DualDynamics}},
    author       = {Oh, YongKyung and Lim, Dong-Young and Kim, Sungil},
    year         = 2025,
    booktitle    = {{AAAI}-25, {Sponsored} by the {Association} for the {Advancement} of {Artificial} {Intelligence}, {February} 25 - {March} 4, 2025, {Philadelphia}, {PA}, {USA}},
    publisher    = {AAAI Press},
    pages        = {19730--19739},
    doi          = {10.1609/AAAI.V39I18.34173},
    editor       = {Walsh, Toby and Shah, Julie and Kolter, Zico}
}

@inproceedings{oh_comprehensive_2025,
    title        = {Comprehensive {Review} of {Neural} {Differential} {Equations} for {Time} {Series} {Analysis}},
    author       = {Oh, YongKyung and Kam, Seungsu and Lee, Jonghun and Lim, Dong-Young and Kim, Sungil and Bui, Alex A. T.},
    year         = 2025,
    month        = aug,
    booktitle    = {Proceedings of the {Thirty}-{Fourth} {International} {Joint} {Conference} on {Artificial} {Intelligence}, {IJCAI}-25},
    publisher    = {International Joint Conferences on Artificial Intelligence Organization},
    pages        = {10621--10631},
    doi          = {10.24963/ijcai.2025/1179},
    editor       = {Kwok, James},
    annote       = {Survey Track}
}
```

If your work additionally relies on a specific external model included in this package, please also cite the original paper for that model — see [`docs/model_reference.md`](docs/model_reference.md) for per-model origin papers and canonical repositories.
