# torch-ists

[![Version](https://img.shields.io/badge/version-0.1.9-blue)](pyproject.toml) [![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml) [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Package status](https://img.shields.io/badge/status-preview-orange)](#project-status)

**PyTorch toolkit for irregularly sampled multivariate time series with continuous-time methods (Neural ODE/CDE/SDE) and discrete-time baselines.**

`torch-ists` provides runtime APIs, benchmark scripts, and benchmark archive loading for irregular time series.

**Related source docs**: [Docs index](docs/README.md) · [API Reference](docs/api_reference.md) · [Model Reference](docs/model_reference.md) · [Scripts Reference](docs/scripts_reference.md) · [Data Reference](docs/data_reference.md) · [Custom Model Reference](docs/custom_model_reference.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

## Overview

The package exposes `ists_classifier` for classification and the lower-level `ists_model` constructor for additional task heads. It includes 47 models spanning sequence baselines, attention-based methods, Neural ODE/CDE/SDE methods, NeuralFlow, DualDynamics and FlowPath, and Stable Neural SDEs.

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

See [Data Reference](docs/data_reference.md) for cache locations and user-provided dataset handling.

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

See [Model Reference](docs/model_reference.md) for which models need `preprocess(X, model_name=...)` versus the direct sequence stack.

Imputation strategy:

- Missing entries on the value channel are imputed before the backbone consumes them. `ists_classifier(..., imputation_strategy="zero")` is the default and matches the established zero-fill baseline (mean-equivalent under training-set standardization).
- `last_observed` (last-observation-carried-forward) and `channel_mean` (per-series observed mean) are experimental opt-in alternatives selected through the same argument.

## Models Included

The package includes 47 models across the groups below. See [Model Reference](docs/model_reference.md) for per-model details.

- **Sequence baselines** — `cnn`, `rnn`, `lstm`, `gru`, `bilstm`, `gru-simple`, `plstm`, `tlstm`, `tglstm` (§2)
- **Attention-based methods** — `transformer`, `transformer-mask`, `sand`, `mtan`, `miam` (§3)
- **Neural ODE methods** — `gru-dt`, `gru-d`, `gru-ode`, `ode-rnn-spline`, `ode-lstm` (§4)
- **ODE-RNN, Latent ODE, and GRU-ODE-Bayes** — `ode-rnn`, `latentode`, `gru-ode-bayes` (§5)
- **Neural CDE** — `neuralcde`, `neuralcde-l`, `neuralcde-r`, `neuralcde-c`, `neuralcde-h` (§6)
- **ANCDE, EXIT, and LEAP** — `ancde`, `exit`, `leap` (§7)
- **NeuralFlow** — `neuralflow`, `neuralflow-r`, `neuralflow-g`, `neuralflow-c` (§8)
- **DualDynamics and FlowPath** — `dualdynamics`, `flowpath`, and `-r/g/c` variants (§9)
- **SDE baselines** — `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t` (§10)
- **Stable Neural SDEs** — `neural_lsde`, `neural_lnsde`, `neural_gsde` (§11)

## Source Checkout Scripts

- `scripts/set_splits.py` — generate reproducible train/validation/test splits
- `scripts/param_search.py` — parameter search for packaged models and datasets; Ray Tune is loaded only for the Ray-backed search path
- `scripts/model_run.py` — train and evaluate a selected packaged model

```bash
PYTHONPATH=. python scripts/set_splits.py --help
PYTHONPATH=. python scripts/param_search.py --help
PYTHONPATH=. python scripts/model_run.py --help
```

## Custom Model Reference

User-owned `nn.Module` models can compose with packaged data, split, and preprocessing utilities — see [Custom Model Reference](docs/custom_model_reference.md).

## Documentation

- `docs/README.md` — docs index for the current package
- `docs/api_reference.md` — exported runtime API
- `docs/model_reference.md` — per-model origin paper, canonical repo, and implementation note
- `docs/scripts_reference.md` — source-checkout CLI scripts: flags, outputs, and pipeline order
- `docs/data_reference.md` — guidance for adding user-provided datasets and NPZ script inputs
- `docs/custom_model_reference.md` — guidance for using custom models alongside the package

Tutorial: <https://nde-for-ts.github.io/> (CIKM 2025).

## Project Status

This is a **preview** release:

- `0.1.9` is a pre-`0.2.0` freeze/consistency release: the same 47-model roster and API as `0.1.8`, with no new methods or interface changes.
- Classification is the primary supported workflow.
- Synthetic missingness (`MCAR`/`MAR`/`MNAR`/`BLOCK`/`BURST`) and robustness result schemas are experimental.

## How to Cite

If you use `torch-ists`, please cite the core papers behind the package:

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

If your work relies on a specific packaged model, also cite the original paper for that model; see [`docs/model_reference.md`](docs/model_reference.md).
