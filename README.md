# torch-ists

[![Version](https://img.shields.io/badge/version-0.4.9-blue)](pyproject.toml) [![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml) [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Package status](https://img.shields.io/badge/status-preview-orange)](#project-status)

**PyTorch toolkit for irregularly sampled multivariate time series with continuous-time methods (Neural ODE/CDE/SDE) and discrete-time baselines.**

It provides runtime APIs, benchmark scripts, and a built-in benchmark archive loader.

**Related source docs**: [Docs index](docs/README.md) · [API Reference](docs/api_reference.md) · [Model Reference](docs/model_reference.md) · [Scripts Reference](docs/scripts_reference.md) · [Custom Data Guide](docs/custom_data_guide.md) · [Custom Model Guide](docs/custom_model_guide.md) · [Third-Party Notices](docs/third_party_notices.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

## Overview

The package exposes `ists_classifier` for classification and the lower-level `ists_model` constructor for additional task heads. It includes 52 models spanning sequence baselines, attention-based methods, Neural ODE/CDE/SDE methods, NeuralFlow, DualDynamics and FlowPath, and Stable Neural SDEs.

## Installation

Requirements:

- `Python >= 3.10`
- `PyTorch >= 2.0`
- `pip >= 22.3` for the pyproject-only build path; upgrade older system pip before installing from source.
- `PYTHONNOUSERSITE=1` for runtime commands

This preview is not published on PyPI. Install it from a source checkout as shown below.

Install the base package:

```bash
PYTHONNOUSERSITE=1 python -m pip install --upgrade pip
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

Neural RDE (`neuralrde-1/2/3`) additionally requires `signatory`. It is kept out of `.[all]` because the PyPI wheel is torch-version-pinned; install the source-build extra only when those models are needed:

```bash
PYTHONNOUSERSITE=1 python -m pip install --no-build-isolation -e .[signatory]
```

## Data Loading

Use `DataLoader` for user-provided NumPy arrays, data frames, or files. Use `torch_ists.benchmark_loader` for built-in benchmark archive datasets:

```python
import numpy as np
from torch_ists import DataLoader

X = np.random.randn(40, 3, 24).astype(np.float32)  # [N, D, L]
y = np.asarray([0, 1] * 20)
bundle = DataLoader.from_numpy(X, y)
print(bundle.X.shape, bundle.y.shape)
```

For a built-in benchmark dataset:

```python
from torch_ists import benchmark_loader

X, y = benchmark_loader.get_data("BasicMotions")
```

See [Custom Data Guide](docs/custom_data_guide.md) for cache locations and user-provided dataset handling.

## Quick Start

```python
import torch
from torch_ists import ists_classifier, set_global_seed

set_global_seed(0)  # optional - for reproducibility

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

To go from a raw array to the forward call, build the sequence stack and continuous-time coefficients with `preprocess`:

```python
from torch_ists import preprocess
# X: your raw array of shape [N, D, L]
X_missing, X_mask, X_delta, coeffs = preprocess(X, model_name="neuralcde")
seq = torch.stack([X_missing, X_mask, X_delta], dim=1)  # [N, 3, L, D]
logits = model(seq, coeffs=coeffs)
```

Coefficient-driven CDE and SDE rows consume the `coeffs` returned by `preprocess(...)`. Direct sequence, attention, and event-driven rows, including ODE-LSTM, the direct ODE-RNN family, NJODE, and LatentSDE, use the sequence stack and accept `coeffs=None`.

See [Model Reference](docs/model_reference.md) for which models need `preprocess(X, model_name=...)` versus the direct sequence stack.

Benchmark loader note: variable-length UCR/UEA panels are padded to the pooled train+test maximum length with `NaN` by default. Use `padding="last"` only when reproducing the older interpolation path.

Missing-value handling:

- `preprocess(...)` preserves missing values as `NaN` in `X_missing` and derives `X_mask`, `X_delta`, and interpolation coefficients from that path.
- Most direct backbones fill missing value-channel entries before the backbone call. `ists_classifier(..., imputation_strategy="zero")` is the default and matches the established zero-fill baseline.
- `neuralflow-*` and `flowpath-*` are strict raw-missing rows: keep missing positions as `NaN` with `X_mask=0`.
- `last_observed` (last-observation-carried-forward) and `channel_mean` (per-series observed mean) are experimental opt-in alternatives selected through the same argument.

## Models Included

The package includes the 52 canonical model names below. See [Model Reference](docs/model_reference.md) for per-model details.

**Canonical model names:**

- **Sequence baselines** — `cnn`, `rnn`, `lstm`, `gru`, `bilstm`, `gru-simple`, `plstm`, `tlstm`, `tglstm` (§1)
- **Attention-based methods** — `transformer`, `transformer-mask`, `sand`, `mtan`, `miam` (§2)
- **Neural ODE methods** — `gru-dt`, `gru-d`, `gru-ode`, `ode-rnn-spline`, `ode-lstm` (§3)
- **ODE-RNN, Latent ODE, GRU-ODE-Bayes, and NJODE** — `ode-rnn`, `latentode`, `gru-ode-bayes`, `njode` (§4)
- **Neural CDE and Neural RDE** — `neuralcde`, `neuralcde-l`, `neuralcde-r`, `neuralcde-c`, `neuralcde-h`, `neuralrde-1`, `neuralrde-2`, `neuralrde-3` (§5)
- **ANCDE, EXIT, and LEAP** — `ancde`, `exit`, `leap` (§6)
- **NeuralFlow** — `neuralflow-r`, `neuralflow-g`, `neuralflow-c` (§7)
- **DualDynamics and FlowPath** — `dualdynamics-r`, `dualdynamics-g`, `dualdynamics-c`, `flowpath-r`, `flowpath-g`, `flowpath-c` (§8)
- **SDE baselines** — `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t`, `latentsde` (§9)
- **Stable Neural SDEs** — `neural_lsde`, `neural_lnsde`, `neural_gsde` (§10)

Compatibility aliases do not add models to the 52-model roster: `neuralflow`, `dualdynamics`, and `flowpath` resolve to their corresponding `-r` variants.

## Custom Model Guide

User-owned `nn.Module` models can compose with packaged data, split, and preprocessing utilities — see [Custom Model Guide](docs/custom_model_guide.md).

## Documentation

- `docs/README.md` — docs index for the current package
- `docs/api_reference.md` — exported runtime API
- `docs/model_reference.md` — per-model origin paper, canonical repo, and implementation note
- `docs/scripts_reference.md` — included scripts: flags, outputs, and pipeline order
- `docs/custom_data_guide.md` — guidance for adding user-provided datasets through the Python API
- `docs/custom_model_guide.md` — guidance for using custom models alongside the package
- `docs/third_party_notices.md` — upstream sources and licenses for third-party code in the package

Tutorial: <https://nde-for-ts.github.io/> (CIKM 2025).

## Project Status

This is a **preview** release:

- Classification is the stable workflow. Regression, forecasting, imputation, outlier detection, and change detection are available through `ists_model` as experimental task heads.
- Synthetic missingness (`MCAR`/`MAR`/`MNAR`/`BLOCK`/`BURST`/`CHANNEL_DROP`) is experimental. The optional 100-condition robustness workflow applies to classification only.
- Broad benchmark parity and performance claims remain future work; this package is still a preview release.

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
