# torch-ists

[![Version](https://img.shields.io/badge/version-0.3.2-blue)](pyproject.toml) [![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml) [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Package status](https://img.shields.io/badge/status-preview-orange)](#project-status)

**PyTorch toolkit for irregularly sampled multivariate time series with continuous-time methods (Neural ODE/CDE/SDE) and discrete-time baselines.**

It provides runtime APIs, benchmark scripts, and a built-in benchmark archive loader.

**Related source docs**: [Docs index](docs/README.md) · [API Reference](docs/api_reference.md) · [Model Reference](docs/model_reference.md) · [Scripts Reference](docs/scripts_reference.md) · [Custom Data Guide](docs/custom_data_guide.md) · [Custom Model Guide](docs/custom_model_guide.md) · [Third-Party Notices](docs/third_party_notices.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

## Overview

The package exposes `ists_classifier` for classification and the lower-level `ists_model` constructor for additional task heads. It includes 50 models spanning sequence baselines, attention-based methods, Neural ODE/CDE/SDE methods, NeuralFlow, DualDynamics and FlowPath, and Stable Neural SDEs.

## Installation

Requirements:

- `Python >= 3.10`
- `PyTorch >= 2.0`
- `pip >= 22.3` for the pyproject-only build path; upgrade older system pip before installing from source.
- `PYTHONNOUSERSITE=1` for runtime and source-checkout script commands

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

**Source-checkout scripts require `torchcde`.** The `scripts/set_splits.py` -> `scripts/param_search.py` -> `scripts/model_run.py` pipeline shares one preprocessing path that builds interpolation coefficients with `torchcde`. This runs for **every** selected model, including the sequence baselines such as `cnn`, so the scripts need `torchcde` even when the model is not a continuous-time method. Install `.[all]` (recommended) or at least `.[tune,cde]`; `.[tune]` on its own can fail with `No module named 'torchcde'`.

NeuralCDE / Neural RDE / ANCDE / EXIT / LEAP additionally require `signatory`. It is kept out of `.[all]` because the PyPI wheel is torch-version-pinned; install the source-build extra only when those paths are needed:

```bash
PYTHONNOUSERSITE=1 python -m pip install --no-build-isolation -e .[signatory]
```

## Data Loading

Use `DataLoader` for user-provided NumPy arrays, data frames, or files. Use `torch_ists.benchmark_loader` for built-in benchmark archive datasets and the source-checkout benchmark scripts:

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

Direct sequence and attention baselines take `coeffs=None`, while continuous-time methods (Neural CDE/ODE/SDE) consume the `coeffs` returned by `preprocess(...)`.

The source-checkout pipeline shares a continuous-time preprocessing path that builds cubic-spline coefficients with `torchcde`, so install the optional backends first:

```bash
PYTHONNOUSERSITE=1 python -m pip install -e .[all]
```

The pipeline runs `set_splits.py` -> `param_search.py` -> `model_run.py` in order; `model_run.py` reads the parameters written by `param_search.py`. The bounded smoke run below uses a small self-contained NPZ example, so it is reproducible without a dataset download:

```bash
# Create a toy NPZ payload: X of shape [N, D, L], y of shape [N]
python -c "import numpy as np, pathlib; p=pathlib.Path('/tmp/torch_ists_demo'); p.mkdir(parents=True, exist_ok=True); rng=np.random.default_rng(0); np.savez(p/'toy.npz', X=rng.normal(size=(40, 3, 24)).astype('float32'), y=np.array([0, 1] * 20))"

# 1) generate the split for the hyperparameter-search seed (seed 42, the new default)
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/set_splits.py \
    --dataset-name Demo --dataset-source npz --dataset-npz /tmp/torch_ists_demo/toy.npz \
    --seed 42 --out-dir /tmp/torch_ists_demo/split

# 2) bounded parameter search (1 sample, 1 epoch; reads the seed-42 split)
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/param_search.py \
    --datasets Demo --dataset-source npz --dataset-npz /tmp/torch_ists_demo/toy.npz \
    --models cnn --seed 42 --num-samples 1 --max-num-epochs 1 --max-runs 1 \
    --split-root /tmp/torch_ists_demo/split --param-root /tmp/torch_ists_demo/params --device cpu

# 3) generate the split for one evaluation seed (the reported runs sweep seeds 0..4)
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/set_splits.py \
    --dataset-name Demo --dataset-source npz --dataset-npz /tmp/torch_ists_demo/toy.npz \
    --seed 0 --out-dir /tmp/torch_ists_demo/split

# 4) train and evaluate the selected model on the seed-0 split (repeat with seeds 1..4 for the full evaluation)
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/model_run.py \
    --datasets Demo --dataset-source npz --dataset-npz /tmp/torch_ists_demo/toy.npz \
    --models cnn --seed 0 --epochs 1 --max-runs 1 \
    --split-root /tmp/torch_ists_demo/split --param-root /tmp/torch_ists_demo/params
```

These are bounded smoke-safe runs; the full defaults (`param_search.py --num-samples 20`, `model_run.py --epochs 100`) and all flags are documented in `docs/scripts_reference.md`. With the default single-seed search, the hyperparameter search seed (default `42`) is disjoint from the standard evaluation seeds (`0..4`), so reported numbers are out-of-sample with respect to hyperparameter selection (aggregated search tunes over `0..4` and is diagnostic, not out-of-sample). To reproduce the ICLR 2024 protocol pass `--seed 0` to step 1 (`set_splits.py`); step 2 (`param_search.py`) keeps its default `--seed 42` because the search seed is held fixed by policy (the param artifact filename does not encode the seed); steps 3 and 4 already run at seed `0`.

The script workflow resolves `batch_size` automatically from the dataset instead of searching it; the resolved value is recorded with the run metadata. See [Scripts Reference](docs/scripts_reference.md).

See [Model Reference](docs/model_reference.md) for which models need `preprocess(X, model_name=...)` versus the direct sequence stack.

Missing-value handling:

- `preprocess(...)` preserves missing values as `NaN` in `X_missing` and derives `X_mask`, `X_delta`, and interpolation coefficients from that path.
- Most direct backbones fill missing value-channel entries before the backbone call. `ists_classifier(..., imputation_strategy="zero")` is the default and matches the established zero-fill baseline.
- `neuralflow-*` and `flowpath-*` are strict raw-missing rows: keep missing positions as `NaN` with `X_mask=0`. The source-checkout scripts handle this automatically.
- `last_observed` (last-observation-carried-forward) and `channel_mean` (per-series observed mean) are experimental opt-in alternatives selected through the same argument.

## Models Included

The package includes 50 models across the groups below. See [Model Reference](docs/model_reference.md) for per-model details.

- **Sequence baselines** — `cnn`, `rnn`, `lstm`, `gru`, `bilstm`, `gru-simple`, `plstm`, `tlstm`, `tglstm` (§1)
- **Attention-based methods** — `transformer`, `transformer-mask`, `sand`, `mtan`, `miam` (§2)
- **Neural ODE methods** — `gru-dt`, `gru-d`, `gru-ode`, `ode-rnn-spline`, `ode-lstm` (§3)
- **ODE-RNN, Latent ODE, and GRU-ODE-Bayes** — `ode-rnn`, `latentode`, `gru-ode-bayes` (§4)
- **Neural CDE and Neural RDE** — `neuralcde`, `neuralcde-l`, `neuralcde-r`, `neuralcde-c`, `neuralcde-h`, `neuralrde-1`, `neuralrde-2`, `neuralrde-3` (§5)
- **ANCDE, EXIT, and LEAP** — `ancde`, `exit`, `leap` (§6)
- **NeuralFlow** — `neuralflow` (residual-flow alias for `neuralflow-r`), `neuralflow-r`, `neuralflow-g`, `neuralflow-c` (§7)
- **DualDynamics and FlowPath** — `dualdynamics`, `flowpath`, and `-r/g/c` variants; the bare `dualdynamics` and `flowpath` names are residual-flow aliases of `dualdynamics-r` and `flowpath-r` (§8)
- **SDE baselines** — `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t` (§9)
- **Stable Neural SDEs** — `neural_lsde`, `neural_lnsde`, `neural_gsde` (§10)

## Source Checkout Scripts

- `scripts/set_splits.py` — generate reproducible train/validation/test splits
- `scripts/param_search.py` — parameter search; Ray Tune is loaded only when the Ray-backed path runs
- `scripts/model_run.py` — train and evaluate a selected packaged model

```bash
PYTHONPATH=. python scripts/set_splits.py --help
PYTHONPATH=. python scripts/param_search.py --help
PYTHONPATH=. python scripts/model_run.py --help
```

## Custom Model Guide

User-owned `nn.Module` models can compose with packaged data, split, and preprocessing utilities — see [Custom Model Guide](docs/custom_model_guide.md).

## Documentation

- `docs/README.md` — docs index for the current package
- `docs/api_reference.md` — exported runtime API
- `docs/model_reference.md` — per-model origin paper, canonical repo, and implementation note
- `docs/scripts_reference.md` — source-checkout CLI scripts: flags, outputs, and pipeline order
- `docs/custom_data_guide.md` — guidance for adding user-provided datasets and NPZ script inputs
- `docs/custom_model_guide.md` — guidance for using custom models alongside the package
- `docs/third_party_notices.md` — upstream sources and licenses for third-party code in the package

Tutorial: <https://nde-for-ts.github.io/> (CIKM 2025).

## Project Status

This is a **preview** release:

- Classification is the primary supported workflow.
- Source-checkout scripts default to model-specific search; constrained search is experimental and writes `_matched200k` params.
- Synthetic missingness (`MCAR`/`MAR`/`MNAR`/`BLOCK`/`BURST`) and robustness result schemas are experimental.
- Stable and parity-verified claims remain future work (`1.0.0` is the stable target).

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
