# torch-ists

[![Version](https://img.shields.io/badge/version-0.5.8-blue)](pyproject.toml) [![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml) [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Package status](https://img.shields.io/badge/status-preview-orange)](#project-status)

**PyTorch toolkit for irregularly sampled multivariate time series with continuous-time methods (Neural ODE/CDE/SDE) and discrete-time baselines.**

It provides runtime APIs, benchmark scripts, and a built-in benchmark archive loader.

**Related source docs**: [Docs index](docs/README.md) · [API Reference](docs/api_reference.md) · [Model Reference](docs/model_reference.md) · [Scripts Reference](docs/scripts_reference.md) · [Custom Data Guide](docs/custom_data_guide.md) · [Custom Model Guide](docs/custom_model_guide.md) · [Attribution and Licenses](docs/attribution_and_licenses.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

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

The package-authored NeuralFlow operators are the default. To use the exact external NFE reference backend, review the upstream repository's terms first; the pinned source does not publish a LICENSE file and is not bundled with this package. Install the required checkout explicitly:

```bash
PYTHONNOUSERSITE=1 python -m pip install "stribor==0.1.0" "torchdiffeq==0.2.1"
PYTHONNOUSERSITE=1 python -m pip install --no-deps "git+https://github.com/mbilos/neural-flows-experiments.git@bd19f7c92461e83521e268c1a235ef845a3dd963"
PYTHONNOUSERSITE=1 python -c "from torch_ists.module.flow.nfe_reference import require_nfe_installation; print(require_nfe_installation()['valid'])"
```

The final command validates the repository URL, commit, dependency versions, import location, and source files without importing the external model code. Activate this backend only in the Python API with `model_params={"flow_backend": "nfe"}`. The source-checkout benchmark scripts always use the package-authored native backend.

**Source-checkout scripts require `torchcde`.** The `scripts/set_splits.py` -> `scripts/param_search.py` -> `scripts/model_run.py` pipeline shares one preprocessing path that builds interpolation coefficients with `torchcde`. This applies to sequence baselines too. Install `.[all]` (recommended) or at least `.[tune,cde]`.

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

Install the `cde` extra before this preprocessing call (`PYTHONNOUSERSITE=1 python -m pip install -e ".[cde]"`). The example is self-contained and derives model dimensions from the raw array:

```python
import numpy as np
import torch
from torch_ists import ists_classifier, preprocess, set_global_seed

set_global_seed(0)  # optional - for reproducibility
X = np.random.default_rng(0).normal(size=(40, 3, 24)).astype(np.float32)  # [N, D, L]
input_dim = int(X.shape[1])
seq_len = int(X.shape[2])

model = ists_classifier(
    model_name="transformer",
    num_class=2,
    input_dim=input_dim,
    seq_len=seq_len,
    device="cpu",
)

X_missing, X_mask, X_delta, _ = preprocess(X, model_name="transformer")
seq = torch.stack([X_missing, X_mask, X_delta], dim=1)  # [N, 3, L, D]
logits = model(seq, coeffs=None)
print(logits.shape)  # torch.Size([40, 2])
```

Coefficient-driven CDE and SDE rows consume the `coeffs` returned by `preprocess(...)`. Direct sequence, attention, and event-driven rows, including ODE-LSTM, the direct ODE-RNN models, Neural Jump ODE (NJ-ODE), and LatentSDE, use the sequence stack and accept `coeffs=None`.

The source-checkout pipeline shares a continuous-time preprocessing path that builds interpolation coefficients with `torchcde`. The following bounded classification workflow can be launched from any working directory after an editable source install. It keeps every generated artifact under `/tmp`:

```bash
TORCH_ISTS_ROOT="$(PYTHONNOUSERSITE=1 python -c 'import pathlib, torch_ists; print(pathlib.Path(torch_ists.__file__).resolve().parents[1])')"
cd /tmp

PYTHONNOUSERSITE=1 python -c "import numpy as np, pathlib; p=pathlib.Path('/tmp/torch_ists_demo'); p.mkdir(parents=True, exist_ok=True); rng=np.random.default_rng(0); np.savez(p/'toy.npz', X=rng.normal(size=(40, 3, 24)).astype('float32'), y=np.array([0, 1] * 20))"

PYTHONNOUSERSITE=1 PYTHONPATH="$TORCH_ISTS_ROOT" python "$TORCH_ISTS_ROOT/scripts/set_splits.py" \
    --dataset-name Demo --dataset-source npz --dataset-npz /tmp/torch_ists_demo/toy.npz \
    --seed 42 --out-dir /tmp/torch_ists_demo/split \
    --contract-dir /tmp/torch_ists_demo/evidence

PYTHONNOUSERSITE=1 PYTHONPATH="$TORCH_ISTS_ROOT" python "$TORCH_ISTS_ROOT/scripts/param_search.py" \
    --datasets Demo --dataset-source npz --dataset-npz /tmp/torch_ists_demo/toy.npz \
    --models cnn --seed 42 --num-samples 1 --max-num-epochs 1 --max-runs 1 \
    --split-root /tmp/torch_ists_demo/split --param-root /tmp/torch_ists_demo/params \
    --contract-dir /tmp/torch_ists_demo/evidence --device cpu

PYTHONNOUSERSITE=1 PYTHONPATH="$TORCH_ISTS_ROOT" python "$TORCH_ISTS_ROOT/scripts/set_splits.py" \
    --dataset-name Demo --dataset-source npz --dataset-npz /tmp/torch_ists_demo/toy.npz \
    --seed 0 --out-dir /tmp/torch_ists_demo/split \
    --contract-dir /tmp/torch_ists_demo/evidence

PYTHONNOUSERSITE=1 PYTHONPATH="$TORCH_ISTS_ROOT" python "$TORCH_ISTS_ROOT/scripts/model_run.py" \
    --datasets Demo --dataset-source npz --dataset-npz /tmp/torch_ists_demo/toy.npz \
    --models cnn --seed 0 --epochs 1 --max-runs 1 \
    --split-root /tmp/torch_ists_demo/split --param-root /tmp/torch_ists_demo/params \
    --out-root /tmp/torch_ists_demo/out --contract-dir /tmp/torch_ists_demo/evidence --device cpu

PYTHONNOUSERSITE=1 PYTHONPATH="$TORCH_ISTS_ROOT" python "$TORCH_ISTS_ROOT/scripts/aggregate_results.py" \
    --evidence-root /tmp/torch_ists_demo/evidence --out-root /tmp/torch_ists_demo/out \
    --output-dir /tmp/torch_ists_demo/aggregate
```

See [Model Reference](docs/model_reference.md) for which models need `preprocess(X, model_name=...)` versus the direct sequence stack.

Benchmark loader note: variable-length UCR/UEA panels are padded to the pooled train+test maximum length with `NaN` by default. Use `padding="last"` only when reproducing the older interpolation path.

Missing-value handling:

- `preprocess(...)` preserves missing values as `NaN` in `X_missing` and derives `X_mask`, `X_delta`, and interpolation coefficients from that path.
- Most direct backbones fill missing value-channel entries before the backbone call. `ists_classifier(..., imputation_strategy="zero")` is the default and matches the established zero-fill baseline.
- `neuralflow-*` and `flowpath-*` are strict raw-missing rows: keep missing positions as `NaN` with `X_mask=0`. They use method-defined mask-aware zero filling and reject other `imputation_strategy` values.
- `last_observed` (last-observation-carried-forward) and `channel_mean` (per-series observed mean) are experimental opt-in alternatives for compatible direct value-channel backbones.

## Models Included

The package includes the 52 canonical model names below. See [Model Reference](docs/model_reference.md) for per-model details.

**Canonical model names:**

- **Sequence baselines** — `cnn`, `rnn`, `lstm`, `gru`, `bilstm`, `gru-simple`, `plstm`, `tlstm`, `tglstm` (§1)
- **Attention-based methods** — `transformer`, `transformer-mask`, `sand`, `mtan`, `miam` (§2)
- **Neural ODE methods** — `gru-dt`, `gru-d`, `gru-ode`, `ode-rnn-spline`, `ode-lstm` (§3)
- **ODE-RNN, Latent ODE, GRU-ODE-Bayes, and NJ-ODE** — `ode-rnn`, `latentode`, `gru-ode-bayes`, `njode` (§4)
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
- `docs/attribution_and_licenses.md` — attribution and licenses for packaged methods, plus benchmark-dataset provenance

Tutorial: <https://nde-for-ts.github.io/> (CIKM 2025).

## Project Status

This is a **preview** release:

- Classification is the stable workflow. Regression, forecasting, imputation, outlier detection, and change detection are available through `ists_model` and the source-checkout scripts as preview workflows with dataset-owned targets and task-isolated artifacts.
- Synthetic missingness (`MCAR`/`MAR`/`MNAR`/`BLOCK`/`BURST`/`CHANNEL_DROP`) is experimental. The optional 100-condition robustness workflow applies to classification only.
- Broad benchmark parity and performance claims remain future work; this package is still a preview release.

## How to Cite

If you use `torch-ists`, please cite the core papers behind the package:

```bibtex
@inproceedings{oh_stable_2024,
    title        = {Stable {Neural} {Stochastic} {Differential} {Equations} in {Analyzing} {Irregular} {Time} {Series} {Data}},
    author       = {Oh, YongKyung and Lim, Dong-Young and Kim, Sungil},
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
