# torch-ists

**PyTorch toolkit for irregular time series classification with neural ODE, CDE, and SDE models.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Status](https://img.shields.io/badge/status-preview-yellow.svg)](#project-status)

## Overview

`torch-ists` provides a unified interface for training and evaluating neural
differential equation models on irregularly-sampled multivariate time series.
Version `0.1.0` ships 25 packaged architectures across sequence, CDE, ODE,
and SDE families, including the **Stable Neural SDE** family from
[Oh et al., ICLR 2024](https://openreview.net/forum?id=4VIgNuQ1pY).

The package centers on the `ists_classifier` classification path. The
lower-level `ists_model` constructor is also available for advanced
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

# CDE backends (packaged NeuralCDE path uses both torchcde and torchdiffeq)
PYTHONNOUSERSITE=1 python -m pip install -e .[cde]

# SDE backends
PYTHONNOUSERSITE=1 python -m pip install -e .[sde]

# Full optional stack for the continuous-time models included in version 0.1.0
PYTHONNOUSERSITE=1 python -m pip install -e .[all]
```

## Quick Start

```python
import torch
from torch_ists import ists_classifier, set_global_seed

set_global_seed(0)

model = ists_classifier(
    model_name="cnn",
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

Version `0.1.0` includes 25 models across six families:

| Family | Models |
|--------|--------|
| **Sequence baselines** | `cnn`, `rnn`, `lstm`, `gru` |
| **Neural CDE** | `neuralcde`, `neuralcde-l`, `neuralcde-r`, `neuralcde-c`, `neuralcde-h` |
| **ODE runtime family** | `gru-dt`, `gru-d`, `gru-ode`, `ode-rnn-spline`, `ode-lstm` |
| **Additional ODE models** | `ode-rnn`, `latentode`, `gru-ode-bayes` |
| **Continuous-time baselines** | `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t` |
| **Stable Neural SDE** ([Oh et al., 2024](https://openreview.net/forum?id=4VIgNuQ1pY)) | `neural_lsde`, `neural_lnsde`, `neural_gsde` |

See [`docs/api_reference_v1.0.md`](docs/api_reference_v1.0.md) for the full
model and API specification.

## Features

- Unified handler architecture across sequence, CDE, ODE, and SDE families
- Reproducible train/validation/test splits with stratification
- Missing-data injection (MCAR) and robustness evaluation utilities
- Hyperparameter search via Ray Tune
- Standalone numpy data loading

## Packaged Scripts

- `scripts/set_splits.py` — generate reproducible train/validation/test splits
- `scripts/param_search.py` — Ray Tune-based parameter search for packaged
  models and datasets
- `scripts/model_run.py` — train and evaluate a selected packaged model

For the bundled `0.1.0` command-line workflow, continuous-time models use
model-specific default solvers. Most users can leave those defaults unchanged.
Advanced solver arguments are mainly for custom experiments, and unsupported
solver settings raise `ValueError` instead of being ignored silently.

## Documentation

- [API Reference](docs/api_reference_v1.0.md) — module/class/function reference
- [Quick Start tutorial](docs/api_reference_v1.0.md#1-quick-start) — minimal
  example walkthrough

## Project Status

This is a **preview** release (`v0.1.0`):

- Core classification pipeline is reproducible
- All 25 packaged models pass smoke and integration tests
- Validation across supported environments is in progress
- Custom-data workflows, additional model families (Flow, LatentSDE, TSDE), and
  stable/parity-verified claims are planned for future releases

`ists_classifier` is the primary supported entry point for classification.
Additional task modes exposed by `ists_model` are available for advanced use,
but they are experimental and may change in future releases.

## Citation

If you use this package in academic work, please cite the following papers:

**Stable Neural SDE (ICLR 2024)** — [arXiv:2402.14989](https://arxiv.org/abs/2402.14989)

```bibtex
@inproceedings{oh_stable_2024,
    title     = {Stable {Neural} {Stochastic} {Differential} {Equations}
                 in {Analyzing} {Irregular} {Time} {Series} {Data}},
    author    = {Oh, YongKyung and Lim, Dong-Young and Kim, Sungil},
    year      = 2024,
    month     = may,
    booktitle = {The {Twelfth} {International} {Conference} on
                 {Learning} {Representations}, {ICLR} 2024,
                 {Vienna}, {Austria}, {May} 7-11, 2024},
    publisher = {OpenReview.net},
    url       = {https://arxiv.org/abs/2402.14989}
}
```

**DualDynamics (AAAI 2025)** — [arXiv:2401.04979](https://arxiv.org/abs/2401.04979)

```bibtex
@inproceedings{oh_dualdynamics_2025,
    title      = {{DualDynamics}: {Synergizing} {Implicit} and {Explicit}
                  {Methods} for {Robust} {Irregular} {Time} {Series} {Analysis}},
    shorttitle = {{DualDynamics}},
    author     = {Oh, YongKyung and Lim, Dong-Young and Kim, Sungil},
    year       = 2025,
    month      = feb,
    booktitle  = {{AAAI}-25, {Sponsored} by the {Association} for the
                  {Advancement} of {Artificial} {Intelligence},
                  {February} 25 - {March} 4, 2025,
                  {Philadelphia}, {PA}, {USA}},
    publisher  = {AAAI Press},
    pages      = {19730--19739},
    doi        = {10.1609/aaai.v39i18.34173},
    editor     = {Walsh, Toby and Shah, Julie and Kolter, Zico},
    url        = {https://arxiv.org/abs/2401.04979}
}
```

**Neural Differential Equations Survey (IJCAI 2025)** — [arXiv:2502.09885](https://arxiv.org/abs/2502.09885)

```bibtex
@inproceedings{oh_comprehensive_2025,
    title     = {Comprehensive {Review} of {Neural} {Differential}
                 {Equations} for {Time} {Series} {Analysis}},
    author    = {Oh, YongKyung and Kam, Seungsu and Lee, Jonghun and
                 Lim, Dong-Young and Kim, Sungil and Bui, Alex A. T.},
    year      = 2025,
    month     = aug,
    booktitle = {Proceedings of the {Thirty}-{Fourth} {International}
                 {Joint} {Conference} on {Artificial} {Intelligence},
                 {IJCAI}-25},
    publisher = {International Joint Conferences on
                 Artificial Intelligence Organization},
    pages     = {10621--10631},
    doi       = {10.24963/ijcai.2025/1179},
    editor    = {Kwok, James},
    note      = {Survey Track},
    url       = {https://arxiv.org/abs/2502.09885}
}
```

## License

Released under the [MIT License](https://opensource.org/licenses/MIT).
Copyright © 2024-2026 YongKyung Oh.

## Acknowledgments

Built on top of [PyTorch](https://pytorch.org/),
[torchdiffeq](https://github.com/rtqichen/torchdiffeq),
[torchcde](https://github.com/patrick-kidger/torchcde),
[torchsde](https://github.com/google-research/torchsde), and
[diffrax](https://github.com/patrick-kidger/diffrax).
