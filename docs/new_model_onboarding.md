# torch-ists New Model Onboarding

> **Package Version**: 0.1.6
> **Scope**: Custom model wrapper guide for user-owned PyTorch models.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Scripts Reference](scripts_reference.md) · [New Data Onboarding](new_data_onboarding.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

This guide shows how to use `torch-ists` data utilities with your own PyTorch model. Custom models run in your project code, while the bundled CLI scripts run the packaged models listed in the [Model Reference](model_reference.md).

---

## Table of Contents

1. [What This Guide Covers](#1-what-this-guide-covers)
2. [Sequence Wrapper Pattern](#2-sequence-wrapper-pattern)
3. [Missing-Value Policy](#3-missing-value-policy)
4. [NDE Wrapper Pattern with Coeffs](#4-nde-wrapper-pattern-with-coeffs)
5. [Using Packaged Splits](#5-using-packaged-splits)
6. [Project Layout](#6-project-layout)
7. [Related References](#7-related-references)

---

## 1. What This Guide Covers

Version `0.1.6` supports custom-model workflows where you:

- define user-owned `torch.nn.Module` implementations outside the `torch_ists` namespace
- use packaged data loading, split utilities, preprocessing helpers, and seed utilities
- run your own training loop and call your custom model directly
- compose `torch_ists` helpers with modules owned by your project

For custom models, keep these responsibilities in your project code:

- model construction and training logic
- command-line entry points for your own experiments
- model metadata and reporting labels for your custom method
- result reporting that distinguishes your custom model from the 47 packaged models shipped in version `0.1.6`

## 2. Sequence Wrapper Pattern

```python
import torch
from torch import nn

from torch_ists import set_global_seed


class ExternalSequenceClassifier(nn.Module):
    def __init__(self, input_dim: int, num_class: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_class)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # Naive baseline only: see the missing-value policy below.
        values = torch.nan_to_num(seq[:, 0], nan=0.0)
        _, hidden = self.encoder(values)
        return self.head(hidden[-1])


set_global_seed(0)
model = ExternalSequenceClassifier(input_dim=8, num_class=2)
seq = torch.randn(16, 3, 64, 8)
logits = model(seq)
```

The expected sequence stack is `[batch, channels, length, features]`. The first channel is the value channel used by the simple wrapper above. Missing values remain the responsibility of the external model or the external training loop unless a packaged preprocessing helper is used explicitly.

## 3. Missing-Value Policy

`torch.nan_to_num(seq[:, 0], nan=0.0)` is a naive direct-sequence baseline policy, not the recommended missing-data policy for new methods. In the bundled script pipeline, values are normalized with training-only statistics before the model forward pass, so replacing `NaN` with `0.0` is equivalent to per-channel mean imputation in raw value space. Outside that normalized workflow it is literal zero-fill.

Do not treat blind `nan_to_num` as a robust imputation method. It discards the distinction between a true zero, a normalized mean value, and a missing observation unless the model separately consumes mask or delta channels.

Recommended custom-model paths:

- Direct naive wrapper: use `coeffs=None` plus an explicit imputation policy in your own data/training code, and keep mask/time-delta channels if your architecture can consume them.
- Missingness-aware discrete wrapper: consume the packaged `(values, mask, delta)` stack directly instead of reading only `seq[:, 0]`.
- NDE-style wrapper: use `preprocess(...)` to build interpolation `coeffs` and pass those coefficients into the custom model.
- Hybrid wrapper: combine explicit imputed values for discrete features with interpolation `coeffs` for the continuous-time path.

## 4. NDE Wrapper Pattern with Coeffs

NDE-style custom models should consume interpolation coefficients rather than relying on blind value imputation. Install the CDE extra when using this pattern:

```bash
PYTHONNOUSERSITE=1 python -m pip install -e .[cde]
```

```python
import torch
from torch import nn
import torchcde

from torch_ists.preprocessing import preprocess


class ExternalVectorField(nn.Module):
    def __init__(self, hidden_dim: int, coeff_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coeff_dim = coeff_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * coeff_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del t
        return self.net(z).view(z.shape[0], self.hidden_dim, self.coeff_dim)


class ExternalCoeffNDEClassifier(nn.Module):
    def __init__(self, coeff_dim: int, num_class: int, hidden_dim: int = 32):
        super().__init__()
        self.initial = nn.Linear(coeff_dim, hidden_dim)
        self.func = ExternalVectorField(hidden_dim=hidden_dim, coeff_dim=coeff_dim)
        self.head = nn.Linear(hidden_dim, num_class)

    def forward(self, coeffs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        control = torchcde.CubicSpline(coeffs, times)
        z0 = self.initial(control.evaluate(times[0]))
        z_t = torchcde.cdeint(X=control, func=self.func, z0=z0, t=times)
        return self.head(z_t[:, -1])


X = torch.randn(16, 8, 64)
X_missing, X_mask, X_delta, coeffs = preprocess(
    X,
    missing_rate=0.0,
    interpolate="hermite",
    use_intensity=False,
)
times = torch.linspace(0, 1, X.shape[-1])
model = ExternalCoeffNDEClassifier(coeff_dim=coeffs.shape[-1], num_class=2)
logits = model(coeffs, times)
```

This pattern keeps the continuous-time path explicit: the external model owns the NDE architecture, while `torch-ists` only supplies the preprocessing helper and interpolation coefficients.

## 5. Using Packaged Splits

The bundled split utilities can still be used with external training code. See [New Data Onboarding](new_data_onboarding.md) for the full dataset shape, NPZ input, and split file format.

```python
from pathlib import Path

from torch_ists import load_splits


train_idx, valid_idx, test_idx = load_splits(Path("split/ECG200/ECG200_0"))
```

For custom NPZ datasets, generate the split artifact with the packaged script, then train the custom model from a user-owned script:

```bash
PYTHONPATH=. python scripts/set_splits.py \
  --dataset-name ToyDataset \
  --dataset-source npz \
  --dataset-npz data/ToyDataset.npz \
  --seed 0
```

`scripts/param_search.py` and `scripts/model_run.py` only accept the model names shipped with the package. Use them for those models, and use external training code for custom models.

## 6. Project Layout

Keep custom code under a project namespace such as `my_project.models` or `experiments.custom_models`. Avoid placing project-specific modules inside an installed `torch_ists` package checkout; that makes upgrades and reproducibility harder.

## 7. Related References

- For packaged model names and method lineage, use [Model Reference](model_reference.md).
- For exported Python APIs, use [API Reference](api_reference.md).
- For bundled CLI scripts, use [Scripts Reference](scripts_reference.md).
- For installation and a quick start, use the [package README](../README.md).

---

**End of New Model Onboarding**
