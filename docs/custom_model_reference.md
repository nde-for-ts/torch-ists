# torch-ists Custom Model Reference

> **Scope**: Custom model wrapper guide for user-owned PyTorch models.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Scripts Reference](scripts_reference.md) · [Data Reference](data_reference.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

This reference shows how to use `torch-ists` data utilities with your own PyTorch model. Custom models run in your project code; the source-checkout CLI runs packaged models only.

---

## Table of Contents

1. [What This Reference Covers](#1-what-this-reference-covers)
2. [Sequence Wrapper Pattern](#2-sequence-wrapper-pattern)
3. [Missing-Value Policy](#3-missing-value-policy)
4. [NDE Wrapper Pattern with Coeffs](#4-nde-wrapper-pattern-with-coeffs)
5. [Using Packaged Splits](#5-using-packaged-splits)

---

## 1. What This Reference Covers

You can compose a user-owned `torch.nn.Module` (outside `torch_ists`) with the package's data loading, split utilities, preprocessing, and seed helpers. Training and any command-line entry points stay in your own code; the source-checkout CLI only accepts packaged model names. The rest of this doc shows the pattern.

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

The sequence stack is `[batch, channels, length, features]`; channel 0 contains values. Missing-value handling belongs to your model or training loop unless you call `preprocess(...)`.

## 3. Missing-Value Policy

`torch.nan_to_num(seq[:, 0], nan=0.0)` is a naive baseline policy. It cannot distinguish a true zero from a missing value unless your model also consumes mask or delta channels.

Recommended custom-model paths:

- Use `coeffs=None` plus your own imputation policy, keeping mask and time-delta channels if your architecture can consume them.
- Consume the `(values, mask, delta)` stack directly instead of reading only `seq[:, 0]`.
- Use `preprocess(...)` to build interpolation `coeffs` and pass those coefficients into the custom model.

## 4. NDE Wrapper Pattern with Coeffs

NDE-style custom models should consume interpolation coefficients. Install the CDE extra when using this pattern:

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

The custom model owns the NDE architecture; `torch-ists` supplies preprocessing and interpolation coefficients.

## 5. Using Packaged Splits

The packaged split utilities can still be used with custom training code. See [Data Reference](data_reference.md) for the full dataset shape, NPZ input, and split file format.

```python
from pathlib import Path

from torch_ists import load_splits


train_idx, valid_idx, test_idx = load_splits(Path("split/ECG200/ECG200_0"))
```

For user-provided NPZ datasets, generate the split artifact with the source-checkout script, then train the custom model from a user-owned script:

```bash
PYTHONPATH=. python scripts/set_splits.py \
  --dataset-name ToyDataset \
  --dataset-source npz \
  --dataset-npz data/ToyDataset.npz \
  --seed 0
```

Use the source-checkout CLI only for packaged models; run custom models from your own training script.

Keep custom model code under a project namespace such as `my_project.models` or `experiments.custom_models`. Avoid placing project-specific modules inside `torch_ists`.

---

**End of Custom Model Reference**
