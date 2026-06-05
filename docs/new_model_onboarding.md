# torch-ists New Model Onboarding

> **Package Version**: 0.1.4
> **Scope**: Public extension boundary for external custom models.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Scripts Reference](scripts_reference.md) · [Project site](https://nde-for-ts.github.io/)

This guide documents how to use `torch-ists` with a custom PyTorch model while keeping the packaged runtime contract stable. It does not add new active model names to the package.

## Public Boundary

Supported in version `0.1.4`:

- user-owned `torch.nn.Module` implementations outside the `torch_ists` namespace
- packaged data loading, split utilities, preprocessing helpers, and seed utilities
- external training loops that call a custom model directly
- wrapper code that composes `torch_ists` helpers with user-owned modules

Not public API in version `0.1.4`:

- handler registration for new model names
- mutation of `MODEL_REGISTRY`, `active_models.yaml`, or packaged handler bootstrap code
- bundled script training for external model names
- package claims that a custom model is part of the 35-model active roster

## Wrapper Pattern

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
        values = torch.nan_to_num(seq[:, 0], nan=0.0)
        _, hidden = self.encoder(values)
        return self.head(hidden[-1])


set_global_seed(0)
model = ExternalSequenceClassifier(input_dim=8, num_class=2)
seq = torch.randn(16, 3, 64, 8)
logits = model(seq)
```

The expected sequence stack is `[batch, channels, length, features]`. The first channel is the value channel used by the simple wrapper above. Missing values remain the responsibility of the external model or the external training loop unless a packaged preprocessing helper is used explicitly.

## Using Packaged Splits

The bundled split utilities can still be used with external training code:

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

`scripts/param_search.py` and `scripts/model_run.py` are scoped to packaged model names in this release. Use them for the active roster, and use external training code for custom models.

## Namespace Rule

Keep custom code under a project namespace such as `my_project.models` or `experiments.custom_models`. Do not add custom modules under `torch_ists` in an installed package checkout unless you are developing a separate package contribution.

## Promotion Rule

A custom model becomes part of the public package only when a later package version explicitly adds:

- a model name in the public active model list
- model metadata and preprocessing policy
- runtime tests for the forward contract
- documentation in the model and API references
- publish payload parity coverage
