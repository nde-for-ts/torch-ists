"""Runtime fault policy helpers.

This module centralizes runtime fault behavior toggles so layer.py can stay
focused on model routing/forward composition.
"""

from __future__ import annotations

import os
import warnings

import torch

STRICT_AUXILIARY_LOSS_ENV = "TORCH_ISTS_STRICT_AUXILIARY_LOSS"


def is_strict_auxiliary_loss_enabled() -> bool:
    value = os.environ.get(STRICT_AUXILIARY_LOSS_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def raise_for_nonfinite_tensor(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    model_name: str,
) -> None:
    nonfinite_mask = ~torch.isfinite(tensor)
    if not bool(nonfinite_mask.any().item()):
        return
    nonfinite_count = int(nonfinite_mask.sum().item())
    total_count = int(tensor.numel())
    raise RuntimeError(
        f"{tensor_name} contains non-finite values for model '{model_name}' "
        f"({nonfinite_count}/{total_count} elements)."
    )


def handle_auxiliary_loss_nonfinite(loss: torch.Tensor, *, model_name: str, event_count: int) -> torch.Tensor:
    if is_strict_auxiliary_loss_enabled():
        raise RuntimeError(
            f"Auxiliary loss contains non-finite values for model '{model_name}' while "
            f"{STRICT_AUXILIARY_LOSS_ENV}=1."
        )
    warnings.warn(
        f"Auxiliary loss contains non-finite values for model '{model_name}'. "
        f"Applying nan_to_num fallback (event={event_count}).",
        RuntimeWarning,
        stacklevel=3,
    )
    return torch.nan_to_num(loss)


__all__ = [
    "STRICT_AUXILIARY_LOSS_ENV",
    "handle_auxiliary_loss_nonfinite",
    "is_strict_auxiliary_loss_enabled",
    "raise_for_nonfinite_tensor",
]
