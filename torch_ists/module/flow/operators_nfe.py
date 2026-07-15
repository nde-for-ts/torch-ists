"""Adapters for the optional external Neural Flows reference implementation."""

from __future__ import annotations

from importlib import import_module
from typing import Iterable

import torch
import torch.nn as nn

from .nfe_reference import NFE_INSTALL_COMMAND, require_nfe_installation


def _make_hidden_dims(hidden_dims: Iterable[int] | None, fallback: int) -> list[int]:
    dims = [int(value) for value in (hidden_dims or []) if int(value) > 0]
    return dims if dims else [int(fallback)]


def _load_reference_class(module_name: str, class_name: str):
    require_nfe_installation()
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The pinned NFE checkout could not be imported because an optional dependency "
            f"is missing: {exc.name}. Install the reference backend with:\n{NFE_INSTALL_COMMAND}"
        ) from exc
    return getattr(module, class_name)


class _BaseNFEAdapter(nn.Module):
    def __init__(self, model: nn.Module, n_layers: int):
        super().__init__()
        self.model = model
        self.backend = "nfe"
        self.num_layers = max(1, int(n_layers))

    @property
    def layers(self):
        layers = getattr(self.model, "layers", None)
        if layers is not None:
            return layers
        transforms = getattr(getattr(self.model, "flow", None), "transforms", None)
        if transforms is not None:
            return transforms
        return tuple(self.model.children())

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)


class ResNetFlow(_BaseNFEAdapter):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: list[int] | None = None,
        **_: object,
    ):
        reference_class = _load_reference_class("nfe.models.flow", "ResNetFlow")
        depth = max(1, int(n_layers))
        model = reference_class(
            dim=int(dim),
            n_layers=depth,
            hidden_dims=_make_hidden_dims(hidden_dims, dim),
            time_net="TimeTanh",
            time_hidden_dim=max(8, int(dim)),
        )
        super().__init__(model=model, n_layers=depth)


class GRUFlow(_BaseNFEAdapter):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: list[int] | None = None,
        **_: object,
    ):
        del hidden_dims
        reference_class = _load_reference_class("nfe.models.gru", "GRUFlow")
        depth = max(1, int(n_layers))
        model = reference_class(
            dim=int(dim),
            n_layers=depth,
            time_net="TimeTanh",
            time_hidden_dim=max(8, int(dim)),
        )
        super().__init__(model=model, n_layers=depth)


class CouplingFlow(_BaseNFEAdapter):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: list[int] | None = None,
        **_: object,
    ):
        reference_class = _load_reference_class("nfe.models.flow", "CouplingFlow")
        depth = max(1, int(n_layers))
        model = reference_class(
            dim=int(dim),
            n_layers=depth,
            hidden_dims=_make_hidden_dims(hidden_dims, dim),
            time_net="TimeTanh",
            time_hidden_dim=max(8, int(dim)),
        )
        super().__init__(model=model, n_layers=depth)


__all__ = ["ResNetFlow", "GRUFlow", "CouplingFlow"]
