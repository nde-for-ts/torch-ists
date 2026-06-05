"""NFE/stribor-backed flow operators for runtime package (guarded)."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ._nfe_guard import ensure_nfe_backend_enabled


def _make_hidden_dims(hidden_dims: Iterable[int] | None, fallback: int) -> list[int]:
    dims = [int(x) for x in (hidden_dims or []) if int(x) > 0]
    return dims if dims else [int(fallback)]


class _BaseNFEAdapter(nn.Module):
    def __init__(self, model: nn.Module, n_layers: int):
        super().__init__()
        self.model = model
        self.backend = "nfe"
        self.num_layers = max(1, int(n_layers))

        layers = getattr(model, "layers", None)
        if isinstance(layers, nn.ModuleList):
            self.layers = layers
            return
        if isinstance(layers, (list, tuple)) and all(isinstance(m, nn.Module) for m in layers):
            self.layers = nn.ModuleList(list(layers))
            return

        transforms = getattr(getattr(model, "flow", None), "transforms", None)
        if isinstance(transforms, (list, tuple)) and all(isinstance(m, nn.Module) for m in transforms):
            self.layers = nn.ModuleList(list(transforms))
            return

        # Fallback for adapter contract introspection only.
        self.layers = nn.ModuleList(nn.Identity() for _ in range(self.num_layers))

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
        ensure_nfe_backend_enabled()
        from ._vendor_nfe.flow import ResNetFlow as VendoredResNetFlow

        depth = max(1, int(n_layers))
        dims = _make_hidden_dims(hidden_dims, dim)
        model = VendoredResNetFlow(
            dim=int(dim),
            n_layers=depth,
            hidden_dims=dims,
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
        ensure_nfe_backend_enabled()
        from ._vendor_nfe.gru import GRUFlow as VendoredGRUFlow

        depth = max(1, int(n_layers))
        model = VendoredGRUFlow(
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
        ensure_nfe_backend_enabled()
        from ._vendor_nfe.flow import CouplingFlow as VendoredCouplingFlow

        depth = max(1, int(n_layers))
        dims = _make_hidden_dims(hidden_dims, dim)
        model = VendoredCouplingFlow(
            dim=int(dim),
            n_layers=depth,
            hidden_dims=dims,
            time_net="TimeTanh",
            time_hidden_dim=max(8, int(dim)),
        )
        super().__init__(model=model, n_layers=depth)


__all__ = [
    "ResNetFlow",
    "GRUFlow",
    "CouplingFlow",
]
