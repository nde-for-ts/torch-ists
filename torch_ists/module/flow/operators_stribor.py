"""Package-owned adapters built directly from the optional MIT Stribor API."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


STRIBOR_VERSION = "0.1.0"


class StriborBackendUnavailableError(RuntimeError):
    """The supported optional Stribor runtime is unavailable."""


def require_stribor():
    try:
        distribution = metadata.distribution("stribor")
    except metadata.PackageNotFoundError as exc:
        raise StriborBackendUnavailableError(
            "The optional Stribor flow backend is unavailable. Install torch-ists "
            "with the 'flow' extra (stribor==0.1.0), or request flow_backend='native'."
        ) from exc
    installed = str(distribution.version)
    if installed != STRIBOR_VERSION:
        raise StriborBackendUnavailableError(
            f"Unsupported stribor version {installed!r}; expected {STRIBOR_VERSION!r}."
        )
    try:
        import stribor as st  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name != "stribor":
            raise
        raise StriborBackendUnavailableError(
            "The optional Stribor flow backend is unavailable. Install torch-ists "
            "with the 'flow' extra (stribor==0.1.0), or request flow_backend='native'."
        ) from exc
    expected_origin = Path(distribution.locate_file("stribor/__init__.py")).resolve()
    actual_file = getattr(st, "__file__", None)
    actual_origin = Path(actual_file).resolve() if actual_file else None
    if actual_origin != expected_origin:
        raise RuntimeError(
            "The imported stribor module does not come from the installed distribution: "
            f"expected={expected_origin}, actual={actual_origin}"
        )
    return st


def _hidden_dims(hidden_dims: Iterable[int] | None, fallback: int) -> list[int]:
    dims = [int(value) for value in (hidden_dims or []) if int(value) > 0]
    return dims if dims else [int(fallback)]


class _DirectResNetFlow(nn.Module):
    def __init__(self, st, dim: int, n_layers: int, hidden_dims: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                st.net.ResNetFlow(
                    dim,
                    hidden_dims,
                    n_layers,
                    activation="ReLU",
                    final_activation=None,
                    time_net="TimeTanh",
                    time_hidden_dim=max(8, dim),
                    invertible=True,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        for layer in self.layers:
            x = layer(x, t)
        return x

    def inverse(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            y = layer.inverse(y, t)
        return y


class _DirectGRUFlowBlock(nn.Module):
    def __init__(self, st, dim: int):
        super().__init__()
        spectral = lambda layer: torch.nn.utils.spectral_norm(  # noqa: E731
            layer, n_power_iterations=5
        )
        self.lin_hh = spectral(nn.Linear(dim + 1, dim))
        self.lin_hz = spectral(nn.Linear(dim + 1, dim))
        self.lin_hr = spectral(nn.Linear(dim + 1, dim))
        self.time_net = st.net.TimeTanh(dim, hidden_dim=max(8, dim))

    def _residual(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([h, t], dim=-1)
        reset = (4.0 / 5.0) * torch.sigmoid(self.lin_hr(joined))
        update = (2.0 / 5.0) * torch.sigmoid(self.lin_hz(joined))
        candidate = torch.tanh(self.lin_hh(torch.cat([reset * h, t], dim=-1)))
        return update * (candidate - h)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return h + self.time_net(t) * self._residual(h, t)

    def inverse(self, y: torch.Tensor, t: torch.Tensor, iterations: int = 100) -> torch.Tensor:
        x = y
        for _ in range(max(1, int(iterations))):
            x = y - self.time_net(t) * self._residual(x, t)
        return x


class _DirectGRUFlow(nn.Module):
    def __init__(self, st, dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DirectGRUFlowBlock(st, dim) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != t.shape[-2]:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        for layer in self.layers:
            x = layer(x, t)
        return x

    def inverse(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            y = layer.inverse(y, t)
        return y


class _DirectCouplingFlow(nn.Module):
    def __init__(self, st, dim: int, n_layers: int, hidden_dims: list[int]):
        super().__init__()
        transforms = []
        for index in range(n_layers):
            transforms.append(
                st.ContinuousAffineCoupling(
                    latent_net=st.net.MLP(dim + 1, hidden_dims, 2 * dim),
                    time_net=st.net.TimeTanh(2 * dim, hidden_dim=max(8, dim)),
                    mask="none" if dim == 1 else f"ordered_{index % 2}",
                )
            )
        self.flow = st.Flow(transforms=transforms)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        return self.flow(x, t=t)[0]

    def inverse(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.flow.inverse(y, t=t)[0]


class _BaseAdapter(nn.Module):
    def __init__(self, model: nn.Module, n_layers: int):
        super().__init__()
        self.model = model
        self.backend = "stribor"
        self.fallback_reason = None
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

    def inverse(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inverse = getattr(self.model, "inverse", None)
        if inverse is None:
            raise NotImplementedError("The selected Stribor operator has no inverse method.")
        return inverse(y, t)


class ResNetFlow(_BaseAdapter):
    def __init__(self, dim: int, n_layers: int, hidden_dims=None, **_: object):
        st = require_stribor()
        depth = max(1, int(n_layers))
        super().__init__(
            _DirectResNetFlow(st, int(dim), depth, _hidden_dims(hidden_dims, dim)),
            depth,
        )


class GRUFlow(_BaseAdapter):
    def __init__(self, dim: int, n_layers: int, hidden_dims=None, **_: object):
        del hidden_dims
        st = require_stribor()
        depth = max(1, int(n_layers))
        super().__init__(_DirectGRUFlow(st, int(dim), depth), depth)


class CouplingFlow(_BaseAdapter):
    def __init__(self, dim: int, n_layers: int, hidden_dims=None, **_: object):
        st = require_stribor()
        depth = max(1, int(n_layers))
        super().__init__(
            _DirectCouplingFlow(st, int(dim), depth, _hidden_dims(hidden_dims, dim)),
            depth,
        )


__all__ = [
    "STRIBOR_VERSION",
    "StriborBackendUnavailableError",
    "require_stribor",
    "ResNetFlow",
    "GRUFlow",
    "CouplingFlow",
]
