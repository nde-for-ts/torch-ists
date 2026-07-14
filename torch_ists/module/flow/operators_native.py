"""Native flow operators for the runtime package."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _make_hidden_dims(hidden_dims: Iterable[int] | None, fallback: int) -> list[int]:
    dims = [int(x) for x in (hidden_dims or []) if int(x) > 0]
    return dims if dims else [int(fallback)]


class _TimeTanh(nn.Module):
    """Time embedding gate used by flow blocks."""

    def __init__(self, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim or max(8, out_dim))
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
            nn.Tanh(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class _MLP(nn.Module):
    """Small MLP helper."""

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResNetFlowBlock(nn.Module):
    def __init__(self, dim: int, hidden_dims: list[int]):
        super().__init__()
        self.residual = _MLP(dim + 1, hidden_dims, dim)
        self.time_gate = _TimeTanh(dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        update = self.residual(torch.cat([x, t], dim=-1))
        return x + self.time_gate(t) * update


class ResNetFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: list[int] | None = None,
        **_: object,
    ):
        super().__init__()
        dims = _make_hidden_dims(hidden_dims, dim)
        self.layers = nn.ModuleList(_ResNetFlowBlock(dim, dims) for _ in range(max(1, int(n_layers))))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != t.shape[-2]:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        for layer in self.layers:
            x = layer(x, t)
        return x


class _GRUFlowBlock(nn.Module):
    """GRU-style invertible residual block (Theorem-style bounded update)."""

    def __init__(self, dim: int):
        super().__init__()
        norm = lambda layer: torch.nn.utils.spectral_norm(layer, n_power_iterations=5)
        self.lin_hh = norm(nn.Linear(dim + 1, dim))
        self.lin_hz = norm(nn.Linear(dim + 1, dim))
        self.lin_hr = norm(nn.Linear(dim + 1, dim))
        self.time_gate = _TimeTanh(dim)
        self.alpha = 2.0 / 5.0
        self.beta = 4.0 / 5.0

    def _residual(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([h, t], dim=-1)
        r = self.beta * torch.sigmoid(self.lin_hr(inp))
        z = self.alpha * torch.sigmoid(self.lin_hz(inp))
        u = torch.tanh(self.lin_hh(torch.cat([r * h, t], dim=-1)))
        return z * (u - h)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return h + self.time_gate(t) * self._residual(h, t)


class GRUFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        **_: object,
    ):
        super().__init__()
        self.layers = nn.ModuleList(_GRUFlowBlock(dim) for _ in range(max(1, int(n_layers))))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != t.shape[-2]:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        for layer in self.layers:
            x = layer(x, t)
        return x


class _CouplingFlowLayer(nn.Module):
    def __init__(self, dim: int, hidden_dims: list[int], parity: int):
        super().__init__()
        self.dim = int(dim)
        self.parity = int(parity) % 2
        self.affine = _MLP(self.dim + 1, hidden_dims, self.dim * 2)

    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            return torch.ones_like(x)
        half = self.dim // 2
        mask = torch.zeros(self.dim, device=x.device, dtype=x.dtype)
        if self.parity == 0:
            mask[:half] = 1.0
        else:
            mask[half:] = 1.0
        return mask.view(*([1] * (x.dim() - 1)), self.dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mask = self._mask(x)
        frozen = x * mask
        params = self.affine(torch.cat([frozen, t], dim=-1))
        shift, log_scale = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * (1.0 - mask)
        shift = shift * (1.0 - mask)
        transformed = x * torch.exp(log_scale) + shift
        return frozen + transformed * (1.0 - mask)


class CouplingFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: list[int] | None = None,
        **_: object,
    ):
        super().__init__()
        dims = _make_hidden_dims(hidden_dims, dim)
        self.layers = nn.ModuleList(
            _CouplingFlowLayer(dim, dims, parity=i) for i in range(max(1, int(n_layers)))
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != t.shape[-2]:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        for layer in self.layers:
            x = layer(x, t)
        return x


__all__ = [
    "ResNetFlow",
    "GRUFlow",
    "CouplingFlow",
]
