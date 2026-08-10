"""Package-authored continuous flow operators."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _make_hidden_dims(hidden_dims: Iterable[int] | None, fallback: int) -> list[int]:
    dims = [int(value) for value in (hidden_dims or []) if int(value) > 0]
    return dims if dims else [int(fallback)]


def _prepare_inputs(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() < 2 or t.dim() != x.dim():
        raise ValueError(
            "flow state and time tensors must have matching ranks >= 2; "
            f"got x={tuple(x.shape)}, t={tuple(t.shape)}"
        )
    if x.shape[-1] != dim:
        raise ValueError(f"flow state last dimension must be {dim}, got {tuple(x.shape)}")
    if t.shape[-1] != 1:
        raise ValueError(f"flow time tensor must end in one time channel, got {tuple(t.shape)}")
    if x.shape[:-2] != t.shape[:-2]:
        raise ValueError(
            "flow state and time leading dimensions must match; "
            f"got x={tuple(x.shape)}, t={tuple(t.shape)}"
        )
    if x.shape[-2] == 1 and t.shape[-2] != 1:
        x = x.expand(*x.shape[:-2], t.shape[-2], dim)
    elif x.shape[-2] != t.shape[-2]:
        raise ValueError(
            "flow state time dimension must be one or match the evaluation grid; "
            f"got x={tuple(x.shape)}, t={tuple(t.shape)}"
        )
    return x, t.to(device=x.device, dtype=x.dtype)


def _prepare_reference_time(t0: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    if t0.dim() == 0:
        return t0.to(state).expand(*state.shape[:-1], 1)
    if t0.dim() != state.dim() or t0.shape[-1] != 1:
        raise ValueError(
            "flow reference time must be scalar or match the state rank with one time channel; "
            f"got state={tuple(state.shape)}, t0={tuple(t0.shape)}"
        )
    if t0.shape[:-2] != state.shape[:-2]:
        raise ValueError(
            "flow reference time leading dimensions must match the state; "
            f"got state={tuple(state.shape)}, t0={tuple(t0.shape)}"
        )
    if t0.shape[-2] == 1 and state.shape[-2] != 1:
        t0 = t0.expand(*t0.shape[:-2], state.shape[-2], 1)
    elif t0.shape[-2] != state.shape[-2]:
        raise ValueError(
            "flow reference time dimension must be one or match the state grid; "
            f"got state={tuple(state.shape)}, t0={tuple(t0.shape)}"
        )
    return t0.to(device=state.device, dtype=state.dtype)


def _validate_inverse_inputs(
    state: torch.Tensor,
    time: torch.Tensor,
    *,
    context: str,
) -> None:
    if not bool(torch.isfinite(state).all()):
        raise ValueError(f"{context} inverse state must be finite")
    if not bool(torch.isfinite(time).all()):
        raise ValueError(f"{context} inverse time must be finite")


def _validate_inverse_result(
    state: torch.Tensor,
    *,
    context: str,
) -> None:
    if not bool(torch.isfinite(state).all()):
        raise RuntimeError(f"{context} inverse produced a non-finite state")


def _fixed_point_inverse(
    y: torch.Tensor,
    t: torch.Tensor,
    *,
    update,
    iterations: int,
    context: str,
) -> torch.Tensor:
    resolved_iterations = max(1, int(iterations))
    x = y
    for _ in range(resolved_iterations):
        x = y - update(x, t)
    _validate_inverse_result(x, context=context)

    residual = x + update(x, t) - y
    _validate_inverse_result(residual, context=f"{context} residual")
    dtype = x.dtype if torch.is_floating_point(x) else torch.float32
    tolerance = 32.0 * torch.finfo(dtype).eps
    scale = torch.maximum(
        x.detach().abs().amax(),
        y.detach().abs().amax(),
    ).clamp_min(1.0)
    if bool((residual.detach().abs().amax() > tolerance * scale).item()):
        raise RuntimeError(
            f"{context} inverse did not converge within {resolved_iterations} iterations"
        )
    return x


class _TimeTanh(nn.Module):
    """Learned zero-origin time modulation."""

    def __init__(self, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        del hidden_dim
        self.scale = nn.Parameter(torch.ones(int(out_dim)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.tanh(t * self.scale)


class _ContractiveLinear(nn.Linear):
    """Linear layer with an explicit operator-norm upper bound of one."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight.norm().clamp_min(1.0)
        return torch.nn.functional.linear(x, self.weight / scale, self.bias)


class _MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        *,
        activation: type[nn.Module],
        contractive: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [int(in_dim), *hidden_dims, int(out_dim)]
        for index in range(len(dims) - 1):
            linear_class = _ContractiveLinear if contractive else nn.Linear
            linear: nn.Module = linear_class(dims[index], dims[index + 1])
            layers.append(linear)
            if index < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResNetFlowBlock(nn.Module):
    def __init__(self, dim: int, hidden_dims: list[int], contraction: float):
        super().__init__()
        if not 0.0 < contraction < 1.0:
            raise ValueError("ResNetFlow contraction must lie strictly between zero and one.")
        self.residual = _MLP(
            dim + 1,
            hidden_dims,
            dim,
            activation=nn.ReLU,
            contractive=True,
        )
        self.time_gate = _TimeTanh(dim)
        self.contraction = float(contraction)

    def _update(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        value = self.residual(torch.cat([x, t], dim=-1))
        return self.contraction * self.time_gate(t) * value

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x + self._update(x, t)

    def inverse(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        iterations: int = 100,
        *,
        context: str = "ResNetFlow",
    ) -> torch.Tensor:
        return _fixed_point_inverse(
            y,
            t,
            update=self._update,
            iterations=iterations,
            context=context,
        )


class ResNetFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: list[int] | None = None,
        contraction: float = 0.25,
        **_: object,
    ):
        super().__init__()
        self.dim = int(dim)
        self.backend = "native"
        self.num_layers = max(1, int(n_layers))
        dims = _make_hidden_dims(hidden_dims, self.dim)
        self.layers = nn.ModuleList(
            _ResNetFlowBlock(self.dim, dims, contraction) for _ in range(self.num_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, t = _prepare_inputs(x, t, dim=self.dim)
        if t0 is not None:
            x = self.inverse(x, _prepare_reference_time(t0, x))
        for layer in self.layers:
            x = layer(x, t)
        return x

    def inverse(self, y: torch.Tensor, t: torch.Tensor, iterations: int = 100) -> torch.Tensor:
        y, t = _prepare_inputs(y, t, dim=self.dim)
        _validate_inverse_inputs(y, t, context="ResNetFlow")
        for reverse_index, layer in enumerate(reversed(self.layers)):
            layer_index = self.num_layers - 1 - reverse_index
            y = layer.inverse(
                y,
                t,
                iterations=iterations,
                context=f"ResNetFlow layer {layer_index}",
            )
        return y


class _GRUFlowBlock(nn.Module):
    """GRU-style invertible residual block."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin_hh = _ContractiveLinear(dim + 1, dim)
        self.lin_hz = _ContractiveLinear(dim + 1, dim)
        self.lin_hr = _ContractiveLinear(dim + 1, dim)
        self.time_gate = _TimeTanh(dim)
        self.alpha = 2.0 / 5.0
        self.beta = 4.0 / 5.0

    def _residual(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([h, t], dim=-1)
        r = self.beta * torch.sigmoid(self.lin_hr(inp))
        z = self.alpha * torch.sigmoid(self.lin_hz(inp))
        u = torch.tanh(self.lin_hh(torch.cat([r * h, t], dim=-1)))
        return z * (u - h)

    def _update(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.time_gate(t) * self._residual(h, t)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return h + self._update(h, t)

    def inverse(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        iterations: int = 100,
        *,
        context: str = "GRUFlow",
    ) -> torch.Tensor:
        return _fixed_point_inverse(
            y,
            t,
            update=self._update,
            iterations=iterations,
            context=context,
        )


class GRUFlow(nn.Module):
    def __init__(self, dim: int, n_layers: int, **_: object):
        super().__init__()
        self.dim = int(dim)
        self.backend = "native"
        self.num_layers = max(1, int(n_layers))
        self.layers = nn.ModuleList(_GRUFlowBlock(self.dim) for _ in range(self.num_layers))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, t = _prepare_inputs(x, t, dim=self.dim)
        if t0 is not None:
            x = self.inverse(x, _prepare_reference_time(t0, x))
        for layer in self.layers:
            x = layer(x, t)
        return x

    def inverse(self, y: torch.Tensor, t: torch.Tensor, iterations: int = 100) -> torch.Tensor:
        y, t = _prepare_inputs(y, t, dim=self.dim)
        _validate_inverse_inputs(y, t, context="GRUFlow")
        for reverse_index, layer in enumerate(reversed(self.layers)):
            layer_index = self.num_layers - 1 - reverse_index
            y = layer.inverse(
                y,
                t,
                iterations=iterations,
                context=f"GRUFlow layer {layer_index}",
            )
        return y


class _CouplingFlowLayer(nn.Module):
    def __init__(self, dim: int, hidden_dims: list[int], parity: int):
        super().__init__()
        self.dim = int(dim)
        self.parity = int(parity) % 2
        self.affine = _MLP(
            self.dim + 1,
            hidden_dims,
            self.dim * 2,
            activation=nn.Tanh,
        )
        self.shift_time_gate = _TimeTanh(self.dim)
        self.scale_time_gate = _TimeTanh(self.dim)

    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(self.dim, device=x.device, dtype=x.dtype)
        if self.dim > 1:
            half = self.dim // 2
            if self.parity == 0:
                mask[:half] = 1.0
            else:
                mask[half:] = 1.0
        return mask.view(*([1] * (x.dim() - 1)), self.dim)

    def _affine_parameters(
        self,
        frozen: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_raw, log_scale_raw = self.affine(torch.cat([frozen, t], dim=-1)).chunk(2, dim=-1)
        transformed = 1.0 - mask
        shift = shift_raw * self.shift_time_gate(t) * transformed
        log_scale = 0.5 * torch.tanh(log_scale_raw) * self.scale_time_gate(t) * transformed
        return shift, log_scale

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mask = self._mask(x)
        frozen = x * mask
        shift, log_scale = self._affine_parameters(frozen, t, mask)
        changed = x * torch.exp(log_scale) + shift
        return frozen + changed * (1.0 - mask)

    def inverse(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        *,
        context: str = "CouplingFlow",
    ) -> torch.Tensor:
        mask = self._mask(y)
        frozen = y * mask
        shift, log_scale = self._affine_parameters(frozen, t, mask)
        _validate_inverse_result(shift, context=f"{context} shift")
        _validate_inverse_result(log_scale, context=f"{context} log-scale")
        changed = (y - shift) * torch.exp(-log_scale)
        result = frozen + changed * (1.0 - mask)
        _validate_inverse_result(result, context=context)
        return result


class CouplingFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: list[int] | None = None,
        **_: object,
    ):
        super().__init__()
        self.dim = int(dim)
        self.backend = "native"
        self.num_layers = max(1, int(n_layers))
        dims = _make_hidden_dims(hidden_dims, self.dim)
        self.layers = nn.ModuleList(
            _CouplingFlowLayer(self.dim, dims, parity=index)
            for index in range(self.num_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, t = _prepare_inputs(x, t, dim=self.dim)
        if t0 is not None:
            x = self.inverse(x, _prepare_reference_time(t0, x))
        for layer in self.layers:
            x = layer(x, t)
        return x

    def inverse(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        y, t = _prepare_inputs(y, t, dim=self.dim)
        _validate_inverse_inputs(y, t, context="CouplingFlow")
        for reverse_index, layer in enumerate(reversed(self.layers)):
            layer_index = self.num_layers - 1 - reverse_index
            y = layer.inverse(y, t, context=f"CouplingFlow layer {layer_index}")
        return y


__all__ = ["ResNetFlow", "GRUFlow", "CouplingFlow"]
