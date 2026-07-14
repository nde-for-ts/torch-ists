"""Additional ODE-family variants used by the runtime package."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torchdiffeq


def _build_mlp(hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = hidden_dim
    for _ in range(max(1, int(num_layers))):
        layers.append(nn.Linear(hidden_dim, width))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(width, hidden_dim))
    return nn.Sequential(*layers)


def _masked_observation_flag(mask_t: torch.Tensor) -> torch.Tensor:
    return mask_t.max(dim=-1).values > 0.5


def _gather_last(hidden_stream: torch.Tensor, final_index: torch.Tensor) -> torch.Tensor:
    gather_index = final_index.view(-1, 1, 1).expand(-1, 1, hidden_stream.size(-1))
    return hidden_stream.gather(dim=1, index=gather_index).squeeze(1)


class _ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.net = _build_mlp(hidden_dim, num_layers)

    def forward(self, _t: torch.Tensor | None, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class _DrivenGRUODEFunc(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lin_xz = nn.Linear(input_dim, hidden_dim)
        self.lin_xn = nn.Linear(input_dim, hidden_dim)
        self.lin_hz = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_hn = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, drive: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.lin_xz(drive) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(drive) + self.lin_hn(z * h))
        return (1.0 - z) * (n - h)


class _FrozenDrivenFunc(nn.Module):
    def __init__(self, func: _DrivenGRUODEFunc, drive: torch.Tensor):
        super().__init__()
        self.func = func
        self.register_buffer("drive", drive)

    def forward(self, _t: torch.Tensor | None, h: torch.Tensor) -> torch.Tensor:
        return self.func(self.drive, h)


def _manual_step(func: nn.Module, h: torch.Tensor, dt: torch.Tensor, method: str) -> torch.Tensor:
    dt = dt.unsqueeze(-1)
    if method == "euler":
        return h + dt * func(None, h)
    if method == "midpoint":
        k1 = func(None, h)
        mid = h + 0.5 * dt * k1
        return h + dt * func(None, mid)
    if method == "rk4":
        k1 = func(None, h)
        k2 = func(None, h + 0.5 * dt * k1)
        k3 = func(None, h + 0.5 * dt * k2)
        k4 = func(None, h + dt * k3)
        return h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    raise ValueError(f"Unsupported fixed-step method: {method}")


def _integrate_scalar_interval(
    func: nn.Module,
    y0: torch.Tensor,
    dt: float,
    *,
    method: str,
    options: dict,
    rtol: float | None,
    atol: float | None,
    adjoint: bool | None,
) -> torch.Tensor:
    if dt <= 0.0:
        return y0
    if method in {"euler", "midpoint", "rk4"}:
        step_size = float(dict(options or {}).get("step_size", 0.0) or 0.0)
        steps = max(1, int(math.ceil(dt / step_size))) if step_size > 0.0 else 1
        current = y0.unsqueeze(0)
        dt_tensor = torch.tensor([dt / steps], dtype=y0.dtype, device=y0.device)
        for _ in range(steps):
            current = _manual_step(func, current, dt_tensor, method)
        return current.squeeze(0)

    t = torch.tensor([0.0, dt], dtype=y0.dtype, device=y0.device)
    odeint_fn = torchdiffeq.odeint_adjoint if bool(adjoint) else torchdiffeq.odeint
    call_kwargs = {"method": method}
    if options:
        call_kwargs["options"] = dict(options)
    if rtol is not None:
        call_kwargs["rtol"] = rtol
    if atol is not None:
        call_kwargs["atol"] = atol
    return odeint_fn(func, y0, t, **call_kwargs)[-1]


def _reconstruct_sample_times(
    seq_delta: torch.Tensor,
    fallback_times: torch.Tensor,
    observation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if seq_delta.dim() != 2:
        raise ValueError(f"seq_delta must be [N, L], got {tuple(seq_delta.shape)}")
    if fallback_times.dim() != 1 or fallback_times.numel() != seq_delta.size(1):
        raise ValueError(
            f"fallback_times must be [L]={seq_delta.size(1)}, got {tuple(fallback_times.shape)}"
        )
    if seq_delta.size(1) == 0:
        return seq_delta.clone()

    delta = torch.nan_to_num(
        seq_delta,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    fallback = fallback_times.to(dtype=delta.dtype, device=delta.device)
    if not bool(torch.isfinite(fallback).all()):
        raise ValueError("fallback_times must be finite")
    if observation_mask is None:
        observed = torch.ones_like(delta, dtype=torch.bool)
    else:
        if tuple(observation_mask.shape) != tuple(delta.shape):
            raise ValueError(
                f"observation_mask must match seq_delta, got {tuple(observation_mask.shape)}"
            )
        observed = observation_mask.to(device=delta.device) > 0.5

    start = fallback[0].expand(delta.size(0), 1)
    if delta.size(1) == 1:
        reconstructed = start
    else:
        elapsed_increment = (delta[:, 1:] - delta[:, :-1]).clamp_min(0.0)
        adjacent_delta = torch.where(
            observed[:, :-1],
            delta[:, 1:],
            elapsed_increment,
        )
        reconstructed = torch.cat(
            [start, start + torch.cumsum(adjacent_delta, dim=1)],
            dim=1,
        )
    fallback_grid = fallback.unsqueeze(0).expand(delta.size(0), -1)
    zero_total = delta.sum(dim=1) <= 0
    return torch.where(zero_total.unsqueeze(1), fallback_grid, reconstructed)


class _RawODERNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.ode_func = _ODEFunc(hidden_dim, num_layers=num_layers)
        self.update = nn.GRUCell(input_dim * 2, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        sample_times: torch.Tensor,
        *,
        method: str,
        options: dict,
        rtol: float | None,
        atol: float | None,
        adjoint: bool | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.update.hidden_size)
        hs = []
        for step in range(seq_len):
            if step > 0:
                next_h = []
                dt = (sample_times[:, step] - sample_times[:, step - 1]).clamp_min(0.0)
                for batch_idx in range(batch_size):
                    next_h.append(
                        _integrate_scalar_interval(
                            self.ode_func,
                            h[batch_idx],
                            float(dt[batch_idx].item()),
                            method=method,
                            options=options,
                            rtol=rtol,
                            atol=atol,
                            adjoint=adjoint,
                        )
                    )
                h = torch.stack(next_h, dim=0)
            obs_input = torch.cat([x[:, step, :] * mask[:, step, :], mask[:, step, :]], dim=-1)
            obs_present = _masked_observation_flag(mask[:, step, :]).unsqueeze(-1)
            updated = self.update(obs_input, h)
            h = torch.where(obs_present, updated, h)
            hs.append(h)
        return torch.stack(hs, dim=1)


class SourceODERNNBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.encoder = _RawODERNNEncoder(input_dim, hidden_dim, num_layers)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        sample_times: torch.Tensor,
        final_index: torch.Tensor,
        *,
        method: str,
        options: dict | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        adjoint: bool | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_stream = self.encoder(
            x,
            mask,
            sample_times,
            method=method,
            options=dict(options or {}),
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
        )
        return self.readout(hidden_stream), _gather_last(hidden_stream, final_index)


class SourceLatentODEBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.encoder = _RawODERNNEncoder(input_dim, hidden_dim, num_layers)
        self.latent_init = nn.Linear(hidden_dim, hidden_dim)
        self.latent_func = _ODEFunc(hidden_dim, num_layers=max(1, num_layers))
        self.readout = nn.Linear(hidden_dim, output_dim)

    def _decode_single(
        self,
        z0: torch.Tensor,
        sample_times: torch.Tensor,
        *,
        method: str,
        options: dict,
        rtol: float | None,
        atol: float | None,
        adjoint: bool | None,
    ) -> torch.Tensor:
        states = [z0]
        current = z0
        for step in range(1, sample_times.numel()):
            dt = float((sample_times[step] - sample_times[step - 1]).clamp_min(0.0).item())
            current = _integrate_scalar_interval(
                self.latent_func,
                current,
                dt,
                method=method,
                options=options,
                rtol=rtol,
                atol=atol,
                adjoint=adjoint,
            )
            states.append(current)
        return torch.stack(states, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        sample_times: torch.Tensor,
        final_index: torch.Tensor,
        *,
        method: str,
        options: dict | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        adjoint: bool | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(
            x,
            mask,
            sample_times,
            method="euler",
            options={},
            rtol=rtol,
            atol=atol,
            adjoint=False,
        )
        z0 = self.latent_init(_gather_last(encoded, final_index))
        trajectories = [
            self._decode_single(
                z0[idx],
                sample_times[idx],
                method=method,
                options=dict(options or {}),
                rtol=rtol,
                atol=atol,
                adjoint=adjoint,
            )
            for idx in range(x.size(0))
        ]
        latent_stream = torch.stack(trajectories, dim=0)
        return self.readout(latent_stream), _gather_last(latent_stream, final_index)


class SourceGRUODEBayesBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update = nn.GRUCell(input_dim * 2, hidden_dim)
        self.drive = nn.Linear(hidden_dim, input_dim * 2)
        self.ode_func = _DrivenGRUODEFunc(input_dim * 2, hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def _evolve(
        self,
        h: torch.Tensor,
        dt: torch.Tensor,
        method: str,
        *,
        options: dict | None,
        rtol: float | None,
        atol: float | None,
        adjoint: bool | None,
    ) -> torch.Tensor:
        drive = self.drive(h)
        if method == "euler":
            return h + dt.unsqueeze(-1) * self.ode_func(drive, h)
        if method == "midpoint":
            k1 = self.ode_func(drive, h)
            mid = h + 0.5 * dt.unsqueeze(-1) * k1
            return h + dt.unsqueeze(-1) * self.ode_func(self.drive(mid), mid)
        if method == "rk4":
            k1 = self.ode_func(drive, h)
            k2 = self.ode_func(drive, h + 0.5 * dt.unsqueeze(-1) * k1)
            k3 = self.ode_func(drive, h + 0.5 * dt.unsqueeze(-1) * k2)
            k4 = self.ode_func(drive, h + dt.unsqueeze(-1) * k3)
            return h + (dt.unsqueeze(-1) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if method == "dopri5":
            next_h = []
            for batch_idx in range(h.size(0)):
                next_h.append(
                    _integrate_scalar_interval(
                        _FrozenDrivenFunc(self.ode_func, drive[batch_idx]),
                        h[batch_idx],
                        float(dt[batch_idx].item()),
                        method="dopri5",
                        options=dict(options or {}),
                        rtol=rtol,
                        atol=atol,
                        adjoint=adjoint,
                    )
                )
            return torch.stack(next_h, dim=0)
        raise ValueError(f"Unsupported GRU-ODE-Bayes method: {method}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        sample_times: torch.Tensor,
        final_index: torch.Tensor,
        *,
        method: str,
        options: dict | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        adjoint: bool | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_dim)
        hs = []
        for step in range(seq_len):
            if step > 0:
                dt = (sample_times[:, step] - sample_times[:, step - 1]).clamp_min(0.0)
                h = self._evolve(
                    h,
                    dt,
                    method,
                    options=options,
                    rtol=rtol,
                    atol=atol,
                    adjoint=adjoint,
                )
            obs_input = torch.cat([x[:, step, :] * mask[:, step, :], mask[:, step, :]], dim=-1)
            obs_present = _masked_observation_flag(mask[:, step, :]).unsqueeze(-1)
            updated = self.update(obs_input, h)
            h = torch.where(obs_present, updated, h)
            hs.append(h)
        hidden_stream = torch.stack(hs, dim=1)
        return self.readout(hidden_stream), _gather_last(hidden_stream, final_index)


__all__ = [
    "SourceODERNNBackbone",
    "SourceLatentODEBackbone",
    "SourceGRUODEBayesBackbone",
    "_reconstruct_sample_times",
]
