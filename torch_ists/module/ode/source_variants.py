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


def _integrate_batch_durations(
    func: nn.Module,
    y0: torch.Tensor,
    durations: torch.Tensor,
    *,
    method: str,
    options: dict,
    rtol: float | None,
    atol: float | None,
    adjoint: bool | None,
) -> torch.Tensor:
    """Integrate autonomous states for per-row signed durations."""
    durations = torch.nan_to_num(
        durations.reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).to(dtype=y0.dtype, device=y0.device)
    if durations.numel() != y0.size(0):
        raise ValueError("Expected one integration duration per batch row.")
    if method in {"euler", "midpoint", "rk4"}:
        resolved_options = dict(options or {})
        configured_step = float(resolved_options.get("step_size", 0.0) or 0.0)
        max_num_steps = int(resolved_options.get("max_num_steps", 10_000))
        if max_num_steps <= 0:
            raise ValueError("fixed-step max_num_steps must be positive")
        max_duration = float(durations.detach().abs().max().item()) if durations.numel() else 0.0
        if configured_step > 0:
            num_steps = max(1, int(math.ceil(max_duration / configured_step)))
            max_step = configured_step
        else:
            num_steps = 1
            max_step = max_duration
        if num_steps > max_num_steps:
            raise ValueError(
                "fixed-step integration requires "
                f"{num_steps} steps but the max_num_steps budget is {max_num_steps}"
            )
        current = y0
        remaining = durations
        for _ in range(num_steps):
            step = remaining.sign() * remaining.abs().clamp(max=max_step)
            current = _manual_step(func, current, step, method)
            remaining = remaining - step
        return current

    result = y0
    nonzero = torch.nonzero(durations != 0, as_tuple=False).reshape(-1)
    if nonzero.numel() == 0:
        return result
    active_durations = durations.index_select(0, nonzero)
    unique_durations, inverse = torch.unique(active_durations, sorted=True, return_inverse=True)
    odeint_fn = torchdiffeq.odeint_adjoint if bool(adjoint) else torchdiffeq.odeint
    for group_index, duration in enumerate(unique_durations):
        members = nonzero.index_select(
            0,
            torch.nonzero(inverse == group_index, as_tuple=False).reshape(-1),
        )
        time_span = torch.stack([duration.new_zeros(()), duration])
        call_kwargs = {"method": method}
        adaptive_options = {
            key: value for key, value in dict(options or {}).items() if key != "step_size"
        }
        adaptive_options.setdefault("max_num_steps", 10_000)
        if adaptive_options:
            call_kwargs["options"] = adaptive_options
        if rtol is not None:
            call_kwargs["rtol"] = rtol
        if atol is not None:
            call_kwargs["atol"] = atol
        solved = odeint_fn(
            func,
            y0.index_select(0, members),
            time_span,
            **call_kwargs,
        )[-1]
        result = result.index_copy(0, members, solved)
    return result


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

    if not bool(torch.isfinite(seq_delta).all()):
        raise ValueError("seq_delta must be finite before source-clock reconstruction")
    if bool((seq_delta < 0).any()):
        raise ValueError("seq_delta must be non-negative before source-clock reconstruction")
    delta = seq_delta
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
        elapsed_increment = delta[:, 1:] - delta[:, :-1]
        adjacent_delta = torch.where(
            observed[:, :-1],
            delta[:, 1:],
            elapsed_increment,
        )
        zero_total = delta.sum(dim=1) <= 0
        invalid_increment = (adjacent_delta <= 0) & ~zero_total.unsqueeze(1)
        if bool(invalid_increment.any()):
            raise ValueError(
                "seq_delta must reconstruct strictly increasing source clocks; "
                "elapsed deltas may reset only after an observed step"
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
                dt = (sample_times[:, step] - sample_times[:, step - 1]).clamp_min(0.0)
                h = _integrate_batch_durations(
                    self.ode_func,
                    h,
                    dt,
                    method=method,
                    options=options,
                    rtol=rtol,
                    atol=atol,
                    adjoint=adjoint,
                )
            step_mask = mask[:, step, :]
            step_values = torch.where(
                step_mask > 0.5,
                torch.nan_to_num(x[:, step, :], nan=0.0, posinf=0.0, neginf=0.0),
                torch.zeros_like(x[:, step, :]),
            )
            obs_input = torch.cat([step_values, step_mask], dim=-1)
            obs_present = _masked_observation_flag(step_mask).unsqueeze(-1)
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
    """Variational Latent ODE with reverse recognition and generative decoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        *,
        observation_std: float = 0.01,
        kl_weight: float = 1.0,
        auxiliary_weight: float = 1e-4,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.observation_std = float(observation_std)
        self.kl_weight = float(kl_weight)
        self.auxiliary_weight = float(auxiliary_weight)
        self.recognition_ode_func = _ODEFunc(hidden_dim, num_layers=max(1, num_layers))
        self.recognition_update = nn.GRUCell(2 * input_dim, hidden_dim)
        self.posterior = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.latent_func = _ODEFunc(hidden_dim, num_layers=max(1, num_layers))
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def _recognize(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        sample_times: torch.Tensor,
        final_index: torch.Tensor,
        *,
        method: str,
        options: dict,
        rtol: float | None,
        atol: float | None,
        adjoint: bool | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = values.shape
        hidden = values.new_zeros(batch_size, self.hidden_dim)
        for step_index in range(seq_len - 1, -1, -1):
            if step_index < seq_len - 1:
                signed_duration = (
                    sample_times[:, step_index] - sample_times[:, step_index + 1]
                )
                signed_duration = torch.where(
                    step_index < final_index,
                    signed_duration,
                    torch.zeros_like(signed_duration),
                )
                hidden = _integrate_batch_durations(
                    self.recognition_ode_func,
                    hidden,
                    signed_duration,
                    method=method,
                    options=options,
                    rtol=rtol,
                    atol=atol,
                    adjoint=adjoint,
                )
            observation = torch.cat(
                [values[:, step_index] * mask[:, step_index], mask[:, step_index]],
                dim=-1,
            )
            updated = self.recognition_update(observation, hidden)
            observed = (
                (step_index <= final_index)
                & (mask[:, step_index] > 0).any(dim=-1)
            ).unsqueeze(-1)
            hidden = torch.where(observed, updated, hidden)

        mean, raw_scale = torch.chunk(self.posterior(hidden), 2, dim=-1)
        scale = torch.nn.functional.softplus(raw_scale) + 1e-4
        has_observation = (mask > 0).any(dim=2).any(dim=1).unsqueeze(-1)
        mean = torch.where(has_observation, mean, torch.zeros_like(mean))
        scale = torch.where(has_observation, scale, torch.ones_like(scale))
        return mean, scale

    def _decode(
        self,
        z0: torch.Tensor,
        sample_times: torch.Tensor,
        final_index: torch.Tensor,
        *,
        method: str,
        options: dict,
        rtol: float | None,
        atol: float | None,
        adjoint: bool | None,
    ) -> torch.Tensor:
        common_grid = torch.allclose(
            sample_times,
            sample_times[:1].expand_as(sample_times),
        )
        common_times = sample_times[0] - sample_times[0, 0]
        strictly_increasing = bool((common_times[1:] > common_times[:-1]).all())
        full_length = bool((final_index == sample_times.size(1) - 1).all())
        if (
            method not in {"euler", "midpoint", "rk4"}
            and common_grid
            and strictly_increasing
            and full_length
        ):
            odeint_fn = torchdiffeq.odeint_adjoint if bool(adjoint) else torchdiffeq.odeint
            call_kwargs = {"method": method}
            adaptive_options = {
                key: value for key, value in dict(options or {}).items() if key != "step_size"
            }
            adaptive_options.setdefault("max_num_steps", 10_000)
            if adaptive_options:
                call_kwargs["options"] = adaptive_options
            if rtol is not None:
                call_kwargs["rtol"] = rtol
            if atol is not None:
                call_kwargs["atol"] = atol
            return odeint_fn(
                self.latent_func,
                z0,
                common_times,
                **call_kwargs,
            ).permute(1, 0, 2)

        states = [z0]
        current = z0
        for step_index in range(1, sample_times.size(1)):
            duration = sample_times[:, step_index] - sample_times[:, step_index - 1]
            duration = torch.where(
                step_index <= final_index,
                duration,
                torch.zeros_like(duration),
            )
            current = _integrate_batch_durations(
                self.latent_func,
                current,
                duration,
                method=method,
                options=options,
                rtol=rtol,
                atol=atol,
                adjoint=adjoint,
            )
            states.append(current)
        return torch.stack(states, dim=1)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        final_index = final_index.to(device=x.device, dtype=torch.long).reshape(-1)
        if final_index.numel() != batch_size:
            raise ValueError("LatentODE final_index must contain one index per sample.")
        final_index = final_index.clamp(min=0, max=max(0, seq_len - 1))
        valid_grid = (
            torch.arange(seq_len, device=x.device).unsqueeze(0)
            <= final_index.unsqueeze(1)
        )
        values = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        observed_mask = (
            (mask > 0.5) & valid_grid.unsqueeze(-1)
        ).to(dtype=values.dtype)
        final_times = sample_times.gather(1, final_index.unsqueeze(1))
        safe_sample_times = torch.where(valid_grid, sample_times, final_times)
        solver_options = dict(options or {})
        posterior_mean, posterior_scale = self._recognize(
            values,
            observed_mask,
            safe_sample_times,
            final_index,
            method=method,
            options=solver_options,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
        )
        if self.training:
            z0 = posterior_mean + posterior_scale * torch.randn_like(posterior_scale)
        else:
            z0 = posterior_mean
        latent_stream = self._decode(
            z0,
            safe_sample_times,
            final_index,
            method=method,
            options=solver_options,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
        )

        reconstruction = self.reconstruction_head(latent_stream)
        observation_std = values.new_tensor(max(self.observation_std, 1e-6))
        reconstruction_nll = 0.5 * (
            ((values - reconstruction) / observation_std).square()
            + 2.0 * torch.log(observation_std)
            + math.log(2.0 * math.pi)
        )
        reconstruction_nll = (
            reconstruction_nll * observed_mask
        ).sum() / observed_mask.sum().clamp_min(1.0)
        posterior_log_scale = torch.log(posterior_scale)
        initial_kl = 0.5 * (
            posterior_mean.square()
            + posterior_scale.square()
            - 2.0 * posterior_log_scale
            - 1.0
        ).sum(dim=-1).mean()
        native_loss = (
            reconstruction_nll + self.kl_weight * initial_kl
        ) * self.auxiliary_weight

        representation_stream = self.readout(latent_stream)
        return representation_stream, self.readout(z0), native_loss


class _GRUObservationCell(nn.Module):
    """Bayesian observation update adapted from the reference implementation."""

    def __init__(self, input_dim: int, hidden_dim: int, prep_hidden: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.prep_hidden = max(1, int(prep_hidden))
        self.gru = nn.GRUCell(self.input_dim * self.prep_hidden, hidden_dim)
        scale = math.sqrt(2.0 / (4.0 + self.prep_hidden))
        self.w_prep = nn.Parameter(
            scale * torch.randn(self.input_dim, 4, self.prep_hidden)
        )
        self.bias_prep = nn.Parameter(
            torch.full((self.input_dim, self.prep_hidden), 0.1)
        )

    @staticmethod
    def _distribution(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = torch.chunk(p, 2, dim=-1)
        return mean, logvar.clamp(min=-12.0, max=12.0)

    def forward(
        self,
        h: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self._distribution(p)
        values = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mask = (mask > 0.5).to(dtype=values.dtype)
        error = (values - mean) * torch.exp(-0.5 * logvar)
        nll = 0.5 * (
            error.square() + logvar + math.log(2.0 * math.pi)
        ) * mask

        features = torch.stack([values, mean, logvar, error], dim=-1)
        prepared = torch.einsum("bdk,dkp->bdp", features, self.w_prep)
        prepared = torch.relu(prepared + self.bias_prep)
        prepared = (prepared * mask.unsqueeze(-1)).reshape(values.size(0), -1)
        updated = self.gru(prepared, h)
        observed = (mask > 0).any(dim=-1, keepdim=True)
        return torch.where(observed, updated, h), nll.sum()


def _gaussian_observation_kl(
    p: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    observation_noise_std: float,
) -> torch.Tensor:
    mean, logvar = torch.chunk(p, 2, dim=-1)
    logvar = logvar.clamp(min=-12.0, max=12.0)
    values = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mask = (mask > 0.5).to(dtype=values.dtype)
    noise_std = values.new_tensor(max(float(observation_noise_std), 1e-6))
    noise_var = noise_std.square()
    variance = torch.exp(logvar)
    kl = (
        torch.log(noise_std)
        - 0.5 * logvar
        + (variance + (mean - values).square()) / (2.0 * noise_var)
        - 0.5
    )
    return (kl * mask).sum()


class SourceGRUODEBayesBackbone(nn.Module):
    """GRU-ODE-Bayes with continuous prediction and Bayesian observation jumps."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        p_hidden: int | None = None,
        prep_hidden: int = 4,
        step_size: float = 0.05,
        observation_noise_std: float = 0.01,
        mixing: float = 1.0,
        auxiliary_weight: float = 1e-3,
        max_substeps: int = 10_000,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.step_size = float(step_size)
        self.observation_noise_std = float(observation_noise_std)
        self.mixing = float(mixing)
        self.auxiliary_weight = float(auxiliary_weight)
        self.max_substeps = int(max_substeps)
        if self.max_substeps <= 0:
            raise ValueError("GRU-ODE-Bayes max_substeps must be positive.")
        prediction_hidden = max(1, int(p_hidden or hidden_dim))

        self.initial_hidden = nn.Parameter(torch.zeros(1, hidden_dim))
        self.p_model = nn.Sequential(
            nn.Linear(hidden_dim, prediction_hidden),
            nn.ReLU(),
            nn.Linear(prediction_hidden, 2 * input_dim),
        )
        self.ode_func = _DrivenGRUODEFunc(2 * input_dim, hidden_dim)
        self.gru_obs = _GRUObservationCell(input_dim, hidden_dim, prep_hidden)
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
        del rtol, atol, adjoint
        if method != "euler":
            raise ValueError("GRU-ODE-Bayes currently supports only Euler integration.")
        durations = torch.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        resolved_options = dict(options or {})
        step_size = float(resolved_options.get("step_size", self.step_size))
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("GRU-ODE-Bayes step_size must be finite and positive.")
        max_duration = float(durations.detach().max().item()) if durations.numel() else 0.0
        num_steps = max(1, int(math.ceil(max_duration / step_size))) if max_duration > 0 else 0
        max_num_steps = int(resolved_options.get("max_num_steps", self.max_substeps))
        if max_num_steps <= 0:
            raise ValueError("GRU-ODE-Bayes max_num_steps must be positive.")
        if num_steps > max_num_steps:
            raise ValueError(
                "GRU-ODE-Bayes integration requires "
                f"{num_steps} steps but the max_num_steps budget is {max_num_steps}."
            )
        remaining = durations
        p = self.p_model(h)
        for _ in range(num_steps):
            step = remaining.clamp(min=0.0, max=step_size)
            candidate = h + step.unsqueeze(-1) * self.ode_func(p, h)
            h = torch.where((step > 0).unsqueeze(-1), candidate, h)
            p = self.p_model(h)
            remaining = (remaining - step).clamp_min(0.0)
        return h

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if method != "euler":
            raise ValueError("GRU-ODE-Bayes currently supports only Euler integration.")
        batch_size, seq_len, _ = x.shape
        values = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        observed_mask = (mask > 0.5).to(dtype=values.dtype)
        h = self.initial_hidden.to(dtype=values.dtype).expand(batch_size, -1)
        hidden_stream = []
        pre_jump_loss = values.new_zeros(())
        post_jump_loss = values.new_zeros(())

        for step_index in range(seq_len):
            if step_index > 0:
                duration = (
                    sample_times[:, step_index] - sample_times[:, step_index - 1]
                ).clamp_min(0.0)
                h = self._evolve(
                    h,
                    duration,
                    method,
                    options=options,
                    rtol=rtol,
                    atol=atol,
                    adjoint=adjoint,
                )

            prediction = self.p_model(h)
            h, observation_nll = self.gru_obs(
                h,
                prediction,
                values[:, step_index],
                observed_mask[:, step_index],
            )
            pre_jump_loss = pre_jump_loss + observation_nll
            posterior_prediction = self.p_model(h)
            post_jump_loss = post_jump_loss + _gaussian_observation_kl(
                posterior_prediction,
                values[:, step_index],
                observed_mask[:, step_index],
                observation_noise_std=self.observation_noise_std,
            )
            hidden_stream.append(h)

        hidden_stream_tensor = torch.stack(hidden_stream, dim=1)
        representation_stream = self.readout(hidden_stream_tensor)
        observed_count = observed_mask.sum().clamp_min(1.0)
        native_loss = (
            pre_jump_loss + self.mixing * post_jump_loss
        ).clamp_min(0.0) / observed_count
        native_loss = native_loss * self.auxiliary_weight
        return (
            representation_stream,
            _gather_last(representation_stream, final_index),
            native_loss,
        )


__all__ = [
    "SourceODERNNBackbone",
    "SourceLatentODEBackbone",
    "SourceGRUODEBayesBackbone",
    "_reconstruct_sample_times",
]
