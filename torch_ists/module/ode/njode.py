"""Neural Jump ODE (NJ-ODE) backbone for the package sequence contract."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _build_network(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for _ in range(max(1, int(num_layers))):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.Tanh())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def _gather_last(stream: torch.Tensor, final_index: torch.Tensor) -> torch.Tensor:
    index = final_index.to(device=stream.device, dtype=torch.long).reshape(-1)
    index = index.clamp(min=0, max=max(0, stream.size(1) - 1))
    gather_index = index.view(-1, 1, 1).expand(-1, 1, stream.size(-1))
    return stream.gather(1, gather_index).squeeze(1)


def njode_observation_loss(
    x: torch.Tensor,
    post_jump: torch.Tensor,
    pre_jump: torch.Tensor,
    mask: torch.Tensor,
    event_counts: torch.Tensor,
    *,
    weight: float = 0.5,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Return the standard NJ-ODE observation-date loss for one grid position."""
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("NJ-ODE loss weight must be between zero and one.")
    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("NJ-ODE loss eps must be finite and positive.")
    if tuple(x.shape) != tuple(post_jump.shape) or tuple(x.shape) != tuple(pre_jump.shape):
        raise ValueError("NJ-ODE loss inputs must have matching [N, D] shapes.")
    if tuple(mask.shape) != tuple(x.shape):
        raise ValueError("NJ-ODE loss mask must match the observation shape.")

    observed_mask = (mask > 0.5).to(dtype=x.dtype, device=x.device)
    values = torch.where(
        observed_mask > 0,
        torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(x),
    )
    prediction_error = torch.sqrt(
        (observed_mask * (values - post_jump).square()).sum(dim=-1) + float(eps)
    )
    jump_size = torch.sqrt(
        (observed_mask * (pre_jump - post_jump).square()).sum(dim=-1) + float(eps)
    )
    event_loss = (
        2.0 * float(weight) * prediction_error
        + 2.0 * (1.0 - float(weight)) * jump_size
    ).square()
    has_event = observed_mask.any(dim=-1).to(dtype=x.dtype)
    counts = event_counts.to(device=x.device, dtype=x.dtype).reshape(-1).clamp_min(1.0)
    if counts.numel() != x.size(0):
        raise ValueError("NJ-ODE event_counts must contain one value per sample.")
    return ((event_loss * has_event) / counts).sum() / max(1, x.size(0))


class _NJODEVectorField(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.net = _build_network(
            input_dim + hidden_dim + 2,
            hidden_dim,
            hidden_dim,
            num_layers,
        )

    def forward(
        self,
        last_x: torch.Tensor,
        hidden: torch.Tensor,
        last_observation_time: torch.Tensor,
        current_time: torch.Tensor,
    ) -> torch.Tensor:
        elapsed = (current_time - last_observation_time).clamp_min(0.0)
        features = torch.cat(
            [
                torch.tanh(last_x),
                torch.tanh(hidden),
                last_observation_time.unsqueeze(-1),
                elapsed.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(features)


class SourceNJODEBackbone(nn.Module):
    """Neural jump ODE with self-imputation and the standard native loss."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        *,
        step_size: float = 0.05,
        loss_weight: float = 0.5,
        auxiliary_weight: float = 1.0,
        loss_eps: float = 1e-10,
        max_substeps: int = 10_000,
    ):
        super().__init__()
        if int(input_dim) <= 0 or int(hidden_dim) <= 0 or int(output_dim) <= 0:
            raise ValueError("NJ-ODE dimensions must be positive.")
        if not math.isfinite(float(step_size)) or float(step_size) <= 0:
            raise ValueError("NJ-ODE step_size must be finite and positive.")
        if not 0.0 <= float(loss_weight) <= 1.0:
            raise ValueError("NJ-ODE loss_weight must be between zero and one.")
        if not math.isfinite(float(auxiliary_weight)) or float(auxiliary_weight) < 0:
            raise ValueError("NJ-ODE auxiliary_weight must be finite and non-negative.")
        if not math.isfinite(float(loss_eps)) or float(loss_eps) <= 0:
            raise ValueError("NJ-ODE loss_eps must be finite and positive.")
        if int(max_substeps) <= 0:
            raise ValueError("NJ-ODE max_substeps must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.step_size = float(step_size)
        self.loss_weight = float(loss_weight)
        self.auxiliary_weight = float(auxiliary_weight)
        self.loss_eps = float(loss_eps)
        self.max_substeps = int(max_substeps)
        self.initial_hidden = nn.Parameter(torch.zeros(1, hidden_dim))
        self.ode_func = _NJODEVectorField(input_dim, hidden_dim, num_layers)
        self.jump = _build_network(2 * input_dim, hidden_dim, hidden_dim, num_layers)
        self.native_readout = nn.Linear(hidden_dim, input_dim)
        self.representation = nn.Linear(hidden_dim, output_dim)
        self._last_function_evaluations = 0

    @staticmethod
    def _expand_times(
        sample_times: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        sample_times = sample_times.to(device=device, dtype=dtype)
        if sample_times.dim() == 1:
            if sample_times.numel() != seq_len:
                raise ValueError(f"NJ-ODE sample_times must have length {seq_len}.")
            sample_times = sample_times.unsqueeze(0).expand(batch_size, -1)
        if sample_times.dim() != 2 or tuple(sample_times.shape) != (batch_size, seq_len):
            raise ValueError(
                "NJ-ODE sample_times must be [L] or [N, L], "
                f"got {tuple(sample_times.shape)}."
            )
        if not bool(torch.isfinite(sample_times).all()):
            raise ValueError("NJ-ODE sample_times must be finite.")
        if seq_len > 1 and bool(((sample_times[:, 1:] - sample_times[:, :-1]) < 0).any()):
            raise ValueError("NJ-ODE sample_times must be non-decreasing.")
        return sample_times

    def _evolve(
        self,
        hidden: torch.Tensor,
        last_x: torch.Tensor,
        last_observation_time: torch.Tensor,
        start_time: torch.Tensor,
        duration: torch.Tensor,
        *,
        step_size: float,
        max_substeps: int,
    ) -> tuple[torch.Tensor, int]:
        duration = torch.nan_to_num(duration, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        max_duration = float(duration.detach().max().item()) if duration.numel() else 0.0
        step_ratio = max_duration / step_size
        required_steps = int(math.ceil(step_ratio - 1e-6)) if max_duration > 0 else 0
        if required_steps > max_substeps:
            raise RuntimeError(
                "NJ-ODE integration exceeded max_substeps; increase step_size or max_substeps."
            )

        remaining = duration
        current_time = start_time
        for _ in range(required_steps):
            step = remaining.clamp(min=0.0, max=step_size)
            drift = self.ode_func(last_x, hidden, last_observation_time, current_time)
            candidate = hidden + step.unsqueeze(-1) * drift
            hidden = torch.where((step > 0).unsqueeze(-1), candidate, hidden)
            remaining = (remaining - step).clamp_min(0.0)
            current_time = current_time + step
        return hidden, required_steps

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
        del rtol, atol
        if str(method).strip().lower() != "euler":
            raise ValueError("NJ-ODE currently supports only Euler integration.")
        if bool(adjoint):
            raise ValueError("NJ-ODE uses explicit Euler substeps and does not use adjoint integration.")
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"NJ-ODE values must be [N, L, {self.input_dim}], got {tuple(x.shape)}."
            )
        if tuple(mask.shape) != tuple(x.shape):
            raise ValueError("NJ-ODE mask must match values shape.")

        solver_options = dict(options or {})
        step_size = float(solver_options.pop("step_size", self.step_size))
        max_substeps = int(
            solver_options.pop("max_substeps", solver_options.pop("max_num_steps", self.max_substeps))
        )
        if solver_options:
            unexpected = ", ".join(sorted(solver_options))
            raise ValueError(f"Unsupported NJ-ODE solver options: {unexpected}")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("NJ-ODE step_size must be finite and positive.")
        if max_substeps <= 0:
            raise ValueError("NJ-ODE max_substeps must be positive.")

        batch_size, seq_len, _ = x.shape
        if seq_len <= 0:
            raise ValueError("NJ-ODE requires a non-empty sequence.")
        times = self._expand_times(
            sample_times,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=x.dtype,
            device=x.device,
        )
        observed_mask = (mask.to(device=x.device) > 0.5).to(dtype=x.dtype)
        values = torch.where(
            observed_mask > 0,
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(x),
        )
        final_index = final_index.to(device=x.device, dtype=torch.long).reshape(-1)
        if final_index.numel() != batch_size:
            raise ValueError("NJ-ODE final_index must contain one index per sample.")
        final_index = final_index.clamp(min=0, max=seq_len - 1)

        event_grid = observed_mask.any(dim=-1)
        has_observation = event_grid.any(dim=-1)
        grid_index = torch.arange(seq_len, device=x.device).unsqueeze(0)
        active_grid = (grid_index <= final_index.unsqueeze(1)) & has_observation.unsqueeze(1)
        valid_events = event_grid & active_grid
        event_counts = valid_events[:, 1:].sum(dim=-1).to(dtype=x.dtype)

        hidden = self.initial_hidden.to(dtype=x.dtype).expand(batch_size, -1)
        pre_jump = self.native_readout(hidden)
        initial_imputation = observed_mask[:, 0] * values[:, 0] + (1.0 - observed_mask[:, 0]) * pre_jump
        jump_candidate = self.jump(
            torch.cat([torch.tanh(initial_imputation), observed_mask[:, 0]], dim=-1)
        )
        initial_event = valid_events[:, 0].unsqueeze(-1)
        hidden = torch.where(initial_event, jump_candidate, hidden)
        post_jump = self.native_readout(hidden)
        last_x = post_jump
        last_observation_time = times[:, 0]
        hidden_stream = [hidden]
        native_loss = values.new_zeros(())
        function_evaluations = 0

        for step_index in range(1, seq_len):
            active = active_grid[:, step_index]
            duration = torch.where(
                active,
                times[:, step_index] - times[:, step_index - 1],
                torch.zeros_like(times[:, step_index]),
            )
            hidden, evaluations = self._evolve(
                hidden,
                last_x,
                last_observation_time,
                times[:, step_index - 1],
                duration,
                step_size=step_size,
                max_substeps=max_substeps,
            )
            function_evaluations += evaluations

            pre_jump = self.native_readout(hidden)
            imputed = (
                observed_mask[:, step_index] * values[:, step_index]
                + (1.0 - observed_mask[:, step_index]) * pre_jump
            )
            jump_candidate = self.jump(
                torch.cat([torch.tanh(imputed), observed_mask[:, step_index]], dim=-1)
            )
            event = valid_events[:, step_index].unsqueeze(-1)
            hidden = torch.where(event, jump_candidate, hidden)
            post_jump = self.native_readout(hidden)
            event_mask = observed_mask[:, step_index] * event.to(dtype=x.dtype)
            native_loss = native_loss + njode_observation_loss(
                values[:, step_index],
                post_jump,
                pre_jump,
                event_mask,
                event_counts,
                weight=self.loss_weight,
                eps=self.loss_eps,
            )
            last_x = torch.where(event, post_jump, last_x)
            last_observation_time = torch.where(
                event.squeeze(-1), times[:, step_index], last_observation_time
            )
            hidden_stream.append(hidden)

        self._last_function_evaluations = function_evaluations
        hidden_stream_tensor = torch.stack(hidden_stream, dim=1)
        representation_stream = self.representation(hidden_stream_tensor)
        auxiliary_loss = native_loss.clamp_min(0.0) * self.auxiliary_weight
        return (
            representation_stream,
            _gather_last(representation_stream, final_index),
            auxiliary_loss,
        )


__all__ = ["SourceNJODEBackbone", "njode_observation_loss"]
