"""Latent SDE runtime with full-path amortized initial inference.

The SDE and path-KL structure is adapted from google-research/torchsde's
Apache-2.0 ``examples/latent_sde.py`` implementation.
"""

from __future__ import annotations

import math
from typing import Any, Tuple, Union

import torch
from torch import distributions, nn
import torch.nn.functional as F
import torchsde


def _stable_division(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    replacement = torch.where(b.detach() >= 0, b.new_full((), epsilon), b.new_full((), -epsilon))
    safe_b = torch.where(b.abs().detach() > epsilon, b, replacement)
    return a / safe_b


class LatentSDE(torchsde.SDEIto):
    """Latent SDE with a mask-aware full-path posterior for the initial state."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        hidden_hidden_channels: int,
        num_hidden_layers: int,
        theta: float = 1.0,
        mu: float = 0.0,
        sigma: float = 0.5,
        kl: bool = True,
        kl_weight: float = 1e-5,
    ):
        super().__init__(noise_type="diagonal")
        if int(input_channels) <= 0:
            raise ValueError("LatentSDE input_channels must be positive.")
        if int(hidden_channels) <= 0:
            raise ValueError("LatentSDE hidden_channels must be positive.")
        if not math.isfinite(float(theta)) or float(theta) <= 0:
            raise ValueError("LatentSDE theta must be finite and positive.")
        if not math.isfinite(float(sigma)) or float(sigma) <= 0:
            raise ValueError("LatentSDE sigma must be finite and positive.")
        if not math.isfinite(float(kl_weight)) or float(kl_weight) < 0:
            raise ValueError("LatentSDE kl_weight must be finite and non-negative.")

        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.kl = bool(kl)
        self.kl_weight = float(kl_weight)
        logvar = math.log(float(sigma) ** 2 / (2.0 * float(theta)))

        prior_shape = (1, self.hidden_channels)
        self.register_buffer("theta", torch.full(prior_shape, float(theta)))
        self.register_buffer("mu", torch.full(prior_shape, float(mu)))
        self.register_buffer("sigma", torch.full(prior_shape, float(sigma)))
        self.register_buffer("py0_mean", torch.full(prior_shape, float(mu)))
        self.register_buffer("py0_logvar", torch.full(prior_shape, logvar))

        recognition_width = max(1, int(hidden_hidden_channels))
        self.recognition_cell = nn.GRUCell(3 * self.input_channels + 1, recognition_width)
        self.posterior_head = nn.Linear(recognition_width, 2 * self.hidden_channels)

        self.linear_in = nn.Linear(self.hidden_channels + 2, recognition_width)
        self.linears = nn.ModuleList(
            nn.Linear(recognition_width, recognition_width)
            for _ in range(max(0, int(num_hidden_layers) - 1))
        )
        self.linear_out = nn.Linear(recognition_width, self.hidden_channels)
        self.embedding = nn.Linear(self.hidden_channels, self.hidden_channels)

    @staticmethod
    def _expand_times(
        times: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        times = times.to(device=device, dtype=dtype)
        if times.dim() == 1:
            if times.numel() != seq_len:
                raise ValueError(f"LatentSDE times must have length {seq_len}.")
            return times.unsqueeze(0).expand(batch_size, -1)
        if times.dim() == 2 and tuple(times.shape) == (batch_size, seq_len):
            return times
        raise ValueError(
            "LatentSDE times must be [L] or [N, L], "
            f"got {tuple(times.shape)}."
        )

    def _infer_initial_posterior(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        times: torch.Tensor,
        final_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.dim() != 3 or values.size(-1) != self.input_channels:
            raise ValueError(
                "LatentSDE values must be [N, L, D] with "
                f"D={self.input_channels}, got {tuple(values.shape)}."
            )
        if tuple(mask.shape) != tuple(values.shape) or tuple(delta.shape) != tuple(values.shape):
            raise ValueError("LatentSDE mask and delta must match values shape.")

        batch_size, seq_len, _ = values.shape
        observed = mask.to(device=values.device) > 0.5
        observed_values = torch.where(
            observed,
            torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(values),
        )
        safe_delta = torch.nan_to_num(
            delta.to(device=values.device, dtype=values.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        expanded_times = self._expand_times(
            times,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=values.dtype,
            device=values.device,
        )
        if not bool(torch.isfinite(expanded_times).all()):
            raise ValueError("LatentSDE times must be finite.")

        final_index = final_index.to(device=values.device, dtype=torch.long).reshape(-1)
        if final_index.numel() != batch_size:
            raise ValueError("LatentSDE final_index must contain one index per sample.")
        final_index = final_index.clamp(min=0, max=max(0, seq_len - 1))
        final_times = expanded_times.gather(1, final_index.unsqueeze(1))
        time_span = (final_times - expanded_times[:, :1]).clamp_min(1e-6)
        normalized_times = (expanded_times - expanded_times[:, :1]) / time_span

        hidden = values.new_zeros(batch_size, self.recognition_cell.hidden_size)
        valid_grid = (
            torch.arange(seq_len, device=values.device).unsqueeze(0)
            <= final_index.unsqueeze(1)
        )
        for step_index in range(seq_len - 1, -1, -1):
            feature = torch.cat(
                [
                    observed_values[:, step_index],
                    observed[:, step_index].to(dtype=values.dtype),
                    safe_delta[:, step_index],
                    normalized_times[:, step_index : step_index + 1],
                ],
                dim=-1,
            )
            candidate = self.recognition_cell(feature, hidden)
            update = (
                valid_grid[:, step_index]
                & observed[:, step_index].any(dim=-1)
            ).unsqueeze(-1)
            hidden = torch.where(update, candidate, hidden)

        posterior_mean, posterior_raw_scale = self.posterior_head(hidden).chunk(2, dim=-1)
        posterior_scale = F.softplus(posterior_raw_scale) + 1e-4
        has_observation = (observed & valid_grid.unsqueeze(-1)).any(dim=(1, 2)).unsqueeze(-1)
        prior_mean = self.py0_mean.to(dtype=values.dtype).expand(batch_size, -1)
        prior_scale = self.py0_std.to(dtype=values.dtype).expand(batch_size, -1)
        posterior_mean = torch.where(has_observation, posterior_mean, prior_mean)
        posterior_scale = torch.where(has_observation, posterior_scale, prior_scale)
        return posterior_mean, posterior_scale

    @staticmethod
    def _time_column(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scalar_t = t.to(device=y.device, dtype=y.dtype).reshape(1, 1)
        return scalar_t.expand(y.size(0), 1)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        time_column = self._time_column(t, y)
        z = self.linear_in(torch.cat((torch.sin(time_column), torch.cos(time_column), y), dim=-1))
        z = z.relu()
        for linear in self.linears:
            z = linear(z).relu()
        return self.linear_out(z)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del t
        return self.sigma.to(dtype=y.dtype).expand_as(y)

    def h(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del t
        return self.theta.to(dtype=y.dtype) * (self.mu.to(dtype=y.dtype) - y)

    def f_aug(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        state = y[:, :-1]
        f_val, g_val, h_val = self.f(t, state), self.g(t, state), self.h(t, state)
        control = _stable_division(f_val - h_val, g_val)
        path_kl = 0.5 * control.square().sum(dim=1, keepdim=True)
        return torch.cat([f_val, path_kl], dim=1)

    def g_aug(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        state = y[:, :-1]
        g_val = self.g(t, state)
        path_noise = state.new_zeros(state.size(0), 1)
        return torch.cat([g_val, path_noise], dim=1)

    @staticmethod
    def _resolve_solver_dt(times: torch.Tensor, explicit_dt: Any) -> float:
        time_diffs = times[1:] - times[:-1]
        if time_diffs.numel() == 0:
            return 0.01
        if not bool(torch.isfinite(time_diffs).all()) or bool((time_diffs <= 0).any()):
            raise ValueError("LatentSDE times must be finite and strictly increasing.")
        if explicit_dt is None:
            min_gap = float(time_diffs.detach().min().item())
            return min(0.01, max(min_gap, 0.001))
        dt = float(explicit_dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("LatentSDE dt must be finite and positive.")
        return dt

    def _integrate(
        self,
        *,
        y0: torch.Tensor,
        times: torch.Tensor,
        names: dict[str, str],
        method: str,
        dt: Any,
        adjoint: bool,
        options: dict[str, Any],
        rtol: Any,
        atol: Any,
    ) -> torch.Tensor:
        if adjoint and method == "srk":
            raise ValueError(
                "LatentSDE does not support adjoint=True with method='srk'; "
                "use Euler adjoint or SRK direct integration."
            )
        solver_dt = self._resolve_solver_dt(times, dt)
        integrate = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        call_kwargs: dict[str, Any] = {
            "sde": self,
            "y0": y0,
            "ts": times,
            "method": method,
            "dt": solver_dt,
            "names": names,
        }
        if options:
            call_kwargs["options"] = options
        if rtol is not None:
            call_kwargs["rtol"] = float(rtol)
        if atol is not None:
            call_kwargs["atol"] = float(atol)
        if adjoint:
            call_kwargs["adjoint_method"] = method
        return integrate(**call_kwargs)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        times: torch.Tensor,
        final_index: torch.Tensor,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        posterior_mean, posterior_scale = self._infer_initial_posterior(
            values, mask, delta, times, final_index
        )
        eps = kwargs.pop("eps", None)
        if eps is None:
            eps = torch.randn_like(posterior_scale)
        else:
            eps = eps.to(device=values.device, dtype=values.dtype)
            if tuple(eps.shape) != tuple(posterior_scale.shape):
                raise ValueError(
                    "LatentSDE eps must match the posterior shape, "
                    f"got {tuple(eps.shape)} instead of {tuple(posterior_scale.shape)}."
                )
        y0 = posterior_mean + posterior_scale * eps

        common_times = times
        if common_times.dim() == 2:
            if not torch.allclose(common_times, common_times[:1].expand_as(common_times)):
                raise ValueError("LatentSDE solver requires a shared time grid across the batch.")
            common_times = common_times[0]
        common_times = common_times.to(device=values.device, dtype=values.dtype)
        if common_times.dim() != 1 or common_times.numel() != values.size(1):
            raise ValueError("LatentSDE solver times must be a one-dimensional input grid.")

        method = str(kwargs.pop("method", "euler")).strip().lower()
        adjoint = bool(kwargs.pop("adjoint", False))
        options = dict(kwargs.pop("options", {}) or {})
        explicit_dt = kwargs.pop("dt", None)
        if explicit_dt is None:
            explicit_dt = options.pop("dt", options.pop("step_size", None))
        else:
            options.pop("dt", None)
            options.pop("step_size", None)
        rtol = kwargs.pop("rtol", None)
        atol = kwargs.pop("atol", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected LatentSDE solver arguments: {unexpected}")

        initial_kl: torch.Tensor | None = None
        if self.kl:
            prior = distributions.Normal(self.py0_mean, self.py0_std)
            posterior = distributions.Normal(posterior_mean, posterior_scale)
            initial_kl = distributions.kl_divergence(posterior, prior).sum(dim=-1)

        if common_times.numel() == 1:
            physical_path = y0.unsqueeze(1)
            out = self.embedding(physical_path)
            if self.kl:
                assert initial_kl is not None
                return out, physical_path, initial_kl.mean().clamp_min(0.0) * self.kl_weight
            return out, physical_path

        if self.kl:
            assert initial_kl is not None
            augmented_y0 = torch.cat([y0, y0.new_zeros(y0.size(0), 1)], dim=-1)
            augmented_path = self._integrate(
                y0=augmented_y0,
                times=common_times,
                names={"drift": "f_aug", "diffusion": "g_aug"},
                method=method,
                dt=explicit_dt,
                adjoint=adjoint,
                options=options,
                rtol=rtol,
                atol=atol,
            )
            physical_path = augmented_path[..., :-1].permute(1, 0, 2)
            path_kl = augmented_path[-1, :, -1]
            auxiliary_loss = (initial_kl + path_kl).mean().clamp_min(0.0) * self.kl_weight
            return self.embedding(physical_path), physical_path, auxiliary_loss

        solved_path = self._integrate(
            y0=y0,
            times=common_times,
            names={"drift": "f", "diffusion": "g"},
            method=method,
            dt=explicit_dt,
            adjoint=adjoint,
            options=options,
            rtol=rtol,
            atol=atol,
        )
        physical_path = solved_path.permute(1, 0, 2)
        return self.embedding(physical_path), physical_path

    @property
    def py0_std(self) -> torch.Tensor:
        return torch.exp(0.5 * self.py0_logvar)


class LatentSDEModel(nn.Module):
    """Wrapper exposing the sequence-layer interface."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        kl: bool,
        theta: float = 1.0,
        mu: float = 0.0,
        sigma: float = 0.5,
        kl_weight: float = 1e-5,
        recognition_hidden: int | None = None,
    ):
        super().__init__()
        self.seq_layer = LatentSDE(
            input_channels=input_dim,
            hidden_channels=hidden_dim,
            hidden_hidden_channels=int(recognition_hidden or hidden_dim),
            num_hidden_layers=num_hidden_layers,
            theta=theta,
            mu=mu,
            sigma=sigma,
            kl=kl,
            kl_weight=kl_weight,
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        times: torch.Tensor,
        final_index: torch.Tensor,
        **kwargs: Any,
    ):
        return self.seq_layer(values, mask, delta, times, final_index, **kwargs)


__all__ = ["LatentSDE", "LatentSDEModel"]
