"""DualDynamics model for runtime package."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchcde

from ..cde._solver import apply_control_aware_cde_solver_defaults

from .operators import build_flow_operator, resolve_flow_backend


class DualDynamics(nn.Module):
    """DualDynamics baseline for `input_option='x'` with coeff-driven control path.

    Runtime contract:
    `forward(x, seq_ts, seq_mask, coeffs, times, **kwargs) -> (out, hidden)`.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_hidden_layers: int,
        output_channels: int,
        func: nn.Module,
        flow_num_layers: int = 2,
        head_depth: int | None = None,
        flow_depth: int | None = None,
        input_option: str = "x",
        flow_option: str = "c",
        flow_backend: str = "native",
    ):
        super().__init__()
        if input_option != "x":
            raise ValueError(
                "Packaged DualDynamics currently supports only input_option='x'. "
                f"Received: {input_option!r}"
            )
        if flow_option not in {"r", "g", "c"}:
            raise ValueError(
                "Packaged DualDynamics supports flow_option in {'r','g','c'}. "
                f"Received: {flow_option!r}"
            )

        self.input_option = input_option
        self.flow_option = flow_option
        self.requested_flow_backend = resolve_flow_backend(flow_backend)
        self.input_channels = int(input_channels)
        self.func = func

        hidden = int(hidden_channels)
        resolved_head_depth = num_hidden_layers if head_depth is None else head_depth
        resolved_flow_depth = flow_num_layers if flow_depth is None else flow_depth
        self.head_depth = max(1, int(resolved_head_depth))
        self.flow_depth = max(1, int(resolved_flow_depth))
        # Compatibility aliases.
        self.num_hidden_layers = self.head_depth
        self.flow_num_layers = self.flow_depth

        self.initial_control = nn.Linear(self.input_channels, hidden)
        self.linears = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(self.head_depth - 1))
        self.linear = nn.Linear(hidden, int(output_channels))

        flow_hidden = [hidden] * self.head_depth
        self.flow_network = build_flow_operator(
            flow_option=flow_option,
            backend=self.requested_flow_backend,
            dim=hidden,
            n_layers=self.flow_depth,
            hidden_dims=flow_hidden,
        )
        # Expose the effective backend (``auto`` -> real stribor/native build).
        self.flow_backend = self.flow_network.backend
        self.used_flow_backend_fallback = (
            self.requested_flow_backend == "auto" and self.flow_backend == "native"
        )
        self.flow_backend_fallback_reason = getattr(
            self.flow_network, "fallback_reason", None
        )

    def _normalize_seq_ts(self, x: torch.Tensor, seq_ts: torch.Tensor) -> torch.Tensor:
        if seq_ts.dim() == 1:
            seq_ts = seq_ts.unsqueeze(0).expand(x.shape[0], -1)
        if seq_ts.dim() == 2:
            seq_ts = seq_ts.unsqueeze(-1)
        if seq_ts.shape[0] != x.shape[0] or seq_ts.shape[1] != x.shape[1]:
            raise ValueError(
                "seq_ts must match x batch/time dimensions. "
                f"got seq_ts={tuple(seq_ts.shape)}, x={tuple(x.shape)}"
            )
        return seq_ts.to(dtype=x.dtype, device=x.device)

    def _resolve_adjoint_params(
        self,
        kwargs: dict[str, object],
        coeffs: torch.Tensor,
    ) -> tuple[object, ...]:
        provided = kwargs.get("adjoint_params")
        if provided is not None:
            if isinstance(provided, tuple):
                return provided
            if isinstance(provided, list):
                return tuple(provided)
            return (provided,)

        params: tuple[object, ...] = tuple(self.parameters())
        return params + (coeffs,)

    def forward(
        self,
        x: torch.Tensor,
        seq_ts: torch.Tensor,
        seq_mask: torch.Tensor,
        coeffs: torch.Tensor | None,
        times: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del seq_mask

        if coeffs is None:
            raise ValueError("DualDynamics models require interpolation coeffs.")
        if x.dim() != 3:
            raise ValueError(f"x must be [N, L, D], got shape={tuple(x.shape)}")
        if times.dim() != 1:
            raise ValueError(f"times must be [L], got shape={tuple(times.shape)}")

        seq_ts = self._normalize_seq_ts(x, seq_ts)
        times = times.to(dtype=x.dtype, device=x.device)

        control_spline = torchcde.CubicSpline(coeffs, times)
        control = control_spline.evaluate(times)
        if control.shape[-1] < self.input_channels:
            raise ValueError(
                "DualDynamics coeff channels are incompatible with input channels. "
                f"coeff_channels={control.shape[-1]}, expected>={self.input_channels}"
            )
        control = control[..., : self.input_channels]

        z0 = self.initial_control(control_spline.evaluate(times[0]))

        method = str(kwargs.get("method", "rk4")).strip().lower() or "rk4"
        options = kwargs.get("options")
        solver_options = dict(options) if isinstance(options, dict) else {}
        solver_kwargs: dict[str, object] = {"method": method}
        if solver_options:
            solver_kwargs["options"] = solver_options
        solver_kwargs = apply_control_aware_cde_solver_defaults(
            times,
            control_spline.grid_points,
            solver_kwargs,
            piecewise_linear=False,
        )
        for key in ("rtol", "atol", "adjoint"):
            if kwargs.get(key) is not None:
                solver_kwargs[key] = kwargs[key]
        if solver_kwargs.get("adjoint", True) is not False:
            solver_kwargs["adjoint_params"] = self._resolve_adjoint_params(kwargs, coeffs)

        try:
            z_t = torchcde.cdeint(
                X=control_spline,
                func=self.func,
                z0=z0,
                t=times,
                **solver_kwargs,
            )
        except TypeError as exc:
            if "adjoint_params" not in str(exc):
                raise
            # torchdiffeq/torchcde version-dependent fallback:
            # some runtimes do not accept adjoint_params.
            solver_kwargs.pop("adjoint_params", None)
            z_t = torchcde.cdeint(
                X=control_spline,
                func=self.func,
                z0=z0,
                t=times,
                **solver_kwargs,
            )

        z = self.flow_network(z_t, seq_ts)
        z = torch.relu(z)
        for linear in self.linears:
            z = torch.relu(linear(z))

        out = self.linear(z)
        return out, z_t
