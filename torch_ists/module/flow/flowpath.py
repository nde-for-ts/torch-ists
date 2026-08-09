"""Paper-aligned FlowPath model for the packaged runtime."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchcde

from ..cde._solver import apply_control_aware_cde_solver_defaults

from .operators import build_flow_operator, resolve_flow_backend


class FlowPath(nn.Module):
    """FlowPath's official ``input_option='x'`` dataflow.

    The incoming ``coeffs`` argument is retained for runtime compatibility but
    is not consumed. The learned path is built from ``[time, x]`` directly.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_hidden_layers: int,
        output_channels: int,
        func: nn.Module,
        flow_num_layers: int = 1,
        head_depth: int | None = None,
        flow_depth: int | None = None,
        input_option: str = "x",
        flow_option: str = "c",
        flow_backend: str = "auto",
    ):
        super().__init__()
        if input_option != "x":
            raise ValueError(
                "Packaged FlowPath supports only input_option='x'. "
                f"Received: {input_option!r}"
            )
        if flow_option not in {"r", "g", "c"}:
            raise ValueError(
                "Packaged FlowPath supports flow_option in {'r','g','c'}. "
                f"Received: {flow_option!r}"
            )

        official_depth = max(1, int(num_hidden_layers))
        if head_depth is not None and max(1, int(head_depth)) != official_depth:
            raise ValueError(
                "FlowPath uses num_hidden_layers as its single official depth axis; "
                "head_depth must be omitted or equal to num_hidden_layers."
            )
        requested_flow_depth = flow_num_layers if flow_depth is None else flow_depth
        if max(1, int(requested_flow_depth)) != 1:
            raise ValueError("FlowPath fixes flow_depth=1 to match the official implementation.")

        self.input_option = input_option
        self.flow_option = flow_option
        self.requested_flow_backend = resolve_flow_backend(flow_backend)
        self.input_channels = int(input_channels)
        self.official_depth = official_depth
        self.head_depth = official_depth
        self.num_hidden_layers = official_depth
        self.flow_depth = 1
        self.flow_num_layers = 1

        hidden = int(hidden_channels)
        self.initial_flow = nn.Linear(self.input_channels, self.input_channels)
        self.initial_control = nn.Linear(self.input_channels, hidden)
        self.flow_network = build_flow_operator(
            flow_option=flow_option,
            backend=self.requested_flow_backend,
            dim=self.input_channels,
            n_layers=1,
            hidden_dims=[hidden] * self.official_depth,
        )
        self.flow_backend = self.flow_network.backend
        self.used_flow_backend_fallback = (
            self.requested_flow_backend == "auto" and self.flow_backend == "native"
        )
        self.linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(output_channels)),
        )
        self.func = func

    @staticmethod
    def _mask_aware_fill(x: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        if seq_mask.dim() == 2:
            seq_mask = seq_mask.unsqueeze(-1)
        if seq_mask.shape != x.shape:
            if seq_mask.shape[:2] == x.shape[:2] and seq_mask.shape[-1] == 1:
                seq_mask = seq_mask.expand_as(x)
            else:
                raise ValueError(
                    "seq_mask must match x for FlowPath. "
                    f"got seq_mask={tuple(seq_mask.shape)}, x={tuple(x.shape)}"
                )
        observed = seq_mask > 0
        if not bool(torch.isfinite(x[observed]).all()):
            raise ValueError("FlowPath observed input values must be finite.")
        return torch.where(observed, x, torch.zeros_like(x))

    @staticmethod
    def _normalize_seq_ts(x: torch.Tensor, seq_ts: torch.Tensor) -> torch.Tensor:
        if seq_ts.dim() == 1:
            seq_ts = seq_ts.unsqueeze(0).expand(x.shape[0], -1)
        if seq_ts.dim() == 2:
            seq_ts = seq_ts.unsqueeze(-1)
        if seq_ts.shape[:2] != x.shape[:2] or seq_ts.shape[-1] != 1:
            raise ValueError(
                "seq_ts must be [N,L] or [N,L,1] and match x. "
                f"got seq_ts={tuple(seq_ts.shape)}, x={tuple(x.shape)}"
            )
        return seq_ts.to(device=x.device, dtype=x.dtype)

    def _learned_path_values(
        self,
        x: torch.Tensor,
        seq_ts: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_filled = self._mask_aware_fill(x, seq_mask)
        normalized_ts = self._normalize_seq_ts(x, seq_ts)
        flow_input = self.initial_flow(torch.cat([normalized_ts, x_filled], dim=-1))
        return self.flow_network(flow_input, normalized_ts)

    @staticmethod
    def _resolve_solver_kwargs(
        kwargs: dict[str, object],
        times: torch.Tensor,
        control_knots: torch.Tensor,
    ) -> dict[str, object]:
        method = str(kwargs.get("method", "rk4")).strip().lower() or "rk4"
        options = kwargs.get("options")
        solver_options = dict(options) if isinstance(options, dict) else {}
        solver_kwargs: dict[str, object] = {"method": method}
        if solver_options:
            solver_kwargs["options"] = solver_options
        solver_kwargs = apply_control_aware_cde_solver_defaults(
            times,
            control_knots,
            solver_kwargs,
            piecewise_linear=False,
        )
        for key in ("rtol", "atol", "adjoint"):
            if kwargs.get(key) is not None:
                solver_kwargs[key] = kwargs[key]
        return solver_kwargs

    def _resolve_adjoint_params(
        self,
        kwargs: dict[str, object],
        learned_coeffs: torch.Tensor,
    ) -> tuple[object, ...]:
        provided = kwargs.get("adjoint_params")
        if provided is not None:
            if isinstance(provided, tuple):
                return provided
            if isinstance(provided, list):
                return tuple(provided)
            return (provided,)
        # The vector field is evaluated inside the adjoint solve. The initial
        # map and prediction head remain on the outer autograd graph, while the
        # learned-path parameters receive gradients through learned_coeffs.
        # Passing those outer parameters directly as adjoint parameters would
        # double-count their gradients.
        return tuple(self.func.parameters()) + (learned_coeffs,)

    def forward(
        self,
        x: torch.Tensor,
        seq_ts: torch.Tensor,
        seq_mask: torch.Tensor,
        coeffs: torch.Tensor | None,
        times: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del coeffs
        if x.dim() != 3:
            raise ValueError(f"x must be [N,L,D], got shape={tuple(x.shape)}")
        if times.dim() != 1 or times.numel() != x.shape[1]:
            raise ValueError(
                f"times must be [L] matching x, got times={tuple(times.shape)}, x={tuple(x.shape)}"
            )

        times = times.to(device=x.device, dtype=x.dtype)
        learned_values = self._learned_path_values(x, seq_ts, seq_mask)
        learned_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            learned_values,
            times,
        )
        learned_control = torchcde.CubicSpline(learned_coeffs, times)
        z0 = self.initial_control(learned_control.evaluate(times[0]))
        solver_kwargs = self._resolve_solver_kwargs(
            kwargs,
            times,
            learned_control.grid_points,
        )
        if solver_kwargs.get("adjoint", True) is not False:
            solver_kwargs["adjoint_params"] = self._resolve_adjoint_params(
                kwargs,
                learned_coeffs,
            )

        try:
            z_t = torchcde.cdeint(
                X=learned_control,
                func=self.func,
                z0=z0,
                t=times,
                **solver_kwargs,
            )
        except TypeError as exc:
            if "adjoint_params" not in str(exc):
                raise
            solver_kwargs.pop("adjoint_params", None)
            z_t = torchcde.cdeint(
                X=learned_control,
                func=self.func,
                z0=z0,
                t=times,
                **solver_kwargs,
            )

        out = self.linear(z_t)
        return out, z_t


__all__ = ["FlowPath"]
