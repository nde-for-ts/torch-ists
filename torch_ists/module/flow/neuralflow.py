"""NeuralFlow model for runtime package."""

from __future__ import annotations

import torch
import torch.nn as nn

from .operators import build_flow_operator, resolve_flow_backend


class NeuralFlow(nn.Module):
    """NeuralFlow baseline for `input_option='x'`.

    Runtime contract:
    `forward(x, seq_ts, seq_mask, coeffs, times, **kwargs) -> (out, hidden)`.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_hidden_layers: int,
        output_channels: int,
        flow_num_layers: int = 2,
        head_depth: int | None = None,
        flow_depth: int | None = None,
        input_option: str = "x",
        flow_option: str = "c",
        flow_backend: str = "auto",
    ):
        super().__init__()
        if input_option != "x":
            raise ValueError(
                "Packaged NeuralFlow currently supports only input_option='x'. "
                f"Received: {input_option!r}"
            )
        if flow_option not in {"r", "g", "c"}:
            raise ValueError(
                "Packaged NeuralFlow supports flow_option in {'r','g','c'}. "
                f"Received: {flow_option!r}"
            )

        self.input_option = input_option
        self.flow_option = flow_option
        self.requested_flow_backend = resolve_flow_backend(flow_backend)

        hidden = int(hidden_channels)
        resolved_head_depth = num_hidden_layers if head_depth is None else head_depth
        resolved_flow_depth = flow_num_layers if flow_depth is None else flow_depth
        self.head_depth = max(1, int(resolved_head_depth))
        self.flow_depth = max(1, int(resolved_flow_depth))
        # Compatibility aliases.
        self.num_hidden_layers = self.head_depth
        self.flow_num_layers = self.flow_depth

        self.initial_flow = nn.Linear(int(input_channels), hidden)
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
        self.flow_backend = self.flow_network.backend
        self.used_flow_backend_fallback = (
            self.requested_flow_backend == "auto" and self.flow_backend == "native"
        )
        # Expose the effective backend (``auto`` -> real stribor/native build).
        self.flow_backend = getattr(self.flow_network, "backend", self.flow_backend)

    @staticmethod
    def _fill_missing_x(x: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        if seq_mask.dim() == 2:
            seq_mask = seq_mask.unsqueeze(-1)
        if seq_mask.shape[:2] != x.shape[:2]:
            raise ValueError(
                "seq_mask must match x batch/time dimensions. "
                f"got seq_mask={tuple(seq_mask.shape)}, x={tuple(x.shape)}"
            )
        missing = seq_mask <= 0
        return torch.where(missing, torch.zeros_like(x), x)

    def forward(
        self,
        x: torch.Tensor,
        seq_ts: torch.Tensor,
        seq_mask: torch.Tensor,
        coeffs: torch.Tensor | None,
        times: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del coeffs, times

        if x.dim() != 3:
            raise ValueError(f"x must be [N, L, D], got shape={tuple(x.shape)}")
        if seq_ts.dim() == 1:
            seq_ts = seq_ts.unsqueeze(0).expand(x.shape[0], -1)
        if seq_ts.dim() == 2:
            seq_ts = seq_ts.unsqueeze(-1)
        if seq_ts.shape[0] != x.shape[0] or seq_ts.shape[1] != x.shape[1]:
            raise ValueError(
                "seq_ts must match x batch/time dimensions. "
                f"got seq_ts={tuple(seq_ts.shape)}, x={tuple(x.shape)}"
            )

        seq_ts = seq_ts.to(dtype=x.dtype, device=x.device)
        x_filled = self._fill_missing_x(x, seq_mask)
        z = self.initial_flow(torch.cat([seq_ts, x_filled], dim=-1))
        z = self.flow_network(z, seq_ts)
        z = torch.relu(z)
        for linear in self.linears:
            z = torch.relu(linear(z))
        out = self.linear(z)
        return out, z
