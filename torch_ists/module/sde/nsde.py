"""Neural SDE solver wrapper for the packaged SDE runtime models."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torchsde


class NeuralSDE(torch.nn.Module):
    """Neural SDE model using torchsde.sdeint."""

    def __init__(
        self,
        func: torch.nn.Module,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        initial: bool = True,
    ):
        super().__init__()
        self.func = func
        self.initial = bool(initial)
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, output_channels),
        )

    @staticmethod
    def _default_dt(times: torch.Tensor) -> float:
        time_diffs = times[1:] - times[:-1]
        return max(time_diffs.min().item(), 1e-3)

    def forward(self, coeffs: torch.Tensor, times: torch.Tensor, **kwargs: Any):
        self.func.set_X(coeffs, times)

        y0 = self.func.X.evaluate(times[0])
        if not self.initial:
            y0 = torch.zeros_like(y0)
        y0 = self.initial_network(y0)

        call_kwargs: Dict[str, Any] = dict(kwargs)
        method = call_kwargs.setdefault("method", "srk")

        if "dt" not in call_kwargs:
            options = call_kwargs.get("options")
            if isinstance(options, dict) and "dt" in options:
                call_kwargs["dt"] = options["dt"]
            else:
                call_kwargs["dt"] = self._default_dt(times)

        if method in {"srk", "euler"} and "options" not in call_kwargs:
            call_kwargs["options"] = {"dt": call_kwargs["dt"]}

        z = torchsde.sdeint(
            sde=self.func,
            y0=y0,
            ts=times,
            **call_kwargs,
        )
        z = z.permute(1, 0, 2)
        out = self.linear(z)
        hn = z[:, -1, :]
        return out, hn
