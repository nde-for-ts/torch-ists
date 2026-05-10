"""SDEModel wrapper for the packaged SDE runtime family."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .diffusion import Diffusion_model
from .nsde import NeuralSDE


class SDEModel(nn.Module):
    """Wrapper exposing seq_layer interface used by handler runtime."""

    def __init__(
        self,
        coeff_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        input_option: int,
        noise_option: int,
        method: Optional[str] = None,
        solver_options: Optional[Dict[str, Any]] = None,
        solver_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.coeff_dim = int(coeff_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_option = int(input_option)
        self.noise_option = int(noise_option)
        self.model_name: Optional[str] = None

        vector_field = Diffusion_model(
            input_channels=self.coeff_dim,
            hidden_channels=self.hidden_dim,
            hidden_hidden_channels=self.hidden_dim,
            num_hidden_layers=int(num_hidden_layers),
            input_option=self.input_option,
            noise_option=self.noise_option,
        )

        self.seq_layer = NeuralSDE(
            func=vector_field,
            input_channels=self.coeff_dim,
            hidden_channels=self.hidden_dim,
            output_channels=self.hidden_dim,
            initial=True,
        )

        self.solver_method = method
        self.solver_options = dict(solver_options or {})
        self.solver_kwargs = dict(solver_kwargs or {})

    def forward(self, coeffs: torch.Tensor, times: torch.Tensor, **kwargs: Any):
        return self.seq_layer(coeffs, times, **kwargs)
