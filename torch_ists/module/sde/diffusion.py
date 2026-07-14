"""Diffusion model primitives for packaged SDE baselines.

This module is adapted from an earlier NSDE implementation and restricted to
runtime functionality required by the packaged SDE baseline models.
"""

from __future__ import annotations

import torch
import torchcde
import torch.nn.functional as F


class Diffusion_model(torch.nn.Module):
    """SDE drift/diffusion vector field.

    Public callers should select readable model names such as ``neural_lnsde``.
    Low-level numeric selectors are derived from those model configs and are
    not accepted as public model names.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        hidden_hidden_channels: int,
        num_hidden_layers: int,
        theta: float = 1.0,
        sigma: float = 1.0,
        input_option: int = 0,
        noise_option: int = 0,
    ):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"

        self.input_option = int(input_option)
        self.noise_option = int(noise_option)
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.monotone_drift = False
        self.diffusion_floor = 0.0

        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)

        if self.input_option in [3, 4, 5, 6]:
            self.linear_in = torch.nn.Linear(hidden_channels + 2, hidden_hidden_channels)
        else:
            self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)

        if self.input_option in [2, 4, 6]:
            self.emb = torch.nn.Linear(hidden_channels * 2, hidden_channels)

        self.linears = torch.nn.ModuleList(
            torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(max(0, num_hidden_layers - 1))
        )
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)

        self.theta = torch.nn.Parameter(torch.tensor([[theta]]), requires_grad=True)
        if self.noise_option in [1, 2, 3]:
            self.sigma = torch.nn.Parameter(torch.tensor([sigma]), requires_grad=True)
        if self.noise_option in [4, 5, 6]:
            self.sigma_diag = torch.nn.Parameter(
                torch.tensor([sigma] * hidden_channels), requires_grad=True
            )

        if self.noise_option in [12, 13]:
            self.noise_t = torch.nn.Linear(2, hidden_channels)
        if self.noise_option in [14, 15]:
            self.noise_y = torch.nn.Linear(hidden_channels + 2, hidden_channels)
        if self.noise_option in [16, 17]:
            self.noise_t = torch.nn.Sequential(
                torch.nn.Linear(2, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        if self.noise_option in [18, 19]:
            self.noise_y = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels + 2, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )

    def set_X(self, coeffs: torch.Tensor, times: torch.Tensor) -> None:
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.f(t, y)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Xt = self.X.evaluate(t)
        Xt = self.initial_network(Xt)

        if self.input_option in [3, 4, 5, 6]:
            if t.dim() == 0:
                t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
            yy = self.linear_in(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))
        else:
            yy = self.linear_in(y)

        if self.input_option == 0:
            z = Xt
        elif self.input_option in [1, 3, 5]:
            z = yy
        elif self.input_option in [2, 4, 6]:
            z = self.emb(torch.cat([yy, Xt], dim=-1))
        else:
            z = yy

        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z)

        if self.input_option in [5, 6]:
            z = z * (1 - torch.nan_to_num(y).sigmoid())

        if self.monotone_drift:
            return -F.softplus(z)
        return z.tanh()

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)

        if self.noise_option == 0:
            noise = torch.zeros(y.size(0), y.size(1), device=y.device)
        elif self.noise_option == 1:
            noise = self.sigma.exp().expand(y.size(0), y.size(1))
        elif self.noise_option == 2:
            noise = self.sigma.exp().expand(y.size(0), y.size(1)) * t
        elif self.noise_option == 3:
            noise = self.sigma.exp().expand(y.size(0), y.size(1)) * y
        elif self.noise_option == 4:
            noise = self.sigma_diag.exp().repeat(y.size(0), 1)
        elif self.noise_option == 5:
            noise = self.sigma_diag.exp().repeat(y.size(0), 1) * t
        elif self.noise_option == 6:
            noise = self.sigma_diag.exp().repeat(y.size(0), 1) * y
        elif self.noise_option == 7:
            noise = torch.sqrt(y)
        elif self.noise_option == 8:
            noise = y ** 3
        elif self.noise_option == 9:
            noise = y.sigmoid()
        elif self.noise_option == 10:
            noise = y.relu()
        elif self.noise_option == 11:
            noise = t * y
        elif self.noise_option == 12:
            noise = self.noise_t(torch.cat([torch.sin(t), torch.cos(t)], dim=-1))
        elif self.noise_option == 13:
            tt = self.noise_t(torch.cat([torch.sin(t), torch.cos(t)], dim=-1))
            noise = tt * y
        elif self.noise_option == 14:
            noise = self.noise_y(torch.cat([torch.sin(t), torch.cos(t), y], dim=-1))
        elif self.noise_option == 15:
            yy = self.noise_y(torch.cat([torch.sin(t), torch.cos(t), y], dim=-1))
            noise = yy * y
        elif self.noise_option == 16:
            tt = self.noise_t(torch.cat([torch.sin(t), torch.cos(t)], dim=-1)).relu()
            noise = tt
        elif self.noise_option == 17:
            tt = self.noise_t(torch.cat([torch.sin(t), torch.cos(t)], dim=-1)).relu()
            noise = tt * y
        elif self.noise_option == 18:
            yy = self.noise_y(torch.cat([torch.sin(t), torch.cos(t), y], dim=-1)).relu()
            noise = yy
        elif self.noise_option == 19:
            yy = self.noise_y(torch.cat([torch.sin(t), torch.cos(t), y], dim=-1)).relu()
            noise = yy * y
        else:
            noise = torch.zeros(y.size(0), y.size(1), device=y.device)

        noise = self.theta.sigmoid() * torch.nan_to_num(noise)
        if float(self.diffusion_floor) > 0.0:
            sign = torch.where(noise < 0, noise.new_tensor(-1.0), noise.new_tensor(1.0))
            noise = sign * (noise.abs() + float(self.diffusion_floor))
        return noise.tanh()
