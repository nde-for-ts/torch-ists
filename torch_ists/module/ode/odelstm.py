"""
ODE-LSTM module adapted from the original implementation.

Source:
- https://github.com/mlech26l/ode-lstms/blob/master/torch_node_cell.py
License: MIT License
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdyn.models import NeuralODE


class ODELSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        solver_type: str = "dopri5",
        solver_kwargs: dict | None = None,
        fixed_step_count: int = 3,
    ):
        super().__init__()
        self.solver_type = solver_type
        self.fixed_step_solver = solver_type.startswith("fixed_")
        self.solver_kwargs = dict(solver_kwargs or {})
        self.fixed_step_count = max(1, int(fixed_step_count))
        self.lstm = nn.LSTMCell(input_size, hidden_size)

        layers = [nn.Linear(hidden_size, hidden_size)]
        for _ in range(max(num_hidden_layers - 1, 0)):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, hidden_size))
        self.f_node = nn.Sequential(*layers)

        self.input_size = input_size
        self.hidden_size = hidden_size
        if not self.fixed_step_solver:
            self.node = NeuralODE(self.f_node, solver=solver_type, **self.solver_kwargs)
        else:
            options = {
                "fixed_euler": self.euler,
                "fixed_heun": self.heun,
                "fixed_rk4": self.rk4,
            }
            if solver_type not in options:
                raise ValueError(f"Unknown solver type '{solver_type}'")
            self.node = options[self.solver_type]

    def forward(self, input_x: torch.Tensor, hx, ts: torch.Tensor):
        new_h, new_c = self.lstm(input_x, hx)
        if self.fixed_step_solver:
            new_h = self.solve_fixed(new_h, ts)
        else:
            new_h = self.solve_adaptive(new_h, ts)

        return new_h, new_c

    def solve_adaptive(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        durations = ts.reshape(-1).to(dtype=x.dtype, device=x.device)
        if durations.numel() != x.size(0):
            raise ValueError(
                f"Expected one ODE duration per sample, got {durations.numel()} for batch {x.size(0)}"
            )
        if not bool(torch.isfinite(durations).all()) or bool((durations < 0).any()):
            raise ValueError("ODE-LSTM durations must be finite and non-negative")

        positive_indices = torch.nonzero(durations > 0, as_tuple=False).reshape(-1)
        if positive_indices.numel() == 0:
            return x
        positive_durations = durations.index_select(0, positive_indices)
        unique_durations, inverse = torch.unique(
            positive_durations,
            sorted=True,
            return_inverse=True,
        )
        time_span = torch.cat([x.new_zeros(1), unique_durations])
        trajectory = self.node.trajectory(x, time_span)
        selected = trajectory[inverse + 1, positive_indices]
        return x.index_copy(0, positive_indices, selected)

    def solve_fixed(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        ts = ts.view(-1, 1)
        step_count = max(1, int(self.fixed_step_count))
        for _ in range(step_count):
            x = self.node(x, ts * (1.0 / step_count))
        return x

    def euler(self, y: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        dy = self.f_node(y)
        return y + delta_t * dy

    def heun(self, y: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        k1 = self.f_node(y)
        k2 = self.f_node(y + delta_t * k1)
        return y + delta_t * 0.5 * (k1 + k2)

    def rk4(self, y: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        k1 = self.f_node(y)
        k2 = self.f_node(y + k1 * delta_t * 0.5)
        k3 = self.f_node(y + k2 * delta_t * 0.5)
        k4 = self.f_node(y + k3 * delta_t)
        return y + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class ODELSTM(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_hidden_layers: int,
        out_feature: int,
        solver_type: str = "dopri5",
        solver_kwargs: dict | None = None,
        fixed_step_count: int = 3,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature

        self.rnn_cell = ODELSTMCell(
            in_features,
            hidden_size,
            num_hidden_layers,
            solver_type=solver_type,
            solver_kwargs=solver_kwargs,
            fixed_step_count=fixed_step_count,
        )
        self.fc = nn.Linear(self.hidden_size, self.out_feature)

    def forward(self, x: torch.Tensor, timespans: torch.Tensor, mask: torch.Tensor | None = None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)

        hidden_state = (
            torch.zeros((batch_size, self.hidden_size), device=device),
            torch.zeros((batch_size, self.hidden_size), device=device),
        )

        outputs = []
        hiddens = []
        for t in range(seq_len):
            inputs = x[:, t]
            ts = timespans[:, t].reshape(batch_size)
            if mask is None:
                hidden_state = self.rnn_cell(inputs, hidden_state, ts)
            else:
                observed_indices = torch.nonzero(mask[:, t].reshape(batch_size) > 0, as_tuple=False)
                observed_indices = observed_indices.reshape(-1)
                if observed_indices.numel() > 0:
                    observed_state = (
                        hidden_state[0].index_select(0, observed_indices),
                        hidden_state[1].index_select(0, observed_indices),
                    )
                    updated_state = self.rnn_cell(
                        inputs.index_select(0, observed_indices),
                        observed_state,
                        ts.index_select(0, observed_indices),
                    )
                    hidden_state = (
                        hidden_state[0].index_copy(0, observed_indices, updated_state[0]),
                        hidden_state[1].index_copy(0, observed_indices, updated_state[1]),
                    )
            current_output = self.fc(hidden_state[0])
            outputs.append(current_output)
            hiddens.append(hidden_state[0])

        outputs = torch.stack(outputs, dim=1)
        hiddens = torch.stack(hiddens, dim=1)
        return outputs, hiddens
