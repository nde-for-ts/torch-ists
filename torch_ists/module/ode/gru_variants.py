"""Paper-aligned GRU variants and ODE-RNN.

Adaptation note:
- Uses controldiffeq NaturalCubicSpline (paper pathway).
- Keeps sequence-return contract used by classifier/runtime.
"""

import torch
import torchdiffeq

from ..cde import controldiffeq


class _GRU(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, use_intensity):
        super().__init__()

        assert (input_channels % 2) == 1, (
            "Input channels must be odd: 1 for time, plus 1 for each actual input, "
            "plus 1 for whether an observation was made for the actual input."
        )

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.use_intensity = use_intensity

        gru_channels = input_channels if use_intensity else (input_channels - 1) // 2
        self.gru_cell = torch.nn.GRUCell(input_size=gru_channels, hidden_size=hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def evolve(self, h, time_diff):
        raise NotImplementedError

    def _step(self, x_i, h, dt, half_num_channels):
        observation = x_i[:, 1 : 1 + half_num_channels].max(dim=1).values > 0.5
        if observation.any():
            x_i_piece = x_i if self.use_intensity else x_i[:, 1 + half_num_channels :]
            x_i_piece = x_i_piece.clone()
            x_i_piece[:, 0] += dt
            new_h = self.gru_cell(x_i_piece, h)
            h = torch.where(observation.unsqueeze(1), new_h, h)
            dt += torch.where(
                observation,
                torch.tensor(0.0, dtype=x_i.dtype, device=x_i.device),
                x_i[:, 0],
            )
        return h, dt

    def forward(self, times, coeffs, final_index, z0=None, stream=True):
        if isinstance(coeffs, torch.Tensor):
            coeffs = torch.chunk(coeffs, chunks=4, dim=-1)

        interp = controldiffeq.NaturalCubicSpline(times, coeffs)
        x = torch.stack([interp.evaluate(t) for t in times], dim=-2)
        half_num_channels = (self.input_channels - 1) // 2

        x[:, 1:, 1 : 1 + half_num_channels] -= x[:, :-1, 1 : 1 + half_num_channels]
        x[:, 0, 0] -= times[0]
        x[:, 1:, 0] -= times[:-1]

        batch_dims = x.shape[:-2]
        if z0 is None:
            z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=x.dtype, device=x.device)

        x_unbound = x.unbind(dim=1)
        h, dt = self._step(
            x_unbound[0],
            z0,
            torch.zeros(*batch_dims, dtype=x.dtype, device=x.device),
            half_num_channels,
        )
        hs = [h]
        time_diffs = times[1:] - times[:-1]
        for time_diff, x_i in zip(time_diffs, x_unbound[1:]):
            h = self.evolve(h, time_diff)
            h, dt = self._step(x_i, h, dt, half_num_channels)
            hs.append(h)

        hidden_stream = torch.stack(hs, dim=1)
        if stream:
            final_out = hidden_stream
        else:
            final_index_indices = final_index.unsqueeze(-1).expand(hidden_stream.size(0), hidden_stream.size(2)).unsqueeze(1)
            final_out = hidden_stream.gather(dim=1, index=final_index_indices).squeeze(1)

        return self.linear(final_out), hidden_stream


class GRU_dt(_GRU):
    def evolve(self, h, time_diff):
        return h


class GRU_D(_GRU):
    def __init__(self, input_channels, hidden_channels, output_channels, use_intensity):
        super().__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            use_intensity=use_intensity,
        )
        self.decay = torch.nn.Linear(1, hidden_channels)

    def evolve(self, h, time_diff):
        return h * torch.exp(-self.decay(time_diff.unsqueeze(0)).squeeze(0).relu())


class _ODERNNFunc(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super().__init__()

        layers = [torch.nn.Linear(hidden_channels, hidden_hidden_channels)]
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(hidden_hidden_channels, hidden_channels))
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, t, x):
        return self.sequential(x)


class ODERNN(_GRU):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        hidden_hidden_channels,
        num_hidden_layers,
        use_intensity,
        solver_method="euler",
        solver_options=None,
        solver_kwargs=None,
    ):
        super().__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            use_intensity=use_intensity,
        )
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.func = _ODERNNFunc(hidden_channels, hidden_hidden_channels, num_hidden_layers)
        self.solver_method = str(solver_method).lower()
        self.solver_options = dict(solver_options or {})
        self.solver_kwargs = dict(solver_kwargs or {})

    def evolve(self, h, time_diff):
        t = torch.tensor([0, time_diff.item()], dtype=time_diff.dtype, device=time_diff.device)
        call_kwargs = {"method": self.solver_method}
        options = dict(self.solver_options)
        dt_value = self.solver_kwargs.get("dt")
        if dt_value is not None and "step_size" not in options:
            options["step_size"] = dt_value
        if options:
            call_kwargs["options"] = options
        for key in ("rtol", "atol"):
            if key in self.solver_kwargs and self.solver_kwargs[key] is not None:
                call_kwargs[key] = self.solver_kwargs[key]

        use_adjoint = bool(self.solver_kwargs.get("adjoint", True))
        odeint_fn = torchdiffeq.odeint_adjoint if use_adjoint else torchdiffeq.odeint
        out = odeint_fn(func=self.func, y0=h, t=t, **call_kwargs)
        return out[1]
