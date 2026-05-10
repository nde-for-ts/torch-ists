"""Paper-aligned Neural CDE metamodel with interpolation-aware controls."""

import torch
import torchcde

from . import controldiffeq


class NeuralCDE(torch.nn.Module):
    def __init__(
        self,
        func,
        input_channels,
        hidden_channels,
        output_channels,
        initial=True,
        control="natural",
    ):
        if isinstance(func, ContinuousRNNConverter):
            hidden_channels = hidden_channels + input_channels

        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.func = func
        self.initial = initial
        self.control = control
        if initial and not isinstance(func, ContinuousRNNConverter):
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def _build_control_path(self, times, coeffs):
        if self.control == "natural":
            if isinstance(coeffs, torch.Tensor):
                coeffs = torch.chunk(coeffs, chunks=4, dim=-1)
            return controldiffeq.NaturalCubicSpline(times, coeffs), coeffs[0]

        if self.control == "linear":
            control = torchcde.LinearInterpolation(coeffs, times)
        elif self.control == "rectilinear":
            rect_times = torch.linspace(0, 1, len(times) * 2 - 1, device=times.device, dtype=times.dtype)
            control = torchcde.LinearInterpolation(coeffs, rect_times)
        elif self.control in {"cubic", "hermite"}:
            control = torchcde.CubicSpline(coeffs, times)
        else:
            raise ValueError(f"Unsupported control mode: {self.control}")
        coeff = torch.chunk(coeffs, chunks=4, dim=-1)[0] if isinstance(coeffs, torch.Tensor) else coeffs[0]
        return control, coeff

    def forward(self, times, coeffs, final_index, z0=None, stream=True, **kwargs):
        control_path, coeff = self._build_control_path(times, coeffs)
        batch_dims = coeff.shape[:-2]

        if not stream:
            assert batch_dims == final_index.shape, (
                "coeff.shape[:-2] must match final_index.shape: "
                f"{batch_dims} != {tuple(final_index.shape)}"
            )

        if z0 is None:
            assert self.initial, "Expected initial=True when z0 is omitted."
            if isinstance(self.func, ContinuousRNNConverter):
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device)
            else:
                z0 = self.initial_network(control_path.evaluate(times[0]))
        else:
            assert not self.initial, "Expected explicit z0 only when initial=False."
            if isinstance(self.func, ContinuousRNNConverter):
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)

        if "method" not in kwargs:
            kwargs["method"] = "rk4"
        if kwargs["method"] in {"rk4", "euler"}:
            kwargs.setdefault("options", {})
            options = kwargs["options"]
            if "step_size" not in options and "grid_constructor" not in options:
                time_diffs = times[1:] - times[:-1]
                step = time_diffs.min().item()
                if kwargs["method"] == "euler":
                    step = max(step, 1e-3)
                options["step_size"] = step

        if self.control == "natural":
            z_t = controldiffeq.cdeint(
                dX_dt=control_path.derivative,
                z0=z0,
                func=self.func,
                t=times,
                **kwargs,
            )
        else:
            z_t = torchcde.cdeint(
                X=control_path,
                z0=z0,
                func=self.func,
                t=times,
                **kwargs,
            )

        if stream:
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        if isinstance(self.func, ContinuousRNNConverter):
            hn = z_t[..., self.input_channels :]
        else:
            hn = z_t

        pred_y = self.linear(z_t)
        return pred_y, hn


class ContinuousRNNConverter(torch.nn.Module):
    # Relies on the first control channel being time.
    def __init__(self, input_channels, hidden_channels, model):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model

        out_base = torch.zeros(self.input_channels + self.hidden_channels, self.input_channels)
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer("out_base", out_base)

    def forward(self, z):
        x = z[..., : self.input_channels]
        h = z[..., self.input_channels :]
        h = h.clamp(-1, 1)
        model_out = self.model(x, h)
        batch_dims = model_out.shape[:-1]
        out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        out[..., self.input_channels :, 0] = model_out
        return out
