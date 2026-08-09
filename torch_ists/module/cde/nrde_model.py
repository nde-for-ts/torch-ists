'''Neural RDE (Rough Differential Equation) runtime module.

Method: Neural Rough Differential Equations for Long Time Series
        (Morrill, Salvi, Kidger, Foster; ICML 2021, arXiv:2009.08295).
Base structure adapted from the Neural CDE metamodel:
        https://github.com/patrick-kidger/NeuralCDE/blob/master/experiments/models/metamodel.py
        Author: Patrick Kidger — License: Apache License 2.0

Only the depth variants neuralrde-{1,2,3} (interpolation=True) are exercised by the runtime; the
interpolation=False (mean-impute / `_m`) path is retained for reference and is not part of the
packaged model set.
'''

import torch
import torchcde

from . import controldiffeq
from ._solver import apply_control_aware_cde_solver_defaults


class NeuralRDE(torch.nn.Module):
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        if isinstance(func, ContinuousRNNConverter):  # ContinuousRNNConverter carries state as concatenated hidden+input channels
            hidden_channels = hidden_channels + input_channels

        super(NeuralRDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):  # ContinuousRNNConverter has no separate initial network
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, x, times, coeffs, final_index=None, z0=None, stream=True, interpolation=True, depth=2, window_length=None, **kwargs):
        # not implemented
        assert stream
        if window_length is None:
            raise ValueError(
                "NeuralRDE.forward requires an explicit log-signature window_length. "
                "Set model_params['logsignature_window_length'] through the runtime handler."
            )
        window_length = max(1, int(window_length))

        # for new xx
        if interpolation:
            interp = torchcde.CubicSpline(coeffs, times)
            X = torch.stack([interp.evaluate(t) for t in times], dim=-2)
            xx = torchcde.logsig_windows(X, depth=depth, window_length=window_length)
        else:
            seq_ts = times.repeat(x.shape[0], 1).to(x.device)
            x = torch.cat([seq_ts.unsqueeze(-1), x], dim=-1)
            xx = torchcde.logsig_windows(x, depth=depth, window_length=window_length)

        control_times = torch.linspace(0, 1, xx.shape[1]).to(xx.device)
        solution_times = times.to(device=xx.device, dtype=xx.dtype)
        coeffs = controldiffeq.natural_cubic_spline_coeffs(control_times, xx)

        # Extract the sizes of the batch dimensions from the coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(control_times, coeffs)

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # ContinuousRNNConverter initializes z0 as zeros
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device)
            else:
                z0 = self.initial_network(cubic_spline.evaluate(control_times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # ContinuousRNNConverter prepends input-channel zeros to z0
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)

        kwargs = apply_control_aware_cde_solver_defaults(
            solution_times,
            control_times,
            kwargs,
            piecewise_linear=False,
        )

        # Actually solve the CDE
        z_t = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   func=self.func,
                                   z0=z0,
                                   t=solution_times,
                                   **kwargs)

        if isinstance(self.func, ContinuousRNNConverter):
            hn = z_t[:, :, self.input_channels:]
        else:
            hn = z_t

        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y, hn


# The first channel is expected to be time.
class ContinuousRNNConverter(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousRNNConverter, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model

        out_base = torch.zeros(self.input_channels + self.hidden_channels, self.input_channels)
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer('out_base', out_base)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        # z is a tensor of shape (..., input_channels + hidden_channels)
        x = z[..., :self.input_channels]
        h = z[..., self.input_channels:]
        h = h.clamp(-1, 1)
        # model_out is a tensor of shape (..., hidden_channels)
        model_out = self.model(x, h)
        batch_dims = model_out.shape[:-1]
        out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        out[..., self.input_channels:, 0] = model_out
        return out
