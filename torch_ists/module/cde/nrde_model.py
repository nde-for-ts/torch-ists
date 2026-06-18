'''Neural RDE (Rough Differential Equation) runtime module.

Method: Neural Rough Differential Equations for Long Time Series
        (Morrill, Salvi, Kidger, Foster; ICML 2021, arXiv:2009.08295).
Base structure adapted from the Neural CDE metamodel:
        https://github.com/patrick-kidger/NeuralCDE/blob/master/experiments/models/metamodel.py
        Author: Patrick Kidger — License: Apache License 2.0

Ported from torch_ists_refactoring/diff_module/NCDE/nrde_model.py (the refactoring SoT) for the
packaged runtime. Only the theory-consistent depth variants (neuralrde-{1,2,3}, interpolation=True)
are exercised by CDEHandler; the interpolation=False (mean-impute / `_m`) path is retained for SoT
fidelity but is out of scope for promotion and is not registered.
'''

import torch
import torchcde

from . import controldiffeq


class NeuralRDE(torch.nn.Module):
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(NeuralRDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, x, times, coeffs, final_index=None, z0=None, stream=True, interpolation=True, depth=2, window_length=10, **kwargs):
        # not implemented
        assert stream

        # for new xx
        if interpolation:
            interp = torchcde.CubicSpline(coeffs, times)
            X = torch.stack([interp.evaluate(t) for t in times], dim=-2)
            xx = torchcde.logsig_windows(X, depth=depth, window_length=window_length)
        else:
            seq_ts = times.repeat(x.shape[0], 1).to(x.device)
            x = torch.cat([seq_ts.unsqueeze(-1), x], dim=-1)
            xx = torchcde.logsig_windows(x, depth=depth, window_length=window_length)

        tt = torch.linspace(0, 1, xx.shape[1]).to(xx.device)
        coeffs = controldiffeq.natural_cubic_spline_coeffs(tt, xx)

        # Extract the sizes of the batch dimensions from the coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(tt, coeffs)

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device)
            else:
                z0 = self.initial_network(cubic_spline.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # continuing adventures in ugly hacks
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # approximation
        if kwargs['method'] == 'euler':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = max(time_diffs.min().item(), 1e-3)

        # Actually solve the CDE
        z_t = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   func=self.func,
                                   z0=z0,
                                   t=tt,
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
