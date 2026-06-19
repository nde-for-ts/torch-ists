'''
https://github.com/alflsowl12/LEAP/blob/main/experiments/models/metamodel.py
Author: Minju Jo

Modificataion as sequence layer
'''

import torch

from . import controldiffeq
from .._solver import apply_advanced_cde_solver_defaults


class ContinuousRNNConverter(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousRNNConverter, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model

        out_base = torch.zeros(
            self.input_channels + self.hidden_channels, self.input_channels
        )
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer("out_base", out_base)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(
            self.input_channels, self.hidden_channels
        )

    def forward(self, z):
        x = z[..., : self.input_channels]
        h = z[..., self.input_channels :]
        h = h.clamp(-1, 1)
        model_out = self.model(x, h)
        batch_dims = model_out.shape[:-1]
        out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        out[..., self.input_channels :, 0] = model_out
        return out


class NeuralCDE_Learnable(torch.nn.Module):
    

    def __init__(
        self,
        func_g,
        func_k,
        func_f,
        mapping,
        input_channels,
        hidden_channels,
        output_channels,
        file,
        initial=True,
    ):
        
        if isinstance(func_k, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(NeuralCDE_Learnable, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.func_g = func_g  # function g of eq(4)
        self.func_k = func_k  # Original CDE

        self.func_f = func_f  # function f of eq(6)
        self.func_init = torch.nn.Linear(input_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, hidden_channels)
        self.mapping = mapping

        self.initial = initial
        if initial and not isinstance(func_k, ContinuousRNNConverter):  
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)
        self.file = file

    def extra_repr(self):
        return (
            "input_channels={}, hidden_channels={}, output_channels={}, initial={}"
            "".format(
                self.input_channels,
                self.hidden_channels,
                self.output_channels,
                self.initial,
            )
        )

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        
        # if len(coeffs) == 4 :
        #     coeff, _, _, _ = coeffs
        # else : 
        #     coeff, _, _ = coeffs
        coeffs = torch.chunk(coeffs, chunks=4, dim=-1)
        coeff = coeffs[0]
        
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, (
                "coeff.shape[:-2] must be the same as final_index.shape. "
                "coeff.shape[:-2]={}, final_index.shape={}"
                "".format(batch_dims, final_index.shape)
            )

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func_k, ContinuousRNNConverter):  
                z0 = torch.zeros(
                    *batch_dims,
                    self.hidden_channels,
                    dtype=coeff.dtype,
                    device=coeff.device
                )
            else:
                z0 = self.initial_network(cubic_spline.evaluate(times[0]))
                path0 = cubic_spline.evaluate(times[0])
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(
                self.func_k, ContinuousRNNConverter
            ):  
                z0_extra = torch.zeros(
                    *batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device
                )
                z0 = torch.cat([z0_extra, z0], dim=-1)
            else:
                self.initial_network = None

                XX = cubic_spline.evaluate(times[0])
                path0 = XX

        if stream:
            t = times
        else:
            sorted_final_index, inverse_final_index = final_index.unique(
                sorted=True, return_inverse=True
            )
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat(
                [
                    times[0].unsqueeze(0),
                    times[sorted_final_index],
                    times[-1].unsqueeze(0),
                ]
            )

        ######################
        # Create Encoder! e(t)
        ######################

        # Upstream LEAP assumes integer-grid time with rk4 step_size=1.0.
        # The current deployment/release runtime builds coeffs on normalized
        # [0, 1] times, so inject a data-driven step size here and leave the
        # low-level 1.0 fallback as provenance-only context.
        kwargs = apply_advanced_cde_solver_defaults(times, kwargs)

        # Equation 4
        Encoder = controldiffeq.cdeint(
            dX_dt=cubic_spline.derivative, z0=z0, func=self.func_k, t=t, **kwargs
        )
        
        if stream:
            for i in range(len(Encoder.shape) - 2, 0, -1):
                Encoder = Encoder.transpose(0, i)

        else:
            final_index_indices = (
                final_index.unsqueeze(-1).expand(Encoder.shape[1:]).unsqueeze(0)
            )
            e_t = Encoder.gather(dim=0, index=final_index_indices).squeeze(0)
            
        h_hat_0 = self.fc(e_t)
        
        # eq(4,5,6) are in cdeint_Learnable
        ode_cde = controldiffeq.cdeint_Learnable(
            dX_dt=cubic_spline,
            path0=path0,
            h_hat_0=h_hat_0,
            z0=z0,
            func_g=self.func_g,
            func_f=self.func_f,
            func_map=self.mapping,
            func_init=self.func_init,
            file=self.file,
            t=times,
            **kwargs
        )

        ode_out, z_t, logpz_t = ode_cde[:3]
        learn_path = ode_cde[3] if len(ode_cde) > 3 else torch.empty(0, device=coeff.device)
        if learn_path.numel() == 0:
            learn_path = coeff.permute(1, 0, 2).detach()

        loss2 = torch.mean(logpz_t)
        mseloss = torch.nn.MSELoss()
        x_t = coeff.permute(1, 0, 2)
        
        # loss1 = mseloss(learn_path[:x_t.shape[0], :, 1:], x_t[:, :, 1:])
        loss1 = mseloss(
            learn_path[:x_t.shape[0], :, 1:],
            x_t[:learn_path.shape[0], :, 1:],
        )
    
        pred_y = self.linear(z_t)
        
        # return pred_y, z_t, loss1, loss2
        return pred_y, z_t, (loss1+loss2)*1e-5
