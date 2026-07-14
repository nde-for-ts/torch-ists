import numpy as np
import torch
import torchdiffeq


class VectorField(torch.nn.Module):
    def __init__(self, dX_dt, func):
        super().__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        self.dX_dt = dX_dt
        self.func = func

    def __call__(self, t, z):
        control_gradient = self.dX_dt(t).float()
        vector_field = self.func(z)
        return (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)


class VectorFieldStack(torch.nn.Module):
    def __init__(self, dX_dt, func):
        super().__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        self.dX_dt = dX_dt
        self.func = func
        self.h_prime_trajectory = []

    def __call__(self, t, z):
        control_gradient = self.dX_dt(t)
        vector_field = self.func(z)
        control_gradient = control_gradient.type(vector_field.type())
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        self.h_prime_trajectory.append(out.detach())
        return out

    def get_h_prime(self):
        if not self.h_prime_trajectory:
            return torch.empty(0)
        return torch.stack(self.h_prime_trajectory, dim=0)


# Backward-compatible alias used by upstream ANCDE modules.
VectorField_stack = VectorFieldStack


class AttentiveVectorField(torch.nn.Module):
    def __init__(self, dX_dt, func_g, X_s, h_prime, timewise, attention, time_max):
        super().__init__()
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func_g must be a torch.nn.Module.")
        self.dX_dt = dX_dt
        self.func_g = func_g
        self.X_s = X_s
        self.h_prime = h_prime
        self.timewise = timewise
        self.attention = attention
        self.time_max = max(float(time_max), 1e-6)

    def _resolve_index(self, t_value: float, length: int) -> int:
        if length <= 1:
            return 0
        ratio = max(0.0, min(1.0, t_value / self.time_max))
        idx = int(round(ratio * (length - 1)))
        return max(0, min(idx, length - 1))

    def __call__(self, t, z):
        y_vec = self.func_g(z)
        control_gradient = self.dX_dt(t).float()

        attn_idx = self._resolve_index(float(t.item()), int(self.attention.shape[0]))
        a_t = self.attention[attn_idx]
        if not self.timewise:
            a_t = a_t.squeeze()

        x_t = self.dX_dt(t)
        d_y_dt_1 = torch.mul(control_gradient, a_t)
        d_y_dt_2_0 = torch.mul(torch.mul(a_t, (1 - a_t)), x_t).to(z.device)

        if self.timewise:
            d_y_dt_2 = torch.mul(d_y_dt_2_0, self.h_prime)
        else:
            if isinstance(self.h_prime, torch.Tensor) and self.h_prime.numel() > 0:
                hp_idx = self._resolve_index(float(t.item()), int(self.h_prime.shape[0]))
                h_prime_t = self.h_prime[hp_idx].to(z.device)
            else:
                h_prime_t = torch.zeros_like(d_y_dt_2_0)
            d_y_dt_2 = torch.mul(d_y_dt_2_0, h_prime_t)

        d_y_dt = (d_y_dt_1 + d_y_dt_2).float()
        return (y_vec @ d_y_dt.unsqueeze(-1)).squeeze(-1)


def _configure_solver_kwargs(kwargs):
    method = kwargs.get("method", "rk4")
    kwargs["method"] = method

    if method == "sym12async":
        raise NotImplementedError("sym12async is not supported for ANCDE.")
    if method == "dopri5":
        kwargs.setdefault("atol", 1e-12)
        kwargs.setdefault("rtol", 1e-6)
    elif method == "rk4":
        kwargs.setdefault("options", {})
        options = kwargs["options"]
        # Upstream ANCDE assumes integer-grid controls, so this low-level
        # `step_size=1.0` fallback is preserved only for provenance. The
        # current packaged wrappers prefill a data-driven step size.
        options.setdefault("step_size", 1.0)
    elif method == "euler":
        kwargs.setdefault("options", {})
        options = kwargs["options"]
        # Preserve the upstream oversampling fallback for provenance; the
        # current packaged wrappers now inject the exact min-gap
        # step size before this branch is reached.
        options.setdefault("step_size", 1e-3)
    return kwargs


def cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs):
    _ = kwargs
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)
    return odeint(func=vector_field, y0=z0, t=t, **kwargs)


def cdeint_final(dX_dt, attention, zz0, X_s, func_f, func_g, t, timewise, adjoint=True, **kwargs):
    del func_f
    z0, y0 = zz0
    _ = dX_dt(torch.zeros(1, dtype=y0.dtype, device=y0.device))
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = AttentiveVectorField(
        dX_dt=dX_dt,
        func_g=func_g,
        X_s=X_s,
        h_prime=torch.empty(0, device=y0.device),
        timewise=timewise,
        attention=attention,
        time_max=float(t[-1].item()),
    )
    return odeint(func=vector_field, y0=z0, t=t, **kwargs)


def ancde_bottom(dX_dt, z0, func, t, file=None, adjoint=True, **kwargs):
    del file
    kwargs = _configure_solver_kwargs(kwargs)

    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    vector_field = func(z0)
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt batch dimensions must match z0.")
    if vector_field.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func batch dimensions must match z0.")
    if vector_field.size(-2) != z0.shape[-1]:
        raise ValueError("func hidden channels must match z0.")
    if vector_field.size(-1) != control_gradient.size(-1):
        raise ValueError("func input channels must match dX_dt output.")
    if control_gradient.requires_grad and adjoint:
        raise ValueError("Control gradients do not backpropagate with adjoint=True.")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    stacked_field = VectorFieldStack(dX_dt=dX_dt, func=func)
    out = odeint(func=stacked_field, y0=z0, t=t, **kwargs)
    return out, stacked_field.get_h_prime()


def ancde(dX_dt, attention, z0, X_s, func_g, h_prime, t, timewise, adjoint=True, **kwargs):
    _ = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = AttentiveVectorField(
        dX_dt=dX_dt,
        func_g=func_g,
        X_s=X_s,
        h_prime=h_prime,
        timewise=timewise,
        attention=attention,
        time_max=float(t[-1].item()),
    )
    return odeint(func=vector_field, y0=z0, t=t, **kwargs)
