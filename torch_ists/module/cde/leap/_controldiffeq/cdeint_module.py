import torch
import torchdiffeq

from .TorchDiffEqPack import odesolve_adjoint_sym12 as odesolve


class VectorField(torch.nn.Module):
    def __init__(self, dX_dt, func):
        super().__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        self.dX_dt = dX_dt
        self.func = func

    def __call__(self, t, z):
        control_gradient = self.dX_dt(t)
        vector_field = self.func(z)
        return (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    return e_dzdx_e.view(y.shape[0], -1).sum(dim=1)


def sample_gaussian_like(y):
    return torch.rand_like(y)


class VectorFieldLearnable(torch.nn.Module):
    def __init__(self, dX_dt, func_f, func_g, func_map, func_init, time_max):
        super().__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func_f must be a torch.nn.Module.")
        self.dX_dt = dX_dt
        self.func_f = func_f
        self.func_g = func_g
        self.func_map = func_map
        self.func_init = func_init
        self.time_max = max(float(time_max), 1e-6)
        self.divergence_fn = divergence_approx
        self._saved_times = []
        self._trajectory = []

    def __call__(self, t, z):
        z0, h_prime_0 = z[0], z[1]
        batchsize = h_prime_0.shape[0]
        noise = sample_gaussian_like(h_prime_0)
        current_t = float(t.item())

        with torch.set_grad_enabled(True):
            h_prime_0 = h_prime_0.requires_grad_(True)
            t = t.requires_grad_(True)
            f = self.func_f(t, h_prime_0)
            divergence = self.divergence_fn(f, h_prime_0, e=noise).view(batchsize, 1)

        h_0 = self.func_map(f)

        # Save trajectory only at integer time steps (matching legacy behavior).
        # Legacy uses `self.time%1 == 0` which saves at t=0,1,2,...
        is_integer_time = abs(current_t - round(current_t)) < 1e-6
        if is_integer_time and current_t not in self._saved_times:
            self._saved_times.append(current_t)
            self._trajectory.append(h_0.detach())

        g = self.func_g(z0)
        dh_dhhat = self.func_map.linear.weight.unsqueeze(0).repeat(batchsize, 1, 1)
        dh_t = dh_dhhat @ f.unsqueeze(-1)
        out = (g @ dh_t).squeeze(-1)
        return (f, out, -divergence)

    def get_trajectory(self, device):
        if not self._trajectory:
            return torch.empty(0, device=device)
        return torch.stack(self._trajectory, dim=0).to(device)


# Backward-compatible alias used by upstream LEAP modules.
VectorField_Learnable = VectorFieldLearnable


def cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs):
    method = kwargs.get("method", "rk4")
    kwargs["method"] = method
    if method == "sym12async":
        kwargs.setdefault("atol", 1e-2)
        kwargs.setdefault("rtol", 1e-3)
    elif method == "dopri5":
        kwargs.setdefault("atol", 1e-12)
        kwargs.setdefault("rtol", 1e-6)
    elif method == "rk4":
        kwargs.setdefault("options", {})
        # Preserve the upstream LEAP integer-grid rk4 default for provenance.
        # The current deployment/release handler now aligns LEAP forward times to
        # the normalized coeff time base, so this fixed step_size only describes
        # the vendored upstream-style branch, not the active runtime time scale.
        kwargs["options"].setdefault("step_size", 1.0)

    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt batch dimensions must match z0.")
    vector_field = func(z0)
    if vector_field.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func batch dimensions must match z0.")
    if vector_field.size(-2) != z0.shape[-1]:
        raise ValueError("func hidden channels must match z0.")
    if vector_field.size(-1) != control_gradient.size(-1):
        raise ValueError("func input channels must match dX_dt output.")
    if control_gradient.requires_grad and adjoint:
        raise ValueError("Control gradients do not backpropagate with adjoint=True.")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)
    if kwargs["method"] == "sym12async":
        for i in range(len(t) - 1):
            kwargs["t0"] = t[i]
            kwargs["t1"] = t[i + 1]
            output = odesolve(vector_field, z0, kwargs)
            if i == 0:
                real_output = torch.cat((z0.unsqueeze(0), output.unsqueeze(0)), dim=0)
            else:
                real_output = torch.cat((real_output, output.unsqueeze(0)), dim=0)
            z0 = output
        return real_output
    return odeint(func=vector_field, y0=z0, t=t, **kwargs)


def cdeint_Learnable(
    dX_dt,
    path0,
    func_g,
    func_f,
    func_map,
    h_hat_0,
    z0,
    t,
    func_init,
    file=None,
    adjoint=True,
    **kwargs,
):
    del file
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorFieldLearnable(
        dX_dt=dX_dt,
        func_g=func_g,
        func_f=func_f,
        func_map=func_map,
        func_init=func_init,
        time_max=t.max().item(),
    )

    method = kwargs.get("method", "rk4")
    kwargs["method"] = method
    if method == "sym12async":
        kwargs.setdefault("atol", 1e-3)
        kwargs.setdefault("rtol", 1e-4)
    elif method == "dopri5":
        kwargs.setdefault("atol", 1e-9)
        kwargs.setdefault("rtol", 1e-7)
    elif method == "rk4":
        kwargs.setdefault("options", {})
        # Preserve the upstream LEAP integer-grid rk4 default for provenance.
        # The current deployment/release handler now aligns LEAP forward times to
        # the normalized coeff time base, so this fixed step_size only describes
        # the vendored upstream-style branch, not the active runtime time scale.
        kwargs["options"].setdefault("step_size", 1.0)

    if kwargs["method"] == "sym12async":
        for i in range(len(t) - 1):
            kwargs["t0"] = t[i]
            kwargs["t1"] = t[i + 1]
            logpz = torch.zeros(h_hat_0.shape[0], 1, requires_grad=True).to(h_hat_0)
            h0 = (z0, h_hat_0, logpz)
            output = odesolve(vector_field, h0, kwargs)
            if i == 0:
                front = torch.cat((h0[0].unsqueeze(0), output[0].unsqueeze(0)), dim=0)
                middle = torch.cat((h0[1].unsqueeze(0), output[1].unsqueeze(0)), dim=0)
                last = torch.cat((h0[2].unsqueeze(0), output[2].unsqueeze(0)), dim=0)
            else:
                front = torch.cat((front, output[0].unsqueeze(0)), dim=0)
                middle = torch.cat((middle, output[1].unsqueeze(0)), dim=0)
                last = torch.cat((last, output[2].unsqueeze(0)), dim=0)
            h0 = output
        real_output = (front, middle, last)
    else:
        logpz = torch.zeros(h_hat_0.shape[0], 1).to(path0)
        h0 = (z0, h_hat_0, logpz)
        real_output = odeint(func=vector_field, y0=h0, t=t, **kwargs)

    learn_path = vector_field.get_trajectory(path0.device)
    return (*real_output, learn_path)
