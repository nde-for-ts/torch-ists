"""VariationalSDEHandler for LatentSDE runtime."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseHandler
from ..module.sde import LatentSDEModel


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "false", "no", "off"}:
            return False
        if lowered in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


class VariationalSDEHandler(BaseHandler):
    """Handler for LatentSDE model.

    Policy:
    - Supported model name: `latentsde` only
    - KL behavior toggled by model_params/config `kl` flag
    - default solver method is Euler with direct differentiation
    - explicit `dt`, `step_size`, and `adjoint` values are validated by the runtime
    """

    name: str = "variational_sde"
    VSDE_ALLOWED_METHODS = {"euler", "milstein", "srk"}
    VSDE_METHOD_ALIASES = {
        "fixed_euler": "euler",
    }

    def can_handle(self, model_name: str) -> bool:
        return model_name == "latentsde"

    @staticmethod
    def _validate_solver_combination(method: str, adjoint: bool) -> None:
        if method == "srk" and adjoint:
            raise ValueError(
                "LatentSDE does not support adjoint=True with method='srk'; "
                "use Euler adjoint or SRK direct integration."
            )

    @property
    def returns_loss(self) -> bool:
        return True

    def create_model(self, model_name: str, config: Dict[str, Any], device: str) -> nn.Module:
        if not self.can_handle(model_name):
            raise ValueError(
                "VariationalSDEHandler supports only ['latentsde']."
            )

        model_params = self._coerce_mapping(config.get("model_params"))
        input_dim = int(config["input_dim"])
        hidden_dim = int(config["hidden_dim"])
        num_hidden_layers = max(1, int(config.get("num_hidden_layers", config.get("num_layers", 3))))

        kl = _coerce_bool(config.get("kl", True), default=True)
        theta = float(config.get("theta", 1.0))
        mu = float(config.get("mu", 0.0))
        sigma = float(config.get("sigma", 0.5))
        kl_weight = float(config.get("kl_weight", 1e-5))
        recognition_hidden = int(config.get("recognition_hidden", hidden_dim))

        model = LatentSDEModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            kl=kl,
            theta=theta,
            mu=mu,
            sigma=sigma,
            kl_weight=kl_weight,
            recognition_hidden=recognition_hidden,
        )

        model.model_name = model_name
        solver = self.resolve_solver_config(
            direct=config,
            model_params=model_params,
            model=None,
            default_method="euler",
            allowed_methods=self.VSDE_ALLOWED_METHODS,
            alias_map=self.VSDE_METHOD_ALIASES,
        )
        solver_adjoint = _coerce_bool(solver["adjoint"], False)
        self._validate_solver_combination(solver["method"], solver_adjoint)
        model.solver_method = solver["method"]
        model.solver_options = dict(solver["options"])
        model.solver_kwargs = {
            key: value
            for key, value in {
                "dt": solver["dt"],
                "rtol": solver["rtol"],
                "atol": solver["atol"],
                "adjoint": solver_adjoint,
            }.items()
            if value is not None
        }
        model.to(device)
        return model

    def forward(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        coeffs: torch.Tensor | None,
        times: torch.Tensor,
        **kwargs: Any,
    ):
        del coeffs

        data = self.preprocess_input(seq, times)
        kwargs.pop("model_name", None)
        model_params = self._coerce_mapping(kwargs.pop("model_params", {}))
        resolved = self.resolve_solver_config(
            direct=kwargs,
            model_params=model_params,
            model=model,
            default_method="euler",
            allowed_methods=self.VSDE_ALLOWED_METHODS,
            alias_map=self.VSDE_METHOD_ALIASES,
        )
        resolved_adjoint = _coerce_bool(resolved["adjoint"], False)
        self._validate_solver_combination(resolved["method"], resolved_adjoint)

        call_kwargs: Dict[str, Any] = {"method": resolved["method"]}
        solver_options = dict(resolved["options"])
        if solver_options:
            call_kwargs["options"] = solver_options
        if resolved["dt"] is not None:
            call_kwargs["dt"] = resolved["dt"]
        if resolved["rtol"] is not None:
            call_kwargs["rtol"] = resolved["rtol"]
        if resolved["atol"] is not None:
            call_kwargs["atol"] = resolved["atol"]
        call_kwargs["adjoint"] = resolved_adjoint

        model.solver_method = resolved["method"]
        model.solver_options = solver_options
        model.solver_kwargs = {
            key: value
            for key, value in {
                "dt": resolved["dt"],
                "rtol": resolved["rtol"],
                "atol": resolved["atol"],
                "adjoint": resolved_adjoint,
            }.items()
            if value is not None
        }

        device = next(model.parameters()).device
        values = data["x"].to(device)
        mask = seq[:, 1, :, :].permute(0, 2, 1).to(device)
        delta = seq[:, 2, :, :].permute(0, 2, 1).to(device)
        final_index = data["final_index"].to(device)
        times = times.to(device)
        result = model.seq_layer(
            values,
            mask,
            delta,
            times,
            final_index,
            **call_kwargs,
        )

        if not isinstance(result, tuple):
            raise RuntimeError("LatentSDE forward must return tuple.")

        if len(result) == 3:
            out, hn, loss = result
            if out.dim() == 2:
                out = out.unsqueeze(1)
            if hn.dim() == 3:
                hn = self.gather_output_at_index(hn, data["final_index"])
            elif hn.dim() != 2:
                hn = self.gather_output_at_index(out, data["final_index"])
            return out, hn, loss

        if len(result) == 2:
            out, hn = result
            if out.dim() == 2:
                out = out.unsqueeze(1)
            if hn.dim() == 3:
                hn = self.gather_output_at_index(hn, data["final_index"])
            elif hn.dim() != 2:
                hn = self.gather_output_at_index(out, data["final_index"])
            return out, hn, out.new_zeros(())

        raise RuntimeError("LatentSDE forward must return 2-tuple or 3-tuple.")


__all__ = ["VariationalSDEHandler"]
