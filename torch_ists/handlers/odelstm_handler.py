"""ODELSTMHandler for ode-lstm model (Pattern-B routing)."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseHandler
from ..config import default_method_for_model as resolve_default_method_for_model
from ..preprocessing import apply_imputation


class ODELSTMHandler(BaseHandler):
    """Dedicated handler for ode-lstm.

    This keeps ODEHandler clean for Pattern-A models:
    - gru-dt, gru-d, ode-rnn-spline, gru-ode
    """

    name: str = "odelstm"
    # Keep `heun` here to match the upstream ode-lstm reference implementation.
    # Do not normalize this set to ODEHandler's generic torchdiffeq policy.
    ODELSTM_ALLOWED_METHODS = {"euler", "heun", "rk4", "dopri5"}
    ODELSTM_METHOD_ALIASES = {
        "fixed_euler": "euler",
        "fixed_heun": "heun",
        "fixed_rk4": "rk4",
        "dopri": "dopri5",
    }
    ODELSTM_TO_SOLVER = {
        "euler": "fixed_euler",
        "heun": "fixed_heun",
        "rk4": "fixed_rk4",
        "dopri5": "dopri5",
    }
    _TORCHDYN_SOLVER_KWARGS = {
        "atol",
        "rtol",
        "sensitivity",
        "solver_adjoint",
        "atol_adjoint",
        "rtol_adjoint",
        "interpolator",
        "seminorm",
    }
    _FIXED_STEP_KEYS = ("fixed_step_count", "solver_substeps", "substeps")

    @classmethod
    def _resolve_odelstm_runtime(
        cls,
        *,
        direct: Dict[str, Any],
        model_params: Dict[str, Any],
        model: nn.Module | None,
        default_method: str,
    ) -> Dict[str, Any]:
        solver = cls().resolve_solver_config(
            direct=direct,
            model_params=model_params,
            model=model,
            default_method=default_method,
            allowed_methods=cls.ODELSTM_ALLOWED_METHODS,
            alias_map=cls.ODELSTM_METHOD_ALIASES,
        )
        method = solver["method"]
        solver_type = cls.ODELSTM_TO_SOLVER[method]
        solver_options = dict(solver["options"])

        solver_kwargs: Dict[str, Any] = {}
        if solver["rtol"] is not None:
            solver_kwargs["rtol"] = float(solver["rtol"])
        if solver["atol"] is not None:
            solver_kwargs["atol"] = float(solver["atol"])
        if solver["adjoint"] is not None:
            solver_kwargs["sensitivity"] = "adjoint" if bool(solver["adjoint"]) else "autograd"
        for key in cls._TORCHDYN_SOLVER_KWARGS:
            if key in solver_options:
                solver_kwargs[key] = solver_options[key]
        existing_solver_kwargs = getattr(model, "solver_kwargs", {}) if model is not None else {}
        if isinstance(existing_solver_kwargs, dict):
            for key in cls._TORCHDYN_SOLVER_KWARGS:
                if key in existing_solver_kwargs and key not in solver_kwargs:
                    solver_kwargs[key] = existing_solver_kwargs[key]

        step_size = solver["step_size"]
        fixed_step_count = solver_options.get("fixed_step_count")
        for key in cls._FIXED_STEP_KEYS:
            if fixed_step_count is None and key in direct:
                fixed_step_count = direct[key]
            if fixed_step_count is None and key in model_params:
                fixed_step_count = model_params[key]
            if fixed_step_count is None and key in solver_options:
                fixed_step_count = solver_options[key]
        if fixed_step_count is None and step_size is not None:
            if method == "dopri5":
                raise ValueError("ode-lstm step_size/dt requires a fixed solver method: euler, heun, or rk4.")
            try:
                fixed_step_count = int(math.ceil(1.0 / max(float(step_size), 1e-12)))
            except (TypeError, ValueError) as exc:
                raise ValueError("ode-lstm step_size/dt must be positive.") from exc
        if fixed_step_count is None and model is not None:
            fixed_step_count = getattr(model, "fixed_step_count", None)
        if fixed_step_count is None:
            fixed_step_count = 3
        fixed_step_count = max(1, int(fixed_step_count))

        unsupported_options = sorted(
            key
            for key in solver_options
            if key not in cls._TORCHDYN_SOLVER_KWARGS and key not in {"step_size", *cls._FIXED_STEP_KEYS}
        )
        if unsupported_options:
            raise ValueError(
                "Unsupported ode-lstm solver option(s): " + ", ".join(unsupported_options)
            )

        if method != "dopri5" and solver_kwargs:
            raise ValueError("ode-lstm tolerance/sensitivity kwargs require adaptive dopri5.")

        return {
            "method": method,
            "solver_type": solver_type,
            "solver_kwargs": solver_kwargs,
            "fixed_step_count": fixed_step_count,
        }

    def can_handle(self, model_name: str) -> bool:
        return model_name == "ode-lstm"

    @property
    def returns_loss(self) -> bool:
        return False

    def create_model(self, model_name: str, config: Dict[str, Any], device: str) -> nn.Module:
        if model_name != "ode-lstm":
            raise ValueError("ODELSTMHandler only supports model_name='ode-lstm'")

        from ..module.ode.odelstm import ODELSTM

        input_dim = int(config["input_dim"])
        hidden_dim = int(config["hidden_dim"])
        num_hidden_layers = max(
            1,
            int(config.get("num_hidden_layers", config.get("num_layers", 1))),
        )
        model_params = self._coerce_mapping(config.get("model_params"))
        solver_runtime = self._resolve_odelstm_runtime(
            direct=config,
            model_params=model_params,
            model=None,
            default_method=resolve_default_method_for_model(model_name),
        )

        model = ODELSTM(
            in_features=input_dim,
            hidden_size=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            out_feature=hidden_dim,
            solver_type=solver_runtime["solver_type"],
            solver_kwargs=solver_runtime["solver_kwargs"],
            fixed_step_count=solver_runtime["fixed_step_count"],
        )
        model.model_name = model_name
        model.solver_method = solver_runtime["method"]
        model.solver_type = solver_runtime["solver_type"]
        model.solver_kwargs = dict(solver_runtime["solver_kwargs"])
        model.fixed_step_count = int(solver_runtime["fixed_step_count"])
        return model

    def forward(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        coeffs: torch.Tensor | None,
        times: torch.Tensor,
        **kwargs: Any,
    ):
        _ = coeffs
        model_params = self._coerce_mapping(kwargs.get("model_params"))
        model_name = kwargs.get("model_name", getattr(model, "model_name", "ode-lstm"))
        solver_runtime = self._resolve_odelstm_runtime(
            direct=kwargs,
            model_params=model_params,
            model=model,
            default_method=resolve_default_method_for_model(model_name),
        )
        requested = solver_runtime["method"]
        configured = self._normalize_solver_method(
            getattr(model, "solver_method", resolve_default_method_for_model(model_name))
        )
        if requested != configured:
            raise ValueError(
                "ode-lstm solver method is fixed at model creation. "
                f"Requested '{requested}', configured '{configured}'."
            )
        if dict(solver_runtime["solver_kwargs"]) != dict(getattr(model, "solver_kwargs", {})):
            raise ValueError("ode-lstm solver kwargs are fixed at model creation.")
        if int(solver_runtime["fixed_step_count"]) != int(getattr(model, "fixed_step_count", 3)):
            raise ValueError("ode-lstm fixed_step_count is fixed at model creation.")
        data = self.preprocess_input(seq, times)
        x = apply_imputation(data["x"], kwargs.get("imputation_strategy", "zero"))
        out, hn = model(x, data["seq_delta"], data["seq_mask"])

        if out.dim() == 2:
            out = out.unsqueeze(1)
        if hn.dim() == 3:
            hn = self.gather_output_at_index(hn, data["final_index"])
        elif hn.dim() != 2:
            hn = self.gather_output_at_index(out, data["final_index"])
        return out, hn
