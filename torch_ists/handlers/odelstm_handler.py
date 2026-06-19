"""ODELSTMHandler for ode-lstm model (Pattern-B routing)."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseHandler
from ..config import default_method_for_model as resolve_default_method_for_model


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
    _UNSUPPORTED_SOLVER_KEYS = ("dt", "step_size", "rtol", "atol", "adjoint")

    @classmethod
    def _reject_unsupported_solver_surface(
        cls,
        direct: Dict[str, Any],
        model_params: Dict[str, Any],
    ) -> None:
        explicit: list[str] = []
        for key in cls._UNSUPPORTED_SOLVER_KEYS:
            if key in direct or key in model_params:
                explicit.append(key)
        for key in ("options", "solver_options"):
            value = direct.get(key)
            if value is None:
                value = model_params.get(key)
            if isinstance(value, dict) and value:
                explicit.append(key)
        if not explicit:
            return
        unique = ", ".join(dict.fromkeys(explicit))
        raise ValueError(
            "ode-lstm only supports explicit solver method at model creation. "
            f"Unsupported solver knobs: {unique}."
        )

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
        self._reject_unsupported_solver_surface(config, model_params)
        solver = self.resolve_solver_config(
            direct=config,
            model_params=model_params,
            model=None,
            default_method=resolve_default_method_for_model(model_name),
            allowed_methods=self.ODELSTM_ALLOWED_METHODS,
            alias_map=self.ODELSTM_METHOD_ALIASES,
        )
        method = solver["method"]
        solver_type = self.ODELSTM_TO_SOLVER[method]

        model = ODELSTM(
            in_features=input_dim,
            hidden_size=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            out_feature=hidden_dim,
            solver_type=solver_type,
        )
        model.model_name = model_name
        model.solver_method = method
        model.solver_type = solver_type
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
        self._reject_unsupported_solver_surface(kwargs, model_params)
        model_name = kwargs.get("model_name", getattr(model, "model_name", "ode-lstm"))
        solver = self.resolve_solver_config(
            direct=kwargs,
            model_params=model_params,
            model=model,
            default_method=resolve_default_method_for_model(model_name),
            allowed_methods=self.ODELSTM_ALLOWED_METHODS,
            alias_map=self.ODELSTM_METHOD_ALIASES,
        )
        requested = solver["method"]
        configured = self._normalize_solver_method(
            getattr(model, "solver_method", resolve_default_method_for_model(model_name))
        )
        if requested != configured:
            raise ValueError(
                "ode-lstm solver method is fixed at model creation. "
                f"Requested '{requested}', configured '{configured}'."
            )
        data = self.preprocess_input(seq, times)
        out, hn = model(data["x"], data["seq_ts"], data["seq_mask"])

        if out.dim() == 2:
            out = out.unsqueeze(1)
        if hn.dim() == 3:
            hn = self.gather_output_at_index(hn, data["final_index"])
        elif hn.dim() != 2:
            hn = self.gather_output_at_index(out, data["final_index"])
        return out, hn
