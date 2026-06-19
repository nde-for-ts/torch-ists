"""ODEHandler for paper ODE-family models in runtime."""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseHandler
from ..config import default_method_for_model as resolve_default_method_for_model
from ..module.ode import GRU_D, GRU_ODE, GRU_dt, NeuralCDE, ODERNN


class ODEHandler(BaseHandler):
    """Handler for ODE-family paper models.

    Supported models:
    - gru-dt
    - gru-d
    - ode-rnn-spline
    - gru-ode
    """

    name: str = "ode"

    ODE_MODELS: Dict[str, Dict[str, Any]] = {
        "gru-dt": {"family": "gru_dt", "use_intensity": True},
        "gru-d": {"family": "gru_d", "use_intensity": True},
        "ode-rnn-spline": {"family": "ode_rnn", "use_intensity": True},
        "gru-ode": {"family": "gru_ode", "use_intensity": False},
    }
    ODE_ALLOWED_METHODS = {"euler", "midpoint", "rk4", "dopri5"}
    ODE_METHOD_ALIASES = {
        "fixed_euler": "euler",
        "fixed_heun": "midpoint",
        "fixed_rk4": "rk4",
        "dopri": "dopri5",
    }
    _INERT_SOLVER_MODELS = frozenset({"gru-d", "gru-dt"})
    _INERT_SOLVER_KEYS = ("method", "dt", "step_size", "rtol", "atol", "adjoint")

    @classmethod
    def _collect_explicit_solver_keys(
        cls,
        direct: Dict[str, Any],
        model_params: Dict[str, Any],
    ) -> list[str]:
        explicit: list[str] = []
        for key in cls._INERT_SOLVER_KEYS:
            if direct.get(key) is not None or model_params.get(key) is not None:
                explicit.append(key)

        for key in ("options", "solver_options"):
            value = direct.get(key)
            if value is None:
                value = model_params.get(key)
            if isinstance(value, dict) and value:
                explicit.append(key)
        return explicit

    @classmethod
    def _warn_inert_solver_surface(
        cls,
        model_name: str,
        direct: Dict[str, Any],
        model_params: Dict[str, Any],
    ) -> None:
        if model_name not in cls._INERT_SOLVER_MODELS:
            return
        explicit = cls._collect_explicit_solver_keys(direct, model_params)
        if not explicit:
            return
        if explicit == ["method"]:
            direct_method = direct.get("method")
            model_params_method = model_params.get("method")
            if (
                model_params_method is None
                and direct_method == resolve_default_method_for_model(model_name)
            ):
                return
        unique = ", ".join(dict.fromkeys(explicit))
        warnings.warn(
            f"{model_name} exposes solver-shaped compatibility knobs, but they have no "
            f"effect on the current runtime: {unique}.",
            RuntimeWarning,
            stacklevel=3,
        )

    def can_handle(self, model_name: str) -> bool:
        return model_name in self.ODE_MODELS

    @property
    def returns_loss(self) -> bool:
        return False

    @staticmethod
    def _control_channels(input_dim: int, use_intensity: bool) -> int:
        return 1 + input_dim * (2 if use_intensity else 1)

    def create_model(self, model_name: str, config: Dict[str, Any], device: str) -> nn.Module:
        if model_name not in self.ODE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. ODEHandler supports: {list(self.ODE_MODELS.keys())}"
            )

        input_dim = int(config["input_dim"])
        hidden_dim = int(config["hidden_dim"])
        num_layers = max(1, int(config.get("num_hidden_layers", config.get("num_layers", 1))))
        model_params = self._coerce_mapping(config.get("model_params"))
        solver = self.resolve_solver_config(
            direct=config,
            model_params=model_params,
            model=None,
            default_method=resolve_default_method_for_model(model_name),
            allowed_methods=self.ODE_ALLOWED_METHODS,
            alias_map=self.ODE_METHOD_ALIASES,
        )
        method = solver["method"]
        solver_options = dict(solver["options"])
        if (
            solver["step_size"] is not None
            and method in {"euler", "midpoint", "rk4"}
            and "step_size" not in solver_options
        ):
            solver_options["step_size"] = solver["step_size"]
        solver_kwargs = {
            key: solver[key]
            for key in ("rtol", "atol", "adjoint", "dt")
            if solver[key] is not None
        }
        self._warn_inert_solver_surface(model_name, config, model_params)

        spec = self.ODE_MODELS[model_name]
        use_intensity = bool(spec["use_intensity"])
        control_channels = self._control_channels(input_dim, use_intensity)
        family = spec["family"]

        if family == "gru_dt":
            model = GRU_dt(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                output_channels=hidden_dim,
                use_intensity=use_intensity,
            )
        elif family == "gru_d":
            model = GRU_D(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                output_channels=hidden_dim,
                use_intensity=use_intensity,
            )
        elif family == "ode_rnn":
            model = ODERNN(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                output_channels=hidden_dim,
                hidden_hidden_channels=hidden_dim,
                num_hidden_layers=num_layers,
                use_intensity=use_intensity,
                solver_method=method,
                solver_options=solver_options,
                solver_kwargs=solver_kwargs,
            )
        elif family == "gru_ode":
            vector_field = GRU_ODE(input_channels=control_channels, hidden_channels=hidden_dim)
            model = NeuralCDE(
                func=vector_field,
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                output_channels=hidden_dim,
                initial=True,
            )
        else:
            raise ValueError(f"Unsupported ODE family: {family}")

        model.model_name = model_name
        model.solver_method = method
        model.solver_options = solver_options
        model.solver_kwargs = solver_kwargs
        return model

    def forward(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        coeffs: torch.Tensor | None,
        times: torch.Tensor,
        **kwargs: Any,
    ):
        if coeffs is None:
            raise ValueError("ODE models require interpolation coeffs.")

        data = self.preprocess_input(seq, times)
        model_params = self._coerce_mapping(kwargs.get("model_params"))
        model_name = kwargs.get("model_name", getattr(model, "model_name", ""))
        solver = self.resolve_solver_config(
            direct=kwargs,
            model_params=model_params,
            model=model,
            default_method=resolve_default_method_for_model(model_name),
            allowed_methods=self.ODE_ALLOWED_METHODS,
            alias_map=self.ODE_METHOD_ALIASES,
        )
        solver_options = dict(solver["options"])
        if (
            solver["step_size"] is not None
            and solver["method"] in {"euler", "midpoint", "rk4"}
            and "step_size" not in solver_options
        ):
            solver_options["step_size"] = solver["step_size"]

        runtime_solver_kwargs = {
            key: solver[key]
            for key in ("rtol", "atol", "adjoint", "dt")
            if solver[key] is not None
        }
        self._warn_inert_solver_surface(str(model_name), kwargs, model_params)
        if model_name == "ode-rnn-spline":
            model.solver_method = solver["method"]
            model.solver_options = solver_options
            model.solver_kwargs = runtime_solver_kwargs

        call_kwargs: Dict[str, Any] = {}
        call_kwargs["method"] = solver["method"]
        if solver_options:
            call_kwargs["options"] = solver_options
        if solver["rtol"] is not None:
            call_kwargs["rtol"] = solver["rtol"]
        if solver["atol"] is not None:
            call_kwargs["atol"] = solver["atol"]
        if solver["adjoint"] is not None:
            call_kwargs["adjoint"] = solver["adjoint"]

        # Only pass solver kwargs that the concrete model forward supports.
        params = inspect.signature(model.forward).parameters
        supports_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not supports_var_kw:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}

        out, hn = model(times, coeffs, data["final_index"], **call_kwargs)

        if out.dim() == 2:
            out = out.unsqueeze(1)
        if hn.dim() == 3:
            hn = self.gather_output_at_index(hn, data["final_index"])
        elif hn.dim() != 2:
            hn = self.gather_output_at_index(out, data["final_index"])

        return out, hn
