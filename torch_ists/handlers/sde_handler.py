"""SDEHandler for the current packaged SDE runtime models."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseHandler
from ..config import (
    MODEL_REGISTRY,
    default_method_for_model as resolve_default_method_for_model,
    get_default_config,
    merge_configs,
)
from ..config.schema import resolve_diffusion_to_numeric, resolve_drift_to_numeric
from ..module.sde import SDEModel


class SDEHandler(BaseHandler):
    """Handler for the packaged SDE runtime family.

    Supported models:
    - neural_lsde, neural_lnsde, neural_gsde
    - naive_cde, neural_ode, neural_ode_t
    - neural_sde, neural_sde_t
    """

    name: str = "sde"

    SDE_MODELS = {
        "neural_lsde",
        "neural_lnsde",
        "neural_gsde",
        "naive_cde",
        "neural_ode",
        "neural_ode_t",
        "neural_sde",
        "neural_sde_t",
    }
    SDE_ALLOWED_METHODS = {"euler", "heun", "milstein", "srk"}
    SDE_METHOD_ALIASES = {
        "fixed_euler": "euler",
        "fixed_heun": "heun",
    }
    STOCHASTIC_KWARG_KEYS = {"bm", "names", "extra", "logqp"}

    @staticmethod
    def _reject_explicit_adjoint(
        model_name: str,
        direct: Dict[str, Any],
        model_params: Dict[str, Any],
    ) -> None:
        if "adjoint" in direct or "adjoint" in model_params:
            raise ValueError(f"{model_name} does not support explicit solver adjoint")

    def can_handle(self, model_name: str) -> bool:
        return model_name in self.SDE_MODELS

    @property
    def returns_loss(self) -> bool:
        return False

    @staticmethod
    def _control_channels(input_dim: int, use_intensity: bool) -> int:
        return 1 + input_dim * (2 if use_intensity else 1)

    @staticmethod
    def _extract_nested_overrides(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        override: Dict[str, Dict[str, Any]] = {}
        for key in ("drift", "diffusion", "solver", "data", "extended"):
            value = config.get(key)
            if isinstance(value, dict):
                override[key] = deepcopy(value)

        model_params = config.get("model_params")
        if isinstance(model_params, dict):
            for key in ("drift", "diffusion", "solver", "data", "extended"):
                value = model_params.get(key)
                if isinstance(value, dict):
                    override[key] = merge_configs(override.get(key, {}), value)
        return override

    @staticmethod
    def _materialize_solver_kwargs(solver_config: Dict[str, Any]) -> Dict[str, Any]:
        solver_kwargs: Dict[str, Any] = {}
        for key in ("dt", "rtol", "atol", "adjoint"):
            if key not in solver_config:
                continue
            value = solver_config[key]
            if value is None or value == "auto":
                continue
            solver_kwargs[key] = value
        return solver_kwargs

    @staticmethod
    def _extract_solver_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
        solver_kwargs: Dict[str, Any] = {}

        model_params = config.get("model_params") if isinstance(config.get("model_params"), dict) else {}

        for key in ("dt", "rtol", "atol", "adjoint", "step_size"):
            if key in config:
                solver_kwargs[key] = config[key]
            elif key in model_params:
                solver_kwargs[key] = model_params[key]
        if "dt" not in solver_kwargs and "step_size" in solver_kwargs:
            solver_kwargs["dt"] = solver_kwargs["step_size"]

        return solver_kwargs

    def create_model(self, model_name: str, config: Dict[str, Any], device: str) -> nn.Module:
        if not self.can_handle(model_name):
            raise ValueError(
                f"Unknown model: {model_name}. SDEHandler supports: {sorted(self.SDE_MODELS)}"
            )

        input_dim = int(config["input_dim"])
        hidden_dim = int(config["hidden_dim"])
        num_hidden_layers = max(1, int(config.get("num_hidden_layers", config.get("num_layers", 1))))
        use_intensity = bool(config.get("use_intensity", False))

        registry_key = model_name
        merged_config = get_default_config("sde")
        merged_config = merge_configs(merged_config, deepcopy(MODEL_REGISTRY.get(registry_key, {})))
        merged_config = merge_configs(merged_config, self._extract_nested_overrides(config))

        drift_config = merged_config.get("drift", {})
        diffusion_config = merged_config.get("diffusion", {})
        solver_config = deepcopy(merged_config.get("solver", {}))

        model_params = self._coerce_mapping(config.get("model_params"))
        self._reject_explicit_adjoint(model_name, config, model_params)
        resolved = self.resolve_solver_config(
            direct=config,
            model_params=model_params,
            model=None,
            default_method=resolve_default_method_for_model(model_name),
            allowed_methods=self.SDE_ALLOWED_METHODS,
            alias_map=self.SDE_METHOD_ALIASES,
        )
        solver_config["method"] = resolved["method"]
        solver_config["options"] = dict(resolved["options"])
        if resolved["dt"] is not None:
            solver_config["dt"] = resolved["dt"]
        elif solver_config.get("dt") == "auto":
            solver_config.pop("dt", None)
        if resolved["rtol"] is not None:
            solver_config["rtol"] = resolved["rtol"]
        if resolved["atol"] is not None:
            solver_config["atol"] = resolved["atol"]
        if resolved["adjoint"] is not None:
            solver_config["adjoint"] = resolved["adjoint"]

        solver_kwargs = self._materialize_solver_kwargs(solver_config)

        input_option = resolve_drift_to_numeric(drift_config)
        noise_option = resolve_diffusion_to_numeric(diffusion_config)
        coeff_dim = self._control_channels(input_dim, use_intensity)

        model = SDEModel(
            coeff_dim=coeff_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            input_option=input_option,
            noise_option=noise_option,
            method=solver_config.get("method"),
            solver_options=solver_config.get("options"),
            solver_kwargs=solver_kwargs,
        )
        model.model_name = model_name
        model.solver_method = solver_config.get("method")
        model.solver_options = dict(solver_config.get("options") or {})
        model.solver_kwargs = dict(solver_kwargs)
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
        if coeffs is None:
            raise ValueError("SDE models require interpolation coeffs.")

        data = self.preprocess_input(seq, times)
        device = next(model.parameters()).device
        coeffs = coeffs.to(device)
        times = times.to(device)

        kwargs.pop("model_name", None)
        model_params = self._coerce_mapping(kwargs.pop("model_params", {}))
        self._reject_explicit_adjoint(str(getattr(model, "model_name", "")), kwargs, model_params)
        resolved = self.resolve_solver_config(
            direct=kwargs,
            model_params=model_params,
            model=model,
            default_method=resolve_default_method_for_model(str(getattr(model, "model_name", ""))),
            allowed_methods=self.SDE_ALLOWED_METHODS,
            alias_map=self.SDE_METHOD_ALIASES,
        )

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
        if resolved["adjoint"] is not None:
            call_kwargs["adjoint"] = resolved["adjoint"]
        for key in self.STOCHASTIC_KWARG_KEYS:
            if key in kwargs and key not in call_kwargs:
                call_kwargs[key] = kwargs[key]

        model.solver_method = resolved["method"]
        model.solver_options = solver_options
        model.solver_kwargs = {
            key: resolved[key]
            for key in ("dt", "rtol", "atol", "adjoint")
            if resolved[key] is not None
        }

        out, hn = model.seq_layer(coeffs, times, **call_kwargs)

        if out.dim() == 2:
            out = out.unsqueeze(1)
        if hn.dim() == 3:
            hn = self.gather_output_at_index(hn, data["final_index"])
        elif hn.dim() != 2:
            hn = self.gather_output_at_index(out, data["final_index"])

        return out, hn
