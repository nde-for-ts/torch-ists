"""CDEHandler for paper CDE-family models in runtime."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base import BaseHandler
from ..config import default_method_for_model as resolve_default_method_for_model
from ..module.cde import (
    FinalTanh as NeuralCDEFinalTanh,
    FinalTanhT,
    NeuralCDE,
)


class CDEHandler(BaseHandler):
    """Handler for CDE-family paper models.

    Supported models:
    - neuralcde
    - neuralcde-l
    - neuralcde-r
    - neuralcde-c
    - neuralcde-h
    """

    name: str = "cde"

    CDE_MODELS: Dict[str, Dict[str, Any]] = {
        "neuralcde": {"family": "neuralcde", "use_intensity": False, "control": "natural", "time_aware_vf": False},
        "neuralcde-l": {"family": "neuralcde", "use_intensity": False, "control": "linear", "time_aware_vf": True},
        "neuralcde-r": {"family": "neuralcde", "use_intensity": False, "control": "rectilinear", "time_aware_vf": True},
        "neuralcde-c": {"family": "neuralcde", "use_intensity": False, "control": "cubic", "time_aware_vf": True},
        "neuralcde-h": {"family": "neuralcde", "use_intensity": False, "control": "hermite", "time_aware_vf": True},
    }
    CDE_ALLOWED_METHODS = {"euler", "midpoint", "rk4", "dopri5"}
    CDE_METHOD_ALIASES = {
        "fixed_euler": "euler",
        "fixed_heun": "midpoint",
        "fixed_rk4": "rk4",
        "dopri": "dopri5",
    }

    def can_handle(self, model_name: str) -> bool:
        return model_name in self.CDE_MODELS

    @property
    def returns_loss(self) -> bool:
        # Per-model loss contract is resolved in ists_layer (has_auxiliary_loss).
        return False

    @staticmethod
    def _control_channels(input_dim: int, use_intensity: bool) -> int:
        return 1 + input_dim * (2 if use_intensity else 1)

    @staticmethod
    def _normalize_sequence_layout(tensor: torch.Tensor, batch_size: int, time_steps: int) -> torch.Tensor:
        if tensor.dim() != 3:
            return tensor
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[1] == batch_size:
            return tensor.transpose(0, 1)
        if tensor.shape[0] == time_steps:
            return tensor.transpose(0, 1)
        return tensor

    def create_model(self, model_name: str, config: Dict[str, Any], device: str) -> nn.Module:
        if model_name not in self.CDE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. CDEHandler supports: {list(self.CDE_MODELS.keys())}"
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
            allowed_methods=self.CDE_ALLOWED_METHODS,
            alias_map=self.CDE_METHOD_ALIASES,
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
        file_path = config.get("file")

        spec = self.CDE_MODELS[model_name]
        use_intensity = bool(spec["use_intensity"])
        control_channels = self._control_channels(input_dim, use_intensity)
        family = spec["family"]

        if family == "neuralcde":
            control_mode = str(config.get("control", spec["control"]))
            if bool(spec.get("time_aware_vf", False)):
                func = FinalTanhT(
                    input_channels=control_channels,
                    hidden_channels=hidden_dim,
                    hidden_hidden_channels=hidden_dim,
                    num_hidden_layers=num_layers,
                )
            else:
                func = NeuralCDEFinalTanh(
                    input_channels=control_channels,
                    hidden_channels=hidden_dim,
                    hidden_hidden_channels=hidden_dim,
                    num_hidden_layers=num_layers,
                )
            model = NeuralCDE(
                func=func,
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                output_channels=hidden_dim,
                initial=True,
                control=control_mode,
            )
        else:
            raise ValueError(f"Unsupported CDE family: {family}")

        model.model_name = model_name
        model.solver_method = method
        model.solver_options = solver_options
        model.solver_kwargs = solver_kwargs
        return model

    def _build_call_kwargs(self, model: nn.Module, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        model_params = self._coerce_mapping(kwargs.get("model_params"))
        model_name = kwargs.get("model_name", getattr(model, "model_name", ""))
        solver = self.resolve_solver_config(
            direct=kwargs,
            model_params=model_params,
            model=model,
            default_method=resolve_default_method_for_model(model_name),
            allowed_methods=self.CDE_ALLOWED_METHODS,
            alias_map=self.CDE_METHOD_ALIASES,
        )
        solver_options = dict(solver["options"])
        if (
            solver["step_size"] is not None
            and solver["method"] in {"euler", "midpoint", "rk4"}
            and "step_size" not in solver_options
        ):
            solver_options["step_size"] = solver["step_size"]

        model.solver_method = solver["method"]
        model.solver_options = solver_options
        model.solver_kwargs = {
            key: solver[key]
            for key in ("rtol", "atol", "adjoint", "dt")
            if solver[key] is not None
        }

        call_kwargs: Dict[str, Any] = {"method": solver["method"]}
        if solver_options:
            call_kwargs["options"] = solver_options
        if solver["rtol"] is not None:
            call_kwargs["rtol"] = solver["rtol"]
        if solver["atol"] is not None:
            call_kwargs["atol"] = solver["atol"]
        if solver["adjoint"] is not None:
            call_kwargs["adjoint"] = solver["adjoint"]

        params = inspect.signature(model.forward).parameters
        supports_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not supports_var_kw:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}
        return call_kwargs

    def _normalize_outputs(
        self,
        out: torch.Tensor,
        hn: torch.Tensor,
        batch_size: int,
        time_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self._normalize_sequence_layout(out, batch_size, time_steps)
        hn = self._normalize_sequence_layout(hn, batch_size, time_steps)

        if out.dim() == 2:
            out = out.unsqueeze(1)
        if hn.dim() == 3:
            hn = hn[:, -1, :]
        elif hn.dim() != 2:
            hn = out[:, -1, :]
        return out, hn

    def forward(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        coeffs: torch.Tensor | None,
        times: torch.Tensor,
        **kwargs: Any,
    ):
        if coeffs is None:
            raise ValueError("CDE models require interpolation coeffs.")

        data = self.preprocess_input(seq, times)
        call_kwargs = self._build_call_kwargs(model, kwargs)

        model_name = kwargs.get("model_name", getattr(model, "model_name", ""))
        out, hn = model(times, coeffs, data["final_index"], **call_kwargs)
        out, hn = self._normalize_outputs(out, hn, seq.shape[0], data["x"].shape[1])
        return out, hn
