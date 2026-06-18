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
    NeuralRDE,
)


class CDEHandler(BaseHandler):
    """Handler for CDE-family paper models.

    Supported models:
    - neuralcde
    - neuralcde-l
    - neuralcde-r
    - neuralcde-c
    - neuralcde-h
    - ancde
    - exit
    - leap
    - neuralrde-1
    - neuralrde-2
    - neuralrde-3
    """

    name: str = "cde"

    CDE_MODELS: Dict[str, Dict[str, Any]] = {
        "neuralcde": {"family": "neuralcde", "use_intensity": False, "control": "natural", "time_aware_vf": False},
        "neuralcde-l": {"family": "neuralcde", "use_intensity": False, "control": "linear", "time_aware_vf": True},
        "neuralcde-r": {"family": "neuralcde", "use_intensity": False, "control": "rectilinear", "time_aware_vf": True},
        "neuralcde-c": {"family": "neuralcde", "use_intensity": False, "control": "cubic", "time_aware_vf": True},
        "neuralcde-h": {"family": "neuralcde", "use_intensity": False, "control": "hermite", "time_aware_vf": True},
        "ancde": {"family": "ancde", "use_intensity": False, "control": "natural"},
        "exit": {"family": "exit", "use_intensity": False, "control": "natural"},
        "leap": {"family": "leap", "use_intensity": False, "control": "natural"},
        "neuralrde-1": {"family": "neuralrde", "use_intensity": False, "control": "natural", "depth": 1},
        "neuralrde-2": {"family": "neuralrde", "use_intensity": False, "control": "natural", "depth": 2},
        "neuralrde-3": {"family": "neuralrde", "use_intensity": False, "control": "natural", "depth": 3},
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
        elif family == "ancde":
            from ..module.cde.ancde.ancde_model import ANCDE
            from ..module.cde.ancde.vector_fields import FinalTanh as ANCDEFinalTanh
            from ..module.cde.ancde.vector_fields import FinalTanh_ff6 as ANCDEFinalTanhFF6

            func_f = ANCDEFinalTanhFF6(
                input_channels=control_channels,
                hidden_atten_channels=hidden_dim,
                hidden_hidden_atten_channels=hidden_dim,
                num_hidden_layers=num_layers,
            )
            func_g = ANCDEFinalTanh(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                hidden_hidden_channels=hidden_dim,
                num_hidden_layers=num_layers,
            )
            model = ANCDE(
                func_f=func_f,
                func_g=func_g,
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                attention_channel=hidden_dim,
                output_channels=hidden_dim,
                file=file_path,
                initial=True,
            )
        elif family == "exit":
            from ..module.cde.exit.exit_model import NeuralCDE_IDEA4
            from ..module.cde.exit.vector_fields import FinalTanh_g as EXITFinalTanhG
            from ..module.cde.exit.vector_fields import ODEFunc_f as EXITODEFuncF

            func_k = NeuralCDEFinalTanh(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                hidden_hidden_channels=hidden_dim,
                num_hidden_layers=num_layers,
            )
            func_g = EXITFinalTanhG(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                hidden_hidden_channels=hidden_dim,
                num_hidden_layers=num_layers,
            )
            func_f = EXITODEFuncF(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                hidden_hidden_channels=hidden_dim,
                num_hidden_layers=num_layers,
            )
            model = NeuralCDE_IDEA4(
                func=func_k,
                func_f=func_f,
                func_g=func_g,
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                output_channels=hidden_dim,
                kinetic_energy_coef=1,
                jacobian_norm2_coef=1,
                div_samples=1,
                initial=True,
            )
        elif family == "leap":
            from ..module.cde.leap.leap_model import NeuralCDE_Learnable
            from ..module.cde.leap.vector_fields import Mapping_f as LEAPMappingF
            from ..module.cde.leap.vector_fields import ODEFunc_ff as LEAPODEFuncFF

            func_k = NeuralCDEFinalTanh(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                hidden_hidden_channels=hidden_dim,
                num_hidden_layers=num_layers,
            )
            func_g = NeuralCDEFinalTanh(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                hidden_hidden_channels=hidden_dim,
                num_hidden_layers=num_layers,
            )
            func_f = LEAPODEFuncFF(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
            )
            mapping = LEAPMappingF(
                input_channels=control_channels,
                hidden_channels=hidden_dim,
            )
            model = NeuralCDE_Learnable(
                func_k=func_k,
                func_g=func_g,
                func_f=func_f,
                mapping=mapping,
                input_channels=control_channels,
                hidden_channels=hidden_dim,
                output_channels=hidden_dim,
                file=file_path,
                initial=True,
            )
        elif family == "neuralrde":
            from signatory import logsignature_channels

            depth = int(spec["depth"])
            signature_channels = logsignature_channels(in_channels=control_channels, depth=depth)
            func = NeuralCDEFinalTanh(
                input_channels=signature_channels,
                hidden_channels=hidden_dim,
                hidden_hidden_channels=hidden_dim,
                num_hidden_layers=num_layers,
            )
            model = NeuralRDE(
                func=func,
                input_channels=signature_channels,
                hidden_channels=hidden_dim,
                output_channels=hidden_dim,
                initial=True,
            )
            model.rde_depth = depth
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
        if model_name == "leap":
            # Upstream LEAP assumes an integer-grid control path, while this
            # package builds coeffs on normalized [0, 1] times. Keep forward-time
            # semantics aligned with the packaged coeffs here.
            out, hn, loss = model(times, coeffs, data["final_index"], **call_kwargs)
            out, hn = self._normalize_outputs(out, hn, seq.shape[0], data["x"].shape[1])
            return out, hn, loss

        if model_name == "exit":
            out, hn, reg = model(times, coeffs, data["final_index"], **call_kwargs)
            out, hn = self._normalize_outputs(out, hn, seq.shape[0], data["x"].shape[1])
            return out, hn, reg

        if model_name.startswith("neuralrde-"):
            depth = int(getattr(model, "rde_depth", self.CDE_MODELS[model_name]["depth"]))
            out, hn = model(
                data["x"],
                times,
                coeffs,
                data["final_index"],
                interpolation=True,
                depth=depth,
                **call_kwargs,
            )
            out, hn = self._normalize_outputs(out, hn, seq.shape[0], data["x"].shape[1])
            return out, hn

        out, hn = model(times, coeffs, data["final_index"], **call_kwargs)
        out, hn = self._normalize_outputs(out, hn, seq.shape[0], data["x"].shape[1])
        return out, hn
