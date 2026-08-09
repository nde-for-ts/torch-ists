"""Handler for source ODE method variants."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseHandler
from ..config import default_method_for_model as resolve_default_method_for_model
from ..module.ode import (
    SourceGRUODEBayesBackbone,
    SourceLatentODEBackbone,
    SourceNJODEBackbone,
    SourceODERNNBackbone,
    _reconstruct_sample_times,
)


class SourceODEHandler(BaseHandler):
    """Handler for source ODE method variants."""

    name: str = "ode"

    SOURCE_MODELS: Dict[str, str] = {
        "ode-rnn": "odernn",
        "latentode": "latentode",
        "gru-ode-bayes": "gru_ode_bayes",
        "njode": "njode",
    }
    ODE_ALLOWED_METHODS = {"euler", "midpoint", "rk4", "dopri5"}
    ODE_METHOD_ALIASES = {
        "fixed_euler": "euler",
        "fixed_heun": "midpoint",
        "fixed_rk4": "rk4",
        "dopri": "dopri5",
    }
    DOPRI5_MAX_NUM_STEPS = 10_000

    @classmethod
    def _allowed_methods_for_model(cls, model_name: str) -> set[str]:
        if model_name in {"gru-ode-bayes", "njode"}:
            return {"euler"}
        return set(cls.ODE_ALLOWED_METHODS)

    @classmethod
    def _with_operational_solver_options(cls, method: str, options: Dict[str, Any]) -> Dict[str, Any]:
        resolved = dict(options)
        if method == "dopri5" and "max_num_steps" not in resolved:
            resolved["max_num_steps"] = cls.DOPRI5_MAX_NUM_STEPS
        return resolved

    @classmethod
    def _resolved_solver_options(cls, solver: Dict[str, Any]) -> Dict[str, Any]:
        options = dict(solver["options"])
        if solver["step_size"] is not None:
            options.setdefault("step_size", float(solver["step_size"]))
        return cls._with_operational_solver_options(solver["method"], options)

    def can_handle(self, model_name: str) -> bool:
        return model_name in self.SOURCE_MODELS

    @property
    def returns_loss(self) -> bool:
        # The layer resolves model-specific auxiliary losses from model policy.
        return False

    def create_model(self, model_name: str, config: Dict[str, Any], device: str) -> nn.Module:
        input_dim = int(config["input_dim"])
        hidden_dim = int(config["hidden_dim"])
        num_layers = max(1, int(config.get("num_hidden_layers", config.get("num_layers", 1))))
        model_params = self._coerce_mapping(config.get("model_params"))
        solver = self.resolve_solver_config(
            direct=config,
            model_params=model_params,
            model=None,
            default_method=resolve_default_method_for_model(model_name),
            allowed_methods=self._allowed_methods_for_model(model_name),
            alias_map=self.ODE_METHOD_ALIASES,
        )
        solver_kwargs = {
            key: solver[key]
            for key in ("rtol", "atol", "adjoint", "dt")
            if solver[key] is not None
        }
        solver_options = self._resolved_solver_options(solver)
        family = self.SOURCE_MODELS[model_name]
        if family == "odernn":
            model = SourceODERNNBackbone(input_dim, hidden_dim, hidden_dim, num_layers)
        elif family == "latentode":
            model = SourceLatentODEBackbone(
                input_dim,
                hidden_dim,
                hidden_dim,
                num_layers,
                observation_std=float(config.get("observation_std", config.get("obsrv_std", 0.01))),
                kl_weight=float(config.get("kl_weight", 1.0)),
                auxiliary_weight=float(config.get("auxiliary_weight", 1e-4)),
            )
        elif family == "gru_ode_bayes":
            configured_step_size = config.get("step_size")
            if configured_step_size is None:
                configured_step_size = config.get("dt")
            if configured_step_size is None:
                configured_step_size = 0.05
            model = SourceGRUODEBayesBackbone(
                input_dim,
                hidden_dim,
                hidden_dim,
                p_hidden=config.get("p_hidden"),
                prep_hidden=int(config.get("prep_hidden", 4)),
                step_size=float(configured_step_size),
                observation_noise_std=float(config.get("observation_noise_std", 0.01)),
                mixing=float(config.get("mixing", 1.0)),
                auxiliary_weight=float(config.get("auxiliary_weight", 1e-3)),
                max_substeps=int(config.get("max_substeps", 10_000)),
            )
        elif family == "njode":
            configured_step_size = config.get("step_size")
            if configured_step_size is None:
                configured_step_size = config.get("dt")
            if configured_step_size is None:
                configured_step_size = 0.05
            model = SourceNJODEBackbone(
                input_dim,
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                step_size=float(configured_step_size),
                loss_weight=float(config.get("loss_weight", 0.5)),
                auxiliary_weight=float(config.get("auxiliary_weight", 1.0)),
                loss_eps=float(config.get("loss_eps", 1e-10)),
                max_substeps=int(config.get("max_substeps", 10_000)),
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported source ODE family: {family}")
        model.model_name = model_name
        model.solver_method = solver["method"]
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
        del coeffs
        data = self.preprocess_input(
            seq,
            times,
            sample_times=kwargs.get("sample_times"),
        )
        model_params = self._coerce_mapping(kwargs.get("model_params"))
        model_name = kwargs.get("model_name", getattr(model, "model_name", ""))
        solver = self.resolve_solver_config(
            direct=kwargs,
            model_params=model_params,
            model=model,
            default_method=resolve_default_method_for_model(model_name),
            allowed_methods=self._allowed_methods_for_model(model_name),
            alias_map=self.ODE_METHOD_ALIASES,
        )
        runtime_solver_kwargs = {
            key: solver[key]
            for key in ("rtol", "atol", "adjoint", "dt")
            if solver[key] is not None
        }
        solver_options = self._resolved_solver_options(solver)
        model.solver_method = solver["method"]
        model.solver_options = solver_options
        model.solver_kwargs = runtime_solver_kwargs

        x = data["x"]
        mask = seq[:, 1, :, :].permute(0, 2, 1)
        if kwargs.get("sample_times") is None:
            step_times = _reconstruct_sample_times(data["seq_delta"], times, data["seq_mask"])
        else:
            step_times = data["seq_ts"]
        final_index = self.get_final_index(mask.max(dim=-1).values, x.size(1))
        return model(
            x,
            mask,
            step_times,
            final_index,
            method=solver["method"],
            options=solver_options,
            rtol=solver["rtol"],
            atol=solver["atol"],
            adjoint=solver["adjoint"],
        )


__all__ = ["SourceODEHandler"]
