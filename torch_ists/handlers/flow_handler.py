"""FlowHandler for packaged NeuralFlow runtime."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseHandler
from ..config.model_data_policy import _resolve_runtime_depth_config
from ..module.flow.neuralflow import NeuralFlow
from ..module.flow.operators import resolve_flow_backend


class FlowHandler(BaseHandler):
    """Handler for canonical `neuralflow-{r,g,c}`, `flowpath-{r,g,c}`, and `dualdynamics-{r,g,c}` models."""

    name: str = "flow"
    _FLOW_OPTIONS = ("r", "g", "c")
    _NEURALFLOW_MODELS = frozenset(f"neuralflow-{flow}" for flow in _FLOW_OPTIONS)
    _FLOWPATH_MODELS = frozenset(f"flowpath-{flow}" for flow in _FLOW_OPTIONS)
    _DUALDYNAMICS_MODELS = frozenset(f"dualdynamics-{flow}" for flow in _FLOW_OPTIONS)
    _SUPPORTED_MODELS = _NEURALFLOW_MODELS | _FLOWPATH_MODELS | _DUALDYNAMICS_MODELS
    _FLOW_SOLVER_DEFAULT_METHOD = "rk4"
    _FLOW_ALLOWED_METHODS = {"euler", "midpoint", "rk4", "dopri5"}
    _FLOW_METHOD_ALIASES = {
        "fixed_euler": "euler",
        "fixed_heun": "midpoint",
        "fixed_rk4": "rk4",
        "dopri": "dopri5",
    }

    @staticmethod
    def _resolve_flow_backend(config: Dict[str, Any], model_params: Dict[str, Any]) -> str:
        return resolve_flow_backend(config.get("flow_backend", model_params.get("flow_backend")))

    def can_handle(self, model_name: str) -> bool:
        return model_name in self._SUPPORTED_MODELS

    @property
    def returns_loss(self) -> bool:
        return False

    @staticmethod
    def _control_channels(input_dim: int, use_intensity: bool) -> int:
        del use_intensity
        # x-only comparator uses [t, x] regardless of intensity path.
        return 1 + input_dim

    @staticmethod
    def _validate_raw_missing_contract(x: torch.Tensor, seq_mask: torch.Tensor) -> None:
        """Enforce raw-missing contract on direct layer path.

        If mask indicates missing points and any corresponding x value is finite,
        input is out of contract (raw missing must remain NaN at missing positions).
        """
        missing_positions = seq_mask == 0
        if not bool(missing_positions.any()):
            return

        missing_x = x[missing_positions]
        if missing_x.numel() == 0:
            return

        if torch.isfinite(missing_x).any():
            raise ValueError(
                "flow comparator/path requires raw x_missing on the x channel. "
                "Detected one or more finite values at missing positions in seq_mask "
                "(likely pre-filled input such as x_filled/nan_to_num)."
            )

    def _build_solver_call_kwargs(self, model: nn.Module, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        model_params = self._coerce_mapping(kwargs.get("model_params"))
        solver = self.resolve_solver_config(
            direct=kwargs,
            model_params=model_params,
            model=model,
            default_method=self._FLOW_SOLVER_DEFAULT_METHOD,
            allowed_methods=self._FLOW_ALLOWED_METHODS,
            alias_map=self._FLOW_METHOD_ALIASES,
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
        adjoint_params = kwargs.get("adjoint_params")
        if adjoint_params is None:
            adjoint_params = model_params.get("adjoint_params")
        if adjoint_params is not None:
            call_kwargs["adjoint_params"] = adjoint_params
        return call_kwargs

    def create_model(self, model_name: str, config: Dict[str, Any], device: str) -> nn.Module:
        if model_name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. FlowHandler supports: {sorted(self._SUPPORTED_MODELS)}"
            )

        input_dim = int(config["input_dim"])
        hidden_dim = int(config["hidden_dim"])
        depth_config = _resolve_runtime_depth_config(
            model_name,
            num_layers=config.get("num_layers", 1),
            num_hidden_layers=config.get("num_hidden_layers"),
            head_depth=config.get("head_depth"),
            flow_num_layers=config.get("flow_num_layers"),
            flow_depth=config.get("flow_depth"),
        )
        head_depth = int(depth_config["head_depth"])
        flow_depth = int(depth_config["flow_depth"])
        num_hidden_layers = int(depth_config["num_hidden_layers"])
        flow_num_layers = int(depth_config["flow_num_layers"])
        use_intensity = bool(config.get("use_intensity", False))
        coeff_dim = self._control_channels(input_dim, use_intensity)
        flow_option = model_name.rsplit("-", 1)[-1]
        model_params = self._coerce_mapping(config.get("model_params"))
        flow_backend = self._resolve_flow_backend(config, model_params)

        if model_name in self._NEURALFLOW_MODELS:
            model = NeuralFlow(
                input_channels=coeff_dim,
                hidden_channels=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                head_depth=head_depth,
                flow_num_layers=flow_num_layers,
                flow_depth=flow_depth,
                output_channels=hidden_dim,
                input_option="x",
                flow_option=flow_option,
                flow_backend=flow_backend,
            )
        elif model_name in self._FLOWPATH_MODELS:
            from ..module.cde import FinalTanhT
            from ..module.flow.flowpath import FlowPath

            solver = self.resolve_solver_config(
                direct=config,
                model_params=model_params,
                model=None,
                default_method=self._FLOW_SOLVER_DEFAULT_METHOD,
                allowed_methods=self._FLOW_ALLOWED_METHODS,
                alias_map=self._FLOW_METHOD_ALIASES,
            )
            func = FinalTanhT(
                input_channels=hidden_dim,
                hidden_channels=hidden_dim,
                hidden_hidden_channels=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )
            model = FlowPath(
                input_channels=coeff_dim,
                hidden_channels=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                head_depth=head_depth,
                flow_num_layers=flow_num_layers,
                flow_depth=flow_depth,
                output_channels=hidden_dim,
                input_option="x",
                flow_option=flow_option,
                flow_backend=flow_backend,
                func=func,
            )
            model.solver_method = solver["method"]
            model.solver_options = dict(solver["options"])
            model.solver_kwargs = {
                key: solver[key]
                for key in ("rtol", "atol", "adjoint", "dt")
                if solver[key] is not None
            }
        else:
            from ..module.cde import FinalTanhT
            from ..module.flow.dualdynamics import DualDynamics

            solver = self.resolve_solver_config(
                direct=config,
                model_params=model_params,
                model=None,
                default_method=self._FLOW_SOLVER_DEFAULT_METHOD,
                allowed_methods=self._FLOW_ALLOWED_METHODS,
                alias_map=self._FLOW_METHOD_ALIASES,
            )
            func = FinalTanhT(
                input_channels=coeff_dim,
                hidden_channels=hidden_dim,
                hidden_hidden_channels=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )
            model = DualDynamics(
                input_channels=coeff_dim,
                hidden_channels=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                head_depth=head_depth,
                flow_num_layers=flow_num_layers,
                flow_depth=flow_depth,
                output_channels=hidden_dim,
                input_option="x",
                flow_option=flow_option,
                flow_backend=flow_backend,
                func=func,
            )
            model.solver_method = solver["method"]
            model.solver_options = dict(solver["options"])
            model.solver_kwargs = {
                key: solver[key]
                for key in ("rtol", "atol", "adjoint", "dt")
                if solver[key] is not None
            }

        model.model_name = model_name
        return model

    def forward(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        coeffs: torch.Tensor | None,
        times: torch.Tensor,
        **kwargs: Any,
    ):
        data = self.preprocess_input(seq, times)
        runtime_model_name = str(getattr(model, "model_name", kwargs.get("model_name", "")))
        strict_raw_models = self._NEURALFLOW_MODELS | self._FLOWPATH_MODELS
        if runtime_model_name in strict_raw_models:
            self._validate_raw_missing_contract(data["x"], data["seq_mask"])

        call_kwargs: Dict[str, Any] = {}
        solver_runtime_models = self._FLOWPATH_MODELS | self._DUALDYNAMICS_MODELS
        if runtime_model_name in solver_runtime_models:
            call_kwargs = self._build_solver_call_kwargs(model, kwargs)

        out, hn = model(
            data["x"],
            data["seq_ts"],
            data["seq_mask"],
            coeffs,
            times,
            **call_kwargs,
        )

        if out.dim() == 2:
            out = out.unsqueeze(1)
        if hn.dim() == 3:
            hn = hn[:, -1, :]
        elif hn.dim() != 2:
            hn = out[:, -1, :]

        return out, hn
