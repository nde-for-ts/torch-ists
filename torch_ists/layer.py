"""Standalone runtime for ``ists_layer``.

Architecture:
- Handler pattern: HandlerRegistry routes model creation to handlers.
- Supported handlers: BasicHandler + SourceODEHandler + ODEHandler + ODELSTMHandler + CDEHandler + FlowHandler + VariationalSDEHandler + DSDEHandler + TSDEHandler + SDEHandler + AttentionHandler.
- Standalone: self-contained handlers and module implementations.

Behavior notes:
- ``use_intensity=None`` resolves from ``get_data_config(model_name)``.
- Active release scope is enforced through packaged runtime metadata.
- Non-finite hidden states and sequence outputs hard-fail with ``RuntimeError`` instead of being silently masked.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional
import warnings

import torch
import torch.nn as nn

from .handlers.registry import HandlerRegistry
from .handlers.bootstrap import configure_handler_registry_bootstrap
from .config.active_models import load_active_models
from .config.alias_policy import (
    is_flow_family_name,
    normalize_flow_alias,
)
from .config.model_data_policy import (
    _resolve_runtime_depth_config,
    get_data_config,
    has_auxiliary_loss,
)
from .config import default_method_for_model as resolve_default_method_for_model
from .config.models import SDE_MODEL_ALIASES
from .config.runtime_fault_policy import (
    handle_auxiliary_loss_nan,
    raise_for_nonfinite_tensor,
)

configure_handler_registry_bootstrap()


LEGACY_SDE_NAME_PATTERN = re.compile(r"^neuralsde_\d+_\d+$")


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "CUDA requested but not available; fallback to CPU.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "cpu"
    return device


def _enforce_active_scope_or_raise(model_name: str, *, allow_flow_extension: bool) -> None:
    del allow_flow_extension
    active_models = set(load_active_models())
    if not active_models:
        raise RuntimeError(
            "Active model scope registry is empty. Reinstall the package or contact the maintainer."
        )
    if model_name not in active_models:
        warnings.warn(
            f"Model '{model_name}' is not available in the current package scope. "
            "Use get_active_model_list() to see supported models.",
            RuntimeWarning,
            stacklevel=3,
        )
        raise ValueError(
            f"Model '{model_name}' is not available in this package. "
            "Use get_active_model_list() to see the list of available models."
        )


def _normalize_model_name_or_raise(model_name: str) -> str:
    # Alias normalization SoT:
    # - flow aliases: config.alias_policy.normalize_flow_alias
    # - SDE aliases: config.models.SDE_MODEL_ALIASES
    if model_name == "latentsde-kl":
        raise ValueError(
            "Model 'latentsde-kl' is not available in this package. "
            "This preview package does not include LatentSDE. "
            "Use get_active_model_list() to see the list of available models."
        )
    normalized_flow_name = normalize_flow_alias(model_name)
    if normalized_flow_name != model_name:
        return normalized_flow_name
    alias = SDE_MODEL_ALIASES.get(model_name)
    if alias is not None:
        return alias
    if LEGACY_SDE_NAME_PATTERN.fullmatch(model_name):
        raise ValueError(
            f"Legacy SDE name {model_name} is not available in this package. "
            f"Allowed aliases: {sorted(SDE_MODEL_ALIASES)}"
        )
    return model_name


def _resolve_use_intensity_flag(
    model_name: str,
    requested_use_intensity: Optional[bool],
) -> bool:
    if requested_use_intensity is not None:
        return bool(requested_use_intensity)
    return bool(get_data_config(model_name)["use_intensity"])


def _default_method_for_model(model_name: str) -> str:
    return resolve_default_method_for_model(model_name)


class ists_layer(nn.Module):
    """Standalone ists_layer for supported models."""

    def __init__(
        self,
        model_name: str = "cnn",
        input_dim: int = 3,
        seq_len: int = 128,
        hidden_dim: int = 32,
        hidden_hidden_dim: int = 32,
        num_layers: int = 1,
        num_hidden_layers: int = 1,
        head_depth: Optional[int] = None,
        flow_depth: Optional[int] = None,
        flow_num_layers: Optional[int] = None,
        bidirectional: bool = False,
        dropout: float = 0.1,
        use_intensity: Optional[bool] = None,
        method: Optional[str] = None,
        file: Optional[str] = None,
        device: str = "cpu",
        model_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        normalized_model_name = _normalize_model_name_or_raise(model_name)
        _enforce_active_scope_or_raise(normalized_model_name, allow_flow_extension=True)

        self.model_name = normalized_model_name
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.hidden_hidden_dim = int(hidden_hidden_dim)
        self.bidirectional = bool(bidirectional)
        self.dropout = float(dropout)
        self.use_intensity = _resolve_use_intensity_flag(
            self.model_name,
            use_intensity,
        )
        self.method = method
        self.file = file
        self.model_params = dict(model_params or {})
        depth_config = _resolve_runtime_depth_config(
            self.model_name,
            num_layers=self.model_params.get("num_layers", num_layers),
            num_hidden_layers=self.model_params.get("num_hidden_layers", num_hidden_layers),
            head_depth=self.model_params.get("head_depth", head_depth),
            flow_num_layers=self.model_params.get("flow_num_layers", flow_num_layers),
            flow_depth=self.model_params.get("flow_depth", flow_depth),
        )
        self.num_layers = int(depth_config["num_layers"])
        self.num_hidden_layers = int(depth_config["num_hidden_layers"])
        self.head_depth = int(depth_config["head_depth"])
        self.flow_depth = int(depth_config["flow_depth"])
        self.flow_num_layers = self.flow_depth

        self.device = _resolve_device(device)

        handler = HandlerRegistry.get_handler(self.model_name)
        config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "seq_len": self.seq_len,
            "num_layers": self.num_layers,
            "num_hidden_layers": self.num_hidden_layers,
            "head_depth": self.head_depth,
            "flow_num_layers": self.flow_num_layers,
            "flow_depth": self.flow_depth,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
            "hidden_hidden_dim": self.hidden_hidden_dim,
            "use_intensity": self.use_intensity,
            "method": self.method,
            "file": self.file,
            "model_params": dict(self.model_params),
        }
        config.update(self.model_params)

        self.seq_layer = handler.create_model(
            model_name=self.model_name,
            config=config,
            device=self.device,
        )
        self.returns_loss = has_auxiliary_loss(self.model_name)
        self.uses_filled = getattr(handler, "uses_filled", False)
        self.aux_nan_event_count = 0

        self.seq_layer.to(self.device)

    def forward(self, seq: torch.Tensor, coeffs: Optional[torch.Tensor] = None, **kwargs):
        if seq.ndim != 4:
            raise ValueError(f"seq must be [N, 3, L, D], got shape {tuple(seq.shape)}")

        seq = seq.permute(0, 1, 3, 2)
        L = seq.shape[3]
        times = torch.linspace(0, 1, L).to(seq.device)

        handler = HandlerRegistry.get_handler(self.model_name)
        forward_kwargs = dict(kwargs)
        direct_method = forward_kwargs.pop("method", None)
        # Reserve internal routing keys to avoid duplicate keyword dispatch.
        forward_kwargs.pop("model_name", None)
        forward_kwargs.pop("model_params", None)
        if "solver_options" not in forward_kwargs and "options" not in forward_kwargs:
            if "solver_options" in self.model_params:
                forward_kwargs["solver_options"] = self.model_params["solver_options"]
            elif "options" in self.model_params:
                forward_kwargs["options"] = self.model_params["options"]
        for solver_key in ("step_size", "dt", "rtol", "atol", "adjoint", "adjoint_params"):
            if solver_key not in forward_kwargs and solver_key in self.model_params:
                forward_kwargs[solver_key] = self.model_params[solver_key]

        method = (
            direct_method
            or self.method
            or self.model_params.get("method")
            or _default_method_for_model(self.model_name)
        )
        result = handler.forward(
            self.seq_layer,
            seq,
            coeffs,
            times,
            model_name=self.model_name,
            method=method,
            model_params=dict(self.model_params),
            **forward_kwargs,
        )

        if isinstance(result, tuple) and len(result) == 3:
            out, hn, loss = result
        else:
            out, hn = result
            loss = None

        raise_for_nonfinite_tensor(
            out,
            tensor_name="layer output",
            model_name=self.model_name,
        )
        raise_for_nonfinite_tensor(
            hn,
            tensor_name="layer hidden state",
            model_name=self.model_name,
        )
        if loss is not None and torch.isnan(loss).any():
            self.aux_nan_event_count += 1
            loss = handle_auxiliary_loss_nan(
                loss,
                model_name=self.model_name,
                event_count=self.aux_nan_event_count,
            )

        if self.returns_loss:
            if loss is None:
                loss = torch.tensor(0.0, device=out.device, dtype=out.dtype)
            return out, hn, loss
        return out, hn
