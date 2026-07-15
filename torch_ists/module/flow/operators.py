"""Flow operator backend selector for runtime package."""

from __future__ import annotations

import torch.nn as nn

from .operators_native import (
    CouplingFlow as NativeCouplingFlow,
    GRUFlow as NativeGRUFlow,
    ResNetFlow as NativeResNetFlow,
)


_FLOW_OPTION_TO_NATIVE_CLASS = {
    "r": NativeResNetFlow,
    "g": NativeGRUFlow,
    "c": NativeCouplingFlow,
}

_FLOW_BACKEND_ALIASES = {
    "native": "native",
    "default": "native",
    "nfe": "nfe",
    "stribor": "nfe",
    "nfe-stibor": "nfe",
    "nfe-stribor": "nfe",
}


def resolve_flow_backend(value: object | None) -> str:
    if value is None:
        return "native"
    backend = str(value).strip().lower()
    normalized = _FLOW_BACKEND_ALIASES.get(backend)
    if normalized is None:
        raise ValueError(
            f"Unsupported flow backend: {value!r}. "
            f"Allowed backends: {sorted(set(_FLOW_BACKEND_ALIASES.values()))}"
        )
    return normalized


def _nfe_class_for_flow_option(flow_option: str):
    from .operators_nfe import CouplingFlow as NFECouplingFlow
    from .operators_nfe import GRUFlow as NFEGRUFlow
    from .operators_nfe import ResNetFlow as NFEResNetFlow

    return {
        "r": NFEResNetFlow,
        "g": NFEGRUFlow,
        "c": NFECouplingFlow,
    }[flow_option]


def build_flow_operator(
    *,
    flow_option: str,
    backend: str,
    dim: int,
    n_layers: int,
    hidden_dims: list[int] | None = None,
) -> nn.Module:
    option = str(flow_option).strip().lower()
    if option not in _FLOW_OPTION_TO_NATIVE_CLASS:
        raise ValueError(
            f"Unsupported flow_option: {flow_option!r}. Allowed: {sorted(_FLOW_OPTION_TO_NATIVE_CLASS)}"
        )

    normalized_backend = resolve_flow_backend(backend)
    if normalized_backend == "native":
        flow_cls = _FLOW_OPTION_TO_NATIVE_CLASS[option]
    else:
        flow_cls = _nfe_class_for_flow_option(option)

    return flow_cls(
        dim=int(dim),
        n_layers=max(1, int(n_layers)),
        hidden_dims=list(hidden_dims or []),
    )


__all__ = [
    "build_flow_operator",
    "resolve_flow_backend",
]
