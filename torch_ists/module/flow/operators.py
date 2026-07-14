"""Flow operator construction for the runtime package."""

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

    resolve_flow_backend(backend)
    flow_cls = _FLOW_OPTION_TO_NATIVE_CLASS[option]

    return flow_cls(
        dim=int(dim),
        n_layers=max(1, int(n_layers)),
        hidden_dims=list(hidden_dims or []),
    )


__all__ = [
    "build_flow_operator",
    "resolve_flow_backend",
]
