"""Flow operator backend selector for runtime package."""

from __future__ import annotations

import torch.nn as nn

from .operators_native import (
    CouplingFlow as NativeCouplingFlow,
    GRUFlow as NativeGRUFlow,
    ResNetFlow as NativeResNetFlow,
)


AUTO_FALLBACK_REASON = "external_unavailable"


_FLOW_OPTION_TO_NATIVE_CLASS = {
    "r": NativeResNetFlow,
    "g": NativeGRUFlow,
    "c": NativeCouplingFlow,
}

_FLOW_BACKEND_ALIASES = {
    "native": "native",
    "stribor": "stribor",
    # Omission is deterministic across environments. ``auto`` remains an
    # explicit Stribor-first request with the documented narrow fallback.
    "default": "native",
    "auto": "auto",
    # Compatibility-only spellings. Runtime state and serialized configuration
    # use the public backend name ``stribor``.
    "nfe": "stribor",
    "nfe-stibor": "stribor",
    "nfe-stribor": "stribor",
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


def _stribor_class_for_flow_option(flow_option: str):
    from .operators_stribor import CouplingFlow as StriborCouplingFlow
    from .operators_stribor import GRUFlow as StriborGRUFlow
    from .operators_stribor import ResNetFlow as StriborResNetFlow

    return {
        "r": StriborResNetFlow,
        "g": StriborGRUFlow,
        "c": StriborCouplingFlow,
    }[flow_option]


def _build_native(option, dim, n_layers, hidden_dims):
    return _FLOW_OPTION_TO_NATIVE_CLASS[option](
        dim=int(dim),
        n_layers=max(1, int(n_layers)),
        hidden_dims=list(hidden_dims or []),
    )


def _build_stribor(option, dim, n_layers, hidden_dims):
    return _stribor_class_for_flow_option(option)(
        dim=int(dim),
        n_layers=max(1, int(n_layers)),
        hidden_dims=list(hidden_dims or []),
    )


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
        # Explicit package-authored operator.
        return _build_native(option, dim, n_layers, hidden_dims)

    if normalized_backend == "stribor":
        # Explicit Stribor backend; fail closed when unavailable so a requested
        # numerical implementation is never silently changed.
        return _build_stribor(option, dim, n_layers, hidden_dims)

    # Explicit ``auto`` tries the supported Stribor backend and falls back only
    # for an expected bootstrap availability condition. Constructor, numerical,
    # and training failures are intentionally not caught here.
    from .operators_stribor import StriborBackendUnavailableError

    try:
        return _build_stribor(option, dim, n_layers, hidden_dims)
    except StriborBackendUnavailableError:
        operator = _build_native(option, dim, n_layers, hidden_dims)
        operator.fallback_reason = AUTO_FALLBACK_REASON
        return operator


__all__ = [
    "AUTO_FALLBACK_REASON",
    "build_flow_operator",
    "resolve_flow_backend",
]
