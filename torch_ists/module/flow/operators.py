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
    "stribor": "stribor",
    # The common default tries the reviewed Stribor backend first and falls back
    # to the package-authored native operator when the pinned checkout is absent
    # or invalid. ``default`` and an unset backend both resolve to ``auto``.
    "default": "auto",
    "auto": "auto",
    # Compatibility-only spellings. Runtime state and serialized configuration
    # use the public backend name ``stribor``.
    "nfe": "stribor",
    "nfe-stibor": "stribor",
    "nfe-stribor": "stribor",
}


def resolve_flow_backend(value: object | None) -> str:
    if value is None:
        return "auto"
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


def _build_native(option, dim, n_layers, hidden_dims):
    return _FLOW_OPTION_TO_NATIVE_CLASS[option](
        dim=int(dim),
        n_layers=max(1, int(n_layers)),
        hidden_dims=list(hidden_dims or []),
    )


def _build_stribor(option, dim, n_layers, hidden_dims):
    return _nfe_class_for_flow_option(option)(
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
        # Explicit reviewed reference backend; fail closed when unavailable so a
        # canonical/paper-facing selection is never silently downgraded.
        return _build_stribor(option, dim, n_layers, hidden_dims)

    # ``auto`` (the common default): try the reviewed Stribor backend, fall back
    # to native when the pinned checkout is absent or invalid. The fallback keeps
    # every flow model runnable without the external checkout; it is a deployment
    # convenience and is not a canonical comparison setting.
    from .nfe_reference import NFEInstallationError
    from .operators_nfe import StriborBackendUnavailableError

    try:
        return _build_stribor(option, dim, n_layers, hidden_dims)
    except (NFEInstallationError, StriborBackendUnavailableError):
        return _build_native(option, dim, n_layers, hidden_dims)


__all__ = [
    "build_flow_operator",
    "resolve_flow_backend",
]
