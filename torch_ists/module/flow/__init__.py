"""Flow module exports for runtime package."""

from __future__ import annotations

from importlib import import_module

from .operators import resolve_flow_backend

_LAZY_EXPORTS = {
    "NeuralFlow": (".neuralflow", "NeuralFlow"),
    "FlowPath": (".flowpath", "FlowPath"),
    "FusionCDE": (".fusioncde", "FusionCDE"),
    "DualDynamics": (".dualdynamics", "DualDynamics"),
}

__all__ = ["NeuralFlow", "FlowPath", "FusionCDE", "DualDynamics", "resolve_flow_backend"]


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
