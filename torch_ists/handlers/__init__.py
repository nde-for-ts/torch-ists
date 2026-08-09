"""Handler module for the torch_ists runtime package.

Exports:
- BaseHandler
- HandlerRegistry
- BasicHandler
- ODEHandler
- SourceODEHandler
- ODELSTMHandler
- CDEHandler
- FlowHandler
- VariationalSDEHandler
- SDEHandler
"""

from __future__ import annotations

from importlib import import_module

from .base import BaseHandler
from .registry import HandlerRegistry
from .bootstrap import configure_handler_registry_bootstrap

_LAZY_EXPORTS = {
    "BasicHandler": (".basic_handler", "BasicHandler"),
    "ODEHandler": (".ode_handler", "ODEHandler"),
    "SourceODEHandler": (".source_ode_handler", "SourceODEHandler"),
    "ODELSTMHandler": (".odelstm_handler", "ODELSTMHandler"),
    "CDEHandler": (".cde_handler", "CDEHandler"),
    "FlowHandler": (".flow_handler", "FlowHandler"),
    "VariationalSDEHandler": (".variational_sde_handler", "VariationalSDEHandler"),
    "SDEHandler": (".sde_handler", "SDEHandler"),
}

__all__ = [
    "BaseHandler",
    "HandlerRegistry",
    "BasicHandler",
    "ODEHandler",
    "SourceODEHandler",
    "ODELSTMHandler",
    "CDEHandler",
    "FlowHandler",
    "VariationalSDEHandler",
    "SDEHandler",
    "model_consumes_coefficients",
    "resolve_model_control_value_transform",
]


def model_consumes_coefficients(model_name: str) -> bool:
    """Return whether the routed model uses interpolation coefficients."""
    handler = HandlerRegistry.get_handler(str(model_name))
    return bool(handler.consumes_coefficients(str(model_name)))


def resolve_model_control_value_transform(model_name: str, requested: str | None):
    """Apply control scaling only to models that consume the generated control."""
    from torch_ists.preprocessing import normalize_control_value_transform

    normalized = normalize_control_value_transform(requested)
    consumes = model_consumes_coefficients(model_name)
    return (normalized if consumes else "identity"), consumes


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


configure_handler_registry_bootstrap()
