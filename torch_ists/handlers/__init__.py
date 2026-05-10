"""Handler module for the torch_ists runtime package.

Exports:
- BaseHandler
- HandlerRegistry
- BasicHandler
- ODEHandler
- SourceODEHandler
- ODELSTMHandler
- CDEHandler
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
    "SDEHandler",
]


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
