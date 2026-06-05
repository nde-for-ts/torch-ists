"""HandlerRegistry bootstrap wiring for the packaged runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Iterable

from ..config.models import (
    ATTENTION_MODELS,
    BASIC_MODELS,
    CDE_MODELS,
    FLOW_MODELS,
    ODE_MODELS,
    ODE_SOURCE_MODELS,
    SDE_BASELINE_MODELS,
    STABLE_SDE_MODELS,
)
from .base import BaseHandler
from .registry import HandlerRegistry

_OPTIONAL_DEPENDENCY_HINTS = {
    "torchdiffeq": "Install `torchdiffeq` or use `pip install \"torch-ists[all]\"`.",
    "torchcde": "Install `torchcde` or use `pip install \"torch-ists[all]\"`.",
    "torchsde": "Install `torchsde` or use `pip install \"torch-ists[all]\"`.",
    "torchdyn": "Install `torchdyn` or use `pip install \"torch-ists[ode]\"`.",
    "stribor": "Install `stribor==0.1.0` only for guarded flow-family NFE backends.",
}


def _build_optional_dependency_error(handler_name: str, exc: ModuleNotFoundError) -> ImportError:
    missing_name = getattr(exc, "name", "") or "unknown"
    if missing_name.startswith("torch_ists"):
        raise exc
    hint = _OPTIONAL_DEPENDENCY_HINTS.get(
        missing_name,
        f"Install `{missing_name}` and retry.",
    )
    return ImportError(
        f"Model family '{handler_name}' requires optional dependency '{missing_name}'. {hint}"
    )


class _LazyHandlerProxy(BaseHandler):
    def __init__(
        self,
        *,
        name: str,
        module_name: str,
        class_name: str,
        supported_models: Iterable[str] = (),
    ):
        self.name = name
        self._module_name = module_name
        self._class_name = class_name
        self._supported_models = frozenset(str(model) for model in supported_models)
        self._handler: BaseHandler | None = None

    def _load_handler(self) -> BaseHandler:
        if self._handler is not None:
            return self._handler
        try:
            module = import_module(self._module_name, __package__)
            handler_cls = getattr(module, self._class_name)
            self._handler = handler_cls()
        except ModuleNotFoundError as exc:
            raise _build_optional_dependency_error(self.name, exc) from exc
        return self._handler

    def can_handle(self, model_name: str) -> bool:
        return model_name in self._supported_models

    def create_model(self, model_name, config, device):
        return self._load_handler().create_model(model_name, config, device)

    def forward(self, model, seq, coeffs, times, **kwargs):
        return self._load_handler().forward(model, seq, coeffs, times, **kwargs)

    @property
    def returns_loss(self) -> bool:
        return self._load_handler().returns_loss

    @property
    def uses_filled(self) -> bool:
        return bool(getattr(self._load_handler(), "uses_filled", False))

    def __getattr__(self, name: str):
        return getattr(self._load_handler(), name)


def bootstrap_handler_registry() -> None:
    HandlerRegistry.clear()
    HandlerRegistry.register(
        _LazyHandlerProxy(
            name="basic",
            module_name=".basic_handler",
            class_name="BasicHandler",
            supported_models=BASIC_MODELS.keys(),
        )
    )
    HandlerRegistry.register(
        _LazyHandlerProxy(
            name="ode",
            module_name=".source_ode_handler",
            class_name="SourceODEHandler",
            supported_models=ODE_SOURCE_MODELS.keys(),
        )
    )
    HandlerRegistry.register(
        _LazyHandlerProxy(
            name="ode",
            module_name=".ode_handler",
            class_name="ODEHandler",
            supported_models=[
                name
                for name, cfg in ODE_MODELS.items()
                if cfg.get("handler") == "ODEHandler" and name != "ode-lstm"
            ],
        )
    )
    HandlerRegistry.register(
        _LazyHandlerProxy(
            name="odelstm",
            module_name=".odelstm_handler",
            class_name="ODELSTMHandler",
            supported_models=("ode-lstm",),
        )
    )
    HandlerRegistry.register(
        _LazyHandlerProxy(
            name="cde",
            module_name=".cde_handler",
            class_name="CDEHandler",
            supported_models=CDE_MODELS.keys(),
        )
    )
    HandlerRegistry.register(
        _LazyHandlerProxy(
            name="attention",
            module_name=".attention_handler",
            class_name="AttentionHandler",
            supported_models=ATTENTION_MODELS.keys(),
        )
    )
    HandlerRegistry.register(
        _LazyHandlerProxy(
            name="flow",
            module_name=".flow_handler",
            class_name="FlowHandler",
            supported_models=FLOW_MODELS.keys(),
        )
    )
    HandlerRegistry.register(
        _LazyHandlerProxy(
            name="sde",
            module_name=".sde_handler",
            class_name="SDEHandler",
            supported_models=tuple(SDE_BASELINE_MODELS.keys()) + tuple(STABLE_SDE_MODELS),
        )
    )


def configure_handler_registry_bootstrap() -> None:
    if hasattr(HandlerRegistry, "set_bootstrap"):
        HandlerRegistry.set_bootstrap(bootstrap_handler_registry, reset=True)
    else:  # pragma: no cover - compatibility fallback for stale runtime registry imports
        bootstrap_handler_registry()


__all__ = ["bootstrap_handler_registry", "configure_handler_registry_bootstrap"]
