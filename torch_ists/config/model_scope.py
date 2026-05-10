"""Model scope helpers for packaged configuration."""

from __future__ import annotations

import copy

from .active_models import load_active_models

ACTIVE_MODELS = load_active_models()


def _normalize_scope_token(raw: str) -> str:
    normalized = str(raw).strip().lower()
    if normalized in {"", "active"}:
        return "active"
    raise ValueError("This package supports only the active model scope.")


def get_active_model_list(scope: str = "active") -> list:
    _normalize_scope_token(scope)
    return copy.deepcopy(ACTIVE_MODELS)


def get_experiment_model_list(variant: str = "active") -> list:
    """Compatibility helper for older packaged integrations."""
    normalized = str(variant).strip().lower()
    if normalized in {"model_run", "param_search"}:
        normalized = "active"
    return get_active_model_list(normalized)
