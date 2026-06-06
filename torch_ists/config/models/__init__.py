"""Model registry for the runtime subset package."""

from typing import Dict, List, Set

from ..model_scope import ACTIVE_MODELS

try:
    from .basic import BASIC_MODELS as _ALL_BASIC_MODELS
except Exception:  # pragma: no cover - subset payload
    _ALL_BASIC_MODELS = {}

try:
    from .attention import ATTENTION_MODELS as _ALL_ATTENTION_MODELS
except Exception:  # pragma: no cover - subset payload
    _ALL_ATTENTION_MODELS = {}

try:
    from .cde import CDE_MODELS as _ALL_CDE_MODELS
except Exception:  # pragma: no cover - subset payload
    _ALL_CDE_MODELS = {}

try:
    from .sde_baseline import SDE_BASELINE_MODELS as _ALL_SDE_BASELINE_MODELS
except Exception:  # pragma: no cover - subset payload
    _ALL_SDE_BASELINE_MODELS = {}

try:
    from .ode import (
        ODE_MODELS as _ALL_ODE_MODELS,
        ODE_SOURCE_MODELS as _ALL_ODE_SOURCE_MODELS,
    )
except Exception:  # pragma: no cover - subset payload
    _ALL_ODE_MODELS = {}
    _ALL_ODE_SOURCE_MODELS = {}

try:
    from .stable_sdes import STABLE_SDE_CONFIGS as _ALL_STABLE_SDE_CONFIGS
except Exception:  # pragma: no cover - subset payload
    _ALL_STABLE_SDE_CONFIGS = {}

try:
    from .flow import FLOW_MODELS as _ALL_FLOW_MODELS
except Exception:  # pragma: no cover - subset payload
    _ALL_FLOW_MODELS = {}


def _select_active_models(model_group: Dict[str, Dict], active_set: Set[str]) -> Dict[str, Dict]:
    return {name: cfg for name, cfg in model_group.items() if name in active_set}


active_set = set(ACTIVE_MODELS)

BASIC_MODELS = _select_active_models(_ALL_BASIC_MODELS, active_set)
ATTENTION_MODELS = _select_active_models(_ALL_ATTENTION_MODELS, active_set)
CDE_MODELS = _select_active_models(_ALL_CDE_MODELS, active_set)
SDE_BASELINE_MODELS = _select_active_models(_ALL_SDE_BASELINE_MODELS, active_set)
ODE_MODELS = _select_active_models(_ALL_ODE_MODELS, active_set)
ODE_SOURCE_MODELS = _select_active_models(_ALL_ODE_SOURCE_MODELS, active_set)
STABLE_SDE_CONFIGS = _select_active_models(_ALL_STABLE_SDE_CONFIGS, active_set)
FLOW_MODELS = _select_active_models(_ALL_FLOW_MODELS, active_set)

VARIATIONAL_SDE_MODELS: Dict[str, Dict] = {}
TSDE_METHOD_CONFIGS: Dict[str, Dict] = {}
# Numeric SDE encodings are internal implementation details in the public package.
STABLE_SDE_LEGACY_ALIASES: Dict[str, str] = {}
STABLE_SDE_ALIASES: Dict[str, str] = STABLE_SDE_LEGACY_ALIASES
SDE_MODEL_ALIASES: Dict[str, str] = {}
SDE_LEGACY_MODEL_ALIASES: Dict[str, str] = SDE_MODEL_ALIASES
STABLE_SDE_MODELS: List[str] = list(STABLE_SDE_CONFIGS.keys())

MODEL_REGISTRY: Dict[str, Dict] = {
    **BASIC_MODELS,
    **ODE_MODELS,
    **ODE_SOURCE_MODELS,
    **CDE_MODELS,
    **ATTENTION_MODELS,
    **SDE_BASELINE_MODELS,
    **STABLE_SDE_CONFIGS,
    **FLOW_MODELS,
}

MODEL_LIST: List[str] = list(MODEL_REGISTRY.keys())
BASELINE_MODEL_LIST: List[str] = list(MODEL_REGISTRY.keys())

registry_set = set(MODEL_REGISTRY.keys())
if registry_set != active_set:
    missing = sorted(active_set - registry_set)
    extra = sorted(registry_set - active_set)
    raise AssertionError(
        "MODEL_REGISTRY is inconsistent with the packaged model registry. "
        f"missing={missing}, extra={extra}"
    )
__all__ = [
    "MODEL_REGISTRY",
    "MODEL_LIST",
    "STABLE_SDE_CONFIGS",
    "TSDE_METHOD_CONFIGS",
    "STABLE_SDE_ALIASES",
    "STABLE_SDE_LEGACY_ALIASES",
    "SDE_MODEL_ALIASES",
    "SDE_LEGACY_MODEL_ALIASES",
    "STABLE_SDE_MODELS",
    "BASIC_MODELS",
    "ODE_MODELS",
    "ODE_SOURCE_MODELS",
    "CDE_MODELS",
    "FLOW_MODELS",
    "SDE_BASELINE_MODELS",
    "VARIATIONAL_SDE_MODELS",
    "ATTENTION_MODELS",
    "BASELINE_MODEL_LIST",
]
