"""Single-source alias policy for runtime-facing model names."""

from __future__ import annotations

FLOW_ALIAS_TO_CANONICAL = {
    "neuralflow": "neuralflow-r",
    "neuralflow_x_r": "neuralflow-r",
    "neuralflow_x_g": "neuralflow-g",
    "neuralflow_x_c": "neuralflow-c",
    "flowpath": "flowpath-r",
    "neuralflowcde_x_r": "flowpath-r",
    "neuralflowcde_x_g": "flowpath-g",
    "neuralflowcde_x_c": "flowpath-c",
    "flowpath_x_r": "flowpath-r",
    "flowpath_x_g": "flowpath-g",
    "flowpath_x_c": "flowpath-c",
    "dualdynamics": "dualdynamics-r",
    "dualdynamics_x_r": "dualdynamics-r",
    "dualdynamics_x_g": "dualdynamics-g",
    "dualdynamics_x_c": "dualdynamics-c",
    "neuralcontrolledflow_x_r": "dualdynamics-r",
    "neuralcontrolledflow_x_g": "dualdynamics-g",
    "neuralcontrolledflow_x_c": "dualdynamics-c",
}

TSDE_METHOD_ALIAS_TO_CANONICAL = {}

FLOW_CANONICAL_MODELS = frozenset(
    {
        "neuralflow-r",
        "neuralflow-g",
        "neuralflow-c",
        "flowpath-r",
        "flowpath-g",
        "flowpath-c",
        "dualdynamics-r",
        "dualdynamics-g",
        "dualdynamics-c",
    }
)

FLOW_STRICT_RAW_CANONICAL_MODELS = frozenset(
    {
        "neuralflow-r",
        "neuralflow-g",
        "neuralflow-c",
        "flowpath-r",
        "flowpath-g",
        "flowpath-c",
    }
)

FLOW_STRICT_RAW_ALIAS_MODELS = frozenset(
    {
        "neuralflow",
        "neuralflow_x_r",
        "neuralflow_x_g",
        "neuralflow_x_c",
        "flowpath",
        "neuralflowcde_x_r",
        "neuralflowcde_x_g",
        "neuralflowcde_x_c",
        "flowpath_x_r",
        "flowpath_x_g",
        "flowpath_x_c",
    }
)

TSDE_CANONICAL_MODELS = frozenset()


def normalize_flow_alias(name: str) -> str:
    """Normalize known flow aliases to canonical names."""
    raw_name = str(name)
    return FLOW_ALIAS_TO_CANONICAL.get(raw_name, raw_name)


def normalize_tsde_method_alias(name: str) -> str:
    """Normalize known TSDE labels to canonical names when present."""
    raw_name = str(name).strip()
    return TSDE_METHOD_ALIAS_TO_CANONICAL.get(raw_name.lower(), raw_name)


def is_flow_family_name(name: str) -> bool:
    """Return True for any supported flow canonical or alias model name."""
    raw_name = str(name)
    if raw_name in FLOW_ALIAS_TO_CANONICAL:
        return True
    return normalize_flow_alias(raw_name) in FLOW_CANONICAL_MODELS


def is_tsde_public_name(name: str) -> bool:
    """Return True for canonical TSDE names or supported alias labels."""
    raw_name = str(name).strip()
    if raw_name.lower() in TSDE_METHOD_ALIAS_TO_CANONICAL:
        return True
    return normalize_tsde_method_alias(raw_name) in TSDE_CANONICAL_MODELS


def is_flow_strict_raw_name(name: str) -> bool:
    """Return True when model must keep raw x_missing (no fill)."""
    raw_name = str(name)
    if raw_name in FLOW_STRICT_RAW_ALIAS_MODELS:
        return True
    return normalize_flow_alias(raw_name) in FLOW_STRICT_RAW_CANONICAL_MODELS


__all__ = [
    "FLOW_ALIAS_TO_CANONICAL",
    "TSDE_METHOD_ALIAS_TO_CANONICAL",
    "FLOW_CANONICAL_MODELS",
    "FLOW_STRICT_RAW_CANONICAL_MODELS",
    "FLOW_STRICT_RAW_ALIAS_MODELS",
    "TSDE_CANONICAL_MODELS",
    "normalize_flow_alias",
    "normalize_tsde_method_alias",
    "is_flow_family_name",
    "is_tsde_public_name",
    "is_flow_strict_raw_name",
]
