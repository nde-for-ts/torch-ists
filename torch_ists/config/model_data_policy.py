"""Model-to-data preprocessing policy for runtime config.

Note:
    Flow alias/raw-missing policy authority moved to ``config.alias_policy``.
    This module keeps compatibility wrappers for existing imports.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .alias_policy import (
    is_flow_family_name,
    is_flow_strict_raw_name,
    normalize_flow_alias,
    normalize_tsde_method_alias,
)

_NATURAL_INTERPOLATION_MODELS = frozenset(
    [
        "gru-dt",
        "gru-d",
        "gru-ode",
        "ode-rnn-spline",
        "ode-rnn",
        "latentode",
        "gru-ode-bayes",
        "neuralcde",
    ]
)

_HERMITE_INTERPOLATION_MODELS = frozenset(
    [
        "neural_lsde",
        "neural_lnsde",
        "neural_gsde",
        "naive_cde",
        "neural_ode",
        "neural_ode_t",
        "neural_sde",
        "neural_sde_t",
        "neuralflow-r",
        "neuralflow-g",
        "neuralflow-c",
        "flowpath-r",
        "flowpath-g",
        "flowpath-c",
        "dualdynamics-r",
        "dualdynamics-g",
        "dualdynamics-c",
    ]
)

_CUBIC_INTERPOLATION_MODELS = frozenset(
    [
        "cnn",
        "rnn",
        "lstm",
        "gru",
        "gru-simple",
        "bilstm",
        "plstm",
        "tlstm",
        "tglstm",
        "transformer",
        "transformer-mask",
        "miam",
        "mtan",
        "sand",
        "ode-lstm",
    ]
)

_INTERPOLATION_OVERRIDES = {
    "neuralcde-l": "linear",
    "neuralcde-r": "rectilinear",
    "neuralcde-c": "cubic",
    "neuralcde-h": "hermite",
}

_INTENSITY_MODELS = frozenset(
    [
        "gru-dt",
        "gru-d",
        "ode-rnn-spline",
    ]
)

_BASIC_SEQUENCE_MODELS = frozenset(
    [
        "cnn",
        "rnn",
        "lstm",
        "gru",
        "gru-simple",
        "bilstm",
        "plstm",
        "tlstm",
        "tglstm",
    ]
)
_ATTENTION_MODELS = frozenset(
    [
        "sand",
        "mtan",
        "miam",
        "transformer",
        "transformer-mask",
    ]
)
_ATTENTION_NUM_LAYERS_MODELS = frozenset(
    [
        "transformer",
        "transformer-mask",
    ]
)

_AUXILIARY_LOSS_MODELS = frozenset()
_FLOW_DEFAULT_DEPTH = 2
_FLOW_DEFAULT_HEAD_DEPTH = 1


def _default_head_depth_for_model(canonical_name: str, *, base_num_layers: int) -> int:
    if is_flow_family_name(canonical_name) and base_num_layers == 1:
        return _FLOW_DEFAULT_HEAD_DEPTH
    return base_num_layers


def _default_flow_depth_for_model(canonical_name: str) -> int:
    if canonical_name.startswith("neuralflow-"):
        return 3
    if canonical_name.startswith("dualdynamics-"):
        return 2
    if canonical_name.startswith("flowpath-"):
        return 1
    return _FLOW_DEFAULT_DEPTH


def _as_positive_int(value: Any, *, default: int) -> int:
    """Best-effort integer coercion with positive floor."""
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _resolve_runtime_depth_config(
    model_name: str,
    *,
    num_layers: Any = 1,
    num_hidden_layers: Any = None,
    head_depth: Any = None,
    flow_num_layers: Any = None,
    flow_depth: Any = None,
) -> Dict[str, Any]:
    """Resolve canonical+compat depth values for runtime wiring.

    Priority:
    - head depth: ``head_depth`` -> ``num_hidden_layers`` -> flow default 1 / ``num_layers``
    - flow depth: ``flow_depth`` -> ``flow_num_layers`` -> family default
      (neuralflow=3, dualdynamics=2, flowpath=1)
    """
    canonical_name = normalize_flow_alias(str(model_name))
    is_flow_model = is_flow_family_name(canonical_name)

    base_num_layers = _as_positive_int(num_layers, default=1)
    default_head_depth = _default_head_depth_for_model(
        canonical_name,
        base_num_layers=base_num_layers,
    )

    if head_depth is None:
        if num_hidden_layers is not None:
            head_depth = num_hidden_layers
        else:
            head_depth = default_head_depth
    resolved_head_depth = _as_positive_int(head_depth, default=default_head_depth)

    if flow_depth is None:
        flow_depth = flow_num_layers
    default_flow_depth = _default_flow_depth_for_model(canonical_name)
    resolved_flow_depth = _as_positive_int(flow_depth, default=default_flow_depth)

    if is_flow_model:
        resolved_num_layers = 1
        resolved_num_hidden_layers = resolved_head_depth
    else:
        resolved_num_layers = base_num_layers
        if num_hidden_layers is None:
            resolved_num_hidden_layers = base_num_layers
        else:
            resolved_num_hidden_layers = _as_positive_int(num_hidden_layers, default=base_num_layers)

    return {
        "is_flow_model": is_flow_model,
        "num_layers": resolved_num_layers,
        "num_hidden_layers": resolved_num_hidden_layers,
        "head_depth": resolved_head_depth,
        "flow_num_layers": resolved_flow_depth,
        "flow_depth": resolved_flow_depth,
    }


def get_data_config(model_name: str) -> dict:
    """Return model-specific preprocessing policy."""
    model_name = normalize_tsde_method_alias(normalize_flow_alias(str(model_name)))
    if model_name in _INTERPOLATION_OVERRIDES:
        interpolation = _INTERPOLATION_OVERRIDES[model_name]
    elif model_name in _HERMITE_INTERPOLATION_MODELS:
        interpolation = "hermite"
    elif model_name in _NATURAL_INTERPOLATION_MODELS:
        interpolation = "natural"
    elif model_name in _CUBIC_INTERPOLATION_MODELS:
        interpolation = "cubic"
    else:
        interpolation = "hermite"

    use_intensity = model_name in _INTENSITY_MODELS
    return {"interpolation": interpolation, "use_intensity": use_intensity}


def resolve_layer_depth(model_name: str, num_layers: int) -> Tuple[int, Optional[int]]:
    """Resolve (num_layers, num_hidden_layers) by model family."""
    model_name = normalize_tsde_method_alias(normalize_flow_alias(str(model_name)))
    if model_name in _BASIC_SEQUENCE_MODELS:
        return num_layers, None
    if model_name in _ATTENTION_MODELS:
        if model_name in _ATTENTION_NUM_LAYERS_MODELS:
            return num_layers, None
        return 1, None
    return 1, num_layers


def has_auxiliary_loss(model_name: str) -> bool:
    """Return True for models that emit auxiliary loss."""
    model_name = normalize_tsde_method_alias(normalize_flow_alias(str(model_name)))
    return model_name in _AUXILIARY_LOSS_MODELS or model_name.startswith("osde_")


def normalize_flow_model_name(model_name: str) -> str:
    """Compatibility wrapper: normalize known flow aliases to canonical names."""
    return normalize_flow_alias(str(model_name))


def requires_raw_missing_input(model_name: str) -> bool:
    """Compatibility wrapper: strict raw-missing policy lookup."""
    return is_flow_strict_raw_name(str(model_name))


def resolve_raw_missing_policy(
    model_name: str,
    runtime_model_name: Optional[str] = None,
) -> bool:
    """Compatibility wrapper: request/runtime-aware strict raw-missing policy."""
    if is_flow_strict_raw_name(str(model_name)):
        return True
    if runtime_model_name is None:
        return False
    return is_flow_strict_raw_name(str(runtime_model_name))
