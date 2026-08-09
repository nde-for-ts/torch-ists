"""Canonical task registry for the runtime package."""

from __future__ import annotations

SUPPORTED_TASKS = frozenset(
    {
        "classification",
        "regression",
        "forecasting",
        "imputation",
        "outlier_detection",
        "change_detection",
    }
)

TASK_DISPLAY_NAMES = {
    "classification": "Classification",
    "regression": "Regression",
    "forecasting": "Forecasting",
    "imputation": "Imputation",
    "outlier_detection": "Outlier Detection",
    "change_detection": "Change Detection",
}

SEQUENCE_TASKS = frozenset({"forecasting", "imputation", "change_detection"})
DATASET_ONLY_TASKS = frozenset({"change_detection"})

_STOCHASTIC_MODEL_NAMES = frozenset({"mtan", "latentode"})


def normalize_task(task: str) -> str:
    normalized = str(task).strip().lower()
    if normalized not in SUPPORTED_TASKS:
        raise NotImplementedError(
            f"Unsupported task: {task!r}. Supported tasks are {sorted(SUPPORTED_TASKS)}."
        )
    return normalized


def get_task_display_name(task: str) -> str:
    normalized = normalize_task(task)
    return TASK_DISPLAY_NAMES[normalized]


def get_model_task_capabilities(model_name: str) -> dict[str, object]:
    """Return the model facts used by preview-task preflight."""
    from .config.model_data_policy import has_auxiliary_loss, requires_raw_missing_input

    normalized_model = str(model_name).strip().lower()
    stochastic = normalized_model in _STOCHASTIC_MODEL_NAMES or "sde" in normalized_model
    return {
        "sample_readout": True,
        "sequence_aligned_output": True,
        "generic_forecast": True,
        "native_future_time": False,
        "ragged_suffix": False,
        "causal_behavior": False,
        "stochastic_evaluation": bool(stochastic),
        "raw_missing_policy": (
            "raw" if requires_raw_missing_input(normalized_model) else "filled"
        ),
        "auxiliary_loss": bool(has_auxiliary_loss(normalized_model)),
    }


def validate_model_task_capability(
    model_name: str,
    task: str,
    *,
    has_ragged_suffix: bool = False,
    requires_native_future_time: bool = False,
    requires_causal_behavior: bool = False,
) -> dict[str, object]:
    """Fail unsupported preview requests before search or model construction."""
    task_name = normalize_task(task)
    capabilities = get_model_task_capabilities(model_name)
    prefix = (
        "Unsupported model/task combination before hyperparameter search: "
        f"model={model_name!r}, task={task_name!r}"
    )
    if task_name in {"regression", "outlier_detection"} and not capabilities["sample_readout"]:
        raise ValueError(f"{prefix}; sample readout is unavailable")
    if task_name in {"imputation", "change_detection"} and not capabilities["sequence_aligned_output"]:
        raise ValueError(f"{prefix}; sequence-aligned output is unavailable")
    if task_name == "forecasting" and not capabilities["generic_forecast"]:
        raise ValueError(f"{prefix}; generic forecasting is unavailable")
    if has_ragged_suffix and not capabilities["ragged_suffix"]:
        raise ValueError(f"{prefix}; ragged suffix inputs are not certified")
    if requires_native_future_time and not capabilities["native_future_time"]:
        raise ValueError(f"{prefix}; native future-time decoding is unavailable")
    if requires_causal_behavior and not capabilities["causal_behavior"]:
        raise ValueError(f"{prefix}; causal behavior is unavailable")
    return {
        "supported": True,
        "model": str(model_name),
        "task": task_name,
        "capabilities": capabilities,
    }
