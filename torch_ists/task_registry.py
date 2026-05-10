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
