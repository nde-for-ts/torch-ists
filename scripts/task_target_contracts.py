"""Task target contracts for non-classification package workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


SUPPORTED_TARGET_TASKS = frozenset(
    {"regression", "forecasting", "imputation", "outlier_detection", "change_detection"}
)

TASK_TARGET_KEYS = {
    "regression": "regression_target",
    "forecasting": "forecasting_target",
    "imputation": "imputation_target",
    "outlier_detection": "outlier_detection_target",
    "change_detection": "change_detection_target",
}

IMPUTATION_MASK_KEY = "imputation_loss_mask"
CHANGE_DETECTION_MASK_KEY = "change_detection_loss_mask"
FORECASTING_TIMESTAMPS_KEY = "forecasting_timestamps"
FORECASTING_ORIGIN_KEY = "forecasting_origin"
FORECASTING_MASK_KEY = "forecasting_loss_mask"


def resolve_task_split_path(
    split_root: str | Path,
    dataset: str,
    seed: int,
    *,
    task: str,
) -> Path:
    """Return the unchanged classification path or a task-isolated preview path."""
    task_name = str(task).strip().lower()
    stem = f"{dataset}_{int(seed)}"
    if task_name != "classification":
        if task_name not in SUPPORTED_TARGET_TASKS:
            raise ValueError(f"Unsupported task for split path: {task!r}")
        stem = f"{stem}__task_{task_name}"
    return Path(split_root) / str(dataset) / stem


def make_fixed_task_splits(
    *,
    num_samples: int,
    seed: int,
    train: float = 0.7,
    valid: float = 0.15,
    test: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic label-independent splits for preview tasks."""
    n = int(num_samples)
    if n < 3:
        raise ValueError("preview task splits require at least 3 samples")
    if min(train, valid, test) <= 0 or abs((train + valid + test) - 1.0) > 1e-8:
        raise ValueError("train/valid/test ratios must be positive and sum to 1.0")
    train_count = int(np.floor(n * float(train)))
    holdout_count = n - train_count
    valid_count = int(np.floor(holdout_count * float(valid) / float(valid + test)))
    test_count = holdout_count - valid_count
    if min(train_count, valid_count, test_count) <= 0:
        raise ValueError(
            "preview task split ratios produce an empty split: "
            f"train={train_count}, valid={valid_count}, test={test_count}"
        )
    indices = np.random.RandomState(int(seed)).permutation(n).astype(np.int64)
    return (
        indices[:train_count],
        indices[train_count : train_count + valid_count],
        indices[train_count + valid_count :],
    )


def infer_task_output_dim(
    *,
    task: str,
    target: torch.Tensor,
    input_dim: int,
) -> int:
    """Infer the shared task-head width from a validated target."""
    task_name = str(task).strip().lower()
    if task_name == "regression":
        if target.ndim != 2 or int(target.shape[1]) <= 0:
            raise ValueError("regression target must have shape [N,R] with R >= 1")
        return int(target.shape[1])
    if task_name in {"forecasting", "imputation"}:
        return int(input_dim)
    if task_name in {"outlier_detection", "change_detection"}:
        return 1
    raise ValueError(f"Output width inference is unsupported for task={task!r}")


def _binary_metric_availability(class_count: int) -> Dict[str, Dict[str, Any]]:
    available = int(class_count) == 2
    status = {"available": available}
    if not available:
        status["reason"] = "requires both binary classes"
    return {"auroc": dict(status), "auprc": dict(status)}


def mask_imputation_inputs(
    inputs: Any,
    supervision_mask: Any,
) -> Any:
    """Hide supervised imputation cells before preprocessing.

    Inputs use the dataset layout ``[N,D,L]`` while target masks use
    ``[N,L,D]``.
    """
    if torch.is_tensor(inputs):
        if inputs.ndim != 3:
            raise ValueError("imputation inputs must have shape [N,D,L]")
        if not torch.is_floating_point(inputs):
            raise ValueError("imputation inputs must use a floating dtype")
        mask = torch.as_tensor(supervision_mask, device=inputs.device)
        expected_shape = (int(inputs.shape[0]), int(inputs.shape[2]), int(inputs.shape[1]))
        if tuple(mask.shape) != expected_shape:
            raise ValueError(
                "imputation supervision mask/input shape mismatch: "
                f"expected mask {expected_shape}, got {tuple(mask.shape)}"
            )
        hidden = inputs.clone()
        hidden.masked_fill_(mask.permute(0, 2, 1) > 0.5, float("nan"))
        return hidden

    values = np.asarray(inputs)
    if values.ndim != 3:
        raise ValueError("imputation inputs must have shape [N,D,L]")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError("imputation inputs must use a floating dtype")
    if torch.is_tensor(supervision_mask):
        mask_array = supervision_mask.detach().cpu().numpy()
    else:
        mask_array = np.asarray(supervision_mask)
    expected_shape = (int(values.shape[0]), int(values.shape[2]), int(values.shape[1]))
    if tuple(mask_array.shape) != expected_shape:
        raise ValueError(
            "imputation supervision mask/input shape mismatch: "
            f"expected mask {expected_shape}, got {tuple(mask_array.shape)}"
        )
    hidden = values.copy()
    hidden[np.transpose(mask_array, (0, 2, 1)) > 0.5] = np.nan
    return hidden


def validate_task_splits(
    *,
    task: str,
    target: torch.Tensor,
    target_mask: Optional[torch.Tensor],
    splits: Dict[str, Any],
) -> Dict[str, Dict[str, int]]:
    """Validate split-level supervision before training starts."""
    task_name = str(task).strip().lower()
    summary: Dict[str, Dict[str, int]] = {}
    for split_name, split_indices in splits.items():
        indices = torch.as_tensor(split_indices, dtype=torch.long, device=target.device)
        if indices.numel() <= 0:
            raise ValueError(f"{split_name} split is empty")
        split_target = target.index_select(0, indices)
        supervised = split_target.reshape(-1)
        if task_name in {"forecasting", "imputation", "change_detection"}:
            if target_mask is None:
                raise ValueError(f"{task_name} requires target_mask")
            split_mask = target_mask.to(target.device).index_select(0, indices).bool()
            supervised = split_target.masked_select(split_mask)
            if supervised.numel() <= 0:
                raise ValueError(f"{split_name} split has zero supervised entries")
        split_summary = {"supervised_element_count": int(supervised.numel())}
        if task_name in {"outlier_detection", "change_detection"}:
            classes = torch.unique(supervised).detach().cpu().tolist()
            class_count = int(len(classes))
            if str(split_name).strip().lower() in {"train", "training"} and set(classes) != {0.0, 1.0}:
                raise ValueError(
                    f"{split_name} split must contain both binary classes for {task_name}"
                )
            split_summary["binary_class_count"] = class_count
            split_summary["metric_availability"] = _binary_metric_availability(class_count)
        summary[str(split_name)] = split_summary
    return summary


def _as_float_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


def _require_key(payload: Any, key: str, *, dataset: str, task: str) -> np.ndarray:
    keys = sorted(payload.files)
    if key not in payload.files:
        raise ValueError(
            f"Target bundle missing key for dataset={dataset}, task={task}: {key}. "
            f"Available keys: {keys}"
        )
    return np.asarray(payload[key])


def _validate_first_dim(name: str, value: np.ndarray, expected_n: int) -> None:
    if value.shape[0] != expected_n:
        raise ValueError(
            f"{name} sample mismatch: expected first dim {expected_n}, got {value.shape[0]}"
        )


def _validate_input_values(
    input_values: Any,
    *,
    num_samples: int,
    seq_len: int,
    input_dim: int,
) -> Optional[np.ndarray]:
    if input_values is None:
        return None
    if torch.is_tensor(input_values):
        input_values = input_values.detach().cpu().numpy()
    values = np.asarray(input_values, dtype=np.float32)
    expected = (int(num_samples), int(input_dim), int(seq_len))
    if tuple(values.shape) != expected:
        raise ValueError(
            f"input_values must have dataset shape [N,D,L]={expected}, got {values.shape}"
        )
    return values


def _validate_regression_target(target: np.ndarray, *, num_samples: int) -> np.ndarray:
    if target.ndim == 1:
        target = target.reshape(num_samples, 1)
    elif target.ndim != 2:
        raise ValueError(f"regression_target must be 1D/2D [N] or [N,R], got {target.shape}")
    _validate_first_dim("regression_target", target, num_samples)
    if target.shape[1] <= 0:
        raise ValueError("regression_target second dimension R must be >= 1")
    if not np.isfinite(target).all():
        raise ValueError("regression_target must be finite")
    return target


def _validate_forecasting_target(
    target: np.ndarray,
    *,
    num_samples: int,
    horizon: int,
    input_dim: int,
) -> np.ndarray:
    if target.ndim != 3:
        raise ValueError(f"forecasting_target must be 3D [N,H,D], got {target.shape}")
    _validate_first_dim("forecasting_target", target, num_samples)
    if target.shape[1] != horizon:
        raise ValueError(
            f"forecasting_target horizon mismatch: expected {horizon}, got {target.shape[1]}"
        )
    if target.shape[2] != input_dim:
        raise ValueError(
            f"forecasting_target feature dim mismatch: expected {input_dim}, got {target.shape[2]}"
        )
    return target


def _validate_forecasting_schedule(
    *,
    timestamps: np.ndarray,
    origin: np.ndarray,
    mask: np.ndarray,
    target: np.ndarray,
    num_samples: int,
    horizon: int,
    input_dim: int,
    input_values: Optional[np.ndarray],
    input_timestamps: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    expected_target_shape = (num_samples, horizon, input_dim)
    if tuple(mask.shape) != expected_target_shape:
        raise ValueError(
            f"forecasting_loss_mask shape mismatch: expected {expected_target_shape}, got {mask.shape}"
        )
    if not np.isfinite(mask).all() or not set(np.unique(mask).tolist()).issubset({0, 0.0, 1, 1.0}):
        raise ValueError("forecasting_loss_mask must be finite binary 0/1")
    supervised = mask > 0.5
    if not supervised.reshape(num_samples, -1).any(axis=1).all():
        raise ValueError("forecasting_loss_mask must supervise at least one value per sample")
    if not np.isfinite(target[supervised]).all():
        raise ValueError("forecasting_target must be finite at supervised positions")

    if tuple(timestamps.shape) != (num_samples, horizon):
        raise ValueError(
            "forecasting_timestamps must have shape "
            f"[{num_samples},{horizon}], got {timestamps.shape}"
        )
    if not np.isfinite(timestamps).all():
        raise ValueError("forecasting_timestamps must be finite")
    if not np.all(np.diff(timestamps, axis=1) > 0):
        raise ValueError("forecasting_timestamps must be strictly increasing per sample")

    origin = np.asarray(origin).reshape(-1)
    if tuple(origin.shape) != (num_samples,):
        raise ValueError(
            f"forecasting_origin must have shape [{num_samples}], got {origin.shape}"
        )
    if not np.isfinite(origin).all():
        raise ValueError("forecasting_origin must be finite")
    if not np.all(timestamps[:, 0] > origin):
        raise ValueError("forecasting_timestamps must be later than forecast origin")

    origin_policy = "declared_origin_only"
    if input_values is not None:
        valid_context = np.isfinite(input_values).any(axis=1)
        if not valid_context.any(axis=1).all():
            raise ValueError("forecasting context must contain an observed value per sample")
        if input_timestamps is None:
            context_times = np.broadcast_to(
                np.arange(input_values.shape[2], dtype=np.float64),
                (num_samples, input_values.shape[2]),
            )
            origin_policy = "last_valid_index_time"
        else:
            if torch.is_tensor(input_timestamps):
                input_timestamps = input_timestamps.detach().cpu().numpy()
            context_times = np.asarray(input_timestamps, dtype=np.float64)
            expected_context_shape = (num_samples, input_values.shape[2])
            if tuple(context_times.shape) != expected_context_shape:
                raise ValueError(
                    "input_timestamps must have shape "
                    f"{expected_context_shape}, got {context_times.shape}"
                )
            if not np.isfinite(context_times[valid_context]).all():
                raise ValueError("input_timestamps must be finite at observed context positions")
            origin_policy = "last_valid_context_timestamp"
        last_indices = np.where(
            valid_context,
            np.arange(valid_context.shape[1])[None, :],
            -1,
        ).max(axis=1)
        context_origin = context_times[np.arange(num_samples), last_indices]
        if not np.allclose(origin, context_origin, rtol=1e-7, atol=1e-9):
            raise ValueError("forecasting_origin must equal the last valid context timestamp")

    cleaned_target = np.where(supervised, target, 0.0).astype(np.float32)
    return cleaned_target, mask.astype(np.float32), origin.astype(np.float64), origin_policy


def _validate_imputation_target(
    target: np.ndarray,
    *,
    num_samples: int,
    seq_len: int,
    input_dim: int,
) -> np.ndarray:
    if target.ndim != 3:
        raise ValueError(f"imputation_target must be 3D [N,L,D], got {target.shape}")
    _validate_first_dim("imputation_target", target, num_samples)
    if target.shape[1] != seq_len or target.shape[2] != input_dim:
        raise ValueError(
            "imputation_target shape mismatch: "
            f"expected [{num_samples},{seq_len},{input_dim}], got {target.shape}"
        )
    if not np.isfinite(target).all():
        raise ValueError("imputation_target must be finite")
    return target


def _validate_imputation_mask(
    mask: np.ndarray,
    *,
    expected_shape: Tuple[int, int, int],
) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError(f"imputation_loss_mask must be 3D [N,L,D], got {mask.shape}")
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"imputation_loss_mask shape mismatch: expected {expected_shape}, got {mask.shape}"
        )
    if not np.isfinite(mask).all():
        raise ValueError("imputation_loss_mask must be finite")
    unique = set(np.unique(mask).tolist())
    if not unique.issubset({0, 0.0, 1, 1.0}):
        raise ValueError(
            "imputation_loss_mask must be binary 0/1. "
            f"found values={sorted(unique)}"
        )
    return mask.astype(np.float32)


def _validate_change_detection_mask(
    mask: np.ndarray,
    *,
    expected_shape: Tuple[int, int, int],
) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError(
            f"change_detection_loss_mask must be 3D [N,L,1], got {mask.shape}"
        )
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            "change_detection_loss_mask shape mismatch: "
            f"expected {expected_shape}, got {mask.shape}"
        )
    if not np.isfinite(mask).all():
        raise ValueError("change_detection_loss_mask must be finite")
    unique = set(np.unique(mask).tolist())
    if not unique.issubset({0, 0.0, 1, 1.0}):
        raise ValueError(
            "change_detection_loss_mask must be binary 0/1. "
            f"found values={sorted(unique)}"
        )
    return mask.astype(np.float32)


def _validate_outlier_detection_target(target: np.ndarray, *, num_samples: int) -> np.ndarray:
    if target.ndim == 1:
        target = target.reshape(num_samples, 1)
    elif target.ndim != 2:
        raise ValueError(
            f"outlier_detection_target must be 1D/2D [N] or [N,1], got {target.shape}"
        )
    _validate_first_dim("outlier_detection_target", target, num_samples)
    if target.shape[1] != 1:
        raise ValueError(
            f"outlier_detection_target second dim must be 1, got {target.shape[1]}"
        )
    if not np.isfinite(target).all():
        raise ValueError("outlier_detection_target must be finite")
    unique = set(np.unique(target).tolist())
    if not unique.issubset({0, 0.0, 1, 1.0}):
        raise ValueError(
            "outlier_detection_target must be binary 0/1. "
            f"found values={sorted(unique)}"
        )
    return target.astype(np.float32)


def _validate_change_detection_target(
    target: np.ndarray,
    *,
    num_samples: int,
    seq_len: int,
) -> np.ndarray:
    if target.ndim != 3:
        raise ValueError(
            f"change_detection_target must be 3D [N,L,1], got {target.shape}"
        )
    _validate_first_dim("change_detection_target", target, num_samples)
    if target.shape[1] != seq_len:
        raise ValueError(
            f"change_detection_target seq_len mismatch: expected {seq_len}, got {target.shape[1]}"
        )
    if target.shape[2] != 1:
        raise ValueError(
            f"change_detection_target feature dim must be 1, got {target.shape[2]}"
        )
    if not np.isfinite(target).all():
        raise ValueError("change_detection_target must be finite")
    unique = set(np.unique(target).tolist())
    if not unique.issubset({0, 0.0, 1, 1.0}):
        raise ValueError(
            "change_detection_target must be binary 0/1. "
            f"found values={sorted(unique)}"
        )
    return target.astype(np.float32)


def load_task_targets_npz(
    *,
    dataset: str,
    targets_root: str | Path,
    task: str,
    horizon: Optional[int],
    num_samples: int,
    seq_len: int,
    input_dim: int,
    input_values: Any = None,
    input_timestamps: Any = None,
    return_forecasting_schedule: bool = False,
):
    """Load and validate dataset-provided targets for non-classification tasks."""
    task_name = str(task).strip().lower()
    if task_name not in SUPPORTED_TARGET_TASKS:
        raise ValueError(
            f"Unsupported task for dataset targets: {task!r}. "
            f"Supported tasks are {sorted(SUPPORTED_TARGET_TASKS)}"
        )

    validated_inputs = _validate_input_values(
        input_values,
        num_samples=num_samples,
        seq_len=seq_len,
        input_dim=input_dim,
    )
    target_path = Path(targets_root) / f"{dataset}.npz"
    if not target_path.exists():
        raise FileNotFoundError(
            f"Target bundle not found for dataset={dataset}: {target_path}"
        )

    with np.load(target_path, allow_pickle=False) as payload:
        target_key = TASK_TARGET_KEYS[task_name]
        raw_target = _require_key(payload, target_key, dataset=dataset, task=task_name)

        target_mask: Optional[np.ndarray] = None
        target_horizon: Optional[int] = None
        forecasting_schedule = None
        forecasting_origin_policy = None
        if task_name == "regression":
            target = _validate_regression_target(raw_target, num_samples=num_samples)
        elif task_name == "forecasting":
            if horizon is None or int(horizon) <= 0:
                raise ValueError(
                    "forecasting requires an explicit positive --forecast-horizon"
                )
            target_horizon = int(horizon)
            target = _validate_forecasting_target(
                raw_target,
                num_samples=num_samples,
                horizon=target_horizon,
                input_dim=input_dim,
            )
            raw_times = _require_key(
                payload, FORECASTING_TIMESTAMPS_KEY, dataset=dataset, task=task_name
            )
            raw_origin = _require_key(
                payload, FORECASTING_ORIGIN_KEY, dataset=dataset, task=task_name
            )
            raw_mask = _require_key(
                payload, FORECASTING_MASK_KEY, dataset=dataset, task=task_name
            )
            target, target_mask, forecast_origin, forecasting_origin_policy = (
                _validate_forecasting_schedule(
                    timestamps=np.asarray(raw_times, dtype=np.float64),
                    origin=np.asarray(raw_origin, dtype=np.float64),
                    mask=np.asarray(raw_mask),
                    target=target,
                    num_samples=num_samples,
                    horizon=target_horizon,
                    input_dim=input_dim,
                    input_values=validated_inputs,
                    input_timestamps=input_timestamps,
                )
            )
            forecasting_schedule = {
                "future_timestamps": torch.as_tensor(raw_times, dtype=torch.float64),
                "forecast_origin": torch.as_tensor(forecast_origin, dtype=torch.float64),
            }
        elif task_name == "imputation":
            target = _validate_imputation_target(
                raw_target,
                num_samples=num_samples,
                seq_len=seq_len,
                input_dim=input_dim,
            )
            raw_mask = _require_key(
                payload,
                IMPUTATION_MASK_KEY,
                dataset=dataset,
                task=task_name,
            )
            target_mask = _validate_imputation_mask(
                raw_mask,
                expected_shape=(num_samples, seq_len, input_dim),
            )
        elif task_name == "outlier_detection":
            target = _validate_outlier_detection_target(raw_target, num_samples=num_samples)
        else:
            target = _validate_change_detection_target(
                raw_target,
                num_samples=num_samples,
                seq_len=seq_len,
            )
            raw_mask = _require_key(
                payload,
                CHANGE_DETECTION_MASK_KEY,
                dataset=dataset,
                task=task_name,
            )
            target_mask = _validate_change_detection_mask(
                raw_mask,
                expected_shape=(num_samples, seq_len, 1),
            )

    valid_time_policy = None
    raw_unit_comparison_count = 0
    if validated_inputs is not None:
        valid_time = np.isfinite(validated_inputs).any(axis=1)
        valid_time_policy = "any_finite_input_channel"
        if target_mask is not None and task_name in {"imputation", "change_detection"}:
            supervised_time = np.asarray(target_mask > 0.5).any(axis=2)
            if np.any(supervised_time & ~valid_time):
                raise ValueError(
                    f"{task_name} supervision mask selects invalid/padded time steps"
                )
        if task_name == "imputation":
            raw_layout = validated_inputs.transpose(0, 2, 1)
            comparable = (target_mask > 0.5) & np.isfinite(raw_layout)
            raw_unit_comparison_count = int(np.count_nonzero(comparable))
            if raw_unit_comparison_count and not np.allclose(
                target[comparable],
                raw_layout[comparable],
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    "imputation_target supervised values must match dataset-native input values"
                )

    target_tensor = _as_float_tensor(target)
    mask_tensor = _as_float_tensor(target_mask) if target_mask is not None else None
    target_spec: Dict[str, Any] = {
        "target_source": "dataset",
        "target_path": str(target_path),
        "task": task_name,
        "target_key": TASK_TARGET_KEYS[task_name],
        "target_shape": [int(x) for x in target_tensor.shape],
        "target_dtype": str(target_tensor.dtype),
        "target_value_space": "dataset_native",
        "valid_time_policy": valid_time_policy,
        "raw_unit_comparison_count": raw_unit_comparison_count,
        "output_dim": infer_task_output_dim(
            task=task_name,
            target=target_tensor,
            input_dim=input_dim,
        ),
        "forecast_horizon": target_horizon if task_name == "forecasting" else None,
        "mask_key": (
            FORECASTING_MASK_KEY
            if task_name == "forecasting"
            else IMPUTATION_MASK_KEY
            if task_name == "imputation"
            else CHANGE_DETECTION_MASK_KEY
            if task_name == "change_detection"
            else None
        ),
        "mask_shape": [int(x) for x in mask_tensor.shape] if mask_tensor is not None else None,
        "mask_dtype": str(mask_tensor.dtype) if mask_tensor is not None else None,
        "forecasting_timestamps_key": (
            FORECASTING_TIMESTAMPS_KEY if task_name == "forecasting" else None
        ),
        "forecasting_origin_key": (
            FORECASTING_ORIGIN_KEY if task_name == "forecasting" else None
        ),
        "forecasting_origin_policy": forecasting_origin_policy,
    }
    result = (target_tensor, mask_tensor, target_spec)
    if return_forecasting_schedule:
        return (*result, forecasting_schedule)
    return result
