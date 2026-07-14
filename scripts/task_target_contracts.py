"""Task target helpers for packaged runs."""

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


def mask_imputation_inputs(
    inputs: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> torch.Tensor:
    """Hide supervised imputation cells before preprocessing.

    Inputs use the dataset layout ``[N,D,L]`` while target masks use
    ``[N,L,D]``.
    """
    if not torch.is_tensor(inputs) or inputs.ndim != 3:
        raise ValueError("imputation inputs must be a tensor with shape [N,D,L]")
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
        if task_name in {"imputation", "change_detection"}:
            if target_mask is None:
                raise ValueError(f"{task_name} requires target_mask")
            split_mask = target_mask.to(target.device).index_select(0, indices).bool()
            supervised = split_target.masked_select(split_mask)
            if supervised.numel() <= 0:
                raise ValueError(f"{split_name} split has zero supervised entries")
        split_summary = {"supervised_element_count": int(supervised.numel())}
        if task_name in {"outlier_detection", "change_detection"}:
            classes = torch.unique(supervised).detach().cpu().tolist()
            if set(classes) != {0.0, 1.0}:
                raise ValueError(
                    f"{split_name} split must contain both binary classes for {task_name}"
                )
            split_summary["binary_class_count"] = 2
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


def _validate_regression_target(target: np.ndarray, *, num_samples: int) -> np.ndarray:
    if target.ndim == 1:
        target = target.reshape(num_samples, 1)
    elif target.ndim != 2:
        raise ValueError(f"regression_target must be 1D/2D [N] or [N,1], got {target.shape}")
    _validate_first_dim("regression_target", target, num_samples)
    if target.shape[1] != 1:
        raise ValueError(f"regression_target second dimension must be 1, got {target.shape[1]}")
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
    if not np.isfinite(target).all():
        raise ValueError("forecasting_target must be finite")
    return target


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
        raise ValueError(f"change_detection_target must be 3D [N,L,1], got {target.shape}")
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
) -> tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    task_name = str(task).strip().lower()
    if task_name not in SUPPORTED_TARGET_TASKS:
        raise ValueError(
            f"Unsupported task for dataset targets: {task!r}. "
            f"Supported tasks are {sorted(SUPPORTED_TARGET_TASKS)}"
        )

    target_path = Path(targets_root) / f"{dataset}.npz"
    if not target_path.exists():
        raise FileNotFoundError(f"Target bundle not found for dataset={dataset}: {target_path}")

    with np.load(target_path, allow_pickle=False) as payload:
        target_key = TASK_TARGET_KEYS[task_name]
        raw_target = _require_key(payload, target_key, dataset=dataset, task=task_name)

        target_mask: Optional[np.ndarray] = None
        target_horizon: Optional[int] = None
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

    target_tensor = _as_float_tensor(target)
    mask_tensor = _as_float_tensor(target_mask) if target_mask is not None else None
    target_spec: Dict[str, Any] = {
        "target_source": "dataset",
        "target_path": str(target_path),
        "task": task_name,
        "target_key": TASK_TARGET_KEYS[task_name],
        "target_shape": [int(x) for x in target_tensor.shape],
        "target_dtype": str(target_tensor.dtype),
        "forecast_horizon": target_horizon if task_name == "forecasting" else None,
        "mask_key": (
            IMPUTATION_MASK_KEY
            if task_name == "imputation"
            else CHANGE_DETECTION_MASK_KEY
            if task_name == "change_detection"
            else None
        ),
        "mask_shape": [int(x) for x in mask_tensor.shape] if mask_tensor is not None else None,
        "mask_dtype": str(mask_tensor.dtype) if mask_tensor is not None else None,
    }
    return target_tensor, mask_tensor, target_spec
