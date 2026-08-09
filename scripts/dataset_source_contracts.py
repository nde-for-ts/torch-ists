"""Dataset source helpers for packaged script execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np


PRIMARY_DATASET_SOURCES = frozenset({"default", "npz"})
COMPAT_DATASET_SOURCE_ALIASES = {"refactoring": "default"}
TIMESTAMP_COMMON_GRID_ATOL = 1e-6


@dataclass(frozen=True)
class DatasetPayload:
    X: np.ndarray
    y: np.ndarray | None
    timestamps: np.ndarray | None = None
    timestamp_summary: dict[str, Any] | None = None


def normalize_dataset_source(dataset_source: str) -> str:
    normalized = str(dataset_source).strip().lower()
    normalized = COMPAT_DATASET_SOURCE_ALIASES.get(normalized, normalized)
    if normalized not in PRIMARY_DATASET_SOURCES:
        raise ValueError(
            f"Unsupported dataset_source: {dataset_source!r}. "
            f"Expected one of {sorted(PRIMARY_DATASET_SOURCES)}"
        )
    return normalized


def _remap_labels_to_contiguous(y: np.ndarray) -> np.ndarray:
    """Remap class labels to contiguous integer indices ``0..K-1``.

    Mirrors the default-source loader (``benchmark_loader`` ``label_dict``) so the
    npz / custom-data path produces the same contiguous integer targets. Handles
    non-contiguous integer, float, or string labels deterministically (sorted
    unique -> ``0..K-1``). Without this, a raw label value (e.g. ``5`` with only
    3 classes) reaches ``nn.CrossEntropyLoss`` as a class index and silently
    corrupts training: out-of-range ``IndexError``, float truncation via
    ``.long()``, or a ``TypeError`` on string labels. Idempotent on already
    contiguous ``0..K-1`` labels, so the built-in default source is unaffected.
    """
    classes = np.unique(y)
    label_to_index = {label: index for index, label in enumerate(classes)}
    return np.array([label_to_index[label] for label in y], dtype=np.int64)


def _validate_x(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [N, D, L], got shape {X.shape}")
    return X


def _validate_y(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D [N], got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"sample mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
    return _remap_labels_to_contiguous(y)


def _validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = _validate_x(X)
    return X, _validate_y(y, X)


def _count_monotonic_failures(timestamps: np.ndarray) -> int:
    failures = 0
    for row in timestamps:
        finite_row = row[np.isfinite(row)]
        if finite_row.size > 1 and np.any(np.diff(finite_row) <= 0):
            failures += 1
    return int(failures)


def _count_duplicate_timestamp_steps(timestamps: np.ndarray) -> int:
    duplicates = 0
    for row in timestamps:
        finite_row = row[np.isfinite(row)]
        if finite_row.size > 1:
            duplicates += int(np.sum(np.diff(finite_row) == 0))
    return int(duplicates)


def _is_common_finite_grid(timestamps: np.ndarray) -> bool:
    if timestamps.ndim != 2 or timestamps.shape[0] == 0:
        return False
    if not np.isfinite(timestamps).all():
        return False
    reference = timestamps[0]
    return bool(all(np.allclose(row, reference, atol=TIMESTAMP_COMMON_GRID_ATOL, rtol=0.0) for row in timestamps[1:]))


def _is_strictly_increasing_grid(timestamps: np.ndarray) -> bool:
    if not _is_common_finite_grid(timestamps):
        return False
    return bool(np.all(np.diff(timestamps[0]) > 0))


def compute_timestamp_delta_scale(
    timestamps: np.ndarray | None,
    indices: Any | None = None,
) -> float | None:
    if timestamps is None:
        return None
    timestamps = np.asarray(timestamps, dtype=np.float32)
    if indices is not None:
        timestamps = timestamps[np.asarray(indices)]
    if timestamps.ndim != 2 or timestamps.shape[1] < 2:
        return 1.0
    left = timestamps[:, :-1]
    right = timestamps[:, 1:]
    diffs = right - left
    valid = np.isfinite(left) & np.isfinite(right) & (diffs > 0)
    positive_diffs = diffs[valid]
    if positive_diffs.size == 0:
        return 1.0
    return float(np.median(positive_diffs))


def _timestamp_delta_scale_fallback(timestamps: np.ndarray | None, indices: Any | None = None) -> bool:
    if timestamps is None:
        return False
    timestamps = np.asarray(timestamps, dtype=np.float32)
    if indices is not None:
        timestamps = timestamps[np.asarray(indices)]
    if timestamps.ndim != 2 or timestamps.shape[1] < 2:
        return True
    left = timestamps[:, :-1]
    right = timestamps[:, 1:]
    diffs = right - left
    valid = np.isfinite(left) & np.isfinite(right) & (diffs > 0)
    return bool(diffs[valid].size == 0)


def resolve_timestamp_delta_metadata(
    timestamps: np.ndarray | None,
    *,
    train_idx: Any | None = None,
    summary: dict[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    scale = compute_timestamp_delta_scale(timestamps, train_idx)
    unit_policy = "none" if timestamps is None else (
        "train_median_positive_step" if train_idx is not None else "dataset_median_positive_step"
    )
    timestamp_summary = dict(summary or summarize_timestamps(timestamps, source="none" if timestamps is None else "npz"))
    timestamp_summary.update(
        {
            "timestamp_delta_unit_policy": unit_policy,
            "timestamp_delta_scale": scale,
            "timestamp_delta_scale_fallback": _timestamp_delta_scale_fallback(timestamps, train_idx),
            "continuous_time_grid_policy": "normalized_index",
            "timestamp_common_grid_atol": TIMESTAMP_COMMON_GRID_ATOL if timestamps is not None else None,
        }
    )
    return scale, timestamp_summary


def summarize_timestamps(
    timestamps: np.ndarray | None,
    *,
    source: str,
    timestamp_delta_scale: float | None = None,
    timestamp_delta_unit_policy: str | None = None,
) -> dict[str, Any]:
    if timestamps is None:
        return {
            "timestamp_source": "none",
            "timestamp_shape": None,
            "timestamp_mode": "absent",
            "timestamp_finite_ratio": None,
            "timestamp_monotonic_failures": 0,
            "timestamp_duplicate_count": 0,
            "timestamp_degenerate_rows": 0,
            "timestamp_degenerate_rows_count": 0,
            "timestamp_min": None,
            "timestamp_max": None,
            "delta_policy": "index",
            "continuous_time_grid_policy": "normalized_index",
            "timestamp_delta_unit_policy": "none",
            "timestamp_delta_scale": None,
            "timestamp_delta_scale_fallback": False,
            "timestamp_common_grid_atol": None,
        }

    finite = np.isfinite(timestamps)
    finite_values = timestamps[finite]
    finite_counts = finite.sum(axis=1)
    common_grid = _is_common_finite_grid(timestamps)
    resolved_scale = (
        compute_timestamp_delta_scale(timestamps)
        if timestamp_delta_scale is None
        else float(timestamp_delta_scale)
    )
    degenerate_rows = int((finite_counts < 2).sum())
    return {
        "timestamp_source": str(source),
        "timestamp_shape": [int(value) for value in timestamps.shape],
        "timestamp_mode": "common" if common_grid else "per_sample",
        "timestamp_finite_ratio": float(finite.mean()) if finite.size else 0.0,
        "timestamp_monotonic_failures": _count_monotonic_failures(timestamps),
        "timestamp_duplicate_count": _count_duplicate_timestamp_steps(timestamps),
        "timestamp_degenerate_rows": degenerate_rows,
        "timestamp_degenerate_rows_count": degenerate_rows,
        "timestamp_min": float(finite_values.min()) if finite_values.size else None,
        "timestamp_max": float(finite_values.max()) if finite_values.size else None,
        "delta_policy": "timestamp",
        "continuous_time_grid_policy": "normalized_index",
        "timestamp_delta_unit_policy": timestamp_delta_unit_policy or "dataset_median_positive_step",
        "timestamp_delta_scale": float(resolved_scale),
        "timestamp_delta_scale_fallback": _timestamp_delta_scale_fallback(timestamps),
        "timestamp_common_grid_atol": TIMESTAMP_COMMON_GRID_ATOL,
    }


def _timestamp_summary(timestamps: np.ndarray | None, *, source: str) -> dict[str, Any]:
    return summarize_timestamps(timestamps, source=source)


def _validate_timestamps(timestamps: np.ndarray | None, X: np.ndarray) -> np.ndarray | None:
    if timestamps is None:
        return None
    timestamps = np.asarray(timestamps, dtype=np.float32)
    expected_shape = (int(X.shape[0]), int(X.shape[2]))
    if timestamps.ndim != 2:
        raise ValueError(f"timestamps must be 2D [N, L], got shape {timestamps.shape}")
    if tuple(timestamps.shape) != expected_shape:
        raise ValueError(f"timestamps shape mismatch: expected {expected_shape}, got {timestamps.shape}")
    failures = _count_monotonic_failures(timestamps)
    if failures:
        raise ValueError("timestamps must be strictly increasing within each sample")
    return timestamps


def load_dataset_npz_payload(
    path: str | Path,
    *,
    task: str = "classification",
) -> DatasetPayload:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"--dataset-npz must point to a .npz file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"dataset npz not found: {path}")
    loaded = np.load(path)
    keys = set(loaded.files)
    x_key = "X" if "X" in keys else "x" if "x" in keys else None
    y_key = "y" if "y" in keys else "Y" if "Y" in keys else None
    if x_key is None:
        raise ValueError(f"dataset npz must include X, got: {sorted(keys)}")

    task_name = str(task).strip().lower()
    if task_name == "classification" and y_key is None:
        raise ValueError(
            "classification dataset npz must include X/y keys; "
            f"got: {sorted(keys)}"
        )

    X = _validate_x(loaded[x_key])
    y = None if y_key is None else _validate_y(loaded[y_key], X)
    timestamps = _validate_timestamps(loaded["timestamps"] if "timestamps" in keys else None, X)
    return DatasetPayload(
        X=X,
        y=y,
        timestamps=timestamps,
        timestamp_summary=summarize_timestamps(timestamps, source="npz"),
    )


def load_dataset_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    payload = load_dataset_npz_payload(path)
    return payload.X, payload.y


def load_default_dataset_payload(dataset_name: str) -> DatasetPayload:
    """Load a packaged built-in dataset."""

    from torch_ists.benchmark_loader import get_data

    X, y = get_data(dataset_name)
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    X, y = _validate_xy(X, y)
    return DatasetPayload(
        X=X,
        y=y,
        timestamps=None,
        timestamp_summary=summarize_timestamps(None, source="none"),
    )


def load_default_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    payload = load_default_dataset_payload(dataset_name)
    return payload.X, payload.y


load_dataset_refactoring = load_default_dataset


def load_dataset_payload(
    *,
    dataset_name: str,
    dataset_source: str,
    dataset_npz: str | Path | None,
    task: str = "classification",
) -> DatasetPayload:
    normalized = normalize_dataset_source(dataset_source)
    if normalized == "npz":
        if not dataset_npz:
            raise ValueError("--dataset-npz is required when --dataset-source=npz")
        return load_dataset_npz_payload(dataset_npz, task=task)
    if normalized == "default":
        return load_default_dataset_payload(dataset_name)
    raise ValueError(f"Unsupported dataset_source: {dataset_source}")


def load_dataset(
    *,
    dataset_name: str,
    dataset_source: str,
    dataset_npz: str | Path | None,
    task: str = "classification",
) -> Tuple[np.ndarray, np.ndarray | None]:
    payload = load_dataset_payload(
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        dataset_npz=dataset_npz,
        task=task,
    )
    return payload.X, payload.y
