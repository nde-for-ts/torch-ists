"""Dataset source helpers for packaged script execution."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


PRIMARY_DATASET_SOURCES = frozenset({"default", "npz"})
COMPAT_DATASET_SOURCE_ALIASES = {"refactoring": "default"}


def normalize_dataset_source(dataset_source: str) -> str:
    normalized = str(dataset_source).strip().lower()
    normalized = COMPAT_DATASET_SOURCE_ALIASES.get(normalized, normalized)
    if normalized not in PRIMARY_DATASET_SOURCES:
        raise ValueError(
            f"Unsupported dataset_source: {dataset_source!r}. "
            f"Expected one of {sorted(PRIMARY_DATASET_SOURCES)}"
        )
    return normalized


def load_dataset_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"--dataset-npz must point to a .npz file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"dataset npz not found: {path}")
    loaded = np.load(path)
    keys = set(loaded.files)
    x_key = "X" if "X" in keys else "x" if "x" in keys else None
    y_key = "y" if "y" in keys else "Y" if "Y" in keys else None
    if x_key is None or y_key is None:
        raise ValueError(f"dataset npz must include X/y keys, got: {sorted(keys)}")

    X = np.asarray(loaded[x_key], dtype=np.float32)
    y = np.asarray(loaded[y_key])
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [N, D, L], got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D [N], got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"sample mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
    return X, y


def load_default_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a packaged built-in dataset."""

    from torch_ists.benchmark_loader import get_data

    X, y = get_data(dataset_name)
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [N, D, L], got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D [N], got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"sample mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
    return X, y


load_dataset_refactoring = load_default_dataset


def load_dataset(
    *,
    dataset_name: str,
    dataset_source: str,
    dataset_npz: str | Path | None,
) -> Tuple[np.ndarray, np.ndarray]:
    normalized = normalize_dataset_source(dataset_source)
    if normalized == "npz":
        if not dataset_npz:
            raise ValueError("--dataset-npz is required when --dataset-source=npz")
        return load_dataset_npz(dataset_npz)
    if normalized == "default":
        return load_default_dataset(dataset_name)
    raise ValueError(f"Unsupported dataset_source: {dataset_source}")
