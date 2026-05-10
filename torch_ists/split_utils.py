"""Split utilities for torch_ists run pipeline."""

from pathlib import Path
from typing import Dict, Tuple
import pickle

import numpy as np
from sklearn.model_selection import train_test_split


def _class_counts(y: np.ndarray) -> Dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def _min_samples_per_class(train: float, valid: float, test: float) -> int:
    requirements = []
    for ratio in (train, valid, test):
        if ratio <= 0.0:
            raise ValueError(
                "train/valid/test ratios must all be > 0 for stratified 3-way split "
                f"(got train={train}, valid={valid}, test={test})"
            )
        requirements.append(int(np.ceil(1.0 / ratio)))
    return int(max(requirements))


def make_stratified_splits(
    y: np.ndarray,
    seed: int,
    train: float = 0.7,
    valid: float = 0.15,
    test: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build reproducible stratified train/valid/test indices."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D [N], got shape {y.shape}")
    if len(y) < 3:
        raise ValueError("y must contain at least 3 samples")

    total = train + valid + test
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train+valid+test must sum to 1.0, got {total}")

    min_per_class = _min_samples_per_class(train, valid, test)
    counts = _class_counts(y)
    too_small = {label: cnt for label, cnt in counts.items() if cnt < min_per_class}
    if too_small:
        raise ValueError(
            "stratified split precheck failed: insufficient samples per class for requested "
            f"train/valid/test={train:.3f}/{valid:.3f}/{test:.3f}. "
            f"minimum_per_class={min_per_class}, class_counts={counts}, failing={too_small}"
        )

    indices = np.arange(len(y))
    try:
        train_idx, holdout_idx = train_test_split(
            indices,
            train_size=train,
            shuffle=True,
            stratify=y,
            random_state=seed,
        )

        valid_frac_in_holdout = valid / (valid + test)
        valid_idx, test_idx = train_test_split(
            holdout_idx,
            train_size=valid_frac_in_holdout,
            shuffle=True,
            stratify=y[holdout_idx],
            random_state=seed,
        )
    except ValueError as exc:
        raise ValueError(
            "stratified split failed after precheck; verify class distribution and split ratios. "
            f"train/valid/test={train:.3f}/{valid:.3f}/{test:.3f}, "
            f"minimum_per_class={min_per_class}, class_counts={counts}"
        ) from exc

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(valid_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def save_splits(path: Path, train_idx: np.ndarray, valid_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Save split indices in pickle format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(valid_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    ]
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_splits(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load split indices in pickle format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")

    with path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, (list, tuple)) or len(payload) != 3:
        raise ValueError(f"invalid split payload in {path}")

    train_idx, valid_idx, test_idx = payload
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(valid_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def get_split_path(split_root: Path, dataset_name: str, seed: int) -> Path:
    """Return canonical split path split/{dataset}/{dataset_seed}."""
    split_root = Path(split_root)
    out_name = f"{dataset_name}_{seed}"
    return split_root / dataset_name / out_name


__all__ = [
    "get_split_path",
    "load_splits",
    "make_stratified_splits",
    "save_splits",
]
