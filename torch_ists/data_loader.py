"""Custom data loading helpers for torch_ists runtime workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .task_registry import normalize_task


_SCALAR_DATAFRAME_TASKS = frozenset(
    {"classification", "regression", "outlier_detection"}
)


def _validate_numpy_target(y: np.ndarray, task: str) -> np.ndarray:
    task_name = normalize_task(task)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D [N] for task={task_name!r}, got shape {y.shape}")
    if task_name == "classification":
        if not np.issubdtype(y.dtype, np.number) or not np.isfinite(y).all():
            raise ValueError("classification targets must be finite and integer-like")
        if not np.equal(y, np.floor(y)).all():
            raise ValueError("classification targets must be integer-like")
        return y.astype(np.int64, copy=False)
    return y


@dataclass
class DatasetBundle:
    """Container for custom dataset payloads."""

    X: np.ndarray
    y: np.ndarray
    timestamps: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def summary(self) -> Dict[str, Any]:
        metadata = dict(self.metadata or {})
        task = normalize_task(str(metadata.get("task", "classification")))
        summary = {
            "num_samples": int(self.X.shape[0]),
            "num_features": int(self.X.shape[1]),
            "seq_len": int(self.X.shape[2]),
            "has_timestamps": self.timestamps is not None,
            "task": task,
            "target_shape": list(self.y.shape[1:]),
            "target_dtype": str(self.y.dtype),
        }
        if task == "classification":
            summary["num_classes"] = int(len(np.unique(self.y)))
        return summary


class DataLoader:
    """Factory-style loader for user-provided datasets."""

    @staticmethod
    def from_numpy(
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        task: str = "classification",
        target_shape: Optional[Sequence[int]] = None,
        target_dtype: Optional[str] = None,
    ) -> DatasetBundle:
        X = np.asarray(X)
        task_name = normalize_task(task)
        y = _validate_numpy_target(np.asarray(y), task_name)

        if X.ndim != 3:
            raise ValueError(f"X must be 3D [N, D, L], got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y sample count mismatch: {X.shape[0]} != {y.shape[0]}")

        if timestamps is not None:
            timestamps = np.asarray(timestamps, dtype=np.float64)
            if timestamps.ndim != 2:
                raise ValueError(f"timestamps must be 2D [N, L], got shape {timestamps.shape}")
            if timestamps.shape[0] != X.shape[0] or timestamps.shape[1] != X.shape[2]:
                raise ValueError(
                    "timestamps shape mismatch: "
                    f"expected [{X.shape[0]}, {X.shape[2]}], got {timestamps.shape}"
                )

        return DatasetBundle(
            X=X,
            y=y,
            timestamps=timestamps,
            metadata={
                "source": "numpy",
                "runtime": "torch_ists",
                "task": task_name,
                "target_shape": list(y.shape[1:]),
                "target_dtype": str(y.dtype),
            },
        )

    @staticmethod
    def from_dataframe(
        df: Any,
        time_col: str,
        feature_cols: Sequence[str],
        label_col: str,
        id_col: Optional[str] = "series_id",
        task: str = "classification",
        target_shape: Optional[Sequence[int]] = None,
        target_dtype: Optional[str] = None,
    ) -> DatasetBundle:
        task_name = normalize_task(task)
        if task_name not in _SCALAR_DATAFRAME_TASKS:
            raise ValueError(
                f"DataFrame loading for task={task_name!r} cannot represent the required "
                "target structure as one scalar target per series; use from_numpy with a "
                "task-owned target payload."
            )
        if target_shape is not None and tuple(target_shape) not in {(), (1,)}:
            raise ValueError("DataFrame loading supports one scalar target per series")

        try:
            import pandas as pd  # type: ignore
        except ImportError as e:
            raise ImportError("pandas is required for from_dataframe") from e

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if time_col not in df.columns:
            raise ValueError(f"Missing time column: {time_col}")
        if label_col not in df.columns:
            raise ValueError(f"Missing label column: {label_col}")
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        if id_col is not None and id_col in df.columns:
            groups = [(sid, g.sort_values(time_col)) for sid, g in df.groupby(id_col, sort=False)]
        else:
            groups = [("single_series", df.sort_values(time_col))]

        target_values = []
        for i, (_, g) in enumerate(groups):
            series_targets = g[label_col].to_numpy()
            if series_targets.ndim != 1 or any(
                np.asarray(value).ndim != 0 for value in series_targets
            ):
                raise ValueError("DataFrame loading supports one scalar target per series")
            first_target = series_targets[0]
            if not np.all(series_targets == first_target):
                raise ValueError(
                    f"{label_col!r} must be constant within each series; "
                    f"series {i} contains multiple targets"
                )
            target_values.append(first_target)

        max_len = max(len(g) for _, g in groups)
        num_samples = len(groups)
        num_features = len(feature_cols)

        X = np.full((num_samples, num_features, max_len), np.nan, dtype=np.float32)
        ts = np.full((num_samples, max_len), np.nan, dtype=np.float64)

        for i, (_, g) in enumerate(groups):
            values = g[list(feature_cols)].to_numpy(dtype=np.float32)
            cur_len = values.shape[0]
            X[i, :, :cur_len] = values.T
            ts[i, :cur_len] = g[time_col].to_numpy(dtype=np.float64)

        raw_targets = np.asarray(target_values)
        if task_name == "classification":
            y = _validate_numpy_target(raw_targets, task_name)
        else:
            if not np.issubdtype(raw_targets.dtype, np.number):
                raise ValueError(f"{task_name} targets must be numeric")
            y = raw_targets

        return DatasetBundle(
            X=X,
            y=y,
            timestamps=ts,
            metadata={
                "source": "dataframe",
                "runtime": "torch_ists",
                "num_groups": num_samples,
                "id_col": id_col if id_col in df.columns else None,
                "task": task_name,
                "target_shape": list(y.shape[1:]),
                "target_dtype": str(y.dtype),
            },
        )

    @staticmethod
    def from_file(path: str, format: Optional[str] = None, **kwargs: Any) -> DatasetBundle:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        fmt = (format or path_obj.suffix.lstrip(".")).lower()
        if fmt == "npz":
            payload = np.load(path_obj, allow_pickle=False)
            keys = set(payload.files)
            x_key = "X" if "X" in keys else "x" if "x" in keys else None
            y_key = "y" if "y" in keys else "Y" if "Y" in keys else None
            if x_key is None or y_key is None:
                raise ValueError(f"npz must contain X/y keys, got {sorted(keys)}")
            timestamps = payload["timestamps"] if "timestamps" in payload else None
            return DataLoader.from_numpy(
                payload[x_key],
                payload[y_key],
                timestamps=timestamps,
                task=kwargs.get("task", "classification"),
                target_shape=kwargs.get("target_shape"),
                target_dtype=kwargs.get("target_dtype"),
            )

        if fmt in {"csv", "parquet"}:
            try:
                import pandas as pd  # type: ignore
            except ImportError as e:
                raise ImportError("pandas is required for csv/parquet loading") from e

            if fmt == "csv":
                df = pd.read_csv(path_obj)
            else:
                df = pd.read_parquet(path_obj)

            required = ("time_col", "feature_cols", "label_col")
            missing = [k for k in required if k not in kwargs]
            if missing:
                raise ValueError(f"Missing kwargs for {fmt} loading: {missing}")

            return DataLoader.from_dataframe(
                df=df,
                time_col=kwargs["time_col"],
                feature_cols=kwargs["feature_cols"],
                label_col=kwargs["label_col"],
                id_col=kwargs.get("id_col", "series_id"),
                task=kwargs.get("task", "classification"),
                target_shape=kwargs.get("target_shape"),
                target_dtype=kwargs.get("target_dtype"),
            )

        raise ValueError(f"Unsupported format: {fmt}")
