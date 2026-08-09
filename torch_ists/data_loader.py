"""Custom data loading helpers for torch_ists runtime workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class DatasetBundle:
    """Container for custom dataset payloads."""

    X: np.ndarray
    y: np.ndarray
    timestamps: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "num_samples": int(self.X.shape[0]),
            "num_features": int(self.X.shape[1]),
            "seq_len": int(self.X.shape[2]),
            "has_timestamps": self.timestamps is not None,
            "num_classes": int(len(np.unique(self.y))),
        }


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
        y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(f"X must be 3D [N, D, L], got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D [N], got shape {y.shape}")
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
                "task": str(task),
                "target_shape": list(target_shape) if target_shape is not None else None,
                "target_dtype": str(target_dtype) if target_dtype is not None else None,
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

        max_len = max(len(g) for _, g in groups)
        num_samples = len(groups)
        num_features = len(feature_cols)

        X = np.full((num_samples, num_features, max_len), np.nan, dtype=np.float32)
        ts = np.full((num_samples, max_len), np.nan, dtype=np.float64)
        y = np.empty((num_samples,), dtype=np.int64)

        for i, (_, g) in enumerate(groups):
            values = g[list(feature_cols)].to_numpy(dtype=np.float32)
            cur_len = values.shape[0]
            X[i, :, :cur_len] = values.T
            ts[i, :cur_len] = g[time_col].to_numpy(dtype=np.float64)
            y[i] = int(g[label_col].to_numpy()[0])

        return DatasetBundle(
            X=X,
            y=y,
            timestamps=ts,
            metadata={
                "source": "dataframe",
                "runtime": "torch_ists",
                "num_groups": num_samples,
                "id_col": id_col if id_col in df.columns else None,
                "task": str(task),
                "target_shape": list(target_shape) if target_shape is not None else None,
                "target_dtype": str(target_dtype) if target_dtype is not None else None,
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
