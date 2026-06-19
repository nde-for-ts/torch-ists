"""Benchmark dataset loader utilities for runtime."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from urllib import request

import numpy as np
import torch

DATA_ROOT_ENV = "TORCH_ISTS_DATA_ROOT"
load_UCR_UEA_dataset = None
load_from_tsfile_to_dataframe = None


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("torch_ists.benchmark_loader requires pandas. Install torch-ists[data].") from exc
    return pd


def _load_ucr_uea_dataset_lazy(*args, **kwargs):
    try:
        from sktime.datasets import load_UCR_UEA_dataset as loader  # type: ignore
    except ImportError as exc:
        raise ImportError("torch_ists.benchmark_loader requires sktime. Install torch-ists[data].") from exc
    return loader(*args, **kwargs)


def _load_from_tsfile_to_dataframe_lazy(*args, **kwargs):
    try:
        from sktime.datasets import load_from_tsfile_to_dataframe as loader  # type: ignore
    except ImportError as exc:
        raise ImportError("torch_ists.benchmark_loader requires sktime. Install torch-ists[data].") from exc
    return loader(*args, **kwargs)


def _require_sktime_loaders():
    primary = load_UCR_UEA_dataset if load_UCR_UEA_dataset is not None else _load_ucr_uea_dataset_lazy
    fallback = (
        load_from_tsfile_to_dataframe
        if load_from_tsfile_to_dataframe is not None
        else _load_from_tsfile_to_dataframe_lazy
    )
    return primary, fallback


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_data_root() -> Path:
    env_root = os.environ.get(DATA_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (_resolve_repo_root() / "data").resolve()


def _build_ts_candidates(dataname: str, data_root: Path):
    return [
        (
            data_root / "Univariate_ts" / f"{dataname}/{dataname}_TRAIN.ts",
            data_root / "Univariate_ts" / f"{dataname}/{dataname}_TEST.ts",
        ),
        (
            data_root / "Multivariate_ts" / f"{dataname}/{dataname}_TRAIN.ts",
            data_root / "Multivariate_ts" / f"{dataname}/{dataname}_TEST.ts",
        ),
    ]


def download():
    """Download benchmark archives into the local data directory."""
    pd = _require_pandas()
    data_root = _resolve_data_root()
    data_root.mkdir(parents=True, exist_ok=True)

    request.urlretrieve(
        "http://www.timeseriesclassification.com/Downloads/Archives/summaryUnivariate.csv",
        str(data_root / "summaryUnivariate.csv"),
    )
    pd.read_csv(data_root / "summaryUnivariate.csv")["problem"].tolist()

    request.urlretrieve(
        "http://www.timeseriesclassification.com/Downloads/Archives/summaryMultivariate.csv",
        str(data_root / "summaryMultivariate.csv"),
    )
    rows = []
    with (data_root / "summaryMultivariate.csv").open() as handle:
        for row in handle:
            rows.append(row.split(",")[:9])
    pd.DataFrame(rows[1:], columns=rows[0])["Problem"].tolist()

    if not (data_root / "Univariate_ts").exists():
        univariate_zip = data_root / "Univariate2018_ts.zip"
        request.urlretrieve(
            "http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip",
            str(univariate_zip),
        )
        with zipfile.ZipFile(univariate_zip, "r") as handle:
            handle.extractall(str(data_root))

    if not (data_root / "Multivariate_ts").exists():
        multivariate_zip = data_root / "Multivariate2018_ts.zip"
        request.urlretrieve(
            "http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip",
            str(multivariate_zip),
        )
        with zipfile.ZipFile(multivariate_zip, "r") as handle:
            handle.extractall(str(data_root))


def _pad(channel, max_len, *, padding: str = "nan"):
    """Pad a 1D channel to ``max_len``.

    ``padding="nan"`` (default): downstream handlers can detect padded
    positions via the NaN sentinel. ``padding="zero"``: explicit 0 fill.
    ``padding="last"``: legacy forward-fill of the last observed value;
    callers using this mode get NO companion mask, so downstream models
    cannot distinguish padded positions from real observations. Prefer
    ``"nan"`` unless reproducing legacy preprocessing.
    """
    channel = torch.as_tensor(channel, dtype=torch.float32)
    if padding == "nan":
        fill_value = float("nan")
    elif padding == "last":
        fill_value = float(channel[-1])
    elif padding == "zero":
        fill_value = 0.0
    else:
        raise ValueError("padding must be one of {'nan', 'last', 'zero'}")
    out = torch.full((max_len,), fill_value, dtype=channel.dtype)
    out[: channel.size(0)] = channel
    return out


def _validate_padding(padding: str) -> str:
    normalized = str(padding).strip().lower()
    if normalized not in {"nan", "last", "zero"}:
        raise ValueError("padding must be one of {'nan', 'last', 'zero'}")
    return normalized


def _panel_lengths(panel):
    lengths = []
    for i in range(panel.shape[0]):
        for j in range(panel.shape[1]):
            lengths.append(len(panel.iloc[i, j]))
    return lengths


def get_data(dataname, *, label_policy: str = "pooled", padding: str = "nan"):
    """Load a benchmark dataset and return tensor X and label array y.

    ``label_policy="pooled"`` and ``padding="nan"`` are the runtime defaults.
    Variable-length series are padded with NaN by default so downstream
    preprocessing can see the observation mask. Use ``label_policy="train"``
    or ``padding="last"`` only to reproduce older benchmark preprocessing.
    """
    pd = _require_pandas()
    padding = _validate_padding(padding)
    load_UCR_UEA_dataset, load_from_tsfile_to_dataframe = _require_sktime_loaders()
    primary_error = None
    try:
        X_train, y_train = load_UCR_UEA_dataset(dataname, "train", return_X_y=True)
        X_test, y_test = load_UCR_UEA_dataset(dataname, "test", return_X_y=True)
    except Exception as exc:
        primary_error = exc
        data_root = _resolve_data_root()
        errors = []
        for train_path, test_path in _build_ts_candidates(dataname, data_root):
            try:
                X_train, y_train = load_from_tsfile_to_dataframe(str(train_path))
                X_test, y_test = load_from_tsfile_to_dataframe(str(test_path))
                break
            except Exception as fallback_exc:
                errors.append(
                    {
                        "train": str(train_path),
                        "test": str(test_path),
                        "error": f"{type(fallback_exc).__name__}: {fallback_exc}",
                    }
                )
        else:
            searched = [f"{item['train']} | {item['test']}" for item in errors]
            raise FileNotFoundError(
                f"Failed to load dataset '{dataname}'. "
                f"Primary sktime loader error: {type(primary_error).__name__}: {primary_error}. "
                f"Fallback data root: {data_root}. "
                f"Searched paths: {searched}."
            ) from primary_error

    if label_policy == "pooled":
        labels = np.unique(np.concatenate([y_train, y_test]))
    elif label_policy == "train":
        labels = np.unique(y_train)
        test_only = np.setdiff1d(np.unique(y_test), labels, assume_unique=True)
        if test_only.size:
            raise ValueError(
                "label_policy='train' but y_test contains labels not seen in "
                f"y_train: {sorted(test_only.tolist())}. Use the default "
                "label_policy='pooled' to build a label dict from the union "
                "of train and test labels."
            )
    else:
        raise ValueError("label_policy must be one of {'pooled', 'train'}")
    label_dict = dict(zip(labels, range(len(labels))))
    y_train = np.array([label_dict[y] for y in y_train])
    y_test = np.array([label_dict[y] for y in y_test])

    len_list = _panel_lengths(X_train) + _panel_lengths(X_test)

    if min(len_list) != max(len_list) and padding == "last":
        max_len = max(len_list)

        def get_same_len(series):
            resized = np.interp(
                np.arange(0, max_len),
                np.linspace(0, max_len, num=len(series)),
                series,
            )
            return pd.Series(resized)

        X_train = X_train.applymap(lambda s: get_same_len(s))
        X_test = X_test.applymap(lambda s: get_same_len(s))

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    lengths = torch.tensor([len(Xi[0]) for Xi in X])
    max_len = lengths.max()

    X = torch.stack(
        [torch.stack([_pad(channel, max_len, padding=padding) for channel in batch], dim=0) for batch in X],
        dim=0,
    )
    return X, y


__all__ = ["download", "get_data"]
