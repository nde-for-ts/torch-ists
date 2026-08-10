"""Benchmark dataset loader utilities for runtime."""

from __future__ import annotations

import os
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from urllib import request

import numpy as np
import torch

DATA_ROOT_ENV = "TORCH_ISTS_DATA_ROOT"
load_UCR_UEA_dataset = None
load_from_tsfile_to_dataframe = None

DOWNLOAD_PROVENANCE = ".torch_ists_download_provenance.json"
BENCHMARK_DOWNLOAD_MANIFEST = (
    {
        "name": "summary-univariate",
        "url": "https://www.timeseriesclassification.com/Downloads/Archives/summaryUnivariate.csv",
        "sha256": "73091048c1e2fb0b383d10fc4069eb9532eeb1b2289f98ede6dcc00dfd99e72d",
        "target": "summaryUnivariate.csv",
        "kind": "file",
    },
    {
        "name": "summary-multivariate",
        "url": "https://www.timeseriesclassification.com/Downloads/Archives/summaryMultivariate.csv",
        "sha256": "32a3b77dadac8eb93bf3dbb2697d071bc0cac4825cdb04ed47b85ee751fb96ca",
        "target": "summaryMultivariate.csv",
        "kind": "file",
    },
    {
        "name": "univariate-archive",
        "url": "https://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip",
        "sha256": "662b07ff7c0ad14c9e541edb345ec3d5ddf0bd483494e18a8fc8d8ca715481dc",
        "target": "Univariate_ts",
        "kind": "zip",
        "archive_root": "Univariate_ts",
    },
    {
        "name": "multivariate-archive",
        "url": "https://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip",
        "sha256": "77a23258e004eb63a6764edda6e68899c1aa1dfbe0dbe5f405af7021d2ee9b2d",
        "target": "Multivariate_ts",
        "kind": "zip",
        "archive_root": "Multivariate_ts",
    },
)


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "torch_ists.benchmark_loader requires pandas. From this source checkout, run "
            "`python -m pip install -e \".[data]\"`."
        ) from exc
    return pd


def _load_ucr_uea_dataset_lazy(*args, **kwargs):
    try:
        from sktime.datasets import load_UCR_UEA_dataset as loader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "torch_ists.benchmark_loader requires sktime. From this source checkout, run "
            "`python -m pip install -e \".[data]\"`."
        ) from exc
    return loader(*args, **kwargs)


def _load_from_tsfile_to_dataframe_lazy(*args, **kwargs):
    try:
        from sktime.datasets import load_from_tsfile_to_dataframe as loader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "torch_ists.benchmark_loader requires sktime. From this source checkout, run "
            "`python -m pip install -e \".[data]\"`."
        ) from exc
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


def _resolve_data_root(data_root=None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_provenance(path: Path) -> dict:
    if not path.exists():
        return {"assets": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid benchmark provenance sidecar: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), dict):
        raise RuntimeError(f"Invalid benchmark provenance sidecar: {path}")
    return payload


def _write_provenance(path: Path, payload: dict) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _safe_extract(handle: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in handle.infolist():
        member_path = (destination / member.filename).resolve()
        if root != member_path and root not in member_path.parents:
            raise RuntimeError(f"Archive contains an unsafe path: {member.filename!r}")
    handle.extractall(destination)


def _cache_is_verified(target: Path, asset: dict, provenance: dict) -> bool:
    record = provenance.get("assets", {}).get(str(asset["name"]))
    return bool(
        target.exists()
        and isinstance(record, dict)
        and record.get("sha256") == asset.get("sha256")
        and record.get("url") == asset.get("url")
        and record.get("target") == asset.get("target")
    )


def download(*, data_root=None, manifest=None):
    """Transactionally install digest-pinned benchmark assets.

    Existing paths are reused only when the provenance sidecar attests the same
    URL, digest, and target. Unverified caches are never moved or overwritten.
    """
    root = _resolve_data_root(data_root)
    root.mkdir(parents=True, exist_ok=True)
    assets = tuple(BENCHMARK_DOWNLOAD_MANIFEST if manifest is None else manifest)
    provenance_path = root / DOWNLOAD_PROVENANCE
    provenance = _read_provenance(provenance_path)

    for raw_asset in assets:
        asset = dict(raw_asset)
        required = {"name", "url", "sha256", "target", "kind"}
        missing = sorted(required - set(asset))
        if missing:
            raise ValueError(f"Download manifest asset is missing fields: {missing}")
        if not str(asset["url"]).startswith("https://"):
            raise ValueError(f"Benchmark download URL must use HTTPS: {asset['url']!r}")
        expected_digest = str(asset["sha256"]).lower()
        if len(expected_digest) != 64:
            raise ValueError(f"Invalid SHA-256 for asset {asset['name']!r}")
        target = root / str(asset["target"])
        if _cache_is_verified(target, asset, provenance):
            continue
        if target.exists():
            raise RuntimeError(
                f"Existing benchmark cache is not attested by the pinned manifest: {target}. "
                "Choose a new data_root or verify and migrate it explicitly."
            )

        with tempfile.TemporaryDirectory(prefix=".torch_ists-download-", dir=str(root)) as temp_dir:
            transaction = Path(temp_dir)
            payload_path = transaction / "payload"
            request.urlretrieve(str(asset["url"]), str(payload_path))
            actual_digest = _sha256(payload_path)
            if actual_digest != expected_digest:
                raise RuntimeError(
                    f"SHA-256 mismatch for {asset['name']!r}: "
                    f"expected={expected_digest}, actual={actual_digest}"
                )

            kind = str(asset["kind"])
            if kind == "file":
                promoted = payload_path
            elif kind == "zip":
                extract_root = transaction / "extract"
                extract_root.mkdir()
                with zipfile.ZipFile(payload_path, "r") as archive:
                    _safe_extract(archive, extract_root)
                archive_root = extract_root / str(asset.get("archive_root", asset["target"]))
                if not archive_root.is_dir() or not any(archive_root.rglob("*.ts")):
                    raise RuntimeError(
                        f"Archive {asset['name']!r} does not contain the expected dataset tree"
                    )
                promoted = archive_root
            else:
                raise ValueError(f"Unsupported download manifest kind: {kind!r}")
            os.replace(promoted, target)

        provenance.setdefault("assets", {})[str(asset["name"])] = {
            "url": str(asset["url"]),
            "sha256": expected_digest,
            "target": str(asset["target"]),
        }
        _write_provenance(provenance_path, provenance)
    return root


def _pad(channel, max_len, *, padding: str = "nan"):
    """Pad a 1D channel to ``max_len``.

    ``padding="nan"`` (default): downstream handlers can detect padded
    positions via the NaN sentinel. ``padding="zero"``: explicit 0 fill.
    ``padding="last"``: final-value forward-fill of the last observed value
    without a companion padding mask, so downstream models cannot distinguish
    padded positions from real observations. Prefer ``"nan"`` unless matching
    historical artifact compatibility.
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


def get_data(
    dataname,
    *,
    label_policy: str = "pooled",
    padding: str = "nan",
    data_root=None,
):
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
    use_explicit_root = data_root is not None
    try:
        if use_explicit_root:
            raise FileNotFoundError("explicit data_root requires the local archive path")
        X_train, y_train = load_UCR_UEA_dataset(dataname, "train", return_X_y=True)
        X_test, y_test = load_UCR_UEA_dataset(dataname, "test", return_X_y=True)
    except Exception as exc:
        primary_error = exc
        resolved_data_root = _resolve_data_root(data_root)
        errors = []
        for train_path, test_path in _build_ts_candidates(dataname, resolved_data_root):
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
                f"Fallback data root: {resolved_data_root}. "
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


__all__ = [
    "BENCHMARK_DOWNLOAD_MANIFEST",
    "DOWNLOAD_PROVENANCE",
    "download",
    "get_data",
]
