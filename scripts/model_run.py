#-*- coding:utf-8 -*-

from __future__ import annotations

import os
import sys
import copy
import hashlib
import math
import time
import datetime
import json
import pickle
import random
import logging
import argparse
import signal
from contextlib import contextmanager
from pathlib import Path

import urllib.request
import zipfile

import numpy as np
import sklearn.model_selection
from sklearn.metrics import *

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal installs
    def tqdm(iterable, *args, **kwargs):
        return iterable

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from _common import (
    CONSTRAINED_CAPACITY_POLICY_NAME,
    CONSTRAINED_MATCHED_CAPACITY_TARGET_NAME,
    DEFAULT_SEARCH_PROFILE,
    build_model_kwargs_from_config,
    configure_runtime_paths,
    default_method_for_model,
    describe_search_profile,
    infer_param_search_profile,
    is_pre_integrity_param_artifact,
    normalize_runtime_baseline_id,
    normalize_search_profile,
    omit_none_runtime_baseline_id,
    resolve_dataset_fixed_batch_policy,
    resolve_learning_rate_policy,
    SINGLE_SEARCH_REPEAT,
    STANDARD_SEARCH_POLICY,
    validate_param_artifact_identity,
    validate_param_artifact_metadata,
)
from dataset_source_contracts import (
    load_dataset_payload,
    normalize_dataset_source,
    resolve_timestamp_delta_metadata,
)
from script_chain_contracts import atomic_pickle_dump, generate_run_id, portable_artifact_reference, write_manifest

sys.dont_write_bytecode = True


def _configure_import_paths() -> None:
    configure_runtime_paths(__file__)


_configure_import_paths()


def _parse_csv_list(raw):
    if raw is None:
        return None
    items = [token.strip() for token in raw.split(",") if token.strip()]
    return items or None


def _normalize_dataset_source_arg(raw: str) -> str:
    try:
        return normalize_dataset_source(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _normalize_param_mode(raw: str) -> str:
    normalized = str(raw).strip().lower()
    if normalized in {"default", "legacy"}:
        return "default"
    if normalized in {"cv", "improved"}:
        return "cv"
    raise argparse.ArgumentTypeError("parameter mode must be one of {'default', 'cv'}.")


def _abspath_or_none(path):
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(str(path)))


def _normalize_search_profile_arg(raw: str) -> str:
    try:
        return normalize_search_profile(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


CONSTRAINED_SEARCH_POLICY = "constrained"
AGGREGATED_SEARCH_REPEAT = "aggregated"
PREDICTION_IDENTITY_SCHEMA = "sample_indices"
UNIVERSAL_100_CELL_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
UNIVERSAL_100_CELL_MECHANISMS = [
    "MCAR",
    "MAR",
    "MNAR",
    "BLOCK",
    "BURST",
    "CHANNEL_DROP",
]
UNIVERSAL_100_CELL_SYNC_OPTIONS = ["false", "true"]


def _normalize_param_policy_arg(raw: str) -> str:
    normalized = str(raw).strip().lower()
    if normalized in {STANDARD_SEARCH_POLICY, CONSTRAINED_SEARCH_POLICY}:
        return normalized
    if normalized == "standard":
        return STANDARD_SEARCH_POLICY
    raise argparse.ArgumentTypeError("param policy must be one of {'default', 'constrained'}.")


def _normalize_param_repeat_arg(raw: str) -> str:
    normalized = str(raw).strip().lower()
    if normalized in {SINGLE_SEARCH_REPEAT, AGGREGATED_SEARCH_REPEAT}:
        return normalized
    raise argparse.ArgumentTypeError("param repeat must be one of {'single', 'aggregated'}.")


def _normalize_param_mode_alias(raw: str) -> tuple[str, str]:
    normalized = str(raw).strip().lower()
    if normalized in {"default", "legacy"}:
        return STANDARD_SEARCH_POLICY, SINGLE_SEARCH_REPEAT
    if normalized in {"cv", "improved"}:
        return STANDARD_SEARCH_POLICY, AGGREGATED_SEARCH_REPEAT
    raise argparse.ArgumentTypeError(
        "--param-mode is deprecated; use --param-policy {default,constrained} "
        "and --param-repeat {single,aggregated}."
    )


def _build_param_artifact_name(
    data_name: str,
    model_name: str,
    *,
    param_policy: str,
    param_repeat: str,
    task: str,
    param_target: str | None = None,
) -> str:
    parts = [data_name, model_name]
    if param_policy == CONSTRAINED_SEARCH_POLICY:
        parts.append(str(param_target) if param_target else CONSTRAINED_MATCHED_CAPACITY_TARGET_NAME)
    if param_repeat == AGGREGATED_SEARCH_REPEAT:
        parts.append(AGGREGATED_SEARCH_REPEAT)
    out_name = "_".join(parts)
    task_name = str(task).strip().lower()
    if task_name != "classification":
        out_name = f"{out_name}__task_{task_name}"
    return out_name


def _artifact_repeat_suffix_stem(artifact_name: str) -> str:
    return str(artifact_name).split("__task_", 1)[0]


def _legacy_aggregated_param_artifact_names(data_name: str, model_name: str, *, task: str) -> list[str]:
    task_name = str(task).strip().lower()
    names = []
    for suffix in ("cv", "improved"):
        out_name = "_".join([data_name, model_name, suffix])
        if task_name != "classification":
            out_name = f"{out_name}__task_{task_name}"
        names.append(out_name)
    return names


def _resolve_param_artifact_path(
    param_root: str,
    data_name: str,
    model_name: str,
    *,
    param_policy: str,
    param_repeat: str,
    task: str,
    param_target: str | None = None,
) -> tuple[str, str]:
    param_name = _build_param_artifact_name(
        data_name,
        model_name,
        param_policy=param_policy,
        param_repeat=param_repeat,
        task=task,
        param_target=param_target,
    )
    param_path = os.path.join(param_root, data_name, param_name)
    if os.path.exists(param_path):
        return param_name, param_path

    if param_policy == STANDARD_SEARCH_POLICY and param_repeat == AGGREGATED_SEARCH_REPEAT:
        for legacy_name in _legacy_aggregated_param_artifact_names(data_name, model_name, task=task):
            legacy_path = os.path.join(param_root, data_name, legacy_name)
            if os.path.exists(legacy_path):
                return legacy_name, legacy_path

    return param_name, param_path


def _expected_param_target_identity(
    *,
    data_name: str,
    task: str,
    target_source: str,
    targets_root: str,
    forecast_horizon,
    model_config: dict,
    dataset_source: str,
    dataset_npz: str | None,
    split_root: str,
    param_repeat: str,
) -> dict:
    task_name = str(task).strip().lower()
    source = "label" if task_name == "classification" else str(target_source).strip().lower()
    if task_name != "classification" and source == "dataset":
        target_path = Path(targets_root) / f"{data_name}.npz"
        if not target_path.exists():
            raise FileNotFoundError(f"Target bundle not found for dataset={data_name}: {target_path}")
    normalized_dataset_source = str(dataset_source).strip().lower()
    if normalized_dataset_source == "npz":
        if dataset_npz is None or not Path(dataset_npz).exists():
            raise FileNotFoundError(f"Dataset NPZ not found: {dataset_npz}")
    split_seeds = (
        [0, 1, 2, 3, 4]
        if param_repeat == AGGREGATED_SEARCH_REPEAT
        else [int(model_config.get("search_algorithm_seed", 42))]
    )
    for split_seed in split_seeds:
        split_path = Path(split_root) / data_name / f"{data_name}_{split_seed}"
        if not split_path.exists():
            raise FileNotFoundError(f"Split artifact not found: {split_path}")
    return {
        "target_source": source,
        "forecast_horizon": (
            int(forecast_horizon)
            if task_name == "forecasting" and forecast_horizon is not None
            else None
        ),
        "dataset_source": normalized_dataset_source,
        "split_seeds": split_seeds,
    }


def _infer_cli_search_policy(model_config: dict) -> str:
    raw = str(model_config.get("search_policy", STANDARD_SEARCH_POLICY)).strip().lower()
    if raw in {STANDARD_SEARCH_POLICY, CONSTRAINED_SEARCH_POLICY}:
        return raw
    if raw == "standard":
        return STANDARD_SEARCH_POLICY
    return STANDARD_SEARCH_POLICY


def _infer_cli_search_repeat(model_config: dict, artifact_name: str | None = None) -> str:
    raw = str(model_config.get("search_repeat", "")).strip().lower()
    if raw in {SINGLE_SEARCH_REPEAT, AGGREGATED_SEARCH_REPEAT}:
        return raw
    if artifact_name:
        stem = _artifact_repeat_suffix_stem(artifact_name)
        if stem.endswith("_aggregated") or stem.endswith("_cv") or stem.endswith("_improved"):
            return AGGREGATED_SEARCH_REPEAT
    return SINGLE_SEARCH_REPEAT



def _requires_flow_raw_input(model_name: str) -> bool:
    from torch_ists.config.alias_policy import is_flow_strict_raw_name

    return is_flow_strict_raw_name(str(model_name))


def _validate_param_search_profile(
    model_config: dict,
    *,
    expected_search_profile: str,
    expected_param_policy: str,
    expected_param_repeat: str,
    expected_data_name: str,
    expected_model_name: str,
    expected_task: str,
    expected_target_source: str,
    expected_forecast_horizon,
    expected_dataset_source: str,
    expected_split_seeds,
    param_path: str,
    allow_legacy_search_profile: bool = False,
) -> str:
    actual = validate_param_artifact_metadata(
        model_config,
        context=f"Param artifact {param_path}",
        allow_legacy_search_profile=allow_legacy_search_profile,
        require_lr=True,
    )
    expected = normalize_search_profile(expected_search_profile)
    if actual != expected:
        raise ValueError(
            "Param artifact search_profile mismatch: "
            f"{param_path} has {actual!r}, expected {expected!r}. "
            "Regenerate params with the current default search, or use an "
            "compatibility profile only for intentional migration checks."
        )
    actual_policy = _infer_cli_search_policy(model_config)
    actual_repeat = _infer_cli_search_repeat(model_config, os.path.basename(param_path))
    if actual_policy != expected_param_policy or actual_repeat != expected_param_repeat:
        raise ValueError(
            "Param artifact search axes mismatch: "
            f"{param_path} has policy={actual_policy!r}, repeat={actual_repeat!r}; "
            f"expected policy={expected_param_policy!r}, repeat={expected_param_repeat!r}."
        )
    if expected_param_policy == CONSTRAINED_SEARCH_POLICY:
        capacity_policy = str(model_config.get("capacity_policy", "")).strip()
        if capacity_policy != CONSTRAINED_CAPACITY_POLICY_NAME:
            raise ValueError(
                "Constrained param artifact is missing current capacity metadata: "
                f"{param_path} has capacity_policy={capacity_policy!r}. "
                "Regenerate params with scripts/param_search.py --search-policy constrained."
            )
    legacy_integrity_shape = is_pre_integrity_param_artifact(model_config)
    if not (allow_legacy_search_profile and legacy_integrity_shape):
        validate_param_artifact_identity(
            model_config,
            data_name=expected_data_name,
            model_name=expected_model_name,
            task=expected_task,
            target_source=expected_target_source,
            forecast_horizon=expected_forecast_horizon,
            dataset_source=expected_dataset_source,
            split_seeds=expected_split_seeds,
            context=f"Param artifact {param_path}",
        )
    return actual


def _param_search_contract_metadata(model_config: dict, search_profile: str) -> dict:
    del search_profile
    return {
        "search_policy": _infer_cli_search_policy(model_config),
        "search_repeat": _infer_cli_search_repeat(model_config),
    }


def _parse_missing_rates(raw):
    if raw is None:
        return [0.0]
    if not isinstance(raw, str):
        values = [float(value) for value in raw]
        if not values:
            raise ValueError("At least one missing rate must be provided.")
        return values
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("At least one missing rate must be provided.")
    return values


def _parse_robustness_mechanisms(raw):
    if raw is None:
        raw = "MCAR"
    if isinstance(raw, str):
        tokens = [token.strip().upper() for token in raw.split(",") if token.strip()]
    else:
        tokens = [str(token).strip().upper() for token in raw if str(token).strip()]
    supported = {"MCAR", "MAR", "MNAR", "BLOCK", "BURST", "CHANNEL_DROP"}
    if not tokens:
        raise ValueError("At least one robustness missingness mechanism must be provided.")
    invalid = sorted(set(tokens) - supported)
    if invalid:
        raise ValueError(f"Unsupported robustness missingness mechanism(s): {', '.join(invalid)}")
    return list(dict.fromkeys(tokens))


def _parse_robustness_channel_sync_options(raw):
    if raw is None:
        raw = "false"
    if isinstance(raw, str):
        tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    else:
        tokens = [str(token).strip().lower() for token in raw if str(token).strip()]
    aliases = {
        "auto": "false",
        "true": "true",
        "t": "true",
        "1": "true",
        "yes": "true",
        "false": "false",
        "f": "false",
        "0": "false",
        "no": "false",
    }
    resolved = []
    for token in tokens:
        if token not in aliases:
            raise ValueError("robustness channel sync options must be drawn from {false,true}")
        resolved.append(aliases[token])
    if not resolved:
        raise ValueError("At least one robustness channel_sync option must be provided.")
    return list(dict.fromkeys(resolved))


def _resolve_channel_sync_options(options, use_raw_flow_input):
    del use_raw_flow_input
    resolved = []
    for option in _parse_robustness_channel_sync_options(options):
        value = option == "true"
        if value not in resolved:
            resolved.append(value)
    return resolved


def _is_zero_missing_rate(rate):
    return abs(float(rate)) < 1e-12


def _effective_robustness_test_rates(rates):
    parsed = _parse_missing_rates(rates)
    positives = []
    for rate in parsed:
        rate = float(rate)
        if _is_zero_missing_rate(rate):
            continue
        if not any(abs(rate - existing) < 1e-12 for existing in positives):
            positives.append(rate)
    positives = sorted(positives)
    return [0.0] + positives


class CellTimeoutError(TimeoutError):
    """Raised when a configured model-run cell exceeds its wall-clock budget."""


class SolverBudgetExcluded(RuntimeError):
    """Raised when a configured solver-budget policy excludes a cell before training."""


@contextmanager
def _cell_timeout(seconds: float):
    budget = float(seconds or 0.0)
    if budget <= 0.0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handle_timeout(signum, frame):
        del signum, frame
        raise CellTimeoutError(f"cell wall-clock budget exceeded ({budget:.1f}s)")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer and previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _runtime_status_for_elapsed(elapsed_s: float, performance_limit_s: float) -> str:
    limit = float(performance_limit_s or 0.0)
    if limit > 0.0 and float(elapsed_s) > limit:
        return "performance_limited"
    return "completed"


def _solver_budget_exclusion_reason(model_name: str, seq_len: int, policy: str, threshold: int) -> str | None:
    normalized_policy = str(policy or "allow").strip().lower()
    if normalized_policy == "allow":
        return None
    if normalized_policy != "exclude-long-adaptive-ode":
        raise ValueError("solver_budget_policy must be one of {'allow','exclude-long-adaptive-ode'}")
    threshold = int(threshold or 0)
    if threshold <= 0 or int(seq_len) < threshold:
        return None
    normalized_model = str(model_name).strip().lower()
    adaptive_models = {"ode-lstm", "latentode"}
    if normalized_model in adaptive_models:
        return (
            f"{normalized_model} excluded by solver budget policy "
            f"for seq_len={int(seq_len)} >= {threshold}"
        )
    return None


def _hash_array_for_cache(value):
    if value is None:
        return None
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.view(np.uint8).tobytes())
    return digest.hexdigest()


def _preprocess_cache_contract(
    *,
    model_name,
    X,
    missing_rate,
    interpolate,
    use_intensity,
    mechanism,
    seed,
    channel_sync,
    timestamps,
    timestamp_delta_scale,
    missingness_reference,
):
    return {
        "package_version": str(globals().get("TORCH_ISTS_PACKAGE_VERSION", "unknown")),
        "model_name": str(model_name),
        "x_hash": _hash_array_for_cache(X),
        "missing_rate": float(missing_rate),
        "interpolate": str(interpolate),
        "use_intensity": bool(use_intensity),
        "mechanism": str(mechanism).upper(),
        "seed": int(seed),
        "channel_sync": bool(channel_sync),
        "timestamps_hash": _hash_array_for_cache(timestamps),
        "timestamp_delta_scale": None if timestamp_delta_scale is None else float(timestamp_delta_scale),
        "missingness_reference_hash": _hash_array_for_cache(missingness_reference),
    }


def _preprocess_with_optional_cache(*, coeff_cache_dir=None, no_coeff_cache=False, cache_events=None, **kwargs):
    model_name = kwargs.pop("model_name", "")
    cache_enabled = bool(coeff_cache_dir) and not bool(no_coeff_cache)
    if not cache_enabled:
        return preprocess(**kwargs)
    cache_root = Path(coeff_cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    contract = _preprocess_cache_contract(model_name=model_name, **kwargs)
    cache_key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = cache_root / f"{cache_key}.pkl"
    event = {"enabled": True, "cache_key": cache_key, "hit": False}
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if not isinstance(cached, dict) or "contract" not in cached or "payload" not in cached:
                raise ValueError("preprocess cache payload is malformed")
            if cached.get("contract") != contract:
                raise ValueError(f"preprocess cache contract mismatch for key={cache_key}")
        except Exception:
            event["corrupt"] = True
        else:
            event["hit"] = True
            if cache_events is not None:
                cache_events.append(event)
            return cached["payload"]
    payload = preprocess(**kwargs)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump({"contract": contract, "payload": payload}, f)
    os.replace(tmp_path, cache_path)
    if cache_events is not None:
        cache_events.append(event)
    return payload


def _summarize_cache_events(cache_events):
    events = list(cache_events or [])
    return {
        "enabled": bool(events),
        "lookups": int(len(events)),
        "hits": int(sum(1 for event in events if event.get("hit"))),
        "misses": int(sum(1 for event in events if not event.get("hit"))),
        "corrupt": int(sum(1 for event in events if event.get("corrupt"))),
    }


def _evaluation_regime_for_missing_rate(rate):
    return "standard" if _is_zero_missing_rate(rate) else "missingness"


def _missing_type_for_missing_rate(rate):
    return "STANDARD" if _is_zero_missing_rate(rate) else "MCAR"


def _axis_token(values):
    return "-".join(
        str(value).strip().lower().replace("-", "m").replace("_", "-").replace(".", "p")
        for value in values
    )


def _build_robustness_axis_suffix(
    robustness_test_rates=None,
    robustness_test_mechanisms=None,
    robustness_channel_sync_options=None,
):
    rates = _effective_robustness_test_rates(robustness_test_rates)
    mechanisms = _parse_robustness_mechanisms(robustness_test_mechanisms)
    sync_options = _parse_robustness_channel_sync_options(robustness_channel_sync_options)
    if all(_is_zero_missing_rate(rate) for rate in rates):
        return ""
    return "__" + "__".join(
        [
            "rates_" + _axis_token(rates),
            "types_" + _axis_token(mechanisms),
            "sync_" + _axis_token(sync_options),
        ]
    )


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation loop. Defaults to one BasicMotions-cnn standard configuration; comma-separated inputs expand the run grid."
    )
    parser.add_argument("seed_positional", nargs="?", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="BasicMotions",
        help="Comma-separated dataset list. Default: BasicMotions.",
    )
    parser.add_argument(
        "--dataset-source",
        type=_normalize_dataset_source_arg,
        default="default",
        metavar="{default,npz}",
        help="Dataset source backend. default=packaged built-in dataset loader, npz=local file payload.",
    )
    parser.add_argument(
        "--dataset-npz",
        type=str,
        default=None,
        help="Path to .npz dataset payload used when --dataset-source=npz.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="cnn",
        help="Comma-separated model list. Default: cnn.",
    )
    parser.add_argument(
        "--missing-rates",
        type=str,
        default="0.0",
        help="Comma-separated missing rates. Default: 0.0.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection. auto uses CUDA when available, cpu forces CPU, cuda requires CUDA.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum number of runs to satisfy gate count.",
    )
    parser.add_argument(
        "--max-runs-mode",
        type=str,
        choices=["completed", "new"],
        default="completed",
        help="Count mode for --max-runs. completed=executed+reused, new=executed only.",
    )
    parser.add_argument(
        "--existing-policy",
        type=str,
        choices=["reuse", "fail"],
        default="reuse",
        help="Behavior when --check finds an output artifact: reuse verifies stored output details; fail raises an error.",
    )
    parser.add_argument(
        "--skip-policy",
        type=str,
        choices=["allow", "fail"],
        default="fail",
        help="Whether skipped dataset/model cells keep exit status zero or fail the run.",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        default=True,
        help="Validate and reuse an existing output file when present (default); see --existing-policy for reuse vs fail behavior.",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Disable output existence check.",
    )
    parser.add_argument(
        "--param-policy",
        type=_normalize_param_policy_arg,
        default=None,
        metavar="{default,constrained}",
        help=(
            "Parameter search policy to load. default=model-specific package search; "
            "constrained=experimental capacity-controlled params."
        ),
    )
    parser.add_argument(
        "--param-repeat",
        type=_normalize_param_repeat_arg,
        default=None,
        metavar="{single,aggregated}",
        help="Parameter repeat policy to load. single uses the single-seed params; aggregated uses fixed-seed aggregation params.",
    )
    parser.add_argument(
        "--param-mode",
        dest="param_mode_alias",
        type=_normalize_param_mode_alias,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--search-profile",
        type=_normalize_search_profile_arg,
        default=DEFAULT_SEARCH_PROFILE,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-legacy-search-profile",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--robustness-mode",
        action="store_true",
        help="Robustness evaluation: train on missing=0.0 and test on configured rates.",
    )
    parser.add_argument(
        "--robustness-profile",
        choices=["custom", "100-cell"],
        default=None,
        help=(
            "Robustness axis profile. --robustness-mode defaults to the 100-cell grid; "
            "custom keeps explicitly configured axes."
        ),
    )
    parser.add_argument(
        "--robustness-test-rates",
        type=str,
        default="0.0",
        help=(
            "Comma-separated positive test missing rates for robustness mode. "
            "The clean 0.0 STANDARD anchor is always included and de-duplicated."
        ),
    )
    parser.add_argument(
        "--robustness-test-mechanisms",
        type=str,
        default="MCAR",
        help=(
            "Comma-separated test missingness mechanisms for robustness mode: "
            "MCAR,MAR,MNAR,BLOCK,BURST,CHANNEL_DROP"
        ),
    )
    parser.add_argument(
        "--robustness-channel-sync-options",
        type=str,
        default="false",
        help=(
            "Comma-separated channel_sync options for robustness mode: false,true. "
            "false masks each channel independently; true shares missing time indices across channels."
        ),
    )
    parser.add_argument(
        "--split-root",
        type=str,
        default="split",
        help="Input split artifact root directory.",
    )
    parser.add_argument(
        "--param-root",
        type=str,
        default="params",
        help="Input parameter artifact root directory.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="out",
        help="Output prediction artifact root directory.",
    )
    parser.add_argument(
        "--contract-dir",
        type=str,
        default="evidence",
        help="Directory for JSON run logs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier for related outputs.",
    )
    parser.add_argument(
        "--runtime-baseline-id",
        type=str,
        default=None,
        help="Optional user-provided run label. Leave unset for shared artifacts.",
    )
    parser.add_argument(
        "--manifest-existing-policy",
        choices=["fail", "reuse", "overwrite"],
        default="fail",
        help="Behavior when the stage manifest for --run-id already exists (default: fail).",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        choices=["full", "summary"],
        default="full",
        help="Prediction payload mode. full=store arrays, summary=store shape/dtype metadata only.",
    )
    parser.add_argument(
        "--coeff-cache-dir",
        type=str,
        default=None,
        help="Optional preprocessing/coefficient cache directory. Disabled unless this path is provided.",
    )
    parser.add_argument(
        "--no-coeff-cache",
        action="store_true",
        help="Disable preprocessing/coefficient cache even when --coeff-cache-dir is set.",
    )
    parser.add_argument(
        "--cell-timeout-seconds",
        type=float,
        default=0.0,
        help="Optional per-cell wall-clock timeout in seconds. 0 disables timeout.",
    )
    parser.add_argument(
        "--performance-limit-seconds",
        type=float,
        default=0.0,
        help="Optional per-cell wall-clock label threshold. Completed cells above this are marked performance_limited.",
    )
    parser.add_argument(
        "--solver-budget-policy",
        type=str,
        choices=["allow", "exclude-long-adaptive-ode"],
        default="allow",
        help="Operational solver-budget policy for long sequence adaptive ODE cells.",
    )
    parser.add_argument(
        "--solver-budget-seq-len-threshold",
        type=int,
        default=0,
        help="Sequence length threshold used by --solver-budget-policy. 0 disables preflight exclusion.",
    )
    parser.add_argument(
        "--task",
        choices=[
            "classification",
            "regression",
            "forecasting",
            "imputation",
            "outlier_detection",
            "change_detection",
        ],
        default="classification",
        help="Task head. Classification is stable; other task heads are experimental.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=None,
        help="Required positive forecasting horizon when --task=forecasting.",
    )
    parser.add_argument(
        "--target-source",
        type=str,
        choices=["dataset", "surrogate"],
        default="dataset",
        help="Target source for non-classification tasks; surrogate targets are rejected.",
    )
    parser.add_argument(
        "--targets-root",
        type=str,
        default="targets",
        help="Root directory for dataset target bundles (<targets-root>/<dataset>.npz).",
    )
    args = parser.parse_args()
    if args.param_mode_alias is not None:
        if args.param_policy is not None or args.param_repeat is not None:
            parser.error("--param-mode is a deprecated alias and cannot be combined with --param-policy or --param-repeat.")
        args.param_policy, args.param_repeat = args.param_mode_alias
    args.param_policy = args.param_policy or STANDARD_SEARCH_POLICY
    args.param_repeat = args.param_repeat or SINGLE_SEARCH_REPEAT
    if args.seed is None:
        args.seed = args.seed_positional if args.seed_positional is not None else 0
    if args.robustness_profile is None:
        args.robustness_profile = "100-cell" if args.robustness_mode else "custom"
    args.datasets = _parse_csv_list(args.datasets)
    args.models = _parse_csv_list(args.models)
    args.missing_rates = _parse_missing_rates(args.missing_rates)
    args.robustness_test_rates = _parse_missing_rates(args.robustness_test_rates)
    args.robustness_test_mechanisms = _parse_robustness_mechanisms(args.robustness_test_mechanisms)
    args.robustness_channel_sync_options = _parse_robustness_channel_sync_options(args.robustness_channel_sync_options)
    if args.robustness_profile == "100-cell":
        args.robustness_mode = True
        args.robustness_test_rates = list(UNIVERSAL_100_CELL_RATES)
        args.robustness_test_mechanisms = list(UNIVERSAL_100_CELL_MECHANISMS)
        args.robustness_channel_sync_options = list(UNIVERSAL_100_CELL_SYNC_OPTIONS)
        args.save_predictions = "full"
    args.dataset_source = normalize_dataset_source(args.dataset_source)
    if args.dataset_source == "npz":
        if args.dataset_npz is None:
            parser.error("--dataset-npz is required when --dataset-source=npz")
        if not args.datasets or len(args.datasets) != 1:
            parser.error("--dataset-source=npz requires exactly one dataset in --datasets")
    args.split_root = os.path.abspath(os.path.expanduser(str(args.split_root)))
    args.param_root = os.path.abspath(os.path.expanduser(str(args.param_root)))
    args.out_root = os.path.abspath(os.path.expanduser(str(args.out_root)))
    args.contract_dir = os.path.abspath(os.path.expanduser(str(args.contract_dir)))
    args.targets_root = os.path.abspath(os.path.expanduser(str(args.targets_root)))
    args.coeff_cache_dir = None if args.coeff_cache_dir is None else os.path.abspath(os.path.expanduser(str(args.coeff_cache_dir)))
    args.dataset_npz = _abspath_or_none(args.dataset_npz)
    return args

# setup seed
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 0
seed_everything(SEED)

# CUDA for PyTorch
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def _resolve_run_device(device_mode: str) -> torch.device:
    normalized = str(device_mode).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    raise ValueError("device must be one of {'auto', 'cpu', 'cuda'}")

# Import from the packaged torch_ists runtime.
from torch_ists.preprocessing import apply_missingness, preprocess
from torch_ists import __version__ as TORCH_ISTS_PACKAGE_VERSION, ists_classifier, ists_model
from torch_ists.channel_drop import build_channel_drop_plan
from _training_runtime import (
    _align_prediction_target,
    apply_train_normalization_stats,
    compute_train_normalization_stats,
    compute_train_missing_summary,
    enforce_train_missing_gate,
    evaluate,
    ists_dataset,
    normalize_with_train_stats,
    run_train_valid_loop,
    train,
)
from task_target_contracts import (
    load_task_targets_npz,
    mask_imputation_inputs,
    validate_task_splits,
)
from torch_ists.config import (
    get_data_config,
    get_active_model_list,
    get_default_selected_datasets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("model_run")


def _expand_mask_or_delta(feature, x_missing):
    if feature.dim() == 3:
        if feature.shape == x_missing.shape:
            return feature
        if feature.shape[:2] == x_missing.shape[:2] and feature.shape[-1] == 1:
            return feature.repeat((1, 1, x_missing.shape[-1]))
        feature = feature[..., 0]
    return feature.unsqueeze(-1).repeat((1, 1, x_missing.shape[-1]))


def _validate_bounded_metric(name, value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"metric {name} must be finite, got {value}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"metric {name} must be in [0, 1], got {value}")
    return value


class OutputArtifactContractError(RuntimeError):
    """Raised when --check finds an output artifact from a different run contract."""


def _build_robustness_axes_metadata(
    *,
    robustness_test_rates,
    robustness_test_mechanisms,
    robustness_channel_sync_options,
    use_raw_flow_input,
    seed,
):
    effective_rates = _effective_robustness_test_rates(robustness_test_rates)
    resolved_sync_options = _resolve_channel_sync_options(
        robustness_channel_sync_options,
        use_raw_flow_input,
    )
    mechanisms = _parse_robustness_mechanisms(robustness_test_mechanisms)
    positive_rate_count = sum(not _is_zero_missing_rate(rate) for rate in effective_rates)
    temporal_mechanism_count = sum(mechanism != "CHANNEL_DROP" for mechanism in mechanisms)
    channel_drop_count = int("CHANNEL_DROP" in mechanisms)
    expected_evaluations = 1 + positive_rate_count * (
        temporal_mechanism_count * len(resolved_sync_options) + channel_drop_count
    )
    is_universal_100_cell = (
        len(effective_rates) == len(UNIVERSAL_100_CELL_RATES)
        and np.allclose(effective_rates, UNIVERSAL_100_CELL_RATES, rtol=0.0, atol=1e-12)
        and mechanisms == UNIVERSAL_100_CELL_MECHANISMS
        and resolved_sync_options == [False, True]
    )
    if is_universal_100_cell and expected_evaluations != 100:
        raise RuntimeError(f"100-cell profile materialized {expected_evaluations} axes")
    return {
        "missing_seed": int(seed),
        "test_missing_rates": [float(rate) for rate in effective_rates],
        "test_missing_types": mechanisms,
        "test_channel_sync_options": resolved_sync_options,
        "expected_evaluations": int(expected_evaluations),
    }


def _build_output_contract(
    *,
    data_name,
    model_name,
    model_parameters,
    seed,
    task,
    target_source,
    forecast_horizon,
    dataset_source,
    data_shape,
    num_class,
    missing_rate,
    evaluation_regime,
    train_missing_type,
    train_channel_sync,
    epochs,
    save_predictions,
    interpolate,
    use_intensity,
    batch_policy,
    lr_policy,
    timestamp_summary=None,
    robustness_axes=None,
    training_loop_policy=None,
):
    task_name = str(task)
    task_target = None
    if task_name != "classification":
        task_target = {
            "source": str(target_source),
            "forecast_horizon": (
                int(forecast_horizon) if task_name == "forecasting" else None
            ),
            "dataset_target_required": True,
            "supervision_mask_required": task_name in {"imputation", "change_detection"},
        }
    return {
        "dataset": str(data_name),
        "dataset_source": str(dataset_source),
        "data_shape": [int(value) for value in data_shape],
        "num_class": int(num_class),
        "model": str(model_name),
        "model_parameters": copy.deepcopy(model_parameters),
        "seed": int(seed),
        "task": task_name,
        "target_source": str(target_source if task_name != "classification" else "label"),
        "forecast_horizon": None if forecast_horizon is None else int(forecast_horizon),
        "task_target": task_target,
        "train_missing_rate": float(missing_rate),
        "train_missing_type": str(train_missing_type).upper(),
        "train_channel_sync": None if train_channel_sync is None else bool(train_channel_sync),
        "evaluation_regime": str(evaluation_regime),
        "epochs": int(epochs),
        "save_predictions": str(save_predictions),
        "prediction_identity_schema": PREDICTION_IDENTITY_SCHEMA,
        "interpolate": str(interpolate),
        "use_intensity": bool(use_intensity),
        "batch_policy": batch_policy,
        "lr_policy": lr_policy,
        "timestamp_summary": timestamp_summary,
        "robustness_axes": robustness_axes,
        "training_loop_policy": training_loop_policy,
    }


def _read_output_payload(output_path):
    with open(output_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise OutputArtifactContractError(
            f"Output artifact {output_path} must contain a dict payload, got {type(payload).__name__}."
        )
    return payload


def _expected_robustness_axis_keys(axes):
    rates = [float(rate) for rate in axes.get("test_missing_rates", [])]
    mechanisms = [str(value).upper() for value in axes.get("test_missing_types", [])]
    sync_options = [bool(value) for value in axes.get("test_channel_sync_options", [])]
    seed = int(axes.get("missing_seed", 0))
    positive_rates = [rate for rate in rates if not _is_zero_missing_rate(rate)]
    expected = [(0.0, "STANDARD", None, seed)]
    for rate in positive_rates:
        for mechanism in mechanisms:
            if mechanism == "CHANNEL_DROP":
                continue
            for channel_sync in sync_options:
                expected.append((rate, mechanism, channel_sync, seed))
    if "CHANNEL_DROP" in mechanisms:
        expected.extend((rate, "CHANNEL_DROP", None, seed) for rate in positive_rates)
    return expected


def _evaluation_axis_key(evaluation):
    return (
        float(evaluation.get("test_missing_rate")),
        str(evaluation.get("test_missing_type", "")).upper(),
        evaluation.get("test_channel_sync"),
        int(evaluation.get("missing_seed")),
    )


def _expected_task_prediction_shape(task, count, expected_contract, output_path):
    data_shape = expected_contract.get("data_shape") if isinstance(expected_contract, dict) else None
    if not isinstance(data_shape, (list, tuple)) or len(data_shape) != 3:
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: task prediction shape details are missing."
        )
    input_dim = int(data_shape[1])
    seq_len = int(data_shape[2])
    if task in {"regression", "outlier_detection"}:
        return (count, 1)
    if task == "forecasting":
        horizon = expected_contract.get("forecast_horizon")
        if horizon is None or int(horizon) <= 0:
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: forecasting horizon is invalid."
            )
        return (count, int(horizon), input_dim)
    if task == "imputation":
        return (count, seq_len, input_dim)
    if task == "change_detection":
        return (count, seq_len, 1)
    raise OutputArtifactContractError(
        f"Output artifact {output_path} cannot be reused: unsupported task {task!r}."
    )


def _validate_prediction_payload(
    predictions,
    *,
    expected_storage,
    task,
    expected_contract=None,
    expected_sample_indices=None,
    expected_y_true=None,
    output_path,
):
    if not isinstance(predictions, dict):
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: missing predictions object."
        )
    storage = predictions.get("storage")
    if storage != expected_storage:
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: prediction storage "
            f"{storage!r} does not match {expected_storage!r}."
        )
    if predictions.get("prediction_identity_schema") != PREDICTION_IDENTITY_SCHEMA:
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: prediction identity schema is invalid."
        )
    if predictions.get("sample_index_dtype") != "int64":
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: sample index dtype is invalid."
        )
    count = predictions.get("sample_index_count")
    if count is None or int(count) < 0:
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: sample_index_count is missing or invalid."
        )
    count = int(count)
    expected_indices = None
    if expected_sample_indices is not None:
        expected_indices = np.ascontiguousarray(
            np.asarray(expected_sample_indices, dtype=np.int64).reshape(-1)
        )
    sample_indices = predictions.get("sample_indices")
    indices = None
    if sample_indices is not None:
        indices = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1))
        if len(indices) != count:
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: sample_index_count does not match "
                "the stored sample indices."
            )
        if expected_indices is not None and not np.array_equal(indices, expected_indices):
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: sample indices do not match current split."
            )
    elif storage == "full" or task != "classification":
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: stored sample indices are required."
        )
    elif expected_indices is not None and len(expected_indices) != count:
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: summary sample indices do not match current split."
        )

    if storage == "full":
        if task == "classification":
            num_class = int((expected_contract or {}).get("num_class", 0))
            y_true_array = np.asarray(predictions.get("y_true"))
            y_pred_array = np.asarray(predictions.get("y_pred"))
            logit_array = np.asarray(predictions.get("logit_list"))
            if y_true_array.shape != (count,):
                raise OutputArtifactContractError(
                    f"Output artifact {output_path} cannot be reused: y_true does not match sample_index_count."
                )
            if expected_y_true is not None and not np.array_equal(
                y_true_array,
                np.asarray(expected_y_true).reshape(-1),
            ):
                raise OutputArtifactContractError(
                    f"Output artifact {output_path} cannot be reused: y_true does not match current split."
                )
            if (
                y_pred_array.shape != (count,)
                or not np.isfinite(y_pred_array).all()
                or not np.equal(y_pred_array, np.floor(y_pred_array)).all()
                or np.any(y_pred_array < 0)
                or np.any(y_pred_array >= num_class)
            ):
                raise OutputArtifactContractError(
                    f"Output artifact {output_path} cannot be reused: y_pred is invalid."
                )
            if logit_array.shape != (count, num_class) or not np.isfinite(logit_array).all():
                raise OutputArtifactContractError(
                    f"Output artifact {output_path} cannot be reused: logit_list is invalid."
                )
        else:
            expected_shape = _expected_task_prediction_shape(
                task,
                count,
                expected_contract,
                output_path,
            )
            if any(predictions.get(key) is not None for key in ("y_true", "y_pred", "logit_list")):
                raise OutputArtifactContractError(
                    f"Output artifact {output_path} cannot be reused: non-classification predictions "
                    "contain classification arrays."
                )
            for key in ("target", "pred"):
                values = predictions.get(key)
                if values is None:
                    raise OutputArtifactContractError(
                        f"Output artifact {output_path} cannot be reused: {key} is missing."
                    )
                array = np.asarray(values)
                if tuple(array.shape) != expected_shape:
                    raise OutputArtifactContractError(
                        f"Output artifact {output_path} cannot be reused: {key} shape "
                        f"{tuple(array.shape)} does not match {expected_shape}."
                    )
                if not np.isfinite(array).all():
                    raise OutputArtifactContractError(
                        f"Output artifact {output_path} cannot be reused: {key} must be finite."
                    )
            if task in {"imputation", "change_detection"}:
                target_mask = predictions.get("target_mask")
                if target_mask is None:
                    raise OutputArtifactContractError(
                        f"Output artifact {output_path} cannot be reused: target_mask is missing."
                    )
                mask_array = np.asarray(target_mask)
                if tuple(mask_array.shape) != expected_shape:
                    raise OutputArtifactContractError(
                        f"Output artifact {output_path} cannot be reused: target_mask shape is invalid."
                    )
                if (
                    not np.isfinite(mask_array).all()
                    or not np.isin(mask_array, [0, 1]).all()
                    or int(np.count_nonzero(mask_array)) <= 0
                ):
                    raise OutputArtifactContractError(
                        f"Output artifact {output_path} cannot be reused: target_mask must contain "
                        "finite binary supervision."
                    )
    else:
        if any(predictions.get(key) is not None for key in ("y_true", "y_pred", "logit_list")):
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: summary predictions contain full arrays."
            )
        if int(predictions.get("n_samples", -1)) != count:
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: sample_index_count does not match "
                "the summary sample count."
            )
        if task == "classification" and int(predictions.get("n_classes", -1)) != int(
            (expected_contract or {}).get("num_class", -1)
        ):
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: summary class count is invalid."
            )
        if task != "classification":
            if any(predictions.get(key) is not None for key in ("target", "pred", "target_mask")):
                raise OutputArtifactContractError(
                    f"Output artifact {output_path} cannot be reused: summary predictions contain task arrays."
                )
            expected_shape = list(
                _expected_task_prediction_shape(task, count, expected_contract, output_path)
            )
            for key in ("task_target_shape", "task_pred_shape"):
                if predictions.get(key) != expected_shape:
                    raise OutputArtifactContractError(
                        f"Output artifact {output_path} cannot be reused: {key} shape is invalid."
                    )
            if task in {"imputation", "change_detection"}:
                if predictions.get("task_mask_shape") != expected_shape:
                    raise OutputArtifactContractError(
                        f"Output artifact {output_path} cannot be reused: task_mask_shape is invalid."
                    )
                if int(predictions.get("supervised_element_count", 0)) <= 0:
                    raise OutputArtifactContractError(
                        f"Output artifact {output_path} cannot be reused: supervised_element_count "
                        "is missing or invalid."
                    )


def _validate_output_payload(
    payload,
    expected_contract,
    output_path,
    *,
    expected_sample_indices=None,
    expected_y_true=None,
):
    for payload_key, contract_key in (
        ("dataset", "dataset"),
        ("model", "model"),
        ("seed", "seed"),
        ("task", "task"),
    ):
        if payload.get(payload_key) != expected_contract.get(contract_key):
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: payload {payload_key} does not "
                "match the stored output details."
            )

    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: evaluations must be a list."
        )
    task = str(expected_contract.get("task", "classification"))
    expected_storage = str(expected_contract.get("save_predictions"))
    regime = str(expected_contract.get("evaluation_regime"))
    axes = expected_contract.get("robustness_axes")
    if regime == "robustness":
        if not isinstance(axes, dict):
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: robustness output details are invalid."
            )
        expected_axes = _expected_robustness_axis_keys(axes)
        expected_count = int(axes.get("expected_evaluations", len(expected_axes)))
        if len(evaluations) != expected_count or len(expected_axes) != expected_count:
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: expected {expected_count} evaluations, "
                f"got {len(evaluations)}."
            )
        try:
            actual_axes = [_evaluation_axis_key(evaluation) for evaluation in evaluations]
        except (TypeError, ValueError) as exc:
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: robustness axis values are invalid."
            ) from exc
        if actual_axes != expected_axes or len(set(actual_axes)) != len(actual_axes):
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: robustness axis order or uniqueness is invalid."
            )
        if expected_count == 100 and expected_storage != "full":
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: 100-cell evaluation requires full predictions."
            )
        if expected_count == 100 and task != "classification":
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: the 100-condition profile "
                "supports classification only."
            )
    elif len(evaluations) != 1:
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: non-robustness output must have one evaluation."
        )

    channel_count = int(expected_contract.get("data_shape", [0, 0])[1])
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            raise OutputArtifactContractError(
                f"Output artifact {output_path} cannot be reused: evaluation rows must be objects."
            )
        _validate_prediction_payload(
            evaluation.get("predictions"),
            expected_storage=expected_storage,
            task=task,
            expected_contract=expected_contract,
            expected_sample_indices=expected_sample_indices,
            expected_y_true=expected_y_true,
            output_path=output_path,
        )
        if regime == "robustness" and evaluation.get("test_missing_type") == "CHANNEL_DROP":
            expected_plan = build_channel_drop_plan(
                channel_count=channel_count,
                rate=float(evaluation.get("test_missing_rate")),
                seed=int(evaluation.get("missing_seed")),
            )
            required = tuple(expected_plan) + ("all_missing_sequence_count",)
            if any(key not in evaluation for key in required):
                raise OutputArtifactContractError(
                    f"Output artifact {output_path} cannot be reused: channel-drop provenance is incomplete."
                )
            for key, expected_value in expected_plan.items():
                actual_value = evaluation.get(key)
                matches = (
                    np.isclose(float(actual_value), float(expected_value))
                    if isinstance(expected_value, float)
                    else actual_value == expected_value
                )
                if not matches:
                    raise OutputArtifactContractError(
                        f"Output artifact {output_path} cannot be reused: channel-drop provenance "
                        f"is inconsistent for {key}."
                    )
            all_missing_count = int(evaluation["all_missing_sequence_count"])
            prediction_count = int(evaluation["predictions"]["sample_index_count"])
            if all_missing_count < 0 or all_missing_count > prediction_count:
                raise OutputArtifactContractError(
                    f"Output artifact {output_path} cannot be reused: all-missing sequence count is invalid."
                )


def _validate_existing_output_contract(
    output_path,
    expected_contract,
    *,
    expected_sample_indices=None,
    expected_y_true=None,
):
    try:
        payload = _read_output_payload(output_path)
    except OutputArtifactContractError:
        raise
    except Exception as exc:
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: failed to read pickle payload ({exc}). "
            "Remove the artifact or rerun with --no-check to overwrite."
        ) from exc

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: missing meta object. "
            "Remove the artifact or rerun with --no-check to overwrite."
        )
    actual_contract = meta.get("output_contract")
    if not isinstance(actual_contract, dict):
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: missing stored output details. "
            "Remove the artifact or rerun with --no-check to overwrite."
        )

    task = str(expected_contract.get("task", "classification"))
    required_fields = (
        "dataset",
        "model",
        "seed",
        "task",
        "train_missing_rate",
        "evaluation_regime",
        "epochs",
        "save_predictions",
    )
    missing = [field for field in required_fields if field not in actual_contract]
    if missing:
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: stored output details are incomplete "
            f"({', '.join(missing)}). "
            "Remove the artifact or rerun with --no-check to overwrite."
        )
    if task != "classification" and not isinstance(actual_contract.get("task_target"), dict):
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: it predates the current "
            "non-classification task semantics. Remove it or rerun with --no-check to overwrite."
        )

    semantic_fields = (
        "dataset",
        "dataset_source",
        "data_shape",
        "num_class",
        "model",
        "model_parameters",
        "seed",
        "task",
        "target_source",
        "forecast_horizon",
        "task_target",
        "train_missing_rate",
        "train_missing_type",
        "train_channel_sync",
        "evaluation_regime",
        "epochs",
        "save_predictions",
        "prediction_identity_schema",
        "interpolate",
        "use_intensity",
        "batch_policy",
        "lr_policy",
        "timestamp_summary",
        "robustness_axes",
        "training_loop_policy",
    )
    mismatches = []
    for field in semantic_fields:
        if field not in actual_contract:
            if task == "classification":
                continue
            if expected_contract.get(field) is None:
                continue
            mismatches.append(field)
            continue
        if actual_contract.get(field) != expected_contract.get(field):
            mismatches.append(field)
    if mismatches:
        raise OutputArtifactContractError(
            f"Output artifact {output_path} cannot be reused: stored output details mismatch "
            f"({', '.join(mismatches)}). "
            "Remove the artifact or rerun with --no-check to overwrite."
        )
    _validate_output_payload(
        payload,
        expected_contract,
        output_path,
        expected_sample_indices=expected_sample_indices,
        expected_y_true=expected_y_true,
    )
    return actual_contract


def _build_output_path(
    data_name,
    missing_rate,
    model_name,
    seed,
    out_root,
    robustness_mode=False,
    task="classification",
    robustness_test_rates=None,
    robustness_test_mechanisms=None,
    robustness_channel_sync_options=None,
):
    task_name = str(task).strip().lower()
    if robustness_mode:
        suffix = _build_robustness_axis_suffix(
            robustness_test_rates=robustness_test_rates,
            robustness_test_mechanisms=robustness_test_mechanisms,
            robustness_channel_sync_options=robustness_channel_sync_options,
        )
        out_name = "_".join([data_name, "robustness", model_name, str(seed)]) + suffix
        if task_name != "classification":
            out_name = f"{out_name}__task_{task_name}"
        return os.path.join(out_root, data_name, "robustness", out_name)
    out_name = "_".join([data_name, str(missing_rate), model_name, str(seed)])
    if task_name != "classification":
        out_name = f"{out_name}__task_{task_name}"
    return os.path.join(out_root, data_name, str(missing_rate), out_name)


def _compute_metrics(y_true, y_pred, logit_list=None):
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    labels = sorted(set(np.asarray(y_true_np).tolist()) | set(np.asarray(y_pred_np).tolist()))
    logit_np = None
    if logit_list is not None:
        logit_np = np.asarray(logit_list, dtype=np.float64)
        if logit_np.ndim == 2 and logit_np.shape[1] > 0:
            max_label_count = (int(max(labels)) + 1) if labels else 0
            labels = list(range(max(int(logit_np.shape[1]), max_label_count)))

    accuracy = _validate_bounded_metric("accuracy", accuracy_score(y_true_np, y_pred_np))
    balanced_accuracy = _validate_bounded_metric(
        "balanced_accuracy",
        balanced_accuracy_score(y_true_np, y_pred_np),
    )
    f1_macro = _validate_bounded_metric("f1", f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    wf1 = _validate_bounded_metric("wf1", f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0))
    precision_macro = _validate_bounded_metric(
        "precision_macro",
        precision_score(y_true_np, y_pred_np, average="macro", zero_division=0),
    )
    recall_macro = _validate_bounded_metric(
        "recall_macro",
        recall_score(y_true_np, y_pred_np, average="macro", zero_division=0),
    )
    precision_weighted = _validate_bounded_metric(
        "precision_weighted",
        precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0),
    )
    recall_weighted = _validate_bounded_metric(
        "recall_weighted",
        recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0),
    )
    confusion = confusion_matrix(y_true_np, y_pred_np, labels=labels)

    cross_entropy = None
    brier_score = None
    mean_confidence = None
    mean_confidence_correct = None
    mean_confidence_incorrect = None
    if logit_np is not None and logit_np.ndim == 2 and logit_np.shape[0] == y_true_np.shape[0]:
        max_logit = np.max(logit_np, axis=1, keepdims=True)
        exp_shifted = np.exp(logit_np - max_logit)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        y_true_int = y_true_np.astype(np.int64)
        if np.all((0 <= y_true_int) & (y_true_int < probs.shape[1])):
            log_probs = logit_np - max_logit - np.log(np.sum(exp_shifted, axis=1, keepdims=True))
            cross_entropy = float(-np.mean(log_probs[np.arange(y_true_int.shape[0]), y_true_int]))
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(y_true_int.shape[0]), y_true_int] = 1.0
            brier_score = float(np.mean(np.sum(np.square(probs - one_hot), axis=1)))
        confidence = np.max(probs, axis=1)
        correct = y_pred_np == y_true_np
        mean_confidence = float(np.mean(confidence))
        mean_confidence_correct = float(np.mean(confidence[correct])) if np.any(correct) else None
        mean_confidence_incorrect = float(np.mean(confidence[~correct])) if np.any(~correct) else None

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1": float(f1_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "wf1": float(wf1),
        "f1_weighted": float(wf1),
        "cross_entropy": cross_entropy,
        "brier_score": brier_score,
        "mean_confidence": mean_confidence,
        "mean_confidence_correct": mean_confidence_correct,
        "mean_confidence_incorrect": mean_confidence_incorrect,
        "n_samples": int(y_true_np.shape[0]),
        "n_classes": int(len(labels)),
        "class_labels": [int(label) for label in labels],
        "confusion_matrix": confusion.astype(int).tolist(),
    }
    for key in ("cross_entropy", "brier_score", "mean_confidence"):
        if metrics[key] is None or not np.isfinite(float(metrics[key])):
            raise ValueError(f"classification metric {key} must be finite")
    return metrics


def _build_task_targets(task, x_missing_norm, y, forecast_horizon=None):
    del x_missing_norm, y, forecast_horizon
    if task == "classification":
        return None
    raise ValueError(
        f"non-classification tasks require --target-source dataset; got task={task!r}"
    )


def _masked_metric_arrays(pred, target, target_mask, *, task: str):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch for task metrics: {pred.shape} != {target.shape}")
    if target_mask is None:
        raise ValueError(f"{task} metrics require target_mask")
    mask = np.asarray(target_mask)
    if mask.shape != target.shape:
        raise ValueError(f"shape mismatch for task mask: {mask.shape} != {target.shape}")
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError(f"{task} target_mask must be finite binary 0/1")
    selected = mask.astype(bool)
    count = int(np.count_nonzero(selected))
    if count <= 0:
        raise ValueError(f"{task} target_mask has zero supervised entries")
    return pred[selected], target[selected], count


def _compute_task_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    target_mask=None,
    task="regression",
):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch for task metrics: {pred.shape} != {target.shape}")
    supervised_count = int(pred.size)
    if target_mask is not None or task == "imputation":
        pred, target, supervised_count = _masked_metric_arrays(
            pred,
            target,
            target_mask,
            task=task,
        )
    diff = pred - target
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    return {"mse": mse, "mae": mae, "supervised_element_count": supervised_count}


def _compute_binary_detection_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    accuracy_key: str,
    target_mask=None,
    task="outlier_detection",
):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch for binary detection metrics: {pred.shape} != {target.shape}")
    supervised_count = int(pred.size)
    if target_mask is not None or task == "change_detection":
        pred, target, supervised_count = _masked_metric_arrays(
            pred,
            target,
            target_mask,
            task=task,
        )

    logits = torch.from_numpy(pred)
    labels = torch.from_numpy(target)
    bce = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item())

    probs = torch.sigmoid(logits).numpy()
    pred_label = probs >= 0.5
    true_label = target >= 0.5
    if np.unique(true_label).size != 2:
        raise ValueError(f"{task} metrics require both binary classes")
    acc = float(np.mean(pred_label == true_label))
    return {
        "bce": bce,
        accuracy_key: acc,
        "balanced_accuracy": float(balanced_accuracy_score(true_label.ravel(), pred_label.ravel())),
        "f1": float(f1_score(true_label.ravel(), pred_label.ravel(), zero_division=0)),
        "auroc": float(roc_auc_score(true_label.ravel(), probs.ravel())),
        "auprc": float(average_precision_score(true_label.ravel(), probs.ravel())),
        "supervised_element_count": supervised_count,
    }


def _prediction_identity_fields(sample_indices, *, include_indices: bool, expected_n: int | None = None) -> dict:
    if sample_indices is None:
        return {
            "prediction_identity_schema": PREDICTION_IDENTITY_SCHEMA,
            "sample_indices": None,
            "sample_index_count": None,
            "sample_index_dtype": None,
        }
    indices = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1))
    if expected_n is not None and int(indices.shape[0]) != int(expected_n):
        raise ValueError(
            f"prediction sample index length mismatch: expected {int(expected_n)}, "
            f"got {int(indices.shape[0])}."
        )
    return {
        "prediction_identity_schema": PREDICTION_IDENTITY_SCHEMA,
        "sample_indices": indices.copy(),
        "sample_index_count": int(indices.shape[0]),
        "sample_index_dtype": str(indices.dtype),
    }


def _build_predictions_payload(y_true, y_pred, logit_list, save_predictions="full", sample_indices=None):
    n_samples = int(y_true.shape[0]) if hasattr(y_true, "shape") else int(len(y_true))
    if save_predictions == "full":
        return {
            "storage": "full",
            **_prediction_identity_fields(sample_indices, include_indices=True, expected_n=n_samples),
            "y_true": y_true,
            "y_pred": y_pred,
            "logit_list": logit_list,
        }
    if save_predictions == "summary":
        n_classes = int(logit_list.shape[1]) if getattr(logit_list, "ndim", 0) >= 2 else 0
        return {
            "storage": "summary",
            **_prediction_identity_fields(sample_indices, include_indices=False, expected_n=n_samples),
            "n_samples": n_samples,
            "n_classes": n_classes,
            "y_true_dtype": str(getattr(y_true, "dtype", "unknown")),
            "y_pred_dtype": str(getattr(y_pred, "dtype", "unknown")),
            "logit_dtype": str(getattr(logit_list, "dtype", "unknown")),
            "y_true": None,
            "y_pred": None,
            "logit_list": None,
        }
    raise ValueError(f"save_predictions must be 'full' or 'summary', got {save_predictions!r}")


def _build_task_predictions_payload(
    task_pred,
    task_target,
    *,
    task_mask,
    save_predictions,
    sample_indices,
):
    pred_np = np.asarray(task_pred)
    target_np = np.asarray(task_target)
    if pred_np.shape != target_np.shape:
        raise ValueError(f"task prediction/target shape mismatch: {pred_np.shape} != {target_np.shape}")
    n_samples = int(pred_np.shape[0]) if pred_np.ndim > 0 else 0
    mask_np = None if task_mask is None else np.asarray(task_mask)
    mask_fields = {}
    if mask_np is not None:
        if mask_np.shape != target_np.shape:
            raise ValueError(f"task mask/target shape mismatch: {mask_np.shape} != {target_np.shape}")
        mask_fields = {
            "task_mask_shape": list(mask_np.shape),
            "task_mask_dtype": str(mask_np.dtype),
            "supervised_element_count": int(np.count_nonzero(mask_np)),
        }
    if save_predictions == "full":
        return {
            "storage": "full",
            **_prediction_identity_fields(sample_indices, include_indices=True, expected_n=n_samples),
            "y_true": None,
            "y_pred": None,
            "logit_list": None,
            "target": task_target,
            "pred": task_pred,
            **({"target_mask": task_mask} if mask_np is not None else {}),
        }
    if save_predictions == "summary":
        return {
            "storage": "summary",
            **_prediction_identity_fields(sample_indices, include_indices=False, expected_n=n_samples),
            "n_samples": n_samples,
            "n_classes": 0,
            "y_true_dtype": None,
            "y_pred_dtype": None,
            "logit_dtype": None,
            "task_pred_shape": list(pred_np.shape),
            "task_target_shape": list(target_np.shape),
            "task_pred_dtype": str(pred_np.dtype),
            "task_target_dtype": str(target_np.dtype),
            **mask_fields,
            "y_true": None,
            "y_pred": None,
            "logit_list": None,
            "target": None,
            "pred": None,
            "target_mask": None,
        }
    raise ValueError(f"save_predictions must be 'full' or 'summary', got {save_predictions!r}")


def _build_missingness_provenance(
    X,
    X_missing,
    mask_observed,
    requested_missing_rate,
    floor_stats=None,
    sample_indices=None,
):
    x_full = np.asarray(X, dtype=np.float32)
    missing_full = np.asarray(X_missing, dtype=np.float32)
    mask_full = np.asarray(mask_observed)
    indices = (
        np.arange(x_full.shape[0], dtype=np.int64)
        if sample_indices is None
        else np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    )
    if np.any(indices < 0) or np.any(indices >= x_full.shape[0]):
        raise ValueError("missingness provenance sample indices are out of range")
    x_np = x_full[indices]
    missing_np = missing_full[indices]
    mask_np = mask_full[indices]
    observed_before = ~np.isnan(x_np)
    observed_value_count = int(np.count_nonzero(observed_before))
    synthetic_missing = observed_before & np.isnan(missing_np)
    realized_missing_count = int(np.count_nonzero(synthetic_missing))
    realized_missing_rate = (
        float(realized_missing_count / observed_value_count)
        if observed_value_count > 0
        else 0.0
    )
    floor_stats = floor_stats or {}
    floor_indices = floor_stats.get("fully_masked_before_floor_indices")
    if floor_indices is None:
        full_floor_count = int(floor_stats.get("fully_masked_before_floor", 0))
        if len(indices) != x_full.shape[0] and full_floor_count > 0:
            raise ValueError("subset floor provenance requires per-sequence indices")
        scoped_floor_count = full_floor_count
    else:
        scoped_floor_count = len(
            set(int(value) for value in floor_indices) & set(indices.tolist())
        )
    provenance = {
        "requested_missing_rate": float(requested_missing_rate),
        "realized_missing_rate": realized_missing_rate,
        "realized_missing_count": realized_missing_count,
        "observed_value_count": observed_value_count,
        "fully_masked_before_floor": int(scoped_floor_count),
        "floor_n_sequences": int(x_np.shape[0]),
    }
    for key in (
        "channel_count",
        "requested_channel_drop_count",
        "dropped_channel_count",
        "dropped_channel_indices",
        "channel_permutation",
        "minimum_one_channel_forced",
        "maximum_one_channel_retained",
        "channel_count_quantized",
        "realized_channel_drop_rate",
        "channel_drop_applicable",
        "analysis_exclusion_reason",
    ):
        if key in floor_stats:
            provenance[key] = floor_stats[key]
    if "all_missing_sequence_count" in floor_stats:
        provenance["all_missing_sequence_count"] = int(
            np.count_nonzero(~(mask_np > 0).any(axis=(1, 2)))
        )
    return provenance


def _build_evaluation_result(
    test_missing_rate,
    y_true=None,
    y_pred=None,
    logit_list=None,
    task_pred=None,
    task_target=None,
    task_mask=None,
    *,
    task="classification",
    test_missing_type="MCAR",
    test_channel_sync=None,
    missing_seed=0,
    save_predictions="full",
    missingness_provenance=None,
    sample_indices=None,
):
    task = str(task).strip().lower()
    if task == "classification":
        metrics = _compute_metrics(y_true, y_pred, logit_list)
        predictions = _build_predictions_payload(
            y_true=y_true,
            y_pred=y_pred,
            logit_list=logit_list,
            save_predictions=save_predictions,
            sample_indices=sample_indices,
        )
    elif task == "outlier_detection":
        binary_metrics = _compute_binary_detection_metrics(
            task_pred,
            task_target,
            accuracy_key="outlier_acc",
        )
        metrics = {
            "bce": float(binary_metrics["bce"]),
            "outlier_acc": float(binary_metrics["outlier_acc"]),
            "balanced_accuracy": float(binary_metrics["balanced_accuracy"]),
            "f1": float(binary_metrics["f1"]),
            "auroc": float(binary_metrics["auroc"]),
            "auprc": float(binary_metrics["auprc"]),
        }
        predictions = _build_task_predictions_payload(
            task_pred,
            task_target,
            task_mask=None,
            save_predictions=save_predictions,
            sample_indices=sample_indices,
        )
    elif task == "change_detection":
        binary_metrics = _compute_binary_detection_metrics(
            task_pred,
            task_target,
            accuracy_key="change_acc",
            target_mask=task_mask,
            task=task,
        )
        metrics = {
            "bce": float(binary_metrics["bce"]),
            "change_acc": float(binary_metrics["change_acc"]),
            "balanced_accuracy": float(binary_metrics["balanced_accuracy"]),
            "f1": float(binary_metrics["f1"]),
            "auroc": float(binary_metrics["auroc"]),
            "auprc": float(binary_metrics["auprc"]),
            "supervised_element_count": int(binary_metrics["supervised_element_count"]),
        }
        predictions = _build_task_predictions_payload(
            task_pred,
            task_target,
            task_mask=task_mask,
            save_predictions=save_predictions,
            sample_indices=sample_indices,
        )
    else:
        task_metrics = _compute_task_metrics(
            task_pred,
            task_target,
            target_mask=task_mask,
            task=task,
        )
        metrics = {
            "mse": float(task_metrics["mse"]),
            "mae": float(task_metrics["mae"]),
            "supervised_element_count": int(task_metrics["supervised_element_count"]),
        }
        predictions = _build_task_predictions_payload(
            task_pred,
            task_target,
            task_mask=task_mask if task == "imputation" else None,
            save_predictions=save_predictions,
            sample_indices=sample_indices,
        )
    result = {
        "missing_seed": int(missing_seed),
        "test_missing_rate": float(test_missing_rate),
        "test_missing_type": str(test_missing_type).upper(),
        "test_channel_sync": None if test_channel_sync is None else bool(test_channel_sync),
    }
    if missingness_provenance is not None:
        result.update(missingness_provenance)
    result.update({"metrics": metrics, "predictions": predictions})
    return result


def _build_result_payload(
    *,
    mode,
    evaluation_regime=None,
    task="classification",
    dataset,
    model_name,
    seed,
    train_missing_rate,
    evaluations,
    epochs,
    interpolate,
    use_intensity,
    device,
    run_id,
    save_predictions,
    train_missing_type="MCAR",
    train_channel_sync=None,
    batch_policy=None,
    lr_policy=None,
    robustness_axes=None,
    output_contract=None,
    timestamp_summary=None,
    aux_nan_event_count=0,
    runtime_baseline_id=None,
    training_loop=None,
    coeff_cache=None,
):
    resolved_regime = str(evaluation_regime or mode)
    resolved_mode = resolved_regime if str(mode).strip().lower() == "standard" else str(mode)
    meta = {
        "task": str(task),
        "task_support_tier": "stable" if str(task) == "classification" else "experimental",
        "epochs": int(epochs),
        "interpolate": str(interpolate),
        "use_intensity": bool(use_intensity),
        "device": str(device),
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "save_predictions": str(save_predictions),
        "evaluation_regime": resolved_regime,
        "batch_policy": batch_policy,
        "lr_policy": lr_policy,
        "robustness_axes": robustness_axes,
        "timestamp_summary": timestamp_summary,
        "aux_nan_event_count": int(aux_nan_event_count),
        "runtime_baseline_id": runtime_baseline_id,
        "training_loop": training_loop,
        "coeff_cache": coeff_cache,
    }
    if output_contract is not None:
        meta["output_contract"] = output_contract
    return omit_none_runtime_baseline_id({
        "mode": resolved_mode,
        "evaluation_regime": resolved_regime,
        "task": str(task),
        "dataset": str(dataset),
        "model": str(model_name),
        "seed": int(seed),
        "train_missing_rate": float(train_missing_rate),
        "train_missing_type": str(train_missing_type).upper(),
        "train_channel_sync": None if train_channel_sync is None else bool(train_channel_sync),
        "evaluations": evaluations,
        "meta": meta,
    })


def _get_aux_nan_event_count(model):
    seq_layer = getattr(model, "seq_layer", None)
    return int(getattr(seq_layer, "aux_nan_event_count", 0))


def run_model(
    data_name,
    missing_rate,
    model_name,
    model_config,
    EPOCHS=100,
    SEED=SEED,
    CHECK=False,
    robustness_mode=False,
    robustness_profile="custom",
    robustness_test_rates=None,
    robustness_test_mechanisms=None,
    robustness_channel_sync_options=None,
    split_root="split",
    out_root="out",
    run_id=None,
    save_predictions="full",
    task="classification",
    forecast_horizon=None,
    target_source="dataset",
    targets_root="targets",
    dataset_source="default",
    dataset_npz=None,
    runtime_baseline_id=None,
    coeff_cache_dir=None,
    no_coeff_cache=False,
    solver_budget_policy="allow",
    solver_budget_seq_len_threshold=0,
):
    task = str(task).strip().lower()
    if robustness_mode:
        robustness_test_rates = _parse_missing_rates(robustness_test_rates)
        robustness_test_mechanisms = _parse_robustness_mechanisms(robustness_test_mechanisms)
        robustness_channel_sync_options = _parse_robustness_channel_sync_options(robustness_channel_sync_options)
    if robustness_mode:
        LOGGER.info(
            "Robustness run start dataset=%s model=%s (train missing=0.0, test rates=%s, types=%s, sync=%s, missing_seed=%s)",
            data_name,
            model_name,
            robustness_test_rates,
            robustness_test_mechanisms,
            robustness_channel_sync_options,
            SEED,
        )
    else:
        LOGGER.info("Run start dataset=%s missing=%.1f model=%s", data_name, missing_rate, model_name)

    if task != "classification" and robustness_mode and robustness_profile == "100-cell":
        raise ValueError("the 100-condition robustness profile supports classification only")

    # check exist
    out_path = _build_output_path(
        data_name,
        missing_rate,
        model_name,
        SEED,
        out_root,
        robustness_mode=robustness_mode,
        task=task,
        robustness_test_rates=robustness_test_rates,
        robustness_test_mechanisms=robustness_test_mechanisms,
        robustness_channel_sync_options=robustness_channel_sync_options,
    )

    # setup out path
    os.makedirs(os.path.join(out_root, data_name), exist_ok=True)
    
    # load data
    dataset_payload = load_dataset_payload(
        dataset_name=data_name,
        dataset_source=dataset_source,
        dataset_npz=dataset_npz,
    )
    X = dataset_payload.X
    Y = dataset_payload.y
    timestamps = dataset_payload.timestamps
    timestamp_summary = dataset_payload.timestamp_summary
    num_data = X.shape[0]
    num_dim = X.shape[1]
    seq_len = X.shape[2]
    num_class = len(np.unique(Y))
    solver_exclusion_reason = _solver_budget_exclusion_reason(
        model_name,
        seq_len,
        solver_budget_policy,
        solver_budget_seq_len_threshold,
    )
    if solver_exclusion_reason is not None:
        raise SolverBudgetExcluded(solver_exclusion_reason)
    coeff_cache_events = []
    
    # check model_name and settings
    data_config = get_data_config(model_name)
    interpolate = data_config['interpolation']
    use_intensity = data_config['use_intensity']
    use_raw_flow_input = _requires_flow_raw_input(model_name)
    evaluation_regime = "robustness" if robustness_mode else _evaluation_regime_for_missing_rate(missing_rate)
    train_missing_type = "STANDARD" if robustness_mode else _missing_type_for_missing_rate(missing_rate)
    train_channel_sync = False
       
    ## data split    
    seed_everything(SEED)
    
    # 0.7/0.15/0.15 train/val/test split
#     train_idx, test_idx = sklearn.model_selection.train_test_split(range(len(Y)), train_size=0.7, shuffle=True, stratify=Y, random_state=SEED)
#     valid_idx, test_idx = sklearn.model_selection.train_test_split(test_idx, train_size=0.5, shuffle=True, stratify=Y[test_idx], random_state=SEED)

    # load splits
    out_name = '_'.join([data_name, str(SEED)])
    split_path = os.path.join(split_root, data_name, out_name)
    with open(split_path, 'rb') as f:
        train_idx, valid_idx, test_idx = pickle.load(f)
    task_targets = None
    target_masks = None
    target_spec = None
    preprocess_X = X
    if task != "classification" and target_source == "dataset":
        task_targets, target_masks, target_spec = load_task_targets_npz(
            dataset=data_name,
            targets_root=targets_root,
            task=task,
            horizon=forecast_horizon,
            num_samples=int(num_data),
            seq_len=int(seq_len),
            input_dim=int(num_dim),
        )
        target_spec = dict(target_spec)
        target_spec["split_supervision"] = validate_task_splits(
            task=task,
            target=task_targets,
            target_mask=target_masks,
            splits={"training": train_idx, "validation": valid_idx, "test": test_idx},
        )
        if task == "imputation":
            preprocess_X = mask_imputation_inputs(X, target_masks)
    elif task != "classification":
        _build_task_targets(
            task=task,
            x_missing_norm=None,
            y=None,
            forecast_horizon=forecast_horizon,
        )
    timestamp_delta_scale, timestamp_summary = resolve_timestamp_delta_metadata(
        timestamps,
        train_idx=train_idx,
        summary=dataset_payload.timestamp_summary,
    )
    batch_policy = resolve_dataset_fixed_batch_policy(
        train_n=len(train_idx),
        input_dim=num_dim,
        seq_len=seq_len,
    )
    batch_size = int(batch_policy["batch_size"])

    # set learning params
    lr_policy = resolve_learning_rate_policy(model_config.get("lr"), require_config=True)
    model_kwargs = build_model_kwargs_from_config(model_name, model_config)
    param_search_contract = _param_search_contract_metadata(
        model_config,
        infer_param_search_profile(model_config),
    )
    robustness_axes = (
        _build_robustness_axes_metadata(
            robustness_test_rates=robustness_test_rates,
            robustness_test_mechanisms=robustness_test_mechanisms,
            robustness_channel_sync_options=robustness_channel_sync_options,
            use_raw_flow_input=use_raw_flow_input,
            seed=SEED,
        )
        if robustness_mode
        else None
    )
    training_loop_policy = {
        "scheduler": "StepLR",
        "scheduler_step_size": 10,
        "scheduler_gamma": 0.5,
        "early_stop_warmup": 22,
        "early_stop_patience": 10,
    }
    output_contract = _build_output_contract(
        data_name=data_name,
        model_name=model_name,
        model_parameters=model_kwargs,
        seed=SEED,
        task=task,
        target_source=target_source,
        forecast_horizon=forecast_horizon,
        dataset_source=dataset_source,
        data_shape=X.shape,
        num_class=num_class,
        missing_rate=0.0 if robustness_mode else missing_rate,
        evaluation_regime=evaluation_regime,
        train_missing_type=train_missing_type,
        train_channel_sync=train_channel_sync,
        epochs=EPOCHS,
        save_predictions=save_predictions,
        interpolate=interpolate,
        use_intensity=use_intensity,
        batch_policy=batch_policy,
        lr_policy=lr_policy,
        timestamp_summary=timestamp_summary,
        robustness_axes=robustness_axes,
        training_loop_policy=training_loop_policy,
    )
    if CHECK and os.path.exists(out_path):
        _validate_existing_output_contract(
            out_path,
            output_contract,
            expected_sample_indices=test_idx,
            expected_y_true=np.asarray(Y)[np.asarray(test_idx, dtype=np.int64)],
        )
        return None
    lr = float(lr_policy["lr"])
    weight_decay = float(lr_policy["weight_decay"])
    LOGGER.info(
        "Batch policy dataset=%s model=%s seed=%s policy=%s batch=%d train_n=%d steps=%d drop=%d sample_elements=%d",
        data_name,
        model_name,
        SEED,
        batch_policy["policy"],
        batch_size,
        batch_policy["train_n"],
        batch_policy["train_steps_per_epoch"],
        batch_policy["dropped_train_samples_per_epoch"],
        batch_policy["sample_elements"],
    )
    LOGGER.info(
        "Learning-rate policy dataset=%s model=%s seed=%s policy=%s source=%s lr=%g weight_decay=%g",
        data_name,
        model_name,
        SEED,
        lr_policy["policy"],
        lr_policy["source"],
        lr,
        weight_decay,
    )
    
    # generate path
    if robustness_mode:
        os.makedirs(os.path.join(out_root, data_name, "robustness"), exist_ok=True)
    else:
        os.makedirs(os.path.join(out_root, data_name, str(missing_rate)), exist_ok=True)

    X_missing, X_mask, X_delta, coeffs = _preprocess_with_optional_cache(
        coeff_cache_dir=coeff_cache_dir,
        no_coeff_cache=no_coeff_cache,
        cache_events=coeff_cache_events,
        model_name=model_name,
        X=preprocess_X,
        missing_rate=missing_rate,
        interpolate=interpolate,
        use_intensity=use_intensity,
        mechanism="MCAR",
        seed=SEED,
        channel_sync=train_channel_sync,
        timestamps=timestamps,
        timestamp_delta_scale=timestamp_delta_scale,
        missingness_reference=preprocess_X[train_idx],
    )
    X_missing_norm, normalization_summary = normalize_with_train_stats(X_missing, train_idx)
    train_missing_summary = compute_train_missing_summary(X_missing, train_idx)
    train_missing_summary.update(normalization_summary)
    if task == "classification":
        task_targets = _build_task_targets(
            task=task,
            x_missing_norm=X_missing_norm,
            y=Y,
            forecast_horizon=forecast_horizon,
        )
    else:
        assert task_targets is not None and target_spec is not None
    if task == "classification":
        target_masks = None
        target_spec = {
            "task": "classification",
            "target_key": "label",
            "target_shape": [int(num_data)],
            "target_dtype": str(np.asarray(Y).dtype),
        }
    target_spec = dict(target_spec)
    target_spec["target_source"] = target_source if task != "classification" else "label"
    supervised_missing_count = None
    if task == "imputation" and target_masks is not None:
        supervised_missing_count = int(target_masks[train_idx, ...].sum().item())
    train_missing_summary["supervised_missing_count"] = (
        int(supervised_missing_count) if supervised_missing_count is not None else None
    )
    enforce_train_missing_gate(
        missing_rate=float(missing_rate),
        summary=train_missing_summary,
        task=task,
        supervised_missing_count=supervised_missing_count,
    )

    train_dataset = ists_dataset(
        Y,
        X_missing_norm,
        X_mask,
        X_delta,
        coeffs,
        train_idx,
        task=task,
        targets=task_targets,
        target_masks=target_masks,
    )
    valid_dataset = ists_dataset(
        Y,
        X_missing_norm,
        X_mask,
        X_delta,
        coeffs,
        valid_idx,
        task=task,
        targets=task_targets,
        target_masks=target_masks,
    )
    test_dataset = ists_dataset(
        Y,
        X_missing_norm,
        X_mask,
        X_delta,
        coeffs,
        test_idx,
        task=task,
        targets=task_targets,
        target_masks=target_masks,
    )

    train_batch = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_batch = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # set tmp_path
    os.makedirs(os.path.join(os.getcwd(), 'tmp'), exist_ok=True)
    tmp_path = os.path.join(os.getcwd(), 'tmp/{}_{}_{}.npy'.format(data_name, missing_rate, str(SEED))) 

    solver_method = default_method_for_model(model_name)

    # set model
    if task == "classification":
        model = ists_classifier(
            model_name=model_name,
            input_dim=num_dim,
            seq_len=seq_len,
            num_class=num_class,
            dropout=0.1,
            use_intensity=use_intensity,
            method=solver_method,
            file=tmp_path,
            device=device,
            **model_kwargs,
        )
    else:
        output_dim = 1 if task in {"regression", "outlier_detection", "change_detection"} else int(num_dim)
        model = ists_model(
            model_name=model_name,
            input_dim=num_dim,
            seq_len=seq_len,
            num_class=num_class,
            output_dim=output_dim,
            task=task,
            forecast_horizon=forecast_horizon,
            dropout=0.1,
            use_intensity=use_intensity,
            method=solver_method,
            file=tmp_path,
            device=device,
            **model_kwargs,
        )
    model = model.to(device)

    # set loss & optimizer
    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task in {"outlier_detection", "change_detection"}:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    training_loop = run_train_valid_loop(
        model_name=model_name,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_batch=train_batch,
        valid_batch=valid_batch,
        interpolate=interpolate,
        use_intensity=use_intensity,
        device=device,
        epochs=EPOCHS,
        task=task,
        scheduler_step_size=10,
        scheduler_gamma=0.5,
        early_stop_warmup=22,
        early_stop_patience=10,
        log_prefix=f"{data_name}/{model_name}",
    )
    best_model_wts = training_loop["best_model_wts"]
    training_loop_metadata = {
        key: value for key, value in training_loop.items() if key != "best_model_wts"
    }

    # using trained model
    try:
        model.load_state_dict(best_model_wts)
    except Exception as e:
        raise RuntimeError(
            f"Failed to restore best weights for {data_name}/{model_name}; "
            "refusing to evaluate final-epoch weights as best checkpoint."
        ) from e

    # Robustness mode: evaluate on multiple test rates
    if robustness_mode:
        effective_robustness_test_rates = list(robustness_axes["test_missing_rates"])
        evaluations = _evaluate_robustness(
            model=model,
            data_name=data_name,
            model_name=model_name,
            SEED=SEED,
            test_missing_rates=effective_robustness_test_rates,
            test_missing_mechanisms=robustness_test_mechanisms,
            channel_sync_options=robustness_channel_sync_options,
            X=X,
            Y=Y,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=test_idx,
            interpolate=interpolate,
            use_intensity=use_intensity,
            device=device,
            criterion=criterion,
            save_predictions=save_predictions,
            task=task,
            task_targets=task_targets,
            target_masks=target_masks,
            batch_size=batch_size,
            timestamps=timestamps,
            timestamp_delta_scale=timestamp_delta_scale,
            coeff_cache_dir=coeff_cache_dir,
            no_coeff_cache=no_coeff_cache,
            coeff_cache_events=coeff_cache_events,
        )
        if len(evaluations) != int(robustness_axes["expected_evaluations"]):
            raise RuntimeError(
                "Robustness evaluation count mismatch: "
                f"expected {robustness_axes['expected_evaluations']}, got {len(evaluations)}"
            )
        coeff_cache_summary = _summarize_cache_events(coeff_cache_events)

        payload = _build_result_payload(
            mode="robustness",
            evaluation_regime="robustness",
            task=task,
            dataset=data_name,
            model_name=model_name,
            seed=SEED,
            train_missing_rate=0.0,
            evaluations=evaluations,
            epochs=EPOCHS,
            interpolate=interpolate,
            use_intensity=use_intensity,
            device=device,
            run_id=run_id,
            save_predictions=save_predictions,
            train_missing_type="STANDARD",
            train_channel_sync=False,
            batch_policy=batch_policy,
            lr_policy=lr_policy,
            robustness_axes=robustness_axes,
            output_contract=output_contract,
            timestamp_summary=timestamp_summary,
            aux_nan_event_count=_get_aux_nan_event_count(model),
            runtime_baseline_id=runtime_baseline_id,
            training_loop=training_loop_metadata,
            coeff_cache=coeff_cache_summary,
        )
        payload.update(param_search_contract)

        # Save robustness results
        output_path = _build_output_path(
            data_name,
            0.0,
            model_name,
            SEED,
            out_root,
            robustness_mode=True,
            task=task,
            robustness_test_rates=effective_robustness_test_rates,
            robustness_test_mechanisms=robustness_test_mechanisms,
            robustness_channel_sync_options=robustness_channel_sync_options,
        )
        atomic_pickle_dump(payload, output_path)

        eval0 = next((x for x in evaluations if abs(x["test_missing_rate"]) < 1e-12), evaluations[0])
        metrics = eval0["metrics"]

        if task == "classification":
            LOGGER.info(
                "Robustness run done dataset=%s model=%s evaluations=%d test_rates=%s test_types=%s acc=%.6f f1=%.6f wf1=%.6f save_predictions=%s",
                data_name,
                model_name,
                len(evaluations),
                [r["test_missing_rate"] for r in evaluations],
                sorted({r["test_missing_type"] for r in evaluations}),
                metrics["accuracy"],
                metrics["f1"],
                metrics["wf1"],
                save_predictions,
            )
        else:
            LOGGER.info(
                "Robustness run done dataset=%s model=%s task=%s evaluations=%d test_rates=%s test_types=%s mse=%.6f mae=%.6f save_predictions=%s",
                data_name,
                model_name,
                task,
                len(evaluations),
                [r["test_missing_rate"] for r in evaluations],
                sorted({r["test_missing_type"] for r in evaluations}),
                metrics["mse"],
                metrics["mae"],
                save_predictions,
            )
        return {
            "dataset": data_name,
            "dataset_source": str(dataset_source),
            "dataset_npz": portable_artifact_reference(dataset_npz, root=True),
            "timestamp_summary": timestamp_summary,
            "task": task,
            "model": model_name,
            **param_search_contract,
            "seed": int(SEED),
            "missing_rate": 0.0,
            "robustness_test_rates": [float(rate) for rate in robustness_test_rates],
            "robustness_test_mechanisms": list(robustness_test_mechanisms),
            "robustness_channel_sync_options": list(robustness_channel_sync_options),
            "robustness_evaluations": int(len(evaluations)),
            "split_path": portable_artifact_reference(split_path),
            "output_path": portable_artifact_reference(output_path),
            "batch_policy": batch_policy,
            "lr_policy": lr_policy,
            "metrics": metrics,
            "train_missing_summary": train_missing_summary,
            "target_spec": None if task == "classification" else target_spec,
            "mode": "robustness",
            "evaluation_regime": "robustness",
            "runtime_baseline_id": runtime_baseline_id,
            "training_loop": training_loop_metadata,
            "coeff_cache": coeff_cache_summary,
        }

    # Normal mode: single test evaluation
    model.eval()
    if task == "classification":
        y_true, y_pred, logit_list = [], [], []
        with torch.no_grad():
            for batch in test_batch:
                y = batch['label'].long().to(device)
                mask = _expand_mask_or_delta(batch['x_mask'], batch['x_missing'])
                delta = _expand_mask_or_delta(batch['x_delta'], batch['x_missing'])
                x_values = batch['x_missing'] if use_raw_flow_input else torch.nan_to_num(batch['x_missing'], 0)
                seq = torch.stack([
                    x_values,
                    mask,
                    delta,
                ], dim=1).to(device)

                output = model(seq, batch['coeffs'].to(device))
                logit = output[0] if isinstance(output, tuple) else output
                y_true.append(y.cpu().numpy())
                y_pred.append(logit.max(1)[1].cpu().numpy())
                logit_list.append(logit.cpu().numpy())

        y_true = np.array([x for y in y_true for x in y]).flatten()
        y_pred = np.array([x for y in y_pred for x in y]).flatten()
        logit_list = np.array([x for y in logit_list for x in y])

        eval_result = _build_evaluation_result(
            task=task,
            test_missing_rate=missing_rate,
            test_missing_type=_missing_type_for_missing_rate(missing_rate),
            test_channel_sync=False,
            y_true=y_true,
            y_pred=y_pred,
            logit_list=logit_list,
            missing_seed=SEED,
            save_predictions=save_predictions,
            sample_indices=test_idx,
        )
        metrics = eval_result["metrics"]
        LOGGER.info(
            "Run done dataset=%s missing=%.1f model=%s acc=%.6f f1=%.6f wf1=%.6f save_predictions=%s",
            data_name,
            missing_rate,
            model_name,
            metrics["accuracy"],
            metrics["f1"],
            metrics["wf1"],
            save_predictions,
        )
    else:
        pred_list, target_list, target_mask_list = [], [], []
        with torch.no_grad():
            for batch in test_batch:
                mask = _expand_mask_or_delta(batch['x_mask'], batch['x_missing'])
                delta = _expand_mask_or_delta(batch['x_delta'], batch['x_missing'])
                x_values = batch['x_missing'] if use_raw_flow_input else torch.nan_to_num(batch['x_missing'], 0)
                seq = torch.stack([
                    x_values,
                    mask,
                    delta,
                ], dim=1).to(device)
                output = model(seq, batch['coeffs'].to(device))
                pred = output[0] if isinstance(output, tuple) else output
                target = batch["target"].to(device).float()
                pred, target = _align_prediction_target(pred, target, task)
                pred_list.append(pred.detach().cpu().numpy())
                target_list.append(target.detach().cpu().numpy())
                if task in {"imputation", "change_detection"}:
                    target_mask = batch.get("target_mask")
                    if target_mask is None or tuple(target_mask.shape) != tuple(target.shape):
                        raise ValueError(f"{task} evaluation requires a target_mask matching target")
                    target_mask_list.append(target_mask.detach().cpu().numpy())

        task_pred = np.concatenate(pred_list, axis=0)
        task_target = np.concatenate(target_list, axis=0)
        task_mask = (
            np.concatenate(target_mask_list, axis=0)
            if target_mask_list
            else None
        )
        eval_result = _build_evaluation_result(
            task=task,
            test_missing_rate=missing_rate,
            test_missing_type=_missing_type_for_missing_rate(missing_rate),
            test_channel_sync=False,
            task_pred=task_pred,
            task_target=task_target,
            task_mask=task_mask,
            missing_seed=SEED,
            save_predictions=save_predictions,
            sample_indices=test_idx,
        )
        metrics = eval_result["metrics"]
        if task == "outlier_detection":
            LOGGER.info(
                "Run done dataset=%s missing=%.1f model=%s task=%s bce=%.6f outlier_acc=%.6f save_predictions=%s",
                data_name,
                missing_rate,
                model_name,
                task,
                metrics["bce"],
                metrics["outlier_acc"],
                save_predictions,
            )
        elif task == "change_detection":
            LOGGER.info(
                "Run done dataset=%s missing=%.1f model=%s task=%s bce=%.6f change_acc=%.6f save_predictions=%s",
                data_name,
                missing_rate,
                model_name,
                task,
                metrics["bce"],
                metrics["change_acc"],
                save_predictions,
            )
        else:
            LOGGER.info(
                "Run done dataset=%s missing=%.1f model=%s task=%s mse=%.6f mae=%.6f save_predictions=%s",
                data_name,
                missing_rate,
                model_name,
                task,
                metrics["mse"],
                metrics["mae"],
                save_predictions,
            )

    output_path = _build_output_path(
        data_name,
        missing_rate,
        model_name,
        SEED,
        out_root,
        task=task,
    )
    coeff_cache_summary = _summarize_cache_events(coeff_cache_events)
    payload = _build_result_payload(
        mode=evaluation_regime,
        evaluation_regime=evaluation_regime,
        task=task,
        dataset=data_name,
        model_name=model_name,
        seed=SEED,
        train_missing_rate=missing_rate,
        evaluations=[eval_result],
        epochs=EPOCHS,
        interpolate=interpolate,
        use_intensity=use_intensity,
        device=device,
        run_id=run_id,
        save_predictions=save_predictions,
        train_missing_type=train_missing_type,
        train_channel_sync=train_channel_sync,
        batch_policy=batch_policy,
        lr_policy=lr_policy,
        output_contract=output_contract,
        timestamp_summary=timestamp_summary,
        aux_nan_event_count=_get_aux_nan_event_count(model),
        runtime_baseline_id=runtime_baseline_id,
        training_loop=training_loop_metadata,
        coeff_cache=coeff_cache_summary,
    )
    payload.update(param_search_contract)
    atomic_pickle_dump(payload, output_path)
    return {
        "dataset": data_name,
        "dataset_source": str(dataset_source),
        "dataset_npz": portable_artifact_reference(dataset_npz, root=True),
        "timestamp_summary": timestamp_summary,
        "task": task,
        "target_source": target_source if task != "classification" else "label",
        "model": model_name,
        **param_search_contract,
        "seed": int(SEED),
        "missing_rate": float(missing_rate),
        "split_path": portable_artifact_reference(split_path),
        "output_path": portable_artifact_reference(output_path),
        "batch_policy": batch_policy,
        "lr_policy": lr_policy,
        "metrics": metrics,
        "task_metrics": metrics if task != "classification" else None,
        "train_missing_summary": train_missing_summary,
        "target_spec": None if task == "classification" else target_spec,
        "mode": evaluation_regime,
        "evaluation_regime": evaluation_regime,
        "train_missing_type": train_missing_type,
        "train_channel_sync": train_channel_sync,
        "runtime_baseline_id": runtime_baseline_id,
        "training_loop": training_loop_metadata,
        "coeff_cache": coeff_cache_summary,
    }


def _evaluate_robustness(
    model,
    data_name,
    model_name,
    SEED,
    test_missing_rates,
    test_missing_mechanisms,
    channel_sync_options,
    X,
    Y,
    train_idx,
    valid_idx,
    test_idx,
    interpolate,
    use_intensity,
    device,
    criterion,
    save_predictions,
    task,
    task_targets,
    target_masks,
    batch_size,
    timestamps=None,
    timestamp_delta_scale=None,
    coeff_cache_dir=None,
    no_coeff_cache=False,
    coeff_cache_events=None,
):
    """
    Evaluate trained model on multiple test missing rates.

    Args:
        model: Trained PyTorch model
        data_name: Dataset name
        model_name: Model name
        SEED: Random seed
        test_missing_rates: List of test missing rates
        test_missing_mechanisms: List of test missingness mechanisms
        channel_sync_options: List of channel_sync tokens ({auto,false,true})
        X: Full dataset features
        Y: Full dataset labels
        train_idx: Training indices (for normalization stats)
        valid_idx: Validation indices (unused, kept for consistency)
        test_idx: Test indices
        interpolate: Interpolation method
        use_intensity: Use intensity flag
        device: PyTorch device
        criterion: Loss function

    Returns:
        list: per-rate evaluation payload list with unified metric schema.
    """
    model.eval()
    evaluations = []
    use_raw_flow_input = _requires_flow_raw_input(model_name)
    test_missing_rates = _effective_robustness_test_rates(test_missing_rates)
    test_missing_mechanisms = _parse_robustness_mechanisms(test_missing_mechanisms)
    positive_test_rates = [
        rate for rate in test_missing_rates if not _is_zero_missing_rate(rate)
    ]
    ordered_test_rates = list(test_missing_rates)
    if "CHANNEL_DROP" in test_missing_mechanisms:
        ordered_test_rates.extend(positive_test_rates)
    missing_seed = int(SEED)
    resolved_channel_sync_options = _resolve_channel_sync_options(
        channel_sync_options,
        use_raw_flow_input,
    )
    train_missing_ref, _, _, _ = _preprocess_with_optional_cache(
        coeff_cache_dir=coeff_cache_dir,
        no_coeff_cache=no_coeff_cache,
        cache_events=coeff_cache_events,
        model_name=model_name,
        X=X,
        missing_rate=0.0,
        interpolate=interpolate,
        use_intensity=use_intensity,
        mechanism="MCAR",
        seed=SEED,
        channel_sync=False,
        timestamps=timestamps,
        timestamp_delta_scale=timestamp_delta_scale,
        missingness_reference=X[train_idx],
    )
    train_stats, _normalization_summary = compute_train_normalization_stats(train_missing_ref, train_idx)

    for axis_rate_index, test_rate in enumerate(ordered_test_rates):
        if _is_zero_missing_rate(test_rate):
            mechanism_options = ["STANDARD"]
            channel_sync_values = [None]
        elif axis_rate_index <= len(positive_test_rates):
            mechanism_options = [
                mechanism for mechanism in test_missing_mechanisms
                if mechanism != "CHANNEL_DROP"
            ]
            channel_sync_values = resolved_channel_sync_options
        else:
            mechanism_options = ["CHANNEL_DROP"]
            channel_sync_values = [None]
        for test_mechanism in mechanism_options:
            mechanism_channel_sync_values = (
                [None] if test_mechanism == "CHANNEL_DROP" else channel_sync_values
            )
            for test_channel_sync in mechanism_channel_sync_values:
                    LOGGER.info(
                        "Robustness eval dataset=%s model=%s test_missing=%.1f type=%s channel_sync=%s missing_seed=%d",
                        data_name,
                        model_name,
                        test_rate,
                        test_mechanism,
                        test_channel_sync,
                        missing_seed,
                    )

                    # Generate missing data for this test axis point.
                    preprocess_mechanism = "MCAR" if _is_zero_missing_rate(test_rate) else test_mechanism
                    preprocess_channel_sync = False if test_channel_sync is None else bool(test_channel_sync)
                    stats_missing_np, stats_mask_np, floor_stats = apply_missingness(
                        X,
                        rate=float(test_rate),
                        mechanism=preprocess_mechanism,
                        seed=missing_seed,
                        channel_sync=preprocess_channel_sync,
                        reference_X=X[train_idx],
                        return_stats=True,
                    )
                    missingness_provenance = _build_missingness_provenance(
                        X,
                        stats_missing_np,
                        stats_mask_np,
                        requested_missing_rate=test_rate,
                        floor_stats=floor_stats,
                        sample_indices=test_idx,
                    )
                    X_missing, X_mask, X_delta, coeffs = _preprocess_with_optional_cache(
                        coeff_cache_dir=coeff_cache_dir,
                        no_coeff_cache=no_coeff_cache,
                        cache_events=coeff_cache_events,
                        model_name=model_name,
                        X=X,
                        missing_rate=test_rate,
                        interpolate=interpolate,
                        use_intensity=use_intensity,
                        mechanism=preprocess_mechanism,
                        seed=missing_seed,
                        channel_sync=preprocess_channel_sync,
                        timestamps=timestamps,
                        timestamp_delta_scale=timestamp_delta_scale,
                        missingness_reference=X[train_idx],
                    )

                    # Reuse the training missing=0.0 statistics for all robustness test axes.
                    X_missing_norm = apply_train_normalization_stats(X_missing, train_stats)

                    # Create test dataset with this missing axis point.
                    test_dataset = ists_dataset(
                        Y,
                        X_missing_norm,
                        X_mask,
                        X_delta,
                        coeffs,
                        test_idx,
                        task=task,
                        targets=task_targets,
                        target_masks=target_masks,
                    )
                    test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

                    if task == "classification":
                        y_true, y_pred, logit_list = [], [], []
                        with torch.no_grad():
                            for batch in test_batch:
                                y = batch['label'].long().to(device)
                                mask = _expand_mask_or_delta(batch['x_mask'], batch['x_missing'])
                                delta = _expand_mask_or_delta(batch['x_delta'], batch['x_missing'])
                                x_values = batch['x_missing'] if use_raw_flow_input else torch.nan_to_num(batch['x_missing'], 0)
                                seq = torch.stack([
                                    x_values,
                                    mask,
                                    delta,
                                ], dim=1).to(device)

                                output = model(seq, batch['coeffs'].to(device))
                                logit = output[0] if isinstance(output, tuple) else output

                                y_true.append(y.cpu().numpy())
                                y_pred.append(logit.max(1)[1].cpu().numpy())
                                logit_list.append(logit.cpu().numpy())

                        y_true = np.array([x for y in y_true for x in y]).flatten()
                        y_pred = np.array([x for y in y_pred for x in y]).flatten()
                        logit_list = np.array([x for y in logit_list for x in y])

                        eval_result = _build_evaluation_result(
                            test_missing_rate=test_rate,
                            y_true=y_true,
                            y_pred=y_pred,
                            logit_list=logit_list,
                            task="classification",
                            test_missing_type=test_mechanism,
                            test_channel_sync=test_channel_sync,
                            missing_seed=missing_seed,
                            save_predictions=save_predictions,
                            missingness_provenance=missingness_provenance,
                            sample_indices=test_idx,
                        )
                    else:
                        pred_list, target_list = [], []
                        with torch.no_grad():
                            for batch in test_batch:
                                mask = _expand_mask_or_delta(batch['x_mask'], batch['x_missing'])
                                delta = _expand_mask_or_delta(batch['x_delta'], batch['x_missing'])
                                x_values = batch['x_missing'] if use_raw_flow_input else torch.nan_to_num(batch['x_missing'], 0)
                                seq = torch.stack([
                                    x_values,
                                    mask,
                                    delta,
                                ], dim=1).to(device)

                                output = model(seq, batch['coeffs'].to(device))
                                pred = output[0] if isinstance(output, tuple) else output
                                target = batch["target"].to(device).float()
                                pred, target = _align_prediction_target(pred, target, task)
                                pred_list.append(pred.detach().cpu().numpy())
                                target_list.append(target.detach().cpu().numpy())

                        task_pred = np.concatenate(pred_list, axis=0)
                        task_target = np.concatenate(target_list, axis=0)
                        eval_result = _build_evaluation_result(
                            test_missing_rate=test_rate,
                            task_pred=task_pred,
                            task_target=task_target,
                            task=task,
                            test_missing_type=test_mechanism,
                            test_channel_sync=test_channel_sync,
                            missing_seed=missing_seed,
                            save_predictions=save_predictions,
                            missingness_provenance=missingness_provenance,
                            sample_indices=test_idx,
                        )
                    evaluations.append(eval_result)
                    metrics = eval_result["metrics"]

                    if task == "classification":
                        LOGGER.info(
                            "Robustness eval dataset=%s model=%s test_missing=%.1f type=%s channel_sync=%s missing_seed=%d acc=%.6f f1=%.6f wf1=%.6f save_predictions=%s",
                            data_name,
                            model_name,
                            test_rate,
                            test_mechanism,
                            test_channel_sync,
                            missing_seed,
                            metrics["accuracy"],
                            metrics["f1"],
                            metrics["wf1"],
                            save_predictions,
                        )
                    else:
                        LOGGER.info(
                            "Robustness eval dataset=%s model=%s test_missing=%.1f type=%s channel_sync=%s missing_seed=%d task=%s mse=%.6f mae=%.6f save_predictions=%s",
                            data_name,
                            model_name,
                            test_rate,
                            test_mechanism,
                            test_channel_sync,
                            missing_seed,
                            task,
                            metrics["mse"],
                            metrics["mae"],
                            save_predictions,
                        )

    return evaluations


def main():
    args = parse_cli_args()
    global device, use_cuda
    device = _resolve_run_device(args.device)
    use_cuda = device.type == "cuda"
    seed = int(args.seed)
    seed_everything(seed)
    LOGGER.info("Run device resolved mode=%s device=%s", args.device, device)
    run_id = args.run_id or generate_run_id("run")
    runtime_baseline_id = normalize_runtime_baseline_id(args.runtime_baseline_id)
    records = []
    reused_records = []
    skipped = []
    policy_violations = []

    if args.datasets is None:
        data_selected = ["BasicMotions"]
    else:
        data_selected = args.datasets

    if args.models is None:
        model_name_list = ["cnn"]
        np.random.shuffle(model_name_list)
    else:
        model_name_list = args.models

    max_runs = args.max_runs if (args.max_runs is not None and args.max_runs > 0) else None
    run_count = 0
    stop_early = False

    def _completed_runs() -> int:
        if args.max_runs_mode == "new":
            return run_count
        return run_count + len(reused_records)

    # Robustness mode: train on missing=0, test on multiple rates
    if args.robustness_mode:
        LOGGER.info(
            "Robustness mode enabled: train on missing=0.0, test rates=%s, test types=%s, channel_sync=%s, missing_seed=%s",
            args.robustness_test_rates,
            args.robustness_test_mechanisms,
            args.robustness_channel_sync_options,
            seed,
        )
        for data_name in data_selected:
            for model_name in model_name_list:
                if max_runs is not None and _completed_runs() >= max_runs:
                    LOGGER.info("Reached max-runs=%d, stopping early.", max_runs)
                    stop_early = True
                    break
                try:
                    cell_start = None
                    param_name, param_path = _resolve_param_artifact_path(
                        args.param_root,
                        data_name,
                        model_name,
                        param_policy=args.param_policy,
                        param_repeat=args.param_repeat,
                        task=args.task,
                    )

                    if not os.path.exists(param_path):
                        raise FileNotFoundError(
                            f"Params file not found: {param_path}. "
                            "Run scripts/param_search.py with "
                            f"--search-policy {args.param_policy} --search-repeat {args.param_repeat} first."
                        )

                    with open(param_path, 'rb') as f:
                        model_config = pickle.load(f)
                    expected_target_identity = _expected_param_target_identity(
                        data_name=data_name,
                        task=args.task,
                        target_source=args.target_source,
                        targets_root=args.targets_root,
                        forecast_horizon=args.forecast_horizon,
                        model_config=model_config,
                        dataset_source=args.dataset_source,
                        dataset_npz=args.dataset_npz,
                        split_root=args.split_root,
                        param_repeat=args.param_repeat,
                    )
                    param_search_profile = _validate_param_search_profile(
                        model_config,
                        expected_search_profile=args.search_profile,
                        expected_param_policy=args.param_policy,
                        expected_param_repeat=args.param_repeat,
                        expected_data_name=data_name,
                        expected_model_name=model_name,
                        expected_task=args.task,
                        expected_target_source=expected_target_identity["target_source"],
                        expected_forecast_horizon=expected_target_identity["forecast_horizon"],
                        expected_dataset_source=expected_target_identity["dataset_source"],
                        expected_split_seeds=expected_target_identity["split_seeds"],
                        param_path=str(param_path),
                        allow_legacy_search_profile=args.allow_legacy_search_profile,
                    )
                    param_search_contract = _param_search_contract_metadata(
                        model_config,
                        param_search_profile,
                    )

                    cell_start = time.monotonic()
                    with _cell_timeout(args.cell_timeout_seconds):
                        run_result = run_model(
                            data_name,
                            missing_rate=0.0,  # Force train on missing=0.0
                            model_name=model_name,
                            model_config=model_config,
                            EPOCHS=args.epochs,
                            SEED=seed,
                            CHECK=args.check,
                            robustness_mode=True,
                            robustness_profile=args.robustness_profile,
                            robustness_test_rates=args.robustness_test_rates,
                            robustness_test_mechanisms=args.robustness_test_mechanisms,
                            robustness_channel_sync_options=args.robustness_channel_sync_options,
                            split_root=args.split_root,
                            out_root=args.out_root,
                            run_id=run_id,
                            save_predictions=args.save_predictions,
                            task=args.task,
                            forecast_horizon=args.forecast_horizon,
                            target_source=args.target_source,
                            targets_root=args.targets_root,
                            dataset_source=args.dataset_source,
                            dataset_npz=args.dataset_npz,
                            runtime_baseline_id=runtime_baseline_id,
                            coeff_cache_dir=args.coeff_cache_dir,
                            no_coeff_cache=args.no_coeff_cache,
                            solver_budget_policy=args.solver_budget_policy,
                            solver_budget_seq_len_threshold=args.solver_budget_seq_len_threshold,
                        )
                    cell_elapsed_s = time.monotonic() - cell_start
                    if run_result is None:
                        output_path = _build_output_path(
                            data_name,
                            0.0,
                            model_name,
                            seed,
                            args.out_root,
                            robustness_mode=True,
                            task=args.task,
                            robustness_test_rates=args.robustness_test_rates,
                            robustness_test_mechanisms=args.robustness_test_mechanisms,
                            robustness_channel_sync_options=args.robustness_channel_sync_options,
                        )
                        split_path = os.path.join(args.split_root, data_name, f"{data_name}_{seed}")
                        if args.existing_policy == "fail":
                            reason = "output artifact already exists (--check, existing-policy=fail)"
                            skipped.append(
                                {
                                    "dataset": data_name,
                                    "model": model_name,
                                    "missing_rate": 0.0,
                                    "robustness_test_rates": [float(rate) for rate in args.robustness_test_rates],
                                    "robustness_test_mechanisms": list(args.robustness_test_mechanisms),
                                    "robustness_channel_sync_options": list(args.robustness_channel_sync_options),
                                    "missing_seed": int(seed),
                                    "mode": "robustness",
                                    "evaluation_regime": "robustness",
                                    "task": args.task,
                                    "reason": reason,
                                }
                            )
                            policy_violations.append(f"{data_name}/{model_name}: {reason}")
                        else:
                            reused_records.append(
                                {
                                    "dataset": data_name,
                                    "dataset_source": str(args.dataset_source),
                                    "dataset_npz": portable_artifact_reference(args.dataset_npz, root=True),
                                    "model": model_name,
                                    "seed": seed,
                                    "missing_rate": 0.0,
                                    "robustness_test_rates": [float(rate) for rate in args.robustness_test_rates],
                                    "robustness_test_mechanisms": list(args.robustness_test_mechanisms),
                                    "robustness_channel_sync_options": list(args.robustness_channel_sync_options),
                                    "task": args.task,
                                    "target_source": args.target_source,
                                    "search_profile": param_search_profile,
                                    "search_policy": args.param_policy,
                                    "search_repeat": args.param_repeat,
                                    **param_search_contract,
                                    "split_path": portable_artifact_reference(split_path),
                                    "param_path": portable_artifact_reference(param_path),
                                    "output_path": portable_artifact_reference(output_path),
                                    "reason": "output artifact already exists (--check)",
                                    "runtime_baseline_id": runtime_baseline_id,
                                    "runtime_status": "reused",
                                }
                            )
                        continue
                    run_result["cell_wall_clock_s"] = float(cell_elapsed_s)
                    run_result["runtime_status"] = _runtime_status_for_elapsed(
                        cell_elapsed_s,
                        args.performance_limit_seconds,
                    )
                    if run_result["runtime_status"] == "performance_limited":
                        run_result["runtime_status_reason"] = (
                            f"cell wall-clock {cell_elapsed_s:.3f}s exceeded "
                            f"performance limit {float(args.performance_limit_seconds):.3f}s"
                        )
                    run_result["param_path"] = portable_artifact_reference(param_path)
                    run_result["search_profile"] = param_search_profile
                    run_result["search_policy"] = args.param_policy
                    run_result["search_repeat"] = args.param_repeat
                    run_result.update(param_search_contract)
                    records.append(run_result)
                    run_count += 1

                except Exception as e:
                    LOGGER.warning("[SKIP] %s/%s: %s", data_name, model_name, e)
                    runtime_status = (
                        "timeout"
                        if isinstance(e, CellTimeoutError)
                        else "excluded_solver_budget"
                        if isinstance(e, SolverBudgetExcluded)
                        else "failed"
                    )
                    cell_elapsed_s = (time.monotonic() - cell_start) if cell_start is not None else None
                    skipped.append(
                        {
                            "dataset": data_name,
                            "model": model_name,
                            "missing_rate": 0.0,
                            "task": args.task,
                            "reason": str(e),
                            "runtime_status": runtime_status,
                            "runtime_status_reason": str(e),
                            "cell_wall_clock_s": cell_elapsed_s,
                        }
                    )
                    if isinstance(e, OutputArtifactContractError):
                        policy_violations.append(f"{data_name}/{model_name}/robustness: {e}")
                    continue
            if stop_early:
                break

    # Normal mode: existing triple loop
    else:
        for data_name in data_selected:
            for missing_rate in args.missing_rates:
                for model_name in model_name_list:
                    if max_runs is not None and _completed_runs() >= max_runs:
                        LOGGER.info("Reached max-runs=%d, stopping early.", max_runs)
                        stop_early = True
                        break
                    try:
                        cell_start = None
                        param_name, param_path = _resolve_param_artifact_path(
                            args.param_root,
                            data_name,
                            model_name,
                            param_policy=args.param_policy,
                            param_repeat=args.param_repeat,
                            task=args.task,
                        )

                        if not os.path.exists(param_path):
                            raise FileNotFoundError(
                                f"Params file not found: {param_path}. "
                                "Run scripts/param_search.py with "
                                f"--search-policy {args.param_policy} --search-repeat {args.param_repeat} first."
                            )

                        with open(param_path, 'rb') as f:
                            model_config = pickle.load(f)
                        expected_target_identity = _expected_param_target_identity(
                            data_name=data_name,
                            task=args.task,
                            target_source=args.target_source,
                            targets_root=args.targets_root,
                            forecast_horizon=args.forecast_horizon,
                            model_config=model_config,
                            dataset_source=args.dataset_source,
                            dataset_npz=args.dataset_npz,
                            split_root=args.split_root,
                            param_repeat=args.param_repeat,
                        )
                        param_search_profile = _validate_param_search_profile(
                            model_config,
                            expected_search_profile=args.search_profile,
                            expected_param_policy=args.param_policy,
                            expected_param_repeat=args.param_repeat,
                            expected_data_name=data_name,
                            expected_model_name=model_name,
                            expected_task=args.task,
                            expected_target_source=expected_target_identity["target_source"],
                            expected_forecast_horizon=expected_target_identity["forecast_horizon"],
                            expected_dataset_source=expected_target_identity["dataset_source"],
                            expected_split_seeds=expected_target_identity["split_seeds"],
                            param_path=str(param_path),
                            allow_legacy_search_profile=args.allow_legacy_search_profile,
                        )
                        param_search_contract = _param_search_contract_metadata(
                            model_config,
                            param_search_profile,
                        )

                        cell_start = time.monotonic()
                        with _cell_timeout(args.cell_timeout_seconds):
                            run_result = run_model(
                                data_name,
                                missing_rate,
                                model_name,
                                model_config,
                                EPOCHS=args.epochs,
                                SEED=seed,
                                CHECK=args.check,
                                split_root=args.split_root,
                                out_root=args.out_root,
                                run_id=run_id,
                                save_predictions=args.save_predictions,
                                task=args.task,
                                forecast_horizon=args.forecast_horizon,
                                target_source=args.target_source,
                                targets_root=args.targets_root,
                                dataset_source=args.dataset_source,
                                dataset_npz=args.dataset_npz,
                                runtime_baseline_id=runtime_baseline_id,
                                coeff_cache_dir=args.coeff_cache_dir,
                                no_coeff_cache=args.no_coeff_cache,
                                solver_budget_policy=args.solver_budget_policy,
                                solver_budget_seq_len_threshold=args.solver_budget_seq_len_threshold,
                            )
                        cell_elapsed_s = time.monotonic() - cell_start
                        if run_result is None:
                            output_path = _build_output_path(
                                data_name,
                                float(missing_rate),
                                model_name,
                                seed,
                                args.out_root,
                                robustness_mode=False,
                                task=args.task,
                            )
                            split_path = os.path.join(args.split_root, data_name, f"{data_name}_{seed}")
                            if args.existing_policy == "fail":
                                reason = "output artifact already exists (--check, existing-policy=fail)"
                                skipped.append(
                                    {
                                        "dataset": data_name,
                                        "model": model_name,
                                        "missing_rate": float(missing_rate),
                                        "task": args.task,
                                        "reason": reason,
                                    }
                                )
                                policy_violations.append(f"{data_name}/{model_name}/{missing_rate}: {reason}")
                            else:
                                reused_records.append(
                                    {
                                        "dataset": data_name,
                                        "dataset_source": str(args.dataset_source),
                                        "dataset_npz": portable_artifact_reference(args.dataset_npz, root=True),
                                        "model": model_name,
                                        "seed": seed,
                                        "missing_rate": float(missing_rate),
                                        "mode": _evaluation_regime_for_missing_rate(missing_rate),
                                        "evaluation_regime": _evaluation_regime_for_missing_rate(missing_rate),
                                        "task": args.task,
                                        "target_source": args.target_source,
                                        "search_profile": param_search_profile,
                                        "search_policy": args.param_policy,
                                        "search_repeat": args.param_repeat,
                                        **param_search_contract,
                                        "split_path": portable_artifact_reference(split_path),
                                        "param_path": portable_artifact_reference(param_path),
                                        "output_path": portable_artifact_reference(output_path),
                                        "reason": "output artifact already exists (--check)",
                                        "runtime_baseline_id": runtime_baseline_id,
                                        "runtime_status": "reused",
                                    }
                                )
                            continue
                        run_result["cell_wall_clock_s"] = float(cell_elapsed_s)
                        run_result["runtime_status"] = _runtime_status_for_elapsed(
                            cell_elapsed_s,
                            args.performance_limit_seconds,
                        )
                        if run_result["runtime_status"] == "performance_limited":
                            run_result["runtime_status_reason"] = (
                                f"cell wall-clock {cell_elapsed_s:.3f}s exceeded "
                                f"performance limit {float(args.performance_limit_seconds):.3f}s"
                            )
                        run_result["param_path"] = portable_artifact_reference(param_path)
                        run_result["search_profile"] = param_search_profile
                        run_result["search_policy"] = args.param_policy
                        run_result["search_repeat"] = args.param_repeat
                        run_result.update(param_search_contract)
                        records.append(run_result)
                        run_count += 1

                    except Exception as e:
                        LOGGER.warning("[SKIP] %s/%s: %s", data_name, model_name, e)
                        runtime_status = (
                            "timeout"
                            if isinstance(e, CellTimeoutError)
                            else "excluded_solver_budget"
                            if isinstance(e, SolverBudgetExcluded)
                            else "failed"
                        )
                        cell_elapsed_s = (time.monotonic() - cell_start) if cell_start is not None else None
                        skipped.append(
                            {
                                "dataset": data_name,
                                "model": model_name,
                                "missing_rate": float(missing_rate),
                                "task": args.task,
                                "reason": str(e),
                                "runtime_status": runtime_status,
                                "runtime_status_reason": str(e),
                                "cell_wall_clock_s": cell_elapsed_s,
                            }
                        )
                        if isinstance(e, OutputArtifactContractError):
                            policy_violations.append(f"{data_name}/{model_name}/{missing_rate}: {e}")
                        continue
                if stop_early:
                    break
            if stop_early:
                break

    exit_code = 0
    completed_runs = _completed_runs()
    if policy_violations:
        LOGGER.error(
            "existing-policy=fail violations=%d. First violation: %s",
            len(policy_violations),
            policy_violations[0],
        )
        exit_code = 2
    elif args.skip_policy == "fail" and skipped:
        LOGGER.error("Skipped runs=%d and --skip-policy=fail.", len(skipped))
        exit_code = 2
    elif max_runs is not None and completed_runs < max_runs:
        LOGGER.warning(
            "Completed runs=%d but requested max-runs=%d (mode=%s). "
            "Check prerequisites: split/*, params/*, out/* artifacts and existing-policy.",
            completed_runs,
            max_runs,
            args.max_runs_mode,
        )
        exit_code = 2
    elif completed_runs == 0:
        LOGGER.error(
            "No run records were produced; refusing status=pass. "
            "Check dataset/model filters, split/param artifacts, run errors, and --skip-policy."
        )
        exit_code = 2

    LOGGER.info(
        "Completed run_model calls=%d (reused=%d, completed=%d, mode=%s)",
        run_count,
        len(reused_records),
        completed_runs,
        args.max_runs_mode,
    )
    manifest_path, _manifest = write_manifest(
        stage="run",
        contract_dir=args.contract_dir,
        run_id=run_id,
        existing_policy=args.manifest_existing_policy,
        payload=omit_none_runtime_baseline_id({
            "status": "pass" if exit_code == 0 else "fail",
            "seed": seed,
            "task": args.task,
            "dataset_source": str(args.dataset_source),
            "dataset_npz": portable_artifact_reference(args.dataset_npz, root=True),
            "target_source": args.target_source if args.task != "classification" else "label",
            "param_policy": args.param_policy,
            "param_repeat": args.param_repeat,
            "search_profile": args.search_profile,
            **describe_search_profile(args.search_profile),
            "split_root": portable_artifact_reference(args.split_root, root=True),
            "param_root": portable_artifact_reference(args.param_root, root=True),
            "out_root": portable_artifact_reference(args.out_root, root=True),
            "records": records,
            "reused_records": reused_records,
            "skipped": skipped,
            "existing_policy": args.existing_policy,
            "skip_policy": args.skip_policy,
            "max_runs_mode": args.max_runs_mode,
            "runtime_baseline_id": runtime_baseline_id,
            "summary": {
                "completed_run_calls": run_count,
                "executed_runs": run_count,
                "reused_runs": len(reused_records),
                "completed_runs": completed_runs,
                "skipped_runs": len(skipped),
                "requested_max_runs": max_runs,
                "stopped_early": bool(stop_early),
                "policy_violation_count": len(policy_violations),
            },
        }),
    )
    LOGGER.info("Run log written: %s", manifest_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
