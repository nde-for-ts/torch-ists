#-*- coding:utf-8 -*-

from __future__ import annotations

import os
import sys
import copy
import math
import time
import datetime
import json
import pickle
import random
import logging
import argparse
from pathlib import Path

import urllib.request
import zipfile

import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import *

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from _common import (
    CONSTRAINED_CAPACITY_POLICY_NAME,
    CONSTRAINED_MATCHED_CAPACITY_DEPTH_LEVEL_VALUES,
    CONSTRAINED_MATCHED_CAPACITY_HIDDEN_DIM_VALUES,
    CONSTRAINED_MATCHED_CAPACITY_LABEL,
    CONSTRAINED_MATCHED_CAPACITY_TARGET,
    CONSTRAINED_MATCHED_CAPACITY_TARGET_NAME,
    CONSTRAINED_PUBLIC_SEARCH_NAME,
    CONSTRAINED_SEARCH_CONTRACT_NAME,
    CONSTRAINED_SEARCH_FAMILY,
    DEFAULT_SEARCH_PROFILE,
    apply_model_scope_variant,
    build_param_artifact_identity,
    build_model_kwargs_from_config,
    compute_param_config_hash,
    configure_runtime_paths,
    default_method_for_model,
    describe_search_profile,
    infer_param_search_profile,
    LEGACY_ICLR2024_SEARCH_PROFILE,
    normalize_search_profile,
    resolve_dataset_fixed_batch_policy,
    resolve_effective_param_config,
    resolve_learning_rate_policy,
    resolve_model_search_space,
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
from script_chain_contracts import (
    generate_run_id,
    stable_hash,
    write_manifest,
)

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


def _normalize_search_profile_arg(raw: str) -> str:
    try:
        return normalize_search_profile(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


CONSTRAINED_SEARCH_POLICY = "constrained"
AGGREGATED_SEARCH_REPEAT = "aggregated"


def _normalize_search_policy_arg(raw: str) -> str:
    normalized = str(raw).strip().lower()
    if normalized in {STANDARD_SEARCH_POLICY, CONSTRAINED_SEARCH_POLICY}:
        return normalized
    if normalized == "standard":
        return STANDARD_SEARCH_POLICY
    raise argparse.ArgumentTypeError("search policy must be one of {'default', 'constrained'}.")


def _normalize_search_repeat_arg(raw: str) -> str:
    normalized = str(raw).strip().lower()
    if normalized in {SINGLE_SEARCH_REPEAT, AGGREGATED_SEARCH_REPEAT}:
        return normalized
    raise argparse.ArgumentTypeError("search repeat must be one of {'single', 'aggregated'}.")


def _normalize_search_mode_alias(raw: str) -> tuple[str, str]:
    normalized = str(raw).strip().lower()
    if normalized in {"default", "legacy"}:
        return STANDARD_SEARCH_POLICY, SINGLE_SEARCH_REPEAT
    if normalized in {"cv", "improved"}:
        return STANDARD_SEARCH_POLICY, AGGREGATED_SEARCH_REPEAT
    raise argparse.ArgumentTypeError(
        "--search-mode is deprecated; use --search-policy {default,constrained} "
        "and --search-repeat {single,aggregated}."
    )


def _fixed_search_seed_set(search_repeat: str, search_seed: int) -> list[int]:
    if search_repeat == AGGREGATED_SEARCH_REPEAT:
        return [0, 1, 2, 3, 4]
    return [int(search_seed)]


def _build_param_artifact_name(
    data_name: str,
    model_name: str,
    *,
    search_policy: str,
    search_repeat: str,
    task: str,
    param_target: str | None = None,
) -> str:
    parts = [data_name, model_name]
    if search_policy == CONSTRAINED_SEARCH_POLICY:
        parts.append(str(param_target) if param_target else CONSTRAINED_MATCHED_CAPACITY_TARGET_NAME)
    if search_repeat == AGGREGATED_SEARCH_REPEAT:
        parts.append(AGGREGATED_SEARCH_REPEAT)
    out_name = "_".join(parts)
    task_name = str(task).strip().lower()
    if task_name != "classification":
        out_name = f"{out_name}__task_{task_name}"
    return out_name


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
    if artifact_name and artifact_name.endswith("_cv"):
        return AGGREGATED_SEARCH_REPEAT
    return SINGLE_SEARCH_REPEAT


def _validate_existing_param_artifact_for_reuse(
    model_config: dict,
    *,
    existing_param_path: str,
    data_name: str,
    model_name: str,
    task: str,
    allow_legacy_search_profile: bool,
) -> str:
    existing_profile = validate_param_artifact_metadata(
        model_config,
        context=f"Param artifact {existing_param_path}",
        allow_legacy_search_profile=allow_legacy_search_profile,
        require_lr=True,
    )
    if not (allow_legacy_search_profile and existing_profile == LEGACY_ICLR2024_SEARCH_PROFILE):
        validate_param_artifact_identity(
            model_config,
            data_name=data_name,
            model_name=model_name,
            task=task,
            context=f"Param artifact {existing_param_path}",
        )
    return existing_profile


def _resolve_model_scope_variant(requested_variant: str) -> str:
    normalized = str(requested_variant).strip().lower()
    if normalized in {"", "active"}:
        return "active"
    raise ValueError("This package supports only the active model scope.")


def _validate_requested_scope(model_names: list[str], *, scope_variant: str) -> None:
    del model_names, scope_variant
    return


def _requires_flow_raw_input(model_name: str) -> bool:
    from torch_ists.config.alias_policy import is_flow_strict_raw_name

    return is_flow_strict_raw_name(str(model_name))


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run parameter search loop. Defaults to one dataset-model configuration; comma-separated inputs expand the run grid."
    )
    parser.add_argument("--seed", type=int, default=42, help="Search seed for --search-repeat single (default 42). --search-repeat aggregated uses the fixed seed set 0-4; no seed-list CLI is exposed.")
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
    parser.add_argument("--missing-rate", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--num-samples", type=int, default=50, help="Ray Tune num_samples.")
    parser.add_argument("--max-num-epochs", type=int, default=10, help="Ray Tune max epochs.")
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
        help="Behavior when a param artifact already exists. The filename does not include the search seed; if you override --seed, back up or delete params/ and pass --existing-policy fail.",
    )
    parser.add_argument(
        "--skip-policy",
        type=str,
        choices=["allow", "fail"],
        default="fail",
        help="Whether skipped dataset/model cells keep exit status zero or fail the run.",
    )
    parser.add_argument("--ray-cpu", type=float, default=16.0, help="CPU per Ray trial.")
    parser.add_argument("--ray-gpu", type=float, default=0.5, help="GPU per Ray trial.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection mode for each Ray trial.",
    )
    parser.add_argument(
        "--search-policy",
        type=_normalize_search_policy_arg,
        default=None,
        metavar="{default,constrained}",
        help=(
            "Search policy. default=model-specific package search; "
            "constrained=experimental capacity-controlled search."
        ),
    )
    parser.add_argument(
        "--search-repeat",
        type=_normalize_search_repeat_arg,
        default=None,
        metavar="{single,aggregated}",
        help="Repeat policy. single uses --seed; aggregated averages the fixed seed set 0-4.",
    )
    parser.add_argument(
        "--search-mode",
        dest="search_mode_alias",
        type=_normalize_search_mode_alias,
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
        "--split-root",
        type=str,
        default="split",
        help="Input split artifact root directory.",
    )
    parser.add_argument(
        "--param-root",
        type=str,
        default="params",
        help="Output parameter artifact root directory.",
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
        "--task",
        type=str,
        choices=["classification"],
        default="classification",
        help="Task head for this package runtime.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--target-source",
        type=str,
        choices=["surrogate"],
        default="surrogate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--targets-root",
        type=str,
        default="targets",
        help="Root directory for dataset target bundles (<targets-root>/<dataset>.npz).",
    )
    parser.add_argument(
        "--model-scope-variant",
        type=str,
        choices=["active"],
        default="active",
        help="Model scope for this package runtime.",
    )
    args = parser.parse_args()
    if args.search_mode_alias is not None:
        if args.search_policy is not None or args.search_repeat is not None:
            parser.error("--search-mode is a deprecated alias and cannot be combined with --search-policy or --search-repeat.")
        args.search_policy, args.search_repeat = args.search_mode_alias
    args.search_policy = args.search_policy or STANDARD_SEARCH_POLICY
    args.search_repeat = args.search_repeat or SINGLE_SEARCH_REPEAT
    if not math.isclose(float(args.missing_rate), 0.0, abs_tol=1e-12):
        parser.error(
            "param_search.py runs the standard search without synthetic missingness injection. "
            "Provided datasets may still contain NaNs; use model_run.py --missing-rates or --robustness-mode for injected-missingness evaluation."
        )
    args.missing_rate = 0.0
    args.datasets = _parse_csv_list(args.datasets)
    args.models = _parse_csv_list(args.models)
    args.dataset_source = normalize_dataset_source(args.dataset_source)
    if not args.datasets:
        parser.error("--datasets must include at least one dataset.")
    if args.dataset_source == "npz":
        if args.dataset_npz is None:
            parser.error("--dataset-npz is required when --dataset-source=npz")
        if len(args.datasets) != 1:
            parser.error("--dataset-source=npz requires exactly one dataset in --datasets")
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
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
    
SEED = 0
seed_everything(SEED)


def _derive_trial_seed(base_seed: int, trial_token: object) -> int:
    """Derive a trial-local initialization seed without changing split seeds."""
    token = stable_hash({"base_seed": int(base_seed), "trial_token": str(trial_token)})
    derived = (int(token[:8], 16) + int(base_seed)) % (2**31 - 1)
    return derived if derived > 0 else int(base_seed)


def _current_tune_trial_token(model_config: dict) -> str:
    try:
        trial_id = tune.get_trial_id()
        if trial_id:
            return str(trial_id)
    except Exception:
        pass
    try:
        from ray import train as ray_train

        trial_id = ray_train.get_context().get_trial_id()
        if trial_id:
            return str(trial_id)
    except Exception:
        pass
    return stable_hash(model_config)

# Import from the packaged torch_ists runtime.
from torch_ists.preprocessing import preprocess
from torch_ists import ists_classifier, ists_model
from _training_runtime import (
    compute_train_missing_summary,
    enforce_train_missing_gate,
    evaluate,
    ists_dataset,
    normalize_with_train_stats,
    train,
)
from task_target_contracts import load_task_targets_npz

from torch_ists.config import (
    get_data_config,
    get_active_model_list,
    get_default_selected_datasets,
)

ray = None
tune = None
ASHAScheduler = None
BasicVariantGenerator = None


def _load_ray_tune():
    global ray, tune, ASHAScheduler, BasicVariantGenerator
    if ray is not None and tune is not None and ASHAScheduler is not None and BasicVariantGenerator is not None:
        return ray, tune, ASHAScheduler, BasicVariantGenerator
    try:
        import ray as _ray
        from ray import tune as _tune
        from ray.tune.search.basic_variant import BasicVariantGenerator as _BasicVariantGenerator
        from ray.tune.schedulers import ASHAScheduler as _ASHAScheduler
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "") or "ray"
        raise ImportError(
            "param_search.py requires Ray Tune for multi-sample search. "
            f"Missing dependency: {missing_name}. Install `torch-ists[tune]` "
            "or `torch-ists[all]`, or run with --num-samples 1 to use the "
            "local Ray-bypass path."
        ) from exc
    ray = _ray
    tune = _tune
    ASHAScheduler = _ASHAScheduler
    BasicVariantGenerator = _BasicVariantGenerator
    return ray, tune, ASHAScheduler, BasicVariantGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("param_search")

os.makedirs('params', exist_ok=True)


def _collect_task_metadata(
    *,
    data_name: str,
    missing_rate: float,
    model_name: str,
    split_root: str,
    seed: int,
    task: str,
    forecast_horizon,
    target_source: str,
    targets_root: str,
    dataset_source: str,
    dataset_npz: str | None,
) -> dict:
    dataset_payload = load_dataset_payload(
        dataset_name=data_name,
        dataset_source=dataset_source,
        dataset_npz=dataset_npz,
    )
    X = dataset_payload.X
    Y = dataset_payload.y
    timestamps = dataset_payload.timestamps
    num_data = int(X.shape[0])
    num_dim = int(X.shape[1])
    seq_len = int(X.shape[2])
    split_path = os.path.join(split_root, data_name, f"{data_name}_{seed}")
    with open(split_path, "rb") as f:
        train_idx, valid_idx, test_idx = pickle.load(f)
    del valid_idx, test_idx
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

    data_config = get_data_config(model_name)
    X_missing, _, _, _ = preprocess(
        X,
        missing_rate=missing_rate,
        interpolate=data_config["interpolation"],
        use_intensity=data_config["use_intensity"],
        mechanism="MCAR",
        seed=seed,
        channel_sync=False,
        timestamps=timestamps,
        timestamp_delta_scale=timestamp_delta_scale,
        missingness_reference=X[train_idx],
    )
    X_missing_norm, normalization_summary = normalize_with_train_stats(X_missing, train_idx)
    train_missing_summary = compute_train_missing_summary(X_missing, train_idx)
    train_missing_summary.update(normalization_summary)

    if task == "classification":
        target_masks = None
        target_spec = {
            "task": "classification",
            "target_key": "label",
            "target_shape": [int(num_data)],
            "target_dtype": str(np.asarray(Y).dtype),
        }
    elif target_source == "dataset":
        _, target_masks, target_spec = load_task_targets_npz(
            dataset=data_name,
            targets_root=targets_root,
            task=task,
            horizon=forecast_horizon,
            num_samples=num_data,
            seq_len=seq_len,
            input_dim=num_dim,
        )
    else:
        if task == "change_detection":
            raise ValueError("change_detection requires --target-source dataset")
        target_masks = None
        target_spec = {
            "task": task,
            "target_key": f"surrogate::{task}",
            "target_shape": None,
            "target_dtype": "torch.float32",
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

    return {
        "batch_policy": batch_policy,
        "train_missing_summary": train_missing_summary,
        "target_spec": target_spec,
        "target_source": str(target_source if task != "classification" else "label"),
        "timestamp_summary": timestamp_summary,
    }


def _normalize_param_config_for_save(
    model_name: str,
    model_config: dict,
    *,
    search_profile: str | None = None,
    artifact_identity: dict | None = None,
    search_provenance: dict | None = None,
) -> dict:
    saved = resolve_effective_param_config(
        model_name,
        model_config,
        search_profile=search_profile,
    )
    if _infer_cli_search_policy(model_config) == CONSTRAINED_SEARCH_POLICY:
        saved["search_contract"] = model_config.get("search_contract", CONSTRAINED_SEARCH_CONTRACT_NAME)
        saved["search_family"] = model_config.get("search_family", CONSTRAINED_SEARCH_FAMILY)
        saved["public_search_name"] = model_config.get("public_search_name", CONSTRAINED_PUBLIC_SEARCH_NAME)
    if artifact_identity is not None:
        saved["artifact_identity"] = dict(artifact_identity)
    if search_provenance is not None:
        saved.update(search_provenance)
    if artifact_identity is not None or search_provenance is not None:
        saved["param_hash"] = compute_param_config_hash(saved)
    return saved


def _build_search_provenance(
    *,
    num_samples: int,
    max_num_epochs: int,
    execution_backend: str,
    search_algorithm: str,
    search_algorithm_seed: int,
) -> dict:
    return {
        "search_num_samples": int(num_samples),
        "search_max_num_epochs": int(max_num_epochs),
        "search_execution_backend": str(execution_backend),
        "search_budget_class": "smoke" if int(num_samples) <= 1 else "production",
        "search_algorithm": str(search_algorithm),
        "search_algorithm_seed": int(search_algorithm_seed),
    }


def _count_model_params_for_config(
    *,
    model_name: str,
    input_dim: int,
    seq_len: int,
    num_class: int,
    task: str,
    forecast_horizon,
    model_config: dict,
) -> int:
    effective = _normalize_param_config_for_save(
        model_name,
        model_config,
        search_profile=model_config.get("search_profile", DEFAULT_SEARCH_PROFILE),
    )
    kwargs = build_model_kwargs_from_config(model_name, effective)
    data_config = get_data_config(model_name)
    common_kwargs = {
        "model_name": model_name,
        "input_dim": int(input_dim),
        "seq_len": int(seq_len),
        "dropout": 0.1,
        "use_intensity": data_config["use_intensity"],
        "method": default_method_for_model(model_name),
        "device": "cpu",
        **kwargs,
    }
    if str(task).strip().lower() == "classification":
        model = ists_classifier(num_class=int(num_class), **common_kwargs)
    else:
        model = ists_model(
            num_class=int(num_class),
            task=task,
            output_dim=max(1, int(input_dim)),
            forecast_horizon=forecast_horizon,
            **common_kwargs,
        )
    return int(sum(param.numel() for param in model.parameters()))


def _candidate_capacity_config(
    *,
    data_name: str,
    missing_rate: float,
    model_name: str,
    task: str,
    search_profile: str,
    search_space: dict,
    hidden_dim: int,
    depth_level: int,
    extra_params: dict,
) -> dict:
    cfg = {
        "data_name": data_name,
        "missing_rate": float(missing_rate),
        "model_name": model_name,
        "task": task,
        "search_profile": search_profile,
        "search_contract": search_space["search_contract"],
        "search_family": search_space["search_family"],
        "public_search_name": search_space["public_search_name"],
        "search_profile_visibility": search_space["search_profile_visibility"],
        "search_policy": CONSTRAINED_SEARCH_POLICY,
        "search_repeat": SINGLE_SEARCH_REPEAT,
        "search_algorithm_policy": search_space["policy"],
        "lr": 1e-3,
        "hidden_dim": int(hidden_dim),
        "depth_level": int(depth_level),
    }
    cfg.update(extra_params)
    return cfg


def _resolve_constrained_capacity_config(
    *,
    data_name: str,
    missing_rate: float,
    model_name: str,
    input_dim: int,
    seq_len: int,
    num_class: int,
    task: str,
    forecast_horizon,
    search_profile: str,
    search_space: dict,
) -> dict:
    target = int(CONSTRAINED_MATCHED_CAPACITY_TARGET)
    lower = int(target * 0.5)
    upper = int(target * 1.5)
    extra_defaults = dict(search_space.get("default_extra_params", {}))
    extra_values = {
        key: list(values)
        for key, values in search_space.get("extra_param_values", {}).items()
    }
    flow_values = extra_values.get("flow_depth") or (
        [int(extra_defaults["flow_depth"])] if "flow_depth" in extra_defaults else [None]
    )
    if "flow_depth" not in extra_defaults:
        flow_values = [None]

    candidates = []
    hidden_dim_values = list(CONSTRAINED_MATCHED_CAPACITY_HIDDEN_DIM_VALUES)
    if search_space.get("depth_collapsed"):
        depth_level_values = [1]
    else:
        depth_level_values = list(CONSTRAINED_MATCHED_CAPACITY_DEPTH_LEVEL_VALUES)

    for hidden_dim in hidden_dim_values:
        for depth_level in depth_level_values:
            for flow_depth in flow_values:
                extra_params = dict(extra_defaults)
                if flow_depth is not None:
                    extra_params["flow_depth"] = int(flow_depth)
                cfg = _candidate_capacity_config(
                    data_name=data_name,
                    missing_rate=missing_rate,
                    model_name=model_name,
                    task=task,
                    search_profile=search_profile,
                    search_space=search_space,
                    hidden_dim=int(hidden_dim),
                    depth_level=int(depth_level),
                    extra_params=extra_params,
                )
                param_count = _count_model_params_for_config(
                    model_name=model_name,
                    input_dim=input_dim,
                    seq_len=seq_len,
                    num_class=num_class,
                    task=task,
                    forecast_horizon=forecast_horizon,
                    model_config=cfg,
                )
                candidates.append(
                    {
                        "hidden_dim": int(hidden_dim),
                        "depth_level": int(depth_level),
                        "flow_depth": None if flow_depth is None else int(flow_depth),
                        "param_count": int(param_count),
                        "distance": abs(int(param_count) - target),
                    }
                )

    if not candidates:
        raise ValueError(f"No constrained capacity candidates generated for model {model_name!r}.")

    selected = min(
        candidates,
        key=lambda row: (
            row["distance"],
            row["param_count"],
            row["hidden_dim"],
            row["depth_level"],
            -1 if row["flow_depth"] is None else row["flow_depth"],
        ),
    )
    min_candidate = min(candidates, key=lambda row: row["param_count"])
    max_candidate = max(candidates, key=lambda row: row["param_count"])
    in_band = lower <= selected["param_count"] <= upper
    metadata = {
        "search_policy_axis": CONSTRAINED_SEARCH_POLICY,
        "capacity_policy": CONSTRAINED_CAPACITY_POLICY_NAME,
        "capacity_target": target,
        "capacity_target_name": CONSTRAINED_MATCHED_CAPACITY_TARGET_NAME,
        "capacity_label": CONSTRAINED_MATCHED_CAPACITY_LABEL,
        "capacity_tolerance": 0.5,
        "capacity_lower": lower,
        "capacity_upper": upper,
        "capacity_hidden_dim_grid": list(hidden_dim_values),
        "capacity_depth_level_grid": list(depth_level_values),
        "capacity_flow_depth_grid": [
            None if value is None else int(value)
            for value in flow_values
        ],
        "capacity_param_count": int(selected["param_count"]),
        "capacity_distance": int(selected["distance"]),
        "capacity_in_band": bool(in_band),
        "capacity_overshoot": bool(min_candidate["param_count"] > upper),
        "capacity_undershoot": bool(max_candidate["param_count"] < lower),
        "capacity_hidden_dim": int(selected["hidden_dim"]),
        "capacity_depth_level": int(selected["depth_level"]),
        "capacity_flow_depth": selected["flow_depth"],
        "params_at_min_carrier": int(min_candidate["param_count"]),
        "params_at_max_carrier": int(max_candidate["param_count"]),
        "candidate_count": int(len(candidates)),
        "input_dim": int(input_dim),
        "seq_len": int(seq_len),
    }
    return {
        "selected": selected,
        "extra_params": {
            key: value
            for key, value in {
                **extra_defaults,
                "flow_depth": selected["flow_depth"],
            }.items()
            if value is not None and (key in extra_defaults or key == "flow_depth")
        },
        "metadata": metadata,
    }


def _apply_constrained_capacity_to_search_space(search_space: dict, capacity_config: dict) -> dict:
    selected = capacity_config["selected"]
    constrained = copy.deepcopy(search_space)
    constrained["default_hidden_dim"] = int(selected["hidden_dim"])
    constrained["default_depth_level"] = int(selected["depth_level"])
    constrained["hidden_dim_values"] = [int(selected["hidden_dim"])]
    constrained["depth_level_values"] = [int(selected["depth_level"])]
    constrained["sampled_axes"] = ["lr"]
    constrained["candidate_axes"] = ["lr"]
    constrained["search_contract"] = CONSTRAINED_SEARCH_CONTRACT_NAME
    constrained["search_family"] = CONSTRAINED_SEARCH_FAMILY
    constrained["public_search_name"] = CONSTRAINED_PUBLIC_SEARCH_NAME
    fixed_extra = dict(capacity_config["extra_params"])
    constrained["default_extra_params"] = fixed_extra
    constrained["extra_param_values"] = {
        key: [value]
        for key, value in fixed_extra.items()
        if key in search_space.get("extra_param_values", {})
    }
    if selected["flow_depth"] is not None:
        constrained["default_flow_depth"] = int(selected["flow_depth"])
        constrained["flow_depth_values"] = [int(selected["flow_depth"])]
    return constrained


def _attach_constrained_capacity_metadata(model_config: dict, capacity_config: dict | None) -> dict:
    if capacity_config is None:
        return model_config
    updated = dict(model_config)
    updated.update(capacity_config["metadata"])
    updated["search_policy_axis"] = CONSTRAINED_SEARCH_POLICY
    updated["capacity_config"] = dict(capacity_config["selected"])
    return updated


_CONSTRAINED_CAPACITY_METADATA_KEYS = (
    "search_policy_axis",
    "capacity_policy",
    "capacity_target",
    "capacity_target_name",
    "capacity_label",
    "capacity_tolerance",
    "capacity_lower",
    "capacity_upper",
    "capacity_hidden_dim_grid",
    "capacity_depth_level_grid",
    "capacity_flow_depth_grid",
    "capacity_param_count",
    "capacity_distance",
    "capacity_in_band",
    "capacity_overshoot",
    "capacity_undershoot",
    "capacity_hidden_dim",
    "capacity_depth_level",
    "capacity_flow_depth",
    "params_at_min_carrier",
    "params_at_max_carrier",
    "candidate_count",
)


def _capacity_metadata_from_config(model_config: dict) -> dict:
    return {
        key: model_config.get(key)
        for key in _CONSTRAINED_CAPACITY_METADATA_KEYS
        if key in model_config
    }


def _build_task_targets(task, x_missing_norm, y, forecast_horizon=None):
    if task == "classification":
        return None
    filled = torch.nan_to_num(x_missing_norm, 0.0)
    if task == "regression":
        return filled.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
    if task == "forecasting":
        horizon = filled.shape[1] if forecast_horizon is None else max(1, int(forecast_horizon))
        horizon = min(horizon, filled.shape[1])
        return filled[:, -horizon:, :]
    if task == "imputation":
        return filled
    if task == "outlier_detection":
        labels = torch.as_tensor(y, device=filled.device)
        normal = torch.tensor(0, device=labels.device, dtype=labels.dtype)
        if not torch.any(labels == normal):
            normal = torch.min(labels)
        return (labels != normal).to(dtype=filled.dtype).unsqueeze(-1)
    if task == "change_detection":
        raise ValueError("change_detection requires --target-source dataset")
    raise ValueError(f"Unsupported task: {task}")


def report_tune_loss(loss_value, *, epoch=None):
    _ray, _tune, *_ = _load_ray_tune()
    metrics = {"loss": float(loss_value)}
    if epoch is not None:
        metrics["epoch"] = int(epoch)
    try:
        _tune.report(**metrics)
        return
    except TypeError:
        pass
    try:
        _tune.report(metrics)
        return
    except TypeError:
        pass
    from ray import train as ray_train

    ray_train.report(metrics)


def _trial_reported_iteration(trial) -> int:
    last_result = getattr(trial, "last_result", None) or {}
    iteration = 0
    for key in ("training_iteration", "epoch"):
        value = last_result.get(key)
        if value is None:
            continue
        try:
            iteration = max(iteration, int(value))
        except (TypeError, ValueError):
            continue
    return iteration


def _trial_reported_loss(trial) -> float | None:
    last_result = getattr(trial, "last_result", None) or {}
    if "loss" not in last_result:
        return None
    try:
        loss = float(last_result["loss"])
    except (TypeError, ValueError):
        return None
    return loss if np.isfinite(loss) else None


class SearchNoCompletedTrialError(RuntimeError):
    """Raised when ASHA prunes every single-repeat trial before max_t."""


def _select_best_ray_trial(result, *, max_num_epochs: int, require_completed: bool):
    if not require_completed:
        return result.get_best_trial("loss", "min", "last")
    candidates = []
    for trial in getattr(result, "trials", []) or []:
        loss = _trial_reported_loss(trial)
        if loss is None:
            continue
        if _trial_reported_iteration(trial) >= int(max_num_epochs):
            candidates.append((loss, trial))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def resolve_trial_device(device_mode, ray_gpu):
    if device_mode == "cpu":
        return torch.device("cpu")
    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested --device cuda but CUDA is unavailable in this execution context."
            )
        if ray_gpu <= 0:
            raise RuntimeError("Requested --device cuda requires --ray-gpu > 0.")
        return torch.device("cuda")
    if ray_gpu > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Helper function for one split seed (used by both single and aggregated repeats)
def _train_single_seed(
    data_name,
    missing_rate,
    model_name,
    model_config,
    EPOCHS,
    seed,
    trial_device,
    interpolate,
    use_intensity,
    split_root,
    task="classification",
    forecast_horizon=None,
    target_source="surrogate",
    targets_root="targets",
    dataset_source="default",
    dataset_npz=None,
    epoch_reporter=None,
):
    """
    Train and evaluate model on a single seed split.
    Returns the best validation loss for this seed.
    """
    # Load data
    dataset_payload = load_dataset_payload(
        dataset_name=data_name,
        dataset_source=dataset_source,
        dataset_npz=dataset_npz,
    )
    X = dataset_payload.X
    Y = dataset_payload.y
    timestamps = dataset_payload.timestamps
    num_data = X.shape[0]
    num_dim = X.shape[1]
    seq_len = X.shape[2]
    num_class = len(np.unique(Y))

    # Load split file
    split_path = os.path.join(split_root, data_name, f'{data_name}_{seed}')
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split file not found: {split_path}. "
            f"Run set_splits.py first to generate split files for seed={seed}."
        )
    with open(split_path, 'rb') as f:
        train_idx, valid_idx, test_idx = pickle.load(f)
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

    # Set learning rate
    lr_policy = resolve_learning_rate_policy(model_config.get("lr"))
    lr = float(lr_policy["lr"])
    weight_decay = float(lr_policy["weight_decay"])
    LOGGER.info(
        "Batch policy dataset=%s model=%s seed=%s policy=%s batch=%d train_n=%d steps=%d drop=%d sample_elements=%d",
        data_name,
        model_name,
        seed,
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
        seed,
        lr_policy["policy"],
        lr_policy["source"],
        lr,
        weight_decay,
    )

    # Preprocess data
    X_missing, X_mask, X_delta, coeffs = preprocess(
        X,
        missing_rate=missing_rate,
        interpolate=interpolate,
        use_intensity=use_intensity,
        mechanism="MCAR",
        seed=seed,
        channel_sync=False,
        timestamps=timestamps,
        timestamp_delta_scale=timestamp_delta_scale,
        missingness_reference=X[train_idx],
    )
    X_missing_norm, normalization_summary = normalize_with_train_stats(X_missing, train_idx)
    train_missing_summary = compute_train_missing_summary(X_missing, train_idx)
    train_missing_summary.update(normalization_summary)

    target_masks = None
    if task == "classification":
        task_targets = None
        target_masks = None
        target_spec = {
            "task": "classification",
            "target_key": "label",
            "target_shape": [int(num_data)],
            "target_dtype": str(np.asarray(Y).dtype),
        }
    elif target_source == "dataset":
        task_targets, target_masks, target_spec = load_task_targets_npz(
            dataset=data_name,
            targets_root=targets_root,
            task=task,
            horizon=forecast_horizon,
            num_samples=int(num_data),
            seq_len=int(seq_len),
            input_dim=int(num_dim),
        )
    else:
        if task == "change_detection":
            raise ValueError("change_detection requires --target-source dataset")
        task_targets = _build_task_targets(
            task=task,
            x_missing_norm=X_missing_norm,
            y=Y,
            forecast_horizon=forecast_horizon,
        )
        target_masks = None
        target_spec = {
            "task": task,
            "target_key": f"surrogate::{task}",
            "target_shape": None,
            "target_dtype": "torch.float32",
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

    # Create datasets
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
    train_batch = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_batch = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Set model params
    model_kwargs = build_model_kwargs_from_config(model_name, model_config)

    # Create tmp path for ANCDE (model-specific file I/O)
    os.makedirs(os.path.join(os.getcwd(), 'tmp'), exist_ok=True)
    tmp_path = os.path.join(os.getcwd(), 'tmp/{}_{}.npy'.format(data_name, str(seed)))

    solver_method = default_method_for_model(model_name)

    # Create model
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
            device=trial_device,
            **model_kwargs
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
            device=trial_device,
            **model_kwargs
        )
    model = model.to(trial_device)

    # Set loss & optimizer
    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task in {"outlier_detection", "change_detection"}:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    patient = 0

    for e in tqdm(range(EPOCHS)):
        train_loss = train(
            model_name, model, optimizer, criterion, train_batch, interpolate, use_intensity, trial_device, task=task
        )
        valid_loss = evaluate(
            model_name, model, criterion, valid_batch, interpolate, use_intensity, trial_device, task=task
        )

        if e % 10 == 0:
            LOGGER.info(
                "Seed=%d Epoch=%d train=%.6f valid=%.6f",
                seed,
                e,
                train_loss,
                valid_loss,
            )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patient = 0
        else:
            patient += 1

        if epoch_reporter is not None:
            epoch_reporter(best_loss, epoch=e + 1)

        if (e > 10) and (patient > 5):
            break

        scheduler.step()

    LOGGER.info("Seed=%d finished with best_loss=%.6f", seed, best_loss)
    return best_loss


# set model
def tune_model(
    data_name,
    missing_rate,
    model_name,
    model_config,
    EPOCHS=100,
    SEED=SEED,
    device_mode="auto",
    ray_gpu=0.0,
    search_repeat=SINGLE_SEARCH_REPEAT,
    split_root="split",
    task="classification",
    forecast_horizon=None,
    target_source="surrogate",
    targets_root="targets",
    dataset_source="default",
    dataset_npz=None,
    trial_seed=None,
):
    trial_device = resolve_trial_device(device_mode, ray_gpu)
    LOGGER.info("Tune start dataset=%s model=%s repeat=%s", data_name, model_name, search_repeat)
    LOGGER.info(
        "Trial device resolved mode=%s ray_gpu=%s device=%s",
        device_mode,
        ray_gpu,
        trial_device,
    )

    # Get model data config
    data_config = get_data_config(model_name)
    interpolate = data_config['interpolation']
    use_intensity = data_config['use_intensity']

    if search_repeat == SINGLE_SEARCH_REPEAT:
        seed_everything(SEED if trial_seed is None else trial_seed)
        reported_epochs = 0

        def _report_epoch(loss_value, *, epoch):
            nonlocal reported_epochs
            reported_epochs += 1
            report_tune_loss(loss_value, epoch=epoch)

        best_loss = _train_single_seed(
            data_name=data_name,
            missing_rate=missing_rate,
            model_name=model_name,
            model_config=model_config,
            EPOCHS=EPOCHS,
            seed=SEED,
            trial_device=trial_device,
            interpolate=interpolate,
            use_intensity=use_intensity,
            split_root=split_root,
            task=task,
            forecast_horizon=forecast_horizon,
            target_source=target_source,
            targets_root=targets_root,
            dataset_source=dataset_source,
            dataset_npz=dataset_npz,
            epoch_reporter=_report_epoch,
        )
        LOGGER.info("Single repeat finished dataset=%s model=%s best_loss=%.6f", data_name, model_name, best_loss)
        if reported_epochs == 0:
            report_tune_loss(best_loss)

    elif search_repeat == AGGREGATED_SEARCH_REPEAT:
        repeat_losses = []
        for seed in _fixed_search_seed_set(search_repeat, SEED):
            fold_trial_seed = seed if trial_seed is None else _derive_trial_seed(trial_seed, seed)
            seed_everything(fold_trial_seed)
            loss = _train_single_seed(
                data_name=data_name,
                missing_rate=missing_rate,
                model_name=model_name,
                model_config=model_config,
                EPOCHS=EPOCHS,
                seed=seed,
                trial_device=trial_device,
                interpolate=interpolate,
                use_intensity=use_intensity,
                split_root=split_root,
                task=task,
                forecast_horizon=forecast_horizon,
                target_source=target_source,
                targets_root=targets_root,
                dataset_source=dataset_source,
                dataset_npz=dataset_npz,
            )
            repeat_losses.append(loss)
            LOGGER.info("Aggregated repeat seed=%d dataset=%s model=%s loss=%.6f", seed, data_name, model_name, loss)

        avg_loss = np.mean(repeat_losses)
        std_loss = np.std(repeat_losses)
        LOGGER.info(
            "Aggregated repeat finished dataset=%s model=%s avg_loss=%.6f±%.6f (seeds=%s)",
            data_name,
            model_name,
            avg_loss,
            std_loss,
            repeat_losses,
        )
        report_tune_loss(avg_loss)

    else:
        raise ValueError(f"Unknown search_repeat: {search_repeat}. Expected 'single' or 'aggregated'.")

    LOGGER.info("Finished training dataset=%s model=%s repeat=%s", data_name, model_name, search_repeat)


def optimize(
    data_name,
    missing_rate,
    model_name,
    split_path,
    split_root="split",
    param_root="params",
    num_samples=50,
    max_num_epochs=10,
    ray_cpu=16.0,
    ray_gpu=0.5,
    device_mode="auto",
    search_policy=STANDARD_SEARCH_POLICY,
    search_repeat=SINGLE_SEARCH_REPEAT,
    search_profile=DEFAULT_SEARCH_PROFILE,
    task="classification",
    forecast_horizon=None,
    target_source="surrogate",
    targets_root="targets",
    dataset_source="default",
    dataset_npz=None,
):
    search_space = resolve_model_search_space(model_name, search_profile)
    search_profile = search_space["search_profile"]
    dataset_payload = load_dataset_payload(
        dataset_name=data_name,
        dataset_source=dataset_source,
        dataset_npz=dataset_npz,
    )
    _, timestamp_summary = resolve_timestamp_delta_metadata(
        dataset_payload.timestamps,
        summary=dataset_payload.timestamp_summary,
    )
    constrained_capacity_config = None
    if search_policy == CONSTRAINED_SEARCH_POLICY:
        X_capacity = dataset_payload.X
        Y_capacity = dataset_payload.y
        input_dim = int(X_capacity.shape[1])
        seq_len = int(X_capacity.shape[2])
        if str(task).strip().lower() == "classification":
            num_class = int(len(np.unique(Y_capacity)))
        else:
            num_class = 1
        constrained_capacity_config = _resolve_constrained_capacity_config(
            data_name=data_name,
            missing_rate=missing_rate,
            model_name=model_name,
            input_dim=input_dim,
            seq_len=seq_len,
            num_class=num_class,
            task=task,
            forecast_horizon=forecast_horizon,
            search_profile=search_profile,
            search_space=search_space,
        )
        search_space = _apply_constrained_capacity_to_search_space(
            search_space,
            constrained_capacity_config,
        )
        LOGGER.info(
            "Constrained search selected %s for %s/%s: hidden_dim=%s depth_level=%s "
            "flow_depth=%s params=%s target=%s",
            CONSTRAINED_MATCHED_CAPACITY_TARGET_NAME,
            data_name,
            model_name,
            constrained_capacity_config["metadata"]["capacity_hidden_dim"],
            constrained_capacity_config["metadata"]["capacity_depth_level"],
            constrained_capacity_config["metadata"]["capacity_flow_depth"],
            constrained_capacity_config["metadata"]["capacity_param_count"],
            constrained_capacity_config["metadata"]["capacity_target"],
        )
    metadata_seed = _fixed_search_seed_set(search_repeat, SEED)[0]
    artifact_identity = build_param_artifact_identity(
        data_name=data_name,
        model_name=model_name,
        task=task,
    )
    if num_samples <= 1:
        LOGGER.info(
            "Using local single-sample optimization path (Ray bypass) for dataset=%s model=%s",
            data_name,
            model_name,
        )
        trial_device = resolve_trial_device(device_mode, ray_gpu)
        model_config = {
            "data_name": data_name,
            "missing_rate": missing_rate,
            "model_name": model_name,
            "task": task,
            "search_profile": search_profile,
            "search_contract": search_space["search_contract"],
            "search_family": search_space["search_family"],
            "public_search_name": search_space["public_search_name"],
            "search_profile_visibility": search_space["search_profile_visibility"],
            "search_policy": search_policy,
            "search_repeat": search_repeat,
            "search_algorithm_policy": search_space["policy"],
            "lr": 1e-3,
            "hidden_dim": int(search_space["default_hidden_dim"]),
            "depth_level": int(search_space["default_depth_level"]),
        }
        model_config.update(search_space.get("default_extra_params", {}))
        model_config = _attach_constrained_capacity_metadata(
            model_config,
            constrained_capacity_config,
        )
        split_root_local = split_root

        if search_repeat == SINGLE_SEARCH_REPEAT:
            best_loss = _train_single_seed(
                data_name=data_name,
                missing_rate=missing_rate,
                model_name=model_name,
                model_config=model_config,
                EPOCHS=max_num_epochs,
                seed=SEED,
                trial_device=trial_device,
                interpolate=get_data_config(model_name)["interpolation"],
                use_intensity=get_data_config(model_name)["use_intensity"],
                split_root=split_root_local,
                task=task,
                forecast_horizon=forecast_horizon,
                target_source=target_source,
                targets_root=targets_root,
                dataset_source=dataset_source,
                dataset_npz=dataset_npz,
            )
        elif search_repeat == AGGREGATED_SEARCH_REPEAT:
            losses = []
            for seed in _fixed_search_seed_set(search_repeat, SEED):
                losses.append(
                    _train_single_seed(
                        data_name=data_name,
                        missing_rate=missing_rate,
                        model_name=model_name,
                        model_config=model_config,
                        EPOCHS=max_num_epochs,
                        seed=seed,
                        trial_device=trial_device,
                        interpolate=get_data_config(model_name)["interpolation"],
                        use_intensity=get_data_config(model_name)["use_intensity"],
                        split_root=split_root_local,
                        task=task,
                        forecast_horizon=forecast_horizon,
                        target_source=target_source,
                        targets_root=targets_root,
                        dataset_source=dataset_source,
                        dataset_npz=dataset_npz,
                    )
                )
            best_loss = float(np.mean(losses))
        else:
            raise ValueError(f"Unknown search_repeat: {search_repeat}")

        metadata = _collect_task_metadata(
            data_name=data_name,
            missing_rate=missing_rate,
            model_name=model_name,
            split_root=split_root,
            seed=metadata_seed,
            task=task,
            forecast_horizon=forecast_horizon,
            target_source=target_source,
            targets_root=targets_root,
            dataset_source=dataset_source,
            dataset_npz=dataset_npz,
        )
        saved_model_config = _normalize_param_config_for_save(
            model_name,
            model_config,
            search_profile=search_profile,
            artifact_identity=artifact_identity,
            search_provenance=_build_search_provenance(
                num_samples=num_samples,
                max_num_epochs=max_num_epochs,
                execution_backend="local-single",
                search_algorithm="local-single",
                search_algorithm_seed=SEED,
            ),
        )
        saved_model_config["timestamp_summary"] = metadata["timestamp_summary"]
        saved_model_config["param_hash"] = compute_param_config_hash(saved_model_config)
        os.makedirs(os.path.join(param_root, data_name), exist_ok=True)
        out_name = _build_param_artifact_name(
            data_name,
            model_name,
            search_policy=search_policy,
            search_repeat=search_repeat,
            task=task,
        )
        param_path = os.path.join(param_root, data_name, out_name)
        with open(param_path, "wb") as f:
            pickle.dump(saved_model_config, f)

        return {
            "best_trial": None,
            "best_loss": float(best_loss),
            "param_config": saved_model_config,
            "param_hash": saved_model_config["param_hash"],
            "param_path": str(param_path),
            "split_path": str(split_path),
            "batch_policy": metadata["batch_policy"],
            "lr_policy": resolve_learning_rate_policy(saved_model_config.get("lr")),
            "search_profile": search_profile,
            "search_contract": saved_model_config["search_contract"],
            "search_family": saved_model_config["search_family"],
            "public_search_name": saved_model_config["public_search_name"],
            "search_profile_visibility": saved_model_config["search_profile_visibility"],
            "search_policy": saved_model_config["search_policy"],
            "search_repeat": saved_model_config["search_repeat"],
            "search_algorithm_policy": saved_model_config["search_algorithm_policy"],
            "search_algorithm": saved_model_config["search_algorithm"],
            "search_algorithm_seed": saved_model_config["search_algorithm_seed"],
            "search_num_samples": saved_model_config["search_num_samples"],
            "search_max_num_epochs": saved_model_config["search_max_num_epochs"],
            "search_execution_backend": saved_model_config["search_execution_backend"],
            "search_budget_class": saved_model_config["search_budget_class"],
            "legacy_grid_version": saved_model_config["legacy_grid_version"],
            "architecture_axes": saved_model_config["architecture_axes"],
            "effective_params": saved_model_config["effective_params"],
            "capacity_metadata": _capacity_metadata_from_config(saved_model_config),
            "train_missing_summary": metadata["train_missing_summary"],
            "target_spec": metadata["target_spec"],
            "target_source": metadata["target_source"],
            "timestamp_summary": metadata["timestamp_summary"],
            "dataset_source": str(dataset_source),
            "dataset_npz": None if dataset_npz is None else str(dataset_npz),
        }

    if device_mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is unavailable.")
    if device_mode == "cuda" and ray_gpu <= 0:
        raise RuntimeError("Requested --device cuda requires --ray-gpu > 0.")
    _load_ray_tune()

    # Propagate runtime import context to Ray workers.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pythonpath = os.environ.get("PYTHONPATH")
    if not pythonpath:
        pythonpath = repo_root
    if not ray.is_initialized():
        runtime_env = {
            "working_dir": repo_root,
            "excludes": [
                "evidence/",
                "out/",
                "params/",
                "split/",
                ".pytest_cache/",
                "canasde/*.ipynb",
            ],
            "env_vars": {
                "PYTHONPATH": pythonpath,
                "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE", "1"),
            },
        }
        ray.init(runtime_env=runtime_env)

    model_config = {
        "data_name": data_name,
        "missing_rate": missing_rate,
        "model_name": model_name,
        "task": task,
        "search_profile": search_profile,
        "search_contract": search_space["search_contract"],
        "search_family": search_space["search_family"],
        "public_search_name": search_space["public_search_name"],
        "search_profile_visibility": search_space["search_profile_visibility"],
        "search_policy": search_policy,
        "search_repeat": search_repeat,
        "search_algorithm_policy": search_space["policy"],
        # "batch_size": None, # determine by the number of the data
        "lr": tune.loguniform(*search_space["lr_bounds"]),
    }
    if constrained_capacity_config is None:
        model_config["hidden_dim"] = tune.choice(search_space["hidden_dim_values"])
        model_config["depth_level"] = tune.choice(search_space["depth_level_values"])
        for param_name, param_values in search_space.get("extra_param_values", {}).items():
            model_config[param_name] = tune.choice(param_values)
    else:
        model_config["hidden_dim"] = int(search_space["default_hidden_dim"])
        model_config["depth_level"] = int(search_space["default_depth_level"])
        model_config.update(search_space.get("default_extra_params", {}))
        model_config = _attach_constrained_capacity_metadata(
            model_config,
            constrained_capacity_config,
        )

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    search_alg = BasicVariantGenerator(random_state=SEED)

    def optimize_model(model_config):
        trial_seed = _derive_trial_seed(SEED, _current_tune_trial_token(model_config))
        tune_model(data_name=data_name, missing_rate=missing_rate, model_name=model_name,
                   model_config=model_config, EPOCHS=max_num_epochs, SEED=SEED,
                   device_mode=device_mode, ray_gpu=ray_gpu, search_repeat=search_repeat,
                   split_root=split_root, task=task, forecast_horizon=forecast_horizon,
                   target_source=target_source, targets_root=targets_root,
                   dataset_source=dataset_source, dataset_npz=dataset_npz,
                   trial_seed=trial_seed)

    result = tune.run(
        tune.with_parameters(optimize_model),
        resources_per_trial={"cpu": ray_cpu, "gpu": ray_gpu},
        config=model_config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        stop={"training_iteration": max_num_epochs, "loss": np.nan},
        raise_on_failed_trial=False,
    )
    
    require_completed_trial = search_repeat == SINGLE_SEARCH_REPEAT
    best_trial = _select_best_ray_trial(
        result,
        max_num_epochs=max_num_epochs,
        require_completed=require_completed_trial,
    )
    if best_trial is None:
        if require_completed_trial:
            message = (
                "No successful Ray trial reached max_t=%d with finite loss dataset=%s model=%s. "
                "Check ASHA pruning/resource failures or lower --max-num-epochs."
            ) % (
                max_num_epochs,
                data_name,
                model_name,
            )
            LOGGER.error(message)
            raise SearchNoCompletedTrialError(message)
        message = (
            "No successful trial with finite loss dataset=%s model=%s. "
            "Check --device/--ray-gpu and runtime resources."
        ) % (
            data_name,
            model_name,
        )
        LOGGER.error(message)
        raise RuntimeError(message)
    if best_trial.last_result is None or "loss" not in best_trial.last_result:
        LOGGER.error(
            "Best trial exists but no loss is reported dataset=%s model=%s.",
            data_name,
            model_name,
        )
        return None
    LOGGER.info("Best trial config: %s", best_trial.config)
    LOGGER.info(
        "Best trial final validation loss: %s iteration=%s completion_required=%s",
        best_trial.last_result["loss"],
        _trial_reported_iteration(best_trial),
        require_completed_trial,
    )
    
    # save params
    os.makedirs(os.path.join(param_root, data_name), exist_ok=True)

    out_name = _build_param_artifact_name(
        data_name,
        model_name,
        search_policy=search_policy,
        search_repeat=search_repeat,
        task=task,
    )

    param_path = os.path.join(param_root, data_name, out_name)
    saved_param_config = _normalize_param_config_for_save(
        model_name,
        best_trial.config,
        search_profile=search_profile,
        artifact_identity=artifact_identity,
        search_provenance=_build_search_provenance(
            num_samples=num_samples,
            max_num_epochs=max_num_epochs,
            execution_backend="ray-tune",
            search_algorithm="BasicVariantGenerator",
            search_algorithm_seed=SEED,
        ),
    )
    saved_param_config["timestamp_summary"] = metadata["timestamp_summary"]
    saved_param_config["param_hash"] = compute_param_config_hash(saved_param_config)
    with open(param_path, 'wb') as f:
        pickle.dump(saved_param_config, f)

    LOGGER.info("Params saved to %s (repeat=%s)", param_path, search_repeat)

    metadata = _collect_task_metadata(
        data_name=data_name,
        missing_rate=missing_rate,
        model_name=model_name,
        split_root=split_root,
        seed=metadata_seed,
        task=task,
        forecast_horizon=forecast_horizon,
        target_source=target_source,
        targets_root=targets_root,
        dataset_source=dataset_source,
        dataset_npz=dataset_npz,
    )

    return {
        "best_trial": best_trial,
        "best_loss": float(best_trial.last_result["loss"]),
        "param_config": saved_param_config,
        "param_hash": saved_param_config["param_hash"],
        "param_path": str(param_path),
        "split_path": str(split_path),
        "batch_policy": metadata["batch_policy"],
        "lr_policy": resolve_learning_rate_policy(saved_param_config.get("lr")),
        "search_profile": search_profile,
        "search_contract": saved_param_config["search_contract"],
        "search_family": saved_param_config["search_family"],
        "public_search_name": saved_param_config["public_search_name"],
        "search_profile_visibility": saved_param_config["search_profile_visibility"],
        "search_policy": saved_param_config["search_policy"],
        "search_repeat": saved_param_config["search_repeat"],
        "search_algorithm_policy": saved_param_config["search_algorithm_policy"],
        "search_algorithm": saved_param_config["search_algorithm"],
        "search_algorithm_seed": saved_param_config["search_algorithm_seed"],
        "search_num_samples": saved_param_config["search_num_samples"],
        "search_max_num_epochs": saved_param_config["search_max_num_epochs"],
        "search_execution_backend": saved_param_config["search_execution_backend"],
        "search_budget_class": saved_param_config["search_budget_class"],
        "legacy_grid_version": saved_param_config["legacy_grid_version"],
        "architecture_axes": saved_param_config["architecture_axes"],
        "effective_params": saved_param_config["effective_params"],
        "capacity_metadata": _capacity_metadata_from_config(saved_param_config),
        "train_missing_summary": metadata["train_missing_summary"],
        "target_spec": metadata["target_spec"],
        "target_source": metadata["target_source"],
        "timestamp_summary": metadata["timestamp_summary"],
        "dataset_source": str(dataset_source),
        "dataset_npz": None if dataset_npz is None else str(dataset_npz),
    }


def main():
    global SEED
    args = parse_cli_args()
    if int(args.num_samples) <= 1:
        LOGGER.warning(
            "param_search is using --num-samples=%d. This creates a smoke/local artifact, "
            "not a production hyperparameter-search artifact.",
            int(args.num_samples),
        )
    else:
        LOGGER.warning(
            "param_search is using --num-samples=%d. This is a production search budget; "
            "use --num-samples 1 --max-num-epochs 1 for a smoke run.",
            int(args.num_samples),
        )
    scope_override = _resolve_model_scope_variant(args.model_scope_variant)
    apply_model_scope_variant(scope_override)
    SEED = int(args.seed)
    seed_everything(SEED)
    run_id = args.run_id or generate_run_id("run")
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
        _validate_requested_scope(model_name_list, scope_variant=scope_override)

    max_runs = args.max_runs if (args.max_runs is not None and args.max_runs > 0) else None
    run_count = 0
    stop_early = False

    def _completed_runs() -> int:
        if args.max_runs_mode == "new":
            return run_count
        return run_count + len(reused_records)

    for data_name in data_selected:
        for model_name in model_name_list:
            if max_runs is not None and _completed_runs() >= max_runs:
                LOGGER.info("Reached max-runs=%d, stopping early.", max_runs)
                stop_early = True
                break

            out_name = _build_param_artifact_name(
                data_name,
                model_name,
                search_policy=args.search_policy,
                search_repeat=args.search_repeat,
                task=args.task,
            )
            split_seeds = _fixed_search_seed_set(args.search_repeat, SEED)
            split_paths = [
                os.path.join(args.split_root, data_name, f"{data_name}_{seed}")
                for seed in split_seeds
            ]
            missing_split_paths = [path for path in split_paths if not os.path.exists(path)]
            split_path = split_paths[0]
            if missing_split_paths:
                reason = "split artifact missing: " + ", ".join(missing_split_paths)
                LOGGER.warning("[SKIP] %s/%s: %s", data_name, model_name, reason)
                skipped.append({"dataset": data_name, "model": model_name, "task": args.task, "reason": reason})
                continue

            if os.path.exists(os.path.join(args.param_root, data_name, out_name)):
                existing_param_path = os.path.join(args.param_root, data_name, out_name)
                if args.existing_policy == "fail":
                    reason = "param artifact already exists (existing-policy=fail)"
                    LOGGER.error("[FAIL] %s/%s: %s", data_name, model_name, reason)
                    skipped.append({"dataset": data_name, "model": model_name, "task": args.task, "reason": reason})
                    policy_violations.append(f"{data_name}/{model_name}: {reason}")
                else:
                    LOGGER.info("Params file already exists: %s/%s (checking reuse details)", data_name, out_name)
                    try:
                        with open(existing_param_path, "rb") as f:
                            existing_cfg = pickle.load(f)
                        param_hash = existing_cfg.get("param_hash")
                    except Exception as exc:
                        reason = f"failed to load existing params for reuse check: {exc}"
                        LOGGER.error("[FAIL] %s/%s: %s", data_name, model_name, reason)
                        skipped.append({"dataset": data_name, "model": model_name, "task": args.task, "reason": reason})
                        policy_violations.append(f"{data_name}/{model_name}: {reason}")
                        continue
                    try:
                        existing_profile = _validate_existing_param_artifact_for_reuse(
                            existing_cfg,
                            existing_param_path=existing_param_path,
                            data_name=data_name,
                            model_name=model_name,
                            task=args.task,
                            allow_legacy_search_profile=args.allow_legacy_search_profile,
                        )
                    except ValueError as exc:
                        reason = f"param artifact reuse check failed: {exc}"
                        LOGGER.error("[FAIL] %s/%s: %s", data_name, model_name, reason)
                        skipped.append({"dataset": data_name, "model": model_name, "task": args.task, "reason": reason})
                        policy_violations.append(f"{data_name}/{model_name}: {reason}")
                        continue
                    existing_contract = describe_search_profile(existing_profile)
                    existing_search_policy = _infer_cli_search_policy(existing_cfg)
                    existing_search_repeat = _infer_cli_search_repeat(existing_cfg, out_name)
                    if existing_profile != args.search_profile:
                        reason = (
                            "param artifact search_profile mismatch "
                            f"(found={existing_profile!r}, expected={args.search_profile!r})"
                        )
                        LOGGER.error("[FAIL] %s/%s: %s", data_name, model_name, reason)
                        skipped.append({"dataset": data_name, "model": model_name, "task": args.task, "reason": reason})
                        policy_violations.append(f"{data_name}/{model_name}: {reason}")
                        continue
                    if existing_search_policy != args.search_policy or existing_search_repeat != args.search_repeat:
                        reason = (
                            "param artifact search axes mismatch "
                            f"(found policy={existing_search_policy!r}, repeat={existing_search_repeat!r}; "
                            f"expected policy={args.search_policy!r}, repeat={args.search_repeat!r})"
                        )
                        LOGGER.error("[FAIL] %s/%s: %s", data_name, model_name, reason)
                        skipped.append({"dataset": data_name, "model": model_name, "task": args.task, "reason": reason})
                        policy_violations.append(f"{data_name}/{model_name}: {reason}")
                        continue
                    existing_seed = existing_cfg.get("search_algorithm_seed")
                    if existing_seed is None or int(existing_seed) != int(SEED):
                        reason = (
                            "param artifact search_algorithm_seed mismatch "
                            f"(found={existing_seed!r}, expected={int(SEED)!r})"
                        )
                        LOGGER.error("[FAIL] %s/%s: %s", data_name, model_name, reason)
                        skipped.append({"dataset": data_name, "model": model_name, "task": args.task, "reason": reason})
                        policy_violations.append(f"{data_name}/{model_name}: {reason}")
                        continue
                    LOGGER.info("Params file already exists: %s/%s (reusing)", data_name, out_name)
                    reused_records.append(
                        {
                            "dataset": data_name,
                            "dataset_source": str(args.dataset_source),
                            "dataset_npz": None if args.dataset_npz is None else str(args.dataset_npz),
                            "model": model_name,
                            "seed": SEED,
                            "search_seed_set": split_seeds,
                            "missing_rate": float(args.missing_rate),
                            "task": args.task,
                            "target_source": args.target_source,
                            "timestamp_summary": existing_cfg.get("timestamp_summary"),
                            "search_profile": existing_profile,
                            "search_contract": existing_cfg.get("search_contract", existing_contract["search_contract"]),
                            "search_family": existing_cfg.get("search_family", existing_contract["search_family"]),
                            "public_search_name": existing_cfg.get("public_search_name", existing_contract["public_search_name"]),
                            "search_profile_visibility": existing_cfg.get("search_profile_visibility", existing_contract["search_profile_visibility"]),
                            "search_policy": existing_search_policy,
                            "search_repeat": existing_search_repeat,
                            "search_algorithm_policy": existing_cfg.get("search_algorithm_policy", existing_cfg.get("search_policy")),
                            "search_algorithm": existing_cfg.get("search_algorithm"),
                            "search_algorithm_seed": existing_cfg.get("search_algorithm_seed"),
                            "search_num_samples": existing_cfg.get("search_num_samples"),
                            "search_max_num_epochs": existing_cfg.get("search_max_num_epochs"),
                            "search_execution_backend": existing_cfg.get("search_execution_backend"),
                            "search_budget_class": existing_cfg.get("search_budget_class"),
                            "legacy_grid_version": existing_cfg.get("legacy_grid_version", existing_profile),
                            "architecture_axes": existing_cfg.get("architecture_axes"),
                            "effective_params": existing_cfg.get("effective_params"),
                            "capacity_metadata": _capacity_metadata_from_config(existing_cfg),
                            "split_path": str(split_path),
                            "param_path": str(existing_param_path),
                            "param_hash": str(param_hash),
                            "reason": "param artifact already exists",
                        }
                    )
                continue

            try:
                search_result = optimize(
                    data_name=data_name,
                    missing_rate=args.missing_rate,
                    model_name=model_name,
                    split_path=split_path,
                    split_root=args.split_root,
                    param_root=args.param_root,
                    num_samples=args.num_samples,
                    max_num_epochs=args.max_num_epochs,
                    ray_cpu=args.ray_cpu,
                    ray_gpu=args.ray_gpu,
                    device_mode=args.device,
                    search_policy=args.search_policy,
                    search_repeat=args.search_repeat,
                    search_profile=args.search_profile,
                    task=args.task,
                    forecast_horizon=args.forecast_horizon,
                    target_source=args.target_source,
                    targets_root=args.targets_root,
                    dataset_source=args.dataset_source,
                    dataset_npz=args.dataset_npz,
                )
                if search_result is None:
                    LOGGER.warning(
                        "[SKIP] %s/%s: no successful trial result (all trials failed or no reported loss)",
                        data_name,
                        model_name,
                    )
                    skipped.append(
                        {
                            "dataset": data_name,
                            "model": model_name,
                            "task": args.task,
                            "reason": "no successful trial result",
                        }
                    )
                    continue
                run_count += 1
                records.append(
                    {
                        "dataset": data_name,
                        "dataset_source": str(search_result["dataset_source"]),
                        "dataset_npz": search_result["dataset_npz"],
                        "model": model_name,
                        "seed": SEED,
                        "search_seed_set": _fixed_search_seed_set(args.search_repeat, SEED),
                        "missing_rate": float(args.missing_rate),
                        "task": args.task,
                        "target_source": str(search_result["target_source"]),
                        "train_missing_summary": search_result["train_missing_summary"],
                        "target_spec": search_result["target_spec"],
                        "timestamp_summary": search_result["timestamp_summary"],
                        "split_path": str(search_result["split_path"]),
                        "param_path": str(search_result["param_path"]),
                        "param_hash": str(search_result["param_hash"]),
                        "batch_policy": search_result["batch_policy"],
                        "lr_policy": search_result["lr_policy"],
                        "search_profile": search_result["search_profile"],
                        "search_contract": search_result["search_contract"],
                        "search_family": search_result["search_family"],
                        "public_search_name": search_result["public_search_name"],
                        "search_profile_visibility": search_result["search_profile_visibility"],
                        "search_policy": search_result["search_policy"],
                        "search_repeat": search_result["search_repeat"],
                        "search_algorithm_policy": search_result["search_algorithm_policy"],
                        "search_algorithm": search_result["search_algorithm"],
                        "search_algorithm_seed": search_result["search_algorithm_seed"],
                        "search_num_samples": search_result["search_num_samples"],
                        "search_max_num_epochs": search_result["search_max_num_epochs"],
                        "search_execution_backend": search_result["search_execution_backend"],
                        "search_budget_class": search_result["search_budget_class"],
                        "legacy_grid_version": search_result["legacy_grid_version"],
                        "architecture_axes": search_result["architecture_axes"],
                        "effective_params": search_result["effective_params"],
                        "capacity_metadata": search_result["capacity_metadata"],
                        "best_loss": float(search_result["best_loss"]),
                    }
                )
            except SearchNoCompletedTrialError as e:
                reason = str(e)
                LOGGER.error("[FAIL] %s/%s: %s", data_name, model_name, reason)
                skipped.append({"dataset": data_name, "model": model_name, "task": args.task, "reason": reason})
                policy_violations.append(f"{data_name}/{model_name}: {reason}")
                continue
            except Exception as e:
                LOGGER.warning("[SKIP] %s/%s: %s", data_name, model_name, e)
                skipped.append({"dataset": data_name, "model": model_name, "task": args.task, "reason": str(e)})
                continue
        if stop_early:
            break

    completed_runs = _completed_runs()
    LOGGER.info(
        "Completed optimize calls=%d (reused=%d, completed=%d, mode=%s)",
        run_count,
        len(reused_records),
        completed_runs,
        args.max_runs_mode,
    )
    exit_code = 0
    if policy_violations:
        LOGGER.error(
            "Parameter artifact errors=%d. First error: %s",
            len(policy_violations),
            policy_violations[0],
        )
        exit_code = 2
    elif args.skip_policy == "fail" and skipped:
        LOGGER.error("Skipped runs=%d and --skip-policy=fail.", len(skipped))
        exit_code = 2
    elif max_runs is not None and completed_runs < max_runs:
        LOGGER.error(
            "Completed runs=%d but requested max-runs=%d (mode=%s). "
            "Check split/params artifacts, trial errors, and existing-policy.",
            completed_runs,
            max_runs,
            args.max_runs_mode,
        )
        exit_code = 2

    search_contract_metadata = describe_search_profile(args.search_profile)
    if args.search_policy == CONSTRAINED_SEARCH_POLICY:
        search_contract_metadata = dict(search_contract_metadata)
        search_contract_metadata.update(
            {
                "search_contract": CONSTRAINED_SEARCH_CONTRACT_NAME,
                "search_family": CONSTRAINED_SEARCH_FAMILY,
                "public_search_name": CONSTRAINED_PUBLIC_SEARCH_NAME,
            }
        )

    manifest_path, _manifest = write_manifest(
        stage="search",
        contract_dir=args.contract_dir,
        run_id=run_id,
        payload={
            "status": "pass" if exit_code == 0 else "fail",
            "seed": SEED,
            "missing_rate": float(args.missing_rate),
            "task": args.task,
            "dataset_source": str(args.dataset_source),
            "dataset_npz": None if args.dataset_npz is None else str(args.dataset_npz),
            "target_source": args.target_source if args.task != "classification" else "label",
            "search_policy": args.search_policy,
            "search_policy_axis": args.search_policy,
            "search_repeat": args.search_repeat,
            "capacity_target": (
                int(CONSTRAINED_MATCHED_CAPACITY_TARGET)
                if args.search_policy == CONSTRAINED_SEARCH_POLICY
                else None
            ),
            "capacity_target_name": (
                CONSTRAINED_MATCHED_CAPACITY_TARGET_NAME
                if args.search_policy == CONSTRAINED_SEARCH_POLICY
                else None
            ),
            "capacity_policy": (
                CONSTRAINED_CAPACITY_POLICY_NAME
                if args.search_policy == CONSTRAINED_SEARCH_POLICY
                else None
            ),
            "search_seed_set": _fixed_search_seed_set(args.search_repeat, SEED),
            "search_profile": args.search_profile,
            **search_contract_metadata,
            "split_root": str(args.split_root),
            "param_root": str(args.param_root),
            "records": records,
            "reused_records": reused_records,
            "skipped": skipped,
            "existing_policy": args.existing_policy,
            "skip_policy": args.skip_policy,
            "max_runs_mode": args.max_runs_mode,
            "summary": {
                "completed_optimize_calls": run_count,
                "executed_runs": run_count,
                "reused_runs": len(reused_records),
                "completed_runs": completed_runs,
                "skipped_runs": len(skipped),
                "requested_max_runs": max_runs,
                "stopped_early": bool(stop_early),
                "policy_violation_count": len(policy_violations),
            },
        },
    )
    LOGGER.info("Search run log written: %s", manifest_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
