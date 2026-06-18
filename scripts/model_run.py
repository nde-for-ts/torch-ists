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
    DEFAULT_SEARCH_PROFILE,
    apply_model_scope_variant,
    build_model_kwargs_from_config,
    configure_runtime_paths,
    default_method_for_model,
    infer_param_search_profile,
    normalize_search_profile,
    resolve_dataset_fixed_batch_policy,
    resolve_learning_rate_policy,
)
from dataset_source_contracts import load_dataset, normalize_dataset_source
from script_chain_contracts import generate_run_id, write_manifest

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


def _normalize_search_profile_arg(raw: str) -> str:
    try:
        return normalize_search_profile(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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


def _validate_param_search_profile(
    model_config: dict,
    *,
    expected_search_profile: str,
    param_path: str,
) -> str:
    actual = infer_param_search_profile(model_config)
    expected = normalize_search_profile(expected_search_profile)
    if actual != expected:
        raise ValueError(
            "Param artifact search_profile mismatch: "
            f"{param_path} has {actual!r}, expected {expected!r}. "
            "Pass --search-profile legacy-iclr2024-v0 only when intentionally "
            "evaluating historical-grid params."
        )
    return actual


def _parse_missing_rates(raw):
    if raw is None:
        return [0.0, 0.3, 0.5, 0.7]
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
    supported = {"MCAR", "MAR", "MNAR", "BLOCK", "BURST"}
    if not tokens:
        raise ValueError("At least one robustness missingness mechanism must be provided.")
    invalid = sorted(set(tokens) - supported)
    if invalid:
        raise ValueError(f"Unsupported robustness missingness mechanism(s): {', '.join(invalid)}")
    return list(dict.fromkeys(tokens))


def _parse_robustness_channel_sync_options(raw):
    if raw is None:
        raw = "auto"
    if isinstance(raw, str):
        tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    else:
        tokens = [str(token).strip().lower() for token in raw if str(token).strip()]
    aliases = {
        "auto": "auto",
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
            raise ValueError("robustness channel sync options must be drawn from {auto,false,true}")
        resolved.append(aliases[token])
    if not resolved:
        raise ValueError("At least one robustness channel_sync option must be provided.")
    return list(dict.fromkeys(resolved))


def _parse_robustness_test_seeds(raw):
    if raw is None:
        raw = "0"
    if isinstance(raw, str):
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in raw if str(token).strip()]
    if not tokens:
        raise ValueError("At least one robustness missingness seed must be provided.")
    return list(dict.fromkeys(int(token) for token in tokens))


def _resolve_channel_sync_options(options, use_raw_flow_input):
    resolved = []
    for option in _parse_robustness_channel_sync_options(options):
        if option == "auto":
            value = bool(use_raw_flow_input)
        elif option == "true":
            value = True
        else:
            value = False
        if value not in resolved:
            resolved.append(value)
    return resolved


def _axis_token(values):
    return "-".join(
        str(value).strip().lower().replace("_", "-").replace(".", "p").replace("-", "m")
        for value in values
    )


def _build_robustness_axis_suffix(
    robustness_test_rates=None,
    robustness_test_mechanisms=None,
    robustness_channel_sync_options=None,
    robustness_test_seeds=None,
):
    rates = _parse_missing_rates(robustness_test_rates)
    mechanisms = _parse_robustness_mechanisms(robustness_test_mechanisms)
    sync_options = _parse_robustness_channel_sync_options(robustness_channel_sync_options)
    seeds = _parse_robustness_test_seeds(robustness_test_seeds)
    if rates == [0.0, 0.3, 0.5, 0.7] and mechanisms == ["MCAR"] and sync_options == ["auto"] and seeds == [0]:
        return ""
    return "__" + "__".join(
        [
            "rates_" + _axis_token(rates),
            "types_" + _axis_token(mechanisms),
            "sync_" + _axis_token(sync_options),
            "mseeds_" + _axis_token(seeds),
        ]
    )


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation loop (default full loop, optional subset mode)."
    )
    parser.add_argument("seed_positional", nargs="?", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset list. Default uses get_default_selected_datasets().",
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
        default=None,
        help="Comma-separated missing rates. Default: 0.0,0.3,0.5,0.7",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run.")
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
        help="Behavior when output artifact already exists with --check.",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        default=True,
        help="Skip run if output file already exists (default: enabled).",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Disable output existence check.",
    )
    parser.add_argument(
        "--param-mode",
        type=_normalize_param_mode,
        default="default",
        metavar="{default,cv}",
        help="Load parameters from the default search or the cv search.",
    )
    parser.add_argument(
        "--search-profile",
        type=_normalize_search_profile_arg,
        default=DEFAULT_SEARCH_PROFILE,
        metavar="{legacy-expanded-v1,legacy-iclr2024-v0}",
        help="Expected search profile of the loaded param artifact.",
    )
    parser.add_argument(
        "--robustness-mode",
        action="store_true",
        help="Robustness evaluation: train on missing=0, test on multiple rates",
    )
    parser.add_argument(
        "--robustness-test-rates",
        type=str,
        default="0.0,0.3,0.5,0.7",
        help="Comma-separated test missing rates for robustness mode",
    )
    parser.add_argument(
        "--robustness-test-mechanisms",
        type=str,
        default="MCAR",
        help="Comma-separated test missingness mechanisms for robustness mode: MCAR,MAR,MNAR,BLOCK,BURST",
    )
    parser.add_argument(
        "--robustness-channel-sync-options",
        type=str,
        default="auto",
        help="Comma-separated channel_sync options for robustness mode: auto,false,true",
    )
    parser.add_argument(
        "--robustness-test-seeds",
        type=str,
        default="0",
        help="Comma-separated missingness RNG seeds for robustness test injections",
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
        help="Directory for run manifests.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier for related artifacts.",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        choices=["full", "summary"],
        default="full",
        help="Prediction payload mode. full=store arrays, summary=store shape/dtype metadata only.",
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
    if args.seed is None:
        args.seed = args.seed_positional if args.seed_positional is not None else 0
    args.datasets = _parse_csv_list(args.datasets)
    args.models = _parse_csv_list(args.models)
    args.missing_rates = _parse_missing_rates(args.missing_rates)
    args.robustness_test_rates = _parse_missing_rates(args.robustness_test_rates)
    args.robustness_test_mechanisms = _parse_robustness_mechanisms(args.robustness_test_mechanisms)
    args.robustness_channel_sync_options = _parse_robustness_channel_sync_options(args.robustness_channel_sync_options)
    args.robustness_test_seeds = _parse_robustness_test_seeds(args.robustness_test_seeds)
    args.dataset_source = normalize_dataset_source(args.dataset_source)
    if args.dataset_source == "npz":
        if args.dataset_npz is None:
            parser.error("--dataset-npz is required when --dataset-source=npz")
        if not args.datasets or len(args.datasets) != 1:
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 0
seed_everything(SEED)

# CUDA for PyTorch
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Import from the packaged torch_ists runtime.
from torch_ists.preprocessing import preprocess
from torch_ists import ists_classifier, ists_model
from _training_runtime import (
    _align_prediction_target,
    compute_train_missing_summary,
    enforce_train_missing_gate,
    evaluate,
    ists_dataset,
    train,
)
from task_target_contracts import load_task_targets_npz
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

os.makedirs('out', exist_ok=True)


def _expand_mask_or_delta(feature, x_missing):
    if feature.dim() == 3:
        feature = feature[..., 0]
    return feature.unsqueeze(-1).repeat((1, 1, x_missing.shape[-1]))


def _validate_bounded_metric(name, value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"metric {name} must be finite, got {value}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"metric {name} must be in [0, 1], got {value}")
    return value


def _build_output_path(
    data_name,
    missing_rate,
    model_name,
    seed,
    out_root,
    robustness_mode=False,
    robustness_test_rates=None,
    robustness_test_mechanisms=None,
    robustness_channel_sync_options=None,
    robustness_test_seeds=None,
):
    if robustness_mode:
        suffix = _build_robustness_axis_suffix(
            robustness_test_rates=robustness_test_rates,
            robustness_test_mechanisms=robustness_test_mechanisms,
            robustness_channel_sync_options=robustness_channel_sync_options,
            robustness_test_seeds=robustness_test_seeds,
        )
        out_name = "_".join([data_name, "robustness", model_name, str(seed)]) + suffix
        return os.path.join(out_root, data_name, "robustness", out_name)
    out_name = "_".join([data_name, str(missing_rate), model_name, str(seed)])
    return os.path.join(out_root, data_name, str(missing_rate), out_name)


def _compute_metrics(y_true, y_pred):
    accuracy = _validate_bounded_metric("accuracy", accuracy_score(y_true, y_pred))
    f1_macro = _validate_bounded_metric("f1", f1_score(y_true, y_pred, average="macro"))
    wf1 = _validate_bounded_metric("wf1", f1_score(y_true, y_pred, average="weighted"))
    return {
        "accuracy": float(accuracy),
        "f1": float(f1_macro),
        "wf1": float(wf1),
        "f1_weighted": float(wf1),
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


def _compute_task_metrics(pred: np.ndarray, target: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch for task metrics: {pred.shape} != {target.shape}")
    diff = pred - target
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    return {"mse": mse, "mae": mae}


def _compute_binary_detection_metrics(pred: np.ndarray, target: np.ndarray, *, accuracy_key: str):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch for binary detection metrics: {pred.shape} != {target.shape}")

    logits = torch.from_numpy(pred)
    labels = torch.from_numpy(target)
    bce = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item())

    probs = 1.0 / (1.0 + np.exp(-pred))
    pred_label = probs >= 0.5
    true_label = target >= 0.5
    acc = float(np.mean(pred_label == true_label))
    return {"bce": bce, accuracy_key: acc}


def _build_predictions_payload(y_true, y_pred, logit_list, save_predictions="full"):
    if save_predictions == "full":
        return {
            "storage": "full",
            "y_true": y_true,
            "y_pred": y_pred,
            "logit_list": logit_list,
        }
    if save_predictions == "summary":
        n_samples = int(y_true.shape[0]) if hasattr(y_true, "shape") else int(len(y_true))
        n_classes = int(logit_list.shape[1]) if getattr(logit_list, "ndim", 0) >= 2 else 0
        return {
            "storage": "summary",
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


def _build_evaluation_result(
    test_missing_rate,
    y_true=None,
    y_pred=None,
    logit_list=None,
    task_pred=None,
    task_target=None,
    *,
    task="classification",
    test_missing_type="MCAR",
    test_channel_sync=None,
    test_missing_seed=0,
    save_predictions="full",
):
    task = str(task).strip().lower()
    if task == "classification":
        metrics = _compute_metrics(y_true, y_pred)
        predictions = _build_predictions_payload(
            y_true=y_true,
            y_pred=y_pred,
            logit_list=logit_list,
            save_predictions=save_predictions,
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
        }
        if save_predictions == "full":
            predictions = {
                "storage": "full",
                "y_true": None,
                "y_pred": None,
                "logit_list": None,
                "target": task_target,
                "pred": task_pred,
            }
        elif save_predictions == "summary":
            task_pred_np = np.asarray(task_pred)
            task_target_np = np.asarray(task_target)
            predictions = {
                "storage": "summary",
                "n_samples": int(task_pred_np.shape[0]) if task_pred_np.ndim > 0 else 0,
                "n_classes": 0,
                "y_true_dtype": None,
                "y_pred_dtype": None,
                "logit_dtype": None,
                "task_pred_shape": list(task_pred_np.shape),
                "task_target_shape": list(task_target_np.shape),
                "task_pred_dtype": str(task_pred_np.dtype),
                "task_target_dtype": str(task_target_np.dtype),
                "y_true": None,
                "y_pred": None,
                "logit_list": None,
                "target": None,
                "pred": None,
            }
        else:
            raise ValueError(f"save_predictions must be 'full' or 'summary', got {save_predictions!r}")
    elif task == "change_detection":
        binary_metrics = _compute_binary_detection_metrics(
            task_pred,
            task_target,
            accuracy_key="change_acc",
        )
        metrics = {
            "bce": float(binary_metrics["bce"]),
            "change_acc": float(binary_metrics["change_acc"]),
        }
        if save_predictions == "full":
            predictions = {
                "storage": "full",
                "y_true": None,
                "y_pred": None,
                "logit_list": None,
                "target": task_target,
                "pred": task_pred,
            }
        elif save_predictions == "summary":
            task_pred_np = np.asarray(task_pred)
            task_target_np = np.asarray(task_target)
            predictions = {
                "storage": "summary",
                "n_samples": int(task_pred_np.shape[0]) if task_pred_np.ndim > 0 else 0,
                "n_classes": 0,
                "y_true_dtype": None,
                "y_pred_dtype": None,
                "logit_dtype": None,
                "task_pred_shape": list(task_pred_np.shape),
                "task_target_shape": list(task_target_np.shape),
                "task_pred_dtype": str(task_pred_np.dtype),
                "task_target_dtype": str(task_target_np.dtype),
                "y_true": None,
                "y_pred": None,
                "logit_list": None,
                "target": None,
                "pred": None,
            }
        else:
            raise ValueError(f"save_predictions must be 'full' or 'summary', got {save_predictions!r}")
    else:
        task_metrics = _compute_task_metrics(task_pred, task_target)
        metrics = {
            "mse": float(task_metrics["mse"]),
            "mae": float(task_metrics["mae"]),
        }
        if save_predictions == "full":
            predictions = {
                "storage": "full",
                "y_true": None,
                "y_pred": None,
                "logit_list": None,
                "target": task_target,
                "pred": task_pred,
            }
        elif save_predictions == "summary":
            task_pred_np = np.asarray(task_pred)
            task_target_np = np.asarray(task_target)
            predictions = {
                "storage": "summary",
                "n_samples": int(task_pred_np.shape[0]) if task_pred_np.ndim > 0 else 0,
                "n_classes": 0,
                "y_true_dtype": None,
                "y_pred_dtype": None,
                "logit_dtype": None,
                "task_pred_shape": list(task_pred_np.shape),
                "task_target_shape": list(task_target_np.shape),
                "task_pred_dtype": str(task_pred_np.dtype),
                "task_target_dtype": str(task_target_np.dtype),
                "y_true": None,
                "y_pred": None,
                "logit_list": None,
                "target": None,
                "pred": None,
            }
        else:
            raise ValueError(f"save_predictions must be 'full' or 'summary', got {save_predictions!r}")
    return {
        "test_missing_rate": float(test_missing_rate),
        "test_missing_type": str(test_missing_type).upper(),
        "test_channel_sync": None if test_channel_sync is None else bool(test_channel_sync),
        "test_missing_seed": int(test_missing_seed),
        "metrics": metrics,
        "predictions": predictions,
    }


def _build_result_payload(
    *,
    mode,
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
):
    return {
        "mode": str(mode),
        "task": str(task),
        "dataset": str(dataset),
        "model": str(model_name),
        "seed": int(seed),
        "train_missing_rate": float(train_missing_rate),
        "train_missing_type": str(train_missing_type).upper(),
        "train_channel_sync": None if train_channel_sync is None else bool(train_channel_sync),
        "evaluations": evaluations,
        "meta": {
            "task": str(task),
            "task_support_tier": "stable" if str(task) == "classification" else "experimental",
            "epochs": int(epochs),
            "interpolate": str(interpolate),
            "use_intensity": bool(use_intensity),
            "device": str(device),
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "save_predictions": str(save_predictions),
            "batch_policy": batch_policy,
            "lr_policy": lr_policy,
            "robustness_axes": robustness_axes,
        },
    }


def run_model(
    data_name,
    missing_rate,
    model_name,
    model_config,
    EPOCHS=100,
    SEED=SEED,
    CHECK=False,
    robustness_mode=False,
    robustness_test_rates=None,
    robustness_test_mechanisms=None,
    robustness_channel_sync_options=None,
    robustness_test_seeds=None,
    split_root="split",
    out_root="out",
    run_id=None,
    save_predictions="full",
    task="classification",
    forecast_horizon=None,
    target_source="surrogate",
    targets_root="targets",
    dataset_source="default",
    dataset_npz=None,
):
    task = str(task).strip().lower()
    if robustness_mode:
        robustness_test_rates = _parse_missing_rates(robustness_test_rates)
        robustness_test_mechanisms = _parse_robustness_mechanisms(robustness_test_mechanisms)
        robustness_channel_sync_options = _parse_robustness_channel_sync_options(robustness_channel_sync_options)
        robustness_test_seeds = _parse_robustness_test_seeds(robustness_test_seeds)
    if robustness_mode:
        LOGGER.info(
            "Robustness run start dataset=%s model=%s (train missing=0.0, test rates=%s, types=%s, sync=%s, missing_seeds=%s)",
            data_name,
            model_name,
            robustness_test_rates,
            robustness_test_mechanisms,
            robustness_channel_sync_options,
            robustness_test_seeds,
        )
    else:
        LOGGER.info("Run start dataset=%s missing=%.1f model=%s", data_name, missing_rate, model_name)

    # check exist
    out_path = _build_output_path(
        data_name,
        missing_rate,
        model_name,
        SEED,
        out_root,
        robustness_mode=robustness_mode,
        robustness_test_rates=robustness_test_rates,
        robustness_test_mechanisms=robustness_test_mechanisms,
        robustness_channel_sync_options=robustness_channel_sync_options,
        robustness_test_seeds=robustness_test_seeds,
    )

    if CHECK and os.path.exists(out_path):
        return None
    
    # setup out path
    os.makedirs(os.path.join(out_root, data_name), exist_ok=True)
    
    # load data
    X, Y = load_dataset(
        dataset_name=data_name,
        dataset_source=dataset_source,
        dataset_npz=dataset_npz,
    )
    num_data = X.shape[0]
    num_dim = X.shape[1]
    seq_len = X.shape[2]
    num_class = len(np.unique(Y))
    
    # check model_name and settings
    data_config = get_data_config(model_name)
    interpolate = data_config['interpolation']
    use_intensity = data_config['use_intensity']
    use_raw_flow_input = _requires_flow_raw_input(model_name)
       
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
    batch_policy = resolve_dataset_fixed_batch_policy(
        train_n=len(train_idx),
        input_dim=num_dim,
        seq_len=seq_len,
    )
    batch_size = int(batch_policy["batch_size"])

    # set learning params
    lr_policy = resolve_learning_rate_policy(model_config.get("lr"))
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

    X_missing, X_mask, X_delta, coeffs = preprocess(
        X,
        missing_rate=missing_rate,
        interpolate=interpolate,
        use_intensity=use_intensity,
        channel_sync=use_raw_flow_input,
    )
    X_train = X_missing[train_idx]

    out = []
    for Xi, train_Xi in zip(X_missing.unbind(dim=-1), X_train.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    X_missing_norm = torch.stack(out, dim=-1)
    train_missing_summary = compute_train_missing_summary(X_missing, train_idx)
    target_masks = None
    if task == "classification":
        task_targets = _build_task_targets(
            task=task,
            x_missing_norm=X_missing_norm,
            y=Y,
            forecast_horizon=forecast_horizon,
        )
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
    valid_batch = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # set params
    model_kwargs = build_model_kwargs_from_config(model_name, model_config)
    
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    patient = 0

    for e in tqdm(range(EPOCHS)):
        train_loss = train(
            model_name, model, optimizer, criterion, train_batch, interpolate, use_intensity, device, task=task
        )
        valid_loss = evaluate(
            model_name, model, criterion, valid_batch, interpolate, use_intensity, device, task=task
        )

        if e % 10 == 0:
            LOGGER.info(
                "Epoch=%d train=%.6f valid=%.6f",
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

        if (e > 20) and (patient > 10):
            break

        scheduler.step()

    # using trained model
    try:
        model.load_state_dict(best_model_wts)
    except Exception as e:
        LOGGER.warning("Failed to load best weights for %s/%s: %s", data_name, model_name, e)

    # Robustness mode: evaluate on multiple test rates
    if robustness_mode:
        evaluations = _evaluate_robustness(
            model=model,
            data_name=data_name,
            model_name=model_name,
            SEED=SEED,
            test_missing_rates=robustness_test_rates,
            test_missing_mechanisms=robustness_test_mechanisms,
            channel_sync_options=robustness_channel_sync_options,
            test_missing_seeds=robustness_test_seeds,
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
        )

        payload = _build_result_payload(
            mode="robustness",
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
            train_missing_type="REGULAR",
            train_channel_sync=use_raw_flow_input,
            batch_policy=batch_policy,
            lr_policy=lr_policy,
            robustness_axes={
                "test_missing_rates": [float(rate) for rate in robustness_test_rates],
                "test_missing_types": list(robustness_test_mechanisms),
                "test_channel_sync_options": _resolve_channel_sync_options(
                    robustness_channel_sync_options,
                    use_raw_flow_input,
                ),
                "test_missing_seeds": [int(seed) for seed in robustness_test_seeds],
            },
        )

        # Save robustness results
        output_path = _build_output_path(
            data_name,
            0.0,
            model_name,
            SEED,
            out_root,
            robustness_mode=True,
            robustness_test_rates=robustness_test_rates,
            robustness_test_mechanisms=robustness_test_mechanisms,
            robustness_channel_sync_options=robustness_channel_sync_options,
            robustness_test_seeds=robustness_test_seeds,
        )
        with open(output_path, 'wb') as f:
            pickle.dump(payload, f)

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
            "dataset_npz": None if dataset_npz is None else str(dataset_npz),
            "task": task,
            "model": model_name,
            "seed": int(SEED),
            "missing_rate": 0.0,
            "robustness_test_rates": [float(rate) for rate in robustness_test_rates],
            "robustness_test_mechanisms": list(robustness_test_mechanisms),
            "robustness_channel_sync_options": list(robustness_channel_sync_options),
            "robustness_test_seeds": [int(seed) for seed in robustness_test_seeds],
            "robustness_evaluations": int(len(evaluations)),
            "split_path": str(split_path),
            "output_path": str(output_path),
            "batch_policy": batch_policy,
            "lr_policy": lr_policy,
            "metrics": metrics,
            "train_missing_summary": train_missing_summary,
            "target_spec": None if task == "classification" else target_spec,
            "mode": "robustness",
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
            y_true=y_true,
            y_pred=y_pred,
            logit_list=logit_list,
            save_predictions=save_predictions,
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
                if pred.dim() == 1:
                    pred = pred.unsqueeze(-1)
                if target.dim() == 1:
                    target = target.unsqueeze(-1)
                if pred.dim() == 3 and target.dim() == 3:
                    length = min(pred.shape[1], target.shape[1])
                    dim = min(pred.shape[2], target.shape[2])
                    pred = pred[:, :length, :dim]
                    target = target[:, :length, :dim]
                else:
                    dim = min(pred.shape[-1], target.shape[-1])
                    pred = pred[..., :dim]
                    target = target[..., :dim]
                pred_list.append(pred.detach().cpu().numpy())
                target_list.append(target.detach().cpu().numpy())

        task_pred = np.concatenate(pred_list, axis=0)
        task_target = np.concatenate(target_list, axis=0)
        eval_result = _build_evaluation_result(
            task=task,
            test_missing_rate=missing_rate,
            task_pred=task_pred,
            task_target=task_target,
            save_predictions=save_predictions,
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

    out_name = '_'.join([data_name, str(missing_rate), model_name, str(SEED)])
    output_path = os.path.join(out_root, data_name, str(missing_rate), out_name)
    payload = _build_result_payload(
        mode="standard",
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
        batch_policy=batch_policy,
        lr_policy=lr_policy,
    )
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)
    return {
        "dataset": data_name,
        "dataset_source": str(dataset_source),
        "dataset_npz": None if dataset_npz is None else str(dataset_npz),
        "task": task,
        "target_source": target_source if task != "classification" else "label",
        "model": model_name,
        "seed": int(SEED),
        "missing_rate": float(missing_rate),
        "split_path": str(split_path),
        "output_path": str(output_path),
        "batch_policy": batch_policy,
        "lr_policy": lr_policy,
        "metrics": metrics,
        "task_metrics": metrics if task != "classification" else None,
        "train_missing_summary": train_missing_summary,
        "target_spec": None if task == "classification" else target_spec,
        "mode": "standard",
    }


def _evaluate_robustness(
    model,
    data_name,
    model_name,
    SEED,
    test_missing_rates,
    test_missing_mechanisms,
    channel_sync_options,
    test_missing_seeds,
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
        test_missing_seeds: List of missingness RNG seeds
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
    test_missing_rates = _parse_missing_rates(test_missing_rates)
    test_missing_mechanisms = _parse_robustness_mechanisms(test_missing_mechanisms)
    test_missing_seeds = _parse_robustness_test_seeds(test_missing_seeds)
    resolved_channel_sync_options = _resolve_channel_sync_options(
        channel_sync_options,
        use_raw_flow_input,
    )
    train_missing_ref, _, _, _ = preprocess(
        X,
        missing_rate=0.0,
        interpolate=interpolate,
        use_intensity=use_intensity,
        channel_sync=use_raw_flow_input,
    )
    train_reference = train_missing_ref[train_idx]
    train_stats = []
    for train_Xi in train_reference.unbind(dim=-1):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()
        std = train_Xi_nonan.std()
        train_stats.append((mean, std))

    for test_rate in test_missing_rates:
        for test_mechanism in test_missing_mechanisms:
            for test_channel_sync in resolved_channel_sync_options:
                for test_missing_seed in test_missing_seeds:
                    LOGGER.info(
                        "Robustness eval dataset=%s model=%s test_missing=%.1f type=%s channel_sync=%s missing_seed=%d",
                        data_name,
                        model_name,
                        test_rate,
                        test_mechanism,
                        test_channel_sync,
                        test_missing_seed,
                    )

                    # Generate missing data for this test axis point.
                    X_missing, X_mask, X_delta, coeffs = preprocess(
                        X,
                        missing_rate=test_rate,
                        interpolate=interpolate,
                        use_intensity=use_intensity,
                        mechanism=test_mechanism,
                        seed=test_missing_seed,
                        channel_sync=test_channel_sync,
                    )

                    # Reuse the training missing=0.0 statistics for all robustness test axes.
                    out = []
                    for Xi, (mean, std) in zip(X_missing.unbind(dim=-1), train_stats):
                        out.append((Xi - mean) / (std + 1e-5))
                    X_missing_norm = torch.stack(out, dim=-1)

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
                    test_batch = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

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
                            test_missing_seed=test_missing_seed,
                            save_predictions=save_predictions,
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
                            test_missing_seed=test_missing_seed,
                            save_predictions=save_predictions,
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
                            test_missing_seed,
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
                            test_missing_seed,
                            task,
                            metrics["mse"],
                            metrics["mae"],
                            save_predictions,
                        )

    return evaluations


def main():
    args = parse_cli_args()
    if (
        args.task == "forecasting"
        and getattr(args, "target_source", "surrogate") == "surrogate"
        and getattr(args, "forecast_horizon", None) is None
    ):
        LOGGER.warning(
            "forecasting with --target-source surrogate and no --forecast-horizon: the surrogate "
            "target is the full sequence, so forecasting degenerates to imputation. Pass "
            "--forecast-horizon < seq_len for genuine forecasting, or use --target-source dataset."
        )
    scope_override = _resolve_model_scope_variant(args.model_scope_variant)
    apply_model_scope_variant(scope_override)
    seed = int(args.seed)
    seed_everything(seed)
    run_id = args.run_id or generate_run_id("run")
    records = []
    reused_records = []
    skipped = []
    policy_violations = []

    if args.datasets is None:
        data_selected = get_default_selected_datasets()
        np.random.shuffle(data_selected)
    else:
        data_selected = args.datasets

    if args.models is None:
        model_name_list = ["cnn"]
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

    # Robustness mode: train on missing=0, test on multiple rates
    if args.robustness_mode:
        LOGGER.info(
            "Robustness mode enabled: train on missing=0.0, test rates=%s, test types=%s, channel_sync=%s, missing_seeds=%s",
            args.robustness_test_rates,
            args.robustness_test_mechanisms,
            args.robustness_channel_sync_options,
            args.robustness_test_seeds,
        )
        for data_name in data_selected:
            for model_name in model_name_list:
                if max_runs is not None and _completed_runs() >= max_runs:
                    LOGGER.info("Reached max-runs=%d, stopping early.", max_runs)
                    stop_early = True
                    break
                try:
                    # Load params file based on param_mode
                    if args.param_mode == "cv":
                        param_name = '_'.join([data_name, model_name, 'cv'])
                    else:
                        param_name = '_'.join([data_name, model_name])

                    param_path = os.path.join(args.param_root, data_name, param_name)
                    if not os.path.exists(param_path):
                        raise FileNotFoundError(
                            f"Params file not found: {param_path}. "
                            f"Run scripts/param_search.py with --search-mode {args.param_mode} first."
                        )

                    with open(param_path, 'rb') as f:
                        model_config = pickle.load(f)
                    param_search_profile = _validate_param_search_profile(
                        model_config,
                        expected_search_profile=args.search_profile,
                        param_path=str(param_path),
                    )

                    run_result = run_model(
                        data_name,
                        missing_rate=0.0,  # Force train on missing=0.0
                        model_name=model_name,
                        model_config=model_config,
                        EPOCHS=args.epochs,
                        SEED=seed,
                        CHECK=args.check,
                        robustness_mode=True,
                        robustness_test_rates=args.robustness_test_rates,
                        robustness_test_mechanisms=args.robustness_test_mechanisms,
                        robustness_channel_sync_options=args.robustness_channel_sync_options,
                        robustness_test_seeds=args.robustness_test_seeds,
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
                    )
                    if run_result is None:
                        output_path = _build_output_path(
                            data_name,
                            0.0,
                            model_name,
                            seed,
                            args.out_root,
                            robustness_mode=True,
                            robustness_test_rates=args.robustness_test_rates,
                            robustness_test_mechanisms=args.robustness_test_mechanisms,
                            robustness_channel_sync_options=args.robustness_channel_sync_options,
                            robustness_test_seeds=args.robustness_test_seeds,
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
                                    "robustness_test_seeds": [int(seed) for seed in args.robustness_test_seeds],
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
                                    "dataset_npz": None if args.dataset_npz is None else str(args.dataset_npz),
                                    "model": model_name,
                                    "seed": seed,
                                    "missing_rate": 0.0,
                                    "robustness_test_rates": [float(rate) for rate in args.robustness_test_rates],
                                    "robustness_test_mechanisms": list(args.robustness_test_mechanisms),
                                    "robustness_channel_sync_options": list(args.robustness_channel_sync_options),
                                    "robustness_test_seeds": [int(seed) for seed in args.robustness_test_seeds],
                                    "task": args.task,
                                    "target_source": args.target_source,
                                    "search_profile": param_search_profile,
                                    "split_path": str(split_path),
                                    "param_path": str(param_path),
                                    "output_path": str(output_path),
                                    "reason": "output artifact already exists (--check)",
                                }
                            )
                        continue
                    run_result["param_path"] = str(param_path)
                    run_result["search_profile"] = param_search_profile
                    records.append(run_result)
                    run_count += 1

                except Exception as e:
                    LOGGER.warning("[SKIP] %s/%s: %s", data_name, model_name, e)
                    skipped.append(
                        {
                            "dataset": data_name,
                            "model": model_name,
                            "missing_rate": 0.0,
                            "task": args.task,
                            "reason": str(e),
                        }
                    )
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
                        # Load params file based on param_mode
                        if args.param_mode == "cv":
                            param_name = '_'.join([data_name, model_name, 'cv'])
                        else:
                            param_name = '_'.join([data_name, model_name])

                        param_path = os.path.join(args.param_root, data_name, param_name)
                        if not os.path.exists(param_path):
                            raise FileNotFoundError(
                                f"Params file not found: {param_path}. "
                                f"Run scripts/param_search.py with --search-mode {args.param_mode} first."
                            )

                        with open(param_path, 'rb') as f:
                            model_config = pickle.load(f)
                        param_search_profile = _validate_param_search_profile(
                            model_config,
                            expected_search_profile=args.search_profile,
                            param_path=str(param_path),
                        )

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
                        )
                        if run_result is None:
                            output_path = _build_output_path(
                                data_name,
                                float(missing_rate),
                                model_name,
                                seed,
                                args.out_root,
                                robustness_mode=False,
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
                                        "dataset_npz": None if args.dataset_npz is None else str(args.dataset_npz),
                                        "model": model_name,
                                        "seed": seed,
                                        "missing_rate": float(missing_rate),
                                        "task": args.task,
                                        "target_source": args.target_source,
                                        "search_profile": param_search_profile,
                                        "split_path": str(split_path),
                                        "param_path": str(param_path),
                                        "output_path": str(output_path),
                                        "reason": "output artifact already exists (--check)",
                                    }
                                )
                            continue
                        run_result["param_path"] = str(param_path)
                        run_result["search_profile"] = param_search_profile
                        records.append(run_result)
                        run_count += 1

                    except Exception as e:
                        LOGGER.warning("[SKIP] %s/%s: %s", data_name, model_name, e)
                        skipped.append(
                            {
                                "dataset": data_name,
                                "model": model_name,
                                "missing_rate": float(missing_rate),
                                "task": args.task,
                                "reason": str(e),
                            }
                        )
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
    elif max_runs is not None and completed_runs < max_runs:
        LOGGER.warning(
            "Completed runs=%d but requested max-runs=%d (mode=%s). "
            "Check prerequisites: split/*, params/*, out/* artifacts and existing-policy.",
            completed_runs,
            max_runs,
            args.max_runs_mode,
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
        payload={
            "status": "pass" if exit_code == 0 else "fail",
            "seed": seed,
            "task": args.task,
            "dataset_source": str(args.dataset_source),
            "dataset_npz": None if args.dataset_npz is None else str(args.dataset_npz),
            "target_source": args.target_source if args.task != "classification" else "label",
            "search_profile": args.search_profile,
            "split_root": str(args.split_root),
            "param_root": str(args.param_root),
            "out_root": str(args.out_root),
            "records": records,
            "reused_records": reused_records,
            "skipped": skipped,
            "existing_policy": args.existing_policy,
            "max_runs_mode": args.max_runs_mode,
            "summary": {
                "completed_run_calls": run_count,
                "executed_runs": run_count,
                "reused_runs": len(reused_records),
                "completed_runs": completed_runs,
                "requested_max_runs": max_runs,
                "stopped_early": bool(stop_early),
                "policy_violation_count": len(policy_violations),
            },
        },
    )
    LOGGER.info("Run manifest written: %s", manifest_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
