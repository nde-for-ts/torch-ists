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

from _common import apply_model_scope_variant, configure_runtime_paths, default_method_for_model
from dataset_source_contracts import load_dataset, normalize_dataset_source
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
    normalized = str(raw).strip().lower()
    if normalized in {"default", "refactoring"}:
        return "default"
    raise argparse.ArgumentTypeError("dataset source must be 'default'.")


def _normalize_search_mode(raw: str) -> str:
    normalized = str(raw).strip().lower()
    if normalized in {"default", "legacy"}:
        return "default"
    if normalized in {"cv", "improved"}:
        return "cv"
    raise argparse.ArgumentTypeError("search mode must be one of {'default', 'cv'}.")


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


def _is_flow_model_name(model_name: str) -> bool:
    from torch_ists.config.alias_policy import is_flow_family_name

    return is_flow_family_name(str(model_name))


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run parameter search loop (default full loop, optional subset mode)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated dataset list.",
    )
    parser.add_argument(
        "--dataset-source",
        type=_normalize_dataset_source_arg,
        default="default",
        metavar="{default}",
        help="Dataset source backend for packaged built-in datasets.",
    )
    parser.add_argument(
        "--dataset-npz",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model list. Default uses get_active_model_list().",
    )
    parser.add_argument("--missing-rate", type=float, default=0.0, help="Missing rate used for tuning.")
    parser.add_argument("--num-samples", type=int, default=20, help="Ray Tune num_samples.")
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
        help="Behavior when param artifact already exists.",
    )
    parser.add_argument("--ray-cpu", type=float, default=16.0, help="CPU per Ray trial.")
    parser.add_argument("--ray-gpu", type=float, default=0.3, help="GPU per Ray trial.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection mode for each Ray trial.",
    )
    parser.add_argument(
        "--search-mode",
        type=_normalize_search_mode,
        default="default",
        metavar="{default,cv}",
        help="Hyperparameter search mode. default uses SEED=0 only; cv uses SEED=0~4 average.",
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
        help="Directory for run manifests.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier for related artifacts.",
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

# Import from the packaged torch_ists runtime.
from torch_ists.preprocessing import preprocess
from torch_ists import ists_classifier, ists_model
from _training_runtime import (
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
    resolve_layer_depth,
    get_default_selected_datasets,
)
from torch_ists.config.model_data_policy import _resolve_runtime_depth_config

ray = None
tune = None
ASHAScheduler = None


def _load_ray_tune():
    global ray, tune, ASHAScheduler
    if ray is not None and tune is not None and ASHAScheduler is not None:
        return ray, tune, ASHAScheduler
    try:
        import ray as _ray
        from ray import tune as _tune
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
    return ray, tune, ASHAScheduler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("param_search")

os.makedirs('params', exist_ok=True)


def _build_depth_model_kwargs(model_name: str, model_config: dict) -> dict:
    depth_cfg = _resolve_runtime_depth_config(
        model_name,
        num_layers=model_config.get("num_layers", 1),
        num_hidden_layers=model_config.get("num_hidden_layers"),
        head_depth=model_config.get("head_depth"),
        flow_num_layers=model_config.get("flow_num_layers"),
        flow_depth=model_config.get("flow_depth"),
    )
    num_layers, num_hidden_layers = resolve_layer_depth(model_name, int(depth_cfg["head_depth"]))
    if num_hidden_layers is None:
        num_hidden_layers = num_layers
    if depth_cfg["is_flow_model"]:
        num_layers = 1
        num_hidden_layers = int(depth_cfg["head_depth"])

    return {
        "num_layers": int(num_layers),
        "num_hidden_layers": int(num_hidden_layers),
        "head_depth": int(depth_cfg["head_depth"]),
        "flow_num_layers": int(depth_cfg["flow_num_layers"]),
        "flow_depth": int(depth_cfg["flow_depth"]),
    }


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
    X, Y = load_dataset(
        dataset_name=data_name,
        dataset_source=dataset_source,
        dataset_npz=dataset_npz,
    )
    num_data = int(X.shape[0])
    num_dim = int(X.shape[1])
    seq_len = int(X.shape[2])
    split_path = os.path.join(split_root, data_name, f"{data_name}_{seed}")
    with open(split_path, "rb") as f:
        train_idx, valid_idx, test_idx = pickle.load(f)
    del valid_idx, test_idx

    data_config = get_data_config(model_name)
    X_missing, _, _, _ = preprocess(
        X,
        missing_rate=missing_rate,
        interpolate=data_config["interpolation"],
        use_intensity=data_config["use_intensity"],
        channel_sync=_requires_flow_raw_input(model_name),
    )
    X_train = X_missing[train_idx]
    out = []
    for Xi, train_Xi in zip(X_missing.unbind(dim=-1), X_train.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    X_missing_norm = torch.stack(out, dim=-1)
    train_missing_summary = compute_train_missing_summary(X_missing, train_idx)

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
        "train_missing_summary": train_missing_summary,
        "target_spec": target_spec,
        "target_source": str(target_source if task != "classification" else "label"),
    }


def _normalize_param_config_for_save(model_name: str, model_config: dict) -> dict:
    saved = dict(model_config)
    if not _is_flow_model_name(model_name):
        return saved
    depth_kwargs = _build_depth_model_kwargs(model_name, model_config)
    saved["num_layers"] = int(depth_kwargs["head_depth"])
    saved["head_depth"] = int(depth_kwargs["head_depth"])
    saved["flow_num_layers"] = int(depth_kwargs["flow_depth"])
    saved["flow_depth"] = int(depth_kwargs["flow_depth"])
    return saved


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


def report_tune_loss(loss_value):
    _ray, _tune, _ = _load_ray_tune()
    metric = float(loss_value)
    try:
        _tune.report(loss=metric)
        return
    except TypeError:
        pass
    try:
        _tune.report({"loss": metric})
        return
    except TypeError:
        pass
    from ray import train as ray_train

    ray_train.report({"loss": metric})


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

# Helper function for single-seed training (used by both default and cv modes)
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
):
    """
    Train and evaluate model on a single seed split.
    Returns the best validation loss for this seed.
    """
    # Load data
    X, Y = load_dataset(
        dataset_name=data_name,
        dataset_source=dataset_source,
        dataset_npz=dataset_npz,
    )
    num_data = X.shape[0]
    num_dim = X.shape[1]
    seq_len = X.shape[2]
    num_class = len(np.unique(Y))

    # Set batch_size
    for i in range(4, 8):
        batch_size = 2**i
        if batch_size > num_data / 16:
            break

    # Set learning rate
    if model_config['lr'] is None:
        lr = 1e-3 * (batch_size / 2**4)
    else:
        lr = model_config['lr']

    # Load split file
    split_path = os.path.join(split_root, data_name, f'{data_name}_{seed}')
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split file not found: {split_path}. "
            f"Run set_splits.py first to generate split files for seed={seed}."
        )
    with open(split_path, 'rb') as f:
        train_idx, valid_idx, test_idx = pickle.load(f)

    # Preprocess data
    X_missing, X_mask, X_delta, coeffs = preprocess(
        X,
        missing_rate=missing_rate,
        interpolate=interpolate,
        use_intensity=use_intensity,
        channel_sync=_requires_flow_raw_input(model_name),
    )
    X_train = X_missing[train_idx]

    # Normalize using train set statistics
    out = []
    for Xi, train_Xi in zip(X_missing.unbind(dim=-1), X_train.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    X_missing_norm = torch.stack(out, dim=-1)
    train_missing_summary = compute_train_missing_summary(X_missing, train_idx)

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
    train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_batch = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Set model params
    depth_kwargs = _build_depth_model_kwargs(model_name, model_config)
    model_kwargs = {
        'hidden_dim': model_config['hidden_dim'],
        'hidden_hidden_dim': model_config['hidden_dim'],
        'num_layers': depth_kwargs['num_layers'],
        'num_hidden_layers': depth_kwargs['num_hidden_layers'],
        'head_depth': depth_kwargs['head_depth'],
        'flow_num_layers': depth_kwargs['flow_num_layers'],
        'flow_depth': depth_kwargs['flow_depth'],
    }

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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lr * 0.01)
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
    search_mode="default",
    split_root="split",
    task="classification",
    forecast_horizon=None,
    target_source="surrogate",
    targets_root="targets",
    dataset_source="default",
    dataset_npz=None,
):
    trial_device = resolve_trial_device(device_mode, ray_gpu)
    LOGGER.info("Tune start dataset=%s model=%s mode=%s", data_name, model_name, search_mode)
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

    if search_mode == "default":
        # Default mode: single-seed search on the requested seed
        seed_everything(SEED)
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
        )
        LOGGER.info("Default mode finished dataset=%s model=%s best_loss=%.6f", data_name, model_name, best_loss)
        report_tune_loss(best_loss)

    elif search_mode == "cv":
        # CV mode: average SEED=0~4 results at higher compute cost
        cv_losses = []
        for seed in range(5):
            seed_everything(seed)
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
            cv_losses.append(loss)
            LOGGER.info("CV mode seed=%d dataset=%s model=%s loss=%.6f", seed, data_name, model_name, loss)

        # Report average loss to Ray Tune
        avg_loss = np.mean(cv_losses)
        std_loss = np.std(cv_losses)
        LOGGER.info(
            "CV mode finished dataset=%s model=%s avg_loss=%.6f±%.6f (seeds=%s)",
            data_name,
            model_name,
            avg_loss,
            std_loss,
            cv_losses,
        )
        report_tune_loss(avg_loss)

    else:
        raise ValueError(f"Unknown search_mode: {search_mode}. Expected 'default' or 'cv'.")

    LOGGER.info("Finished training dataset=%s model=%s mode=%s", data_name, model_name, search_mode)


def optimize(
    data_name,
    missing_rate,
    model_name,
    split_path,
    split_root="split",
    param_root="params",
    num_samples=20,
    max_num_epochs=10,
    ray_cpu=16.0,
    ray_gpu=0.3,
    device_mode="auto",
    search_mode="default",
    task="classification",
    forecast_horizon=None,
    target_source="surrogate",
    targets_root="targets",
    dataset_source="default",
    dataset_npz=None,
):
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
            "lr": 1e-3,
            "hidden_dim": 32,
            "num_layers": 1,
        }
        split_root_local = split_root

        if search_mode == "default":
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
        elif search_mode == "cv":
            losses = []
            for seed in range(5):
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
            raise ValueError(f"Unknown search_mode: {search_mode}")

        saved_model_config = _normalize_param_config_for_save(model_name, model_config)
        os.makedirs(os.path.join(param_root, data_name), exist_ok=True)
        if search_mode == "cv":
            out_name = "_".join([data_name, model_name, "cv"])
        else:
            out_name = "_".join([data_name, model_name])
        param_path = os.path.join(param_root, data_name, out_name)
        with open(param_path, "wb") as f:
            pickle.dump(saved_model_config, f)

        metadata = _collect_task_metadata(
            data_name=data_name,
            missing_rate=missing_rate,
            model_name=model_name,
            split_root=split_root,
            seed=SEED,
            task=task,
            forecast_horizon=forecast_horizon,
            target_source=target_source,
            targets_root=targets_root,
            dataset_source=dataset_source,
            dataset_npz=dataset_npz,
        )

        return {
            "best_trial": None,
            "best_loss": float(best_loss),
            "param_config": saved_model_config,
            "param_hash": stable_hash(saved_model_config),
            "param_path": str(param_path),
            "split_path": str(split_path),
            "train_missing_summary": metadata["train_missing_summary"],
            "target_spec": metadata["target_spec"],
            "target_source": metadata["target_source"],
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
        # "batch_size": None, # determine by the number of the data
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden_dim": tune.choice([16, 32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3, 4]),
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    def optimize_model(model_config):
        tune_model(data_name=data_name, missing_rate=missing_rate, model_name=model_name,
                   model_config=model_config, EPOCHS=max_num_epochs, SEED=SEED,
                   device_mode=device_mode, ray_gpu=ray_gpu, search_mode=search_mode,
                   split_root=split_root, task=task, forecast_horizon=forecast_horizon,
                   target_source=target_source, targets_root=targets_root,
                   dataset_source=dataset_source, dataset_npz=dataset_npz)

    result = tune.run(
        tune.with_parameters(optimize_model),
        resources_per_trial={"cpu": ray_cpu, "gpu": ray_gpu},
        config=model_config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        stop={"training_iteration": max_num_epochs, "loss": np.nan},
        raise_on_failed_trial=False,
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    if best_trial is None:
        LOGGER.error(
            "No successful trial with finite loss dataset=%s model=%s. "
            "Check --device/--ray-gpu and runtime resources.",
            data_name,
            model_name,
        )
        return None
    if best_trial.last_result is None or "loss" not in best_trial.last_result:
        LOGGER.error(
            "Best trial exists but no loss is reported dataset=%s model=%s.",
            data_name,
            model_name,
        )
        return None
    LOGGER.info("Best trial config: %s", best_trial.config)
    LOGGER.info("Best trial final validation loss: %s", best_trial.last_result["loss"])
    
    # save params
    os.makedirs(os.path.join(param_root, data_name), exist_ok=True)

    # Distinguish default vs cv params by suffix
    if search_mode == "cv":
        out_name = '_'.join([data_name, model_name, 'cv'])
    else:
        out_name = '_'.join([data_name, model_name])

    param_path = os.path.join(param_root, data_name, out_name)
    saved_param_config = _normalize_param_config_for_save(model_name, best_trial.config)
    with open(param_path, 'wb') as f:
        pickle.dump(saved_param_config, f)

    LOGGER.info("Params saved to %s (mode=%s)", param_path, search_mode)

    metadata = _collect_task_metadata(
        data_name=data_name,
        missing_rate=missing_rate,
        model_name=model_name,
        split_root=split_root,
        seed=SEED,
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
        "param_hash": stable_hash(saved_param_config),
        "param_path": str(param_path),
        "split_path": str(split_path),
        "train_missing_summary": metadata["train_missing_summary"],
        "target_spec": metadata["target_spec"],
        "target_source": metadata["target_source"],
        "dataset_source": str(dataset_source),
        "dataset_npz": None if dataset_npz is None else str(dataset_npz),
    }


def main():
    global SEED
    args = parse_cli_args()
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
        data_selected = get_default_selected_datasets()
        np.random.shuffle(data_selected)
    else:
        data_selected = args.datasets

    if args.models is None:
        model_name_list = get_active_model_list()
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

            # Check if params file already exists (skip logic)
            if args.search_mode == "cv":
                out_name = '_'.join([data_name, model_name, 'cv'])
            else:
                out_name = '_'.join([data_name, model_name])

            split_path = os.path.join(args.split_root, data_name, f"{data_name}_{SEED}")
            if not os.path.exists(split_path):
                reason = f"split artifact missing: {split_path}"
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
                    LOGGER.info("Params file already exists: %s/%s (reusing)", data_name, out_name)
                    param_hash = ""
                    try:
                        with open(existing_param_path, "rb") as f:
                            existing_cfg = pickle.load(f)
                        param_hash = stable_hash(existing_cfg)
                    except Exception as exc:
                        LOGGER.warning(
                            "Failed to load existing params for hash %s/%s: %s",
                            data_name,
                            out_name,
                            exc,
                        )
                    reused_records.append(
                        {
                            "dataset": data_name,
                            "dataset_source": str(args.dataset_source),
                            "dataset_npz": None if args.dataset_npz is None else str(args.dataset_npz),
                            "model": model_name,
                            "seed": SEED,
                            "missing_rate": float(args.missing_rate),
                            "task": args.task,
                            "target_source": args.target_source,
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
                    search_mode=args.search_mode,
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
                        "missing_rate": float(args.missing_rate),
                        "task": args.task,
                        "target_source": str(search_result["target_source"]),
                        "train_missing_summary": search_result["train_missing_summary"],
                        "target_spec": search_result["target_spec"],
                        "split_path": str(search_result["split_path"]),
                        "param_path": str(search_result["param_path"]),
                        "param_hash": str(search_result["param_hash"]),
                        "best_loss": float(search_result["best_loss"]),
                    }
                )
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
            "existing-policy=fail violations=%d. First violation: %s",
            len(policy_violations),
            policy_violations[0],
        )
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
            "split_root": str(args.split_root),
            "param_root": str(args.param_root),
            "records": records,
            "reused_records": reused_records,
            "skipped": skipped,
            "existing_policy": args.existing_policy,
            "max_runs_mode": args.max_runs_mode,
            "summary": {
                "completed_optimize_calls": run_count,
                "executed_runs": run_count,
                "reused_runs": len(reused_records),
                "completed_runs": completed_runs,
                "requested_max_runs": max_runs,
                "stopped_early": bool(stop_early),
                "policy_violation_count": len(policy_violations),
            },
        },
    )
    LOGGER.info("Search manifest written: %s", manifest_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
