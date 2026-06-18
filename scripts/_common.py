#!/usr/bin/env python
"""Shared bootstrap helpers for packaged scripts."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

MODEL_SCOPE_VARIANT_ENV = "TORCH_ISTS_MODEL_SCOPE_VARIANT"
BATCH_SIZE_POLICY_NAME = "dataset-derived-fixed-v1"
LEARNING_RATE_POLICY_NAME = "fixed-lr-fallback-v1"
WEIGHT_DECAY_POLICY_NAME = "lr-proportional-weight-decay-v1"
__all__ = [
    "configure_runtime_paths",
    "apply_model_scope_variant",
    "default_method_for_model",
    "resolve_dataset_fixed_batch_policy",
    "resolve_learning_rate_policy",
]


def configure_runtime_paths(caller_file: str | Path) -> Path:
    """Ensure packaged runtime resolution for script execution."""
    caller_path = Path(caller_file).resolve()
    scripts_root = caller_path.parent
    repo_root = scripts_root.parent
    runtime_root = Path(
        os.environ.get("TORCH_ISTS_RUNTIME_ROOT", str(repo_root))
    ).resolve()

    for path in (runtime_root, repo_root, scripts_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    runtime_pkg = runtime_root / "torch_ists"
    init_py = runtime_pkg / "__init__.py"
    if not init_py.exists():
        raise FileNotFoundError(
            f"Runtime package not found: {init_py}. "
            "Set TORCH_ISTS_RUNTIME_ROOT to a packaged runtime root."
        )

    current = sys.modules.get("torch_ists")
    current_file = Path(getattr(current, "__file__", "")) if current else None
    if (
        current is None
        or not current_file
        or runtime_pkg.resolve() not in current_file.resolve().parents
    ):
        spec = importlib.util.spec_from_file_location(
            "torch_ists",
            str(init_py),
            submodule_search_locations=[str(runtime_pkg)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to build import spec for {init_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["torch_ists"] = module
        spec.loader.exec_module(module)

    return repo_root


def apply_model_scope_variant(variant: str | None) -> str:
    normalized = "" if variant is None else str(variant).strip().lower()
    if normalized in {"", "active"}:
        os.environ.pop(MODEL_SCOPE_VARIANT_ENV, None)
        return "active"
    raise ValueError("This package supports only the active model scope.")


def _floor_power_of_two_at_most(value: float) -> int:
    if value < 1:
        return 1
    return 2 ** int(math.floor(math.log2(value)))


def _avoid_singleton_last_batch(
    train_n: int,
    batch_size: int,
    *,
    target_batch_size: float,
    max_allowed_batch_size: int,
) -> tuple[int, bool]:
    if batch_size <= 1 or train_n <= batch_size or train_n % batch_size != 1:
        return batch_size, False

    upper = max(2, min(train_n, int(max_allowed_batch_size)))
    candidates = [candidate for candidate in range(2, upper + 1) if train_n % candidate != 1]
    if not candidates:
        return batch_size, False
    selected = min(candidates, key=lambda candidate: (abs(candidate - target_batch_size), abs(candidate - batch_size)))
    return int(selected), selected != batch_size


def resolve_dataset_fixed_batch_policy(
    *,
    train_n: int,
    input_dim: int,
    seq_len: int,
    target_steps: int = 8,
    min_batch_size: int = 4,
    max_batch_size: int = 128,
) -> dict:
    """Resolve the dataset-derived fixed batch policy used by benchmark scripts."""
    train_n = int(train_n)
    if train_n <= 0:
        raise ValueError(f"train_n must be positive, got {train_n}.")

    input_dim = max(1, int(input_dim))
    seq_len = max(1, int(seq_len))
    target_steps = max(1, int(target_steps))
    min_batch_size = max(1, int(min_batch_size))
    max_batch_size = max(min_batch_size, int(max_batch_size))

    batch_size = _floor_power_of_two_at_most(train_n / target_steps)
    batch_size = max(min_batch_size, min(max_batch_size, batch_size))

    sample_elements = input_dim * seq_len
    memory_cap = max_batch_size
    if sample_elements >= 250_000:
        memory_cap = 8
    elif sample_elements >= 100_000:
        memory_cap = 16
    elif sample_elements >= 50_000:
        memory_cap = 32
    batch_size = min(batch_size, memory_cap)

    target_batch_size = train_n / target_steps
    initial_batch_size = max(1, min(train_n, batch_size))
    batch_size, singleton_adjusted = _avoid_singleton_last_batch(
        train_n,
        initial_batch_size,
        target_batch_size=target_batch_size,
        max_allowed_batch_size=memory_cap,
    )
    remainder = train_n % batch_size
    last_batch_size = batch_size if remainder == 0 else remainder
    train_steps = int(math.ceil(train_n / batch_size))
    dropped = 0
    return {
        "policy": BATCH_SIZE_POLICY_NAME,
        "batch_size": int(batch_size),
        "initial_batch_size": int(initial_batch_size),
        "singleton_remainder_adjusted": bool(singleton_adjusted),
        "train_n": int(train_n),
        "target_steps": int(target_steps),
        "min_batch_size": int(min_batch_size),
        "max_batch_size": int(max_batch_size),
        "input_dim": int(input_dim),
        "seq_len": int(seq_len),
        "sample_elements": int(sample_elements),
        "memory_cap": int(memory_cap),
        "drop_last": False,
        "last_batch_size": int(last_batch_size),
        "singleton_last_batch": bool(last_batch_size == 1 and train_n > batch_size),
        "train_steps_per_epoch": int(train_steps),
        "dropped_train_samples_per_epoch": int(dropped),
        "dropped_train_fraction_per_epoch": float(dropped / train_n),
    }


def resolve_learning_rate_policy(
    config_lr,
    *,
    default_lr=1e-3,
    weight_decay_ratio=0.01,
) -> dict:
    """Resolve optimizer LR without coupling fallback behavior to batch size."""
    if config_lr is None:
        lr = float(default_lr)
        source = "fixed_fallback"
    else:
        lr = float(config_lr)
        source = "config"
    if lr <= 0:
        raise ValueError(f"learning rate must be positive, got {lr}.")
    weight_decay = float(lr * weight_decay_ratio)
    return {
        "policy": LEARNING_RATE_POLICY_NAME,
        "source": source,
        "lr": float(lr),
        "config_lr": None if config_lr is None else float(config_lr),
        "default_lr": float(default_lr),
        "batch_size_coupled": False,
        "weight_decay_policy": WEIGHT_DECAY_POLICY_NAME,
        "weight_decay_ratio": float(weight_decay_ratio),
        "weight_decay": float(weight_decay),
    }


def default_method_for_model(model_name: str) -> str:
    try:
        from torch_ists.config import default_method_for_model as runtime_default_method_for_model

        return runtime_default_method_for_model(model_name)
    except Exception:
        normalized = str(model_name).strip().lower()
        if normalized in {
            "naive_cde",
            "neural_ode",
            "neural_ode_t",
            "neural_sde",
            "neural_sde_t",
            "neural_lsde",
            "neural_lnsde",
            "neural_gsde",
        }:
            return "srk"
        if normalized == "ode-rnn-spline":
            return "rk4"
        if normalized == "ode-rnn":
            return "euler"
        if normalized == "latentode":
            return "dopri5"
        if normalized == "gru-ode-bayes":
            return "euler"
        if normalized == "ode-lstm":
            return "dopri5"
        if normalized in {"neuralcde", "neuralcde-l", "neuralcde-r", "neuralcde-c", "neuralcde-h"}:
            return "rk4"
        return "euler"
