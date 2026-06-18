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
LEGACY_ICLR2024_SEARCH_PROFILE = "legacy-iclr2024-v0"
LEGACY_EXPANDED_SEARCH_PROFILE = "legacy-expanded-v1"
DEFAULT_SEARCH_PROFILE = LEGACY_EXPANDED_SEARCH_PROFILE
MODEL_SPECIFIC_SEARCH_POLICY_NAME = "model-specific-effective-params-v1"
__all__ = [
    "configure_runtime_paths",
    "apply_model_scope_variant",
    "default_method_for_model",
    "resolve_dataset_fixed_batch_policy",
    "resolve_learning_rate_policy",
    "normalize_search_profile",
    "infer_param_search_profile",
    "resolve_model_search_space",
    "resolve_effective_param_config",
    "build_model_kwargs_from_config",
]

_SEARCH_PROFILE_ALIASES = {
    "default": LEGACY_EXPANDED_SEARCH_PROFILE,
    "legacy": LEGACY_EXPANDED_SEARCH_PROFILE,
    "expanded": LEGACY_EXPANDED_SEARCH_PROFILE,
    "legacy-expanded": LEGACY_EXPANDED_SEARCH_PROFILE,
    LEGACY_EXPANDED_SEARCH_PROFILE: LEGACY_EXPANDED_SEARCH_PROFILE,
    "iclr2024": LEGACY_ICLR2024_SEARCH_PROFILE,
    "legacy-iclr2024": LEGACY_ICLR2024_SEARCH_PROFILE,
    LEGACY_ICLR2024_SEARCH_PROFILE: LEGACY_ICLR2024_SEARCH_PROFILE,
}

_SEARCH_PROFILE_GRIDS = {
    LEGACY_ICLR2024_SEARCH_PROFILE: {
        "hidden_dim_values": [16, 32, 64, 128],
        "depth_level_values": [1, 2, 3, 4],
        "default_hidden_dim": 32,
        "default_depth_level": 1,
        "lr_bounds": [1e-4, 1e-2],
    },
    LEGACY_EXPANDED_SEARCH_PROFILE: {
        "hidden_dim_values": [32, 64, 96, 128, 192, 256],
        "depth_level_values": [1, 2, 3, 4, 5],
        "default_hidden_dim": 32,
        "default_depth_level": 1,
        "lr_bounds": [1e-4, 1e-2],
    },
}

_DEPTH_COLLAPSED_MODELS = frozenset(
    [
        "gru-dt",
        "gru-d",
        "gru-ode",
        "gru-ode-bayes",
        "mtan",
    ]
)
_SAND_MODELS = frozenset(["sand"])
_MIAM_MODELS = frozenset(["miam"])
_FLOW_PREFIXES = ("neuralflow-", "dualdynamics-", "flowpath-")


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


def normalize_search_profile(raw: str | None) -> str:
    normalized = DEFAULT_SEARCH_PROFILE if raw is None else str(raw).strip().lower()
    try:
        return _SEARCH_PROFILE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(_SEARCH_PROFILE_GRIDS))
        raise ValueError(f"Unknown search profile: {raw!r}. Allowed: {allowed}.") from exc


def infer_param_search_profile(model_config: dict | None) -> str:
    """Infer profile for saved params; missing profile means historical ICLR grid."""
    if not model_config:
        return LEGACY_ICLR2024_SEARCH_PROFILE
    raw = model_config.get("search_profile")
    if raw in {None, ""}:
        return LEGACY_ICLR2024_SEARCH_PROFILE
    return normalize_search_profile(raw)


def _resolve_config_search_profile(config: dict, search_profile: str | None) -> str:
    raw = config.get("search_profile")
    if raw not in {None, ""}:
        return normalize_search_profile(raw)
    if search_profile is not None:
        return normalize_search_profile(search_profile)
    return LEGACY_ICLR2024_SEARCH_PROFILE


def _canonical_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    try:
        from torch_ists.config.alias_policy import normalize_flow_alias

        normalized = normalize_flow_alias(normalized)
    except Exception:
        pass
    return normalized


def _is_flow_model_name(model_name: str) -> bool:
    normalized = _canonical_model_name(model_name)
    try:
        from torch_ists.config.alias_policy import is_flow_family_name

        return bool(is_flow_family_name(normalized))
    except Exception:
        return normalized.startswith(_FLOW_PREFIXES)


def _default_flow_depth_for_model(model_name: str) -> int:
    normalized = _canonical_model_name(model_name)
    if normalized.startswith("neuralflow-"):
        return 3
    if normalized.startswith("dualdynamics-"):
        return 2
    if normalized.startswith("flowpath-"):
        return 1
    return 2


def _resolve_layer_depth(model_name: str, depth_level: int) -> tuple[int, int]:
    try:
        from torch_ists.config import resolve_layer_depth

        num_layers, num_hidden_layers = resolve_layer_depth(model_name, int(depth_level))
    except Exception:
        num_layers, num_hidden_layers = int(depth_level), None
    if num_hidden_layers is None:
        num_hidden_layers = num_layers
    return int(num_layers), int(num_hidden_layers)


def resolve_model_search_space(model_name: str, search_profile: str | None = None) -> dict:
    """Return the meaningful sampled axes for a model under a search profile."""
    profile = normalize_search_profile(search_profile)
    grid = _SEARCH_PROFILE_GRIDS[profile]
    normalized_model = _canonical_model_name(model_name)
    depth_values = list(grid["depth_level_values"])
    if normalized_model in _DEPTH_COLLAPSED_MODELS:
        depth_values = [1]
    return {
        "policy": MODEL_SPECIFIC_SEARCH_POLICY_NAME,
        "search_profile": profile,
        "model_name": normalized_model,
        "hidden_dim_values": list(grid["hidden_dim_values"]),
        "depth_level_values": depth_values,
        "default_hidden_dim": int(grid["default_hidden_dim"]),
        "default_depth_level": int(depth_values[0] if depth_values else grid["default_depth_level"]),
        "lr_bounds": list(grid["lr_bounds"]),
        "sampled_axes": ["lr", "hidden_dim", "depth_level"],
        "depth_collapsed": bool(normalized_model in _DEPTH_COLLAPSED_MODELS),
    }


def _coerce_positive_int(value, *, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def resolve_effective_param_config(
    model_name: str,
    model_config: dict,
    *,
    search_profile: str | None = None,
) -> dict:
    """Resolve sampled search axes into the parameters consumed by model handlers."""
    config = dict(model_config)
    profile = _resolve_config_search_profile(config, search_profile)
    space = resolve_model_search_space(model_name, profile)
    normalized_model = _canonical_model_name(model_name)

    hidden_dim = _coerce_positive_int(
        config.get("hidden_dim"),
        default=space["default_hidden_dim"],
    )
    raw_depth = config.get(
        "depth_level",
        config.get("head_depth", config.get("num_layers", space["default_depth_level"])),
    )
    depth_level = _coerce_positive_int(raw_depth, default=space["default_depth_level"])
    if normalized_model in _DEPTH_COLLAPSED_MODELS:
        depth_level = 1

    model_params = dict(config.get("model_params") or {})
    if _is_flow_model_name(normalized_model):
        flow_depth = _default_flow_depth_for_model(normalized_model)
        num_layers = 1
        num_hidden_layers = depth_level
        head_depth = depth_level
    elif normalized_model in _SAND_MODELS:
        flow_depth = _default_flow_depth_for_model(normalized_model)
        num_layers = 1
        num_hidden_layers = 1
        head_depth = 1
        model_params["n_layers"] = int(depth_level)
        model_params["d_model"] = int(hidden_dim)
    elif normalized_model in _MIAM_MODELS:
        flow_depth = _default_flow_depth_for_model(normalized_model)
        num_layers = 1
        num_hidden_layers = 1
        head_depth = 1
        model_params["num_stack"] = int(depth_level)
    else:
        flow_depth = _default_flow_depth_for_model(normalized_model)
        num_layers, num_hidden_layers = _resolve_layer_depth(normalized_model, depth_level)
        head_depth = depth_level

    saved = dict(config)
    saved["search_profile"] = profile
    saved["search_policy"] = MODEL_SPECIFIC_SEARCH_POLICY_NAME
    saved["legacy_grid_version"] = profile
    saved["hidden_dim"] = int(hidden_dim)
    saved["depth_level"] = int(depth_level)
    saved["num_layers"] = int(num_layers)
    saved["num_hidden_layers"] = int(num_hidden_layers)
    saved["head_depth"] = int(head_depth)
    saved["flow_num_layers"] = int(flow_depth)
    saved["flow_depth"] = int(flow_depth)
    saved["model_params"] = model_params
    saved["architecture_axes"] = {
        "policy": MODEL_SPECIFIC_SEARCH_POLICY_NAME,
        "width_axis": "hidden_dim",
        "depth_axis": "depth_level",
        "hidden_dim_values": list(space["hidden_dim_values"]),
        "depth_level_values": list(space["depth_level_values"]),
        "depth_collapsed": bool(space["depth_collapsed"]),
    }
    saved["effective_params"] = {
        "hidden_dim": int(hidden_dim),
        "depth_level": int(depth_level),
        "num_layers": int(num_layers),
        "num_hidden_layers": int(num_hidden_layers),
        "head_depth": int(head_depth),
        "flow_num_layers": int(flow_depth),
        "flow_depth": int(flow_depth),
        "model_params": dict(model_params),
    }
    return saved


def build_model_kwargs_from_config(model_name: str, model_config: dict) -> dict:
    """Build kwargs accepted by ists_model/ists_classifier from a saved config."""
    effective = resolve_effective_param_config(
        model_name,
        model_config,
        search_profile=infer_param_search_profile(model_config),
    )
    hidden_dim = int(effective["hidden_dim"])
    return {
        "hidden_dim": hidden_dim,
        "hidden_hidden_dim": hidden_dim,
        "num_layers": int(effective["num_layers"]),
        "num_hidden_layers": int(effective["num_hidden_layers"]),
        "head_depth": int(effective["head_depth"]),
        "flow_num_layers": int(effective["flow_num_layers"]),
        "flow_depth": int(effective["flow_depth"]),
        "model_params": dict(effective.get("model_params") or {}),
    }


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
