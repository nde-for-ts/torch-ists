#!/usr/bin/env python
"""Shared runtime/bootstrap helpers for script entrypoints."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import sys
from numbers import Integral
from pathlib import Path

import numpy as np

BATCH_SIZE_POLICY_NAME = "dataset-derived-fixed"
LEARNING_RATE_POLICY_NAME = "fixed-lr-fallback"
WEIGHT_DECAY_POLICY_NAME = "lr-proportional-weight-decay"
REPRODUCTION_SEARCH_PROFILE = "reproduction"
EXPANDED_SEARCH_PROFILE = "expanded"
MODEL_NATIVE_SEARCH_PROFILE = "model-native"
DEFAULT_SEARCH_PROFILE = MODEL_NATIVE_SEARCH_PROFILE
MODEL_SPECIFIC_SEARCH_POLICY_NAME = "model-native-effective-params"
STANDARD_SEARCH_POLICY = "default"
SINGLE_SEARCH_REPEAT = "single"
DEFAULT_MODEL_NATIVE_HIDDEN_DIM_VALUES = [32, 64, 96, 128, 160, 192, 224, 256]
CONSTRAINED_MATCHED_CAPACITY_HIDDEN_DIM_VALUES = list(range(16, 257, 16))
CONSTRAINED_MATCHED_CAPACITY_DEPTH_LEVEL_VALUES = [1, 2, 3, 4, 5]
CONSTRAINED_MATCHED_CAPACITY_TARGET = 200_000
CONSTRAINED_MATCHED_CAPACITY_TARGET_NAME = "constrained"
CONSTRAINED_MATCHED_CAPACITY_LABEL = "capacity-controlled"
CONSTRAINED_CAPACITY_POLICY_NAME = "capacity-controlled"
CONSTRAINED_SEARCH_CONTRACT_NAME = "capacity-controlled-search"
CONSTRAINED_SEARCH_FAMILY = "capacity-controlled"
CONSTRAINED_PUBLIC_SEARCH_NAME = "experimental capacity-controlled search"
DEFAULT_SEARCH_CONTRACT_NAME = "model-specific-default"
HISTORICAL_SEARCH_CONTRACT_NAME = "historical-reproducibility"
PUBLIC_SEARCH_PROFILE_VISIBILITY = "public"
COMPATIBILITY_SEARCH_PROFILE_VISIBILITY = "compatibility"
INTERNAL_SEARCH_ARTIFACT_KEYS = frozenset(
    {
        "search_profile",
        "search_contract",
        "search_family",
        "public_search_name",
        "search_profile_visibility",
        "search_algorithm_policy",
        "search_grid_version",
        "architecture_axes",
    }
)

_SEARCH_POLICY_ALIASES = {
    "default": MODEL_NATIVE_SEARCH_PROFILE,
    "model-native": MODEL_NATIVE_SEARCH_PROFILE,
    "model-specific": MODEL_NATIVE_SEARCH_PROFILE,
    MODEL_NATIVE_SEARCH_PROFILE: MODEL_NATIVE_SEARCH_PROFILE,
    "expanded": EXPANDED_SEARCH_PROFILE,
    EXPANDED_SEARCH_PROFILE: EXPANDED_SEARCH_PROFILE,
    REPRODUCTION_SEARCH_PROFILE: REPRODUCTION_SEARCH_PROFILE,
}

_SEARCH_POLICY_GRIDS = {
    MODEL_NATIVE_SEARCH_PROFILE: {
        "hidden_dim_values": list(DEFAULT_MODEL_NATIVE_HIDDEN_DIM_VALUES),
        "depth_level_values": [1, 2, 3, 4, 5],
        "flow_depth_values": [1, 2, 3],
        "default_hidden_dim": 32,
        "default_depth_level": 1,
        "lr_bounds": [1e-4, 1e-2],
    },
    REPRODUCTION_SEARCH_PROFILE: {
        "hidden_dim_values": [16, 32, 64, 128],
        "depth_level_values": [1, 2, 3, 4],
        "default_hidden_dim": 32,
        "default_depth_level": 1,
        "lr_bounds": [1e-4, 1e-2],
    },
    EXPANDED_SEARCH_PROFILE: {
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
_BASIC_SEQUENCE_MODELS = frozenset(
    [
        "cnn",
        "rnn",
        "lstm",
        "gru",
        "gru-simple",
        "bilstm",
        "plstm",
        "tlstm",
        "tglstm",
    ]
)
_SAND_MODELS = frozenset(["sand"])
_MIAM_MODELS = frozenset(["miam"])
_MTAN_MODELS = frozenset(["mtan"])
_TRANSFORMER_MODELS = frozenset(["transformer", "transformer-mask"])
_NEURALRDE_DEPTH = {
    "neuralrde-1": 1,
    "neuralrde-2": 2,
    "neuralrde-3": 3,
}
_CDE_CONTROL_BY_MODEL = {
    "neuralcde": "natural",
    "neuralcde-l": "linear",
    "neuralcde-r": "rectilinear",
    "neuralcde-c": "cubic",
    "neuralcde-h": "hermite",
    "ancde": "natural",
    "exit": "natural",
    "leap": "natural",
}
_SDE_MODELS = frozenset(
    [
        "naive_cde",
        "neural_ode",
        "neural_ode_t",
        "neural_sde",
        "neural_sde_t",
        "neural_lsde",
        "neural_lnsde",
        "neural_gsde",
    ]
)
_ODE_SOLVER_MODELS = frozenset(
    [
        "ode-rnn-spline",
        "ode-lstm",
        "ode-rnn",
        "latentode",
        "gru-ode-bayes",
    ]
)
_CDE_SOLVER_MODELS = frozenset(_CDE_CONTROL_BY_MODEL) | frozenset(_NEURALRDE_DEPTH)
_FLOW_PREFIXES = ("neuralflow-", "dualdynamics-", "flowpath-", "fusioncde-")
DEFAULT_NEURALRDE_LOGSIGNATURE_WINDOW_LENGTH = 10

_MODEL_NATIVE_EXTRA_PARAM_CANDIDATES = {
    "transformer": {"n_heads": [1, 2, 4, 8]},
    "transformer-mask": {"n_heads": [1, 2, 4, 8]},
    "sand": {"n_heads": [1, 2, 4], "factor": [8, 16]},
    "miam": {"n_heads": [1, 2, 4], "n_iter": [1, 2]},
    "mtan": {"n_heads": [1, 2, 4], "embed_time": [16, 32]},
}

_MODEL_NATIVE_EXTRA_PARAM_DEFAULTS = {
    "transformer": {"n_heads": 4},
    "transformer-mask": {"n_heads": 4},
    "sand": {"n_heads": 4, "factor": 16},
    "miam": {"n_heads": 1, "n_iter": 1},
    "mtan": {"n_heads": 1, "embed_time": 16},
    "neuralrde-1": {"logsignature_window_length": DEFAULT_NEURALRDE_LOGSIGNATURE_WINDOW_LENGTH},
    "neuralrde-2": {"logsignature_window_length": DEFAULT_NEURALRDE_LOGSIGNATURE_WINDOW_LENGTH},
    "neuralrde-3": {"logsignature_window_length": DEFAULT_NEURALRDE_LOGSIGNATURE_WINDOW_LENGTH},
}


def _read_json_file(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_public_mirror_head(root: Path) -> str | None:
    git_dir = root / ".git"
    if not git_dir.exists():
        return None
    try:
        if git_dir.is_file():
            raw = git_dir.read_text(encoding="utf-8").strip()
            if not raw.startswith("gitdir:"):
                return None
            git_dir = (root / raw.split(":", 1)[1].strip()).resolve()
        head_path = git_dir / "HEAD"
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref_path = git_dir / head.split(":", 1)[1].strip()
            if ref_path.exists():
                value = ref_path.read_text(encoding="utf-8").strip()
                return value or None
            packed_refs = git_dir / "packed-refs"
            if packed_refs.exists():
                ref_name = head.split(":", 1)[1].strip()
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref_name:
                        return parts[0]
            return None
        return head or None
    except Exception:
        return None


def normalize_runtime_baseline_id(runtime_baseline_id: str | None = None) -> str | None:
    """Return a caller-provided runtime label without adding generated metadata."""
    if runtime_baseline_id is None:
        return None
    value = str(runtime_baseline_id).strip()
    return value or None


def omit_none_runtime_baseline_id(payload):
    """Remove unset runtime labels before writing artifacts or manifests."""
    if isinstance(payload, dict):
        cleaned = {}
        for key, value in payload.items():
            if key == "runtime_baseline_id" and value is None:
                continue
            cleaned[key] = omit_none_runtime_baseline_id(value)
        return cleaned
    if isinstance(payload, list):
        return [omit_none_runtime_baseline_id(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(omit_none_runtime_baseline_id(value) for value in payload)
    return payload


def _path_is_under(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _module_paths(module: object) -> list[Path]:
    paths: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        paths.append(Path(module_file))

    module_path = getattr(module, "__path__", None)
    if module_path:
        paths.extend(Path(path) for path in module_path)

    spec = getattr(module, "__spec__", None)
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    if locations:
        paths.extend(Path(path) for path in locations)

    return paths


def _module_is_from_runtime(module: object, runtime_roots: list[Path]) -> bool:
    return any(_path_is_under(path, runtime_roots) for path in _module_paths(module))


def _clear_runtime_modules() -> None:
    for name in list(sys.modules):
        if name == "torch_ists" or name.startswith("torch_ists."):
            sys.modules.pop(name, None)


def _clear_incompatible_runtime_submodules(runtime_roots: list[Path]) -> None:
    for name, module in list(sys.modules.items()):
        if name == "torch_ists" or not name.startswith("torch_ists."):
            continue
        if not _module_is_from_runtime(module, runtime_roots):
            sys.modules.pop(name, None)


def configure_runtime_paths(caller_file: str | Path) -> Path:
    """Resolve this source checkout and an optional explicit runtime override."""
    caller_path = Path(caller_file).resolve()
    public_checkout_root = caller_path.parents[1]
    env_runtime_root = os.environ.get("TORCH_ISTS_RUNTIME_ROOT")
    runtime_root = (
        Path(env_runtime_root).resolve()
        if env_runtime_root
        else public_checkout_root.resolve()
    )
    repo_root = public_checkout_root.resolve()
    scripts_root = public_checkout_root / "scripts"

    for path in (runtime_root, repo_root, scripts_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    runtime_pkg = runtime_root / "torch_ists"
    init_py = runtime_pkg / "__init__.py"
    if not init_py.exists():
        raise FileNotFoundError(
            f"Runtime package not found: {init_py}. "
            "Set TORCH_ISTS_RUNTIME_ROOT to a valid package root."
        )

    current = sys.modules.get("torch_ists")
    current_file = Path(getattr(current, "__file__", "")) if current else None
    runtime_roots = [runtime_pkg.resolve()]
    current_matches_runtime = (
        current is not None
        and current_file
        and _path_is_under(current_file, runtime_roots)
    )
    if not current_matches_runtime:
        _clear_runtime_modules()
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
    else:
        _clear_incompatible_runtime_submodules(runtime_roots)

    return repo_root


def normalize_search_profile(raw: str | None) -> str:
    normalized = DEFAULT_SEARCH_PROFILE if raw is None else str(raw).strip().lower()
    try:
        return _SEARCH_POLICY_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError("Unsupported search settings.") from exc


def describe_search_profile(raw: str | None) -> dict:
    """Return compatibility metadata for a search configuration."""
    profile = normalize_search_profile(raw)
    if profile == REPRODUCTION_SEARCH_PROFILE:
        return {
            "search_contract": HISTORICAL_SEARCH_CONTRACT_NAME,
            "search_family": "historical_reproducibility",
            "search_profile_visibility": COMPATIBILITY_SEARCH_PROFILE_VISIBILITY,
            "public_search_name": "historical reproducibility",
        }
    if profile == EXPANDED_SEARCH_PROFILE:
        return {
            "search_contract": DEFAULT_SEARCH_CONTRACT_NAME,
            "search_family": "historical_reproducibility",
            "search_profile_visibility": COMPATIBILITY_SEARCH_PROFILE_VISIBILITY,
            "public_search_name": "expanded historical search",
        }
    return {
        "search_contract": DEFAULT_SEARCH_CONTRACT_NAME,
        "search_family": "default",
        "search_profile_visibility": PUBLIC_SEARCH_PROFILE_VISIBILITY,
        "public_search_name": "default model-specific search",
    }


def infer_param_search_profile(model_config: dict | None) -> str:
    """Infer the search settings needed to read a saved parameter artifact."""
    if not model_config:
        raise ValueError(
            "Param artifact is empty or missing; cannot infer search_profile. "
            "Regenerate params with scripts/param_search.py."
        )
    raw = model_config.get("search_profile")
    if raw in {None, ""}:
        if _is_semantic_param_artifact(model_config):
            return DEFAULT_SEARCH_PROFILE
        raise ValueError(
            "Param artifact is missing required field 'search_profile'. "
            "Regenerate params with the current default search, or pass the hidden "
            "compatibility flag only for intentional older-artifact audits."
        )
    return normalize_search_profile(raw)


def _is_semantic_param_artifact(model_config: dict | None) -> bool:
    if not isinstance(model_config, dict):
        return False
    return (
        isinstance(model_config.get("artifact_identity"), dict)
        and isinstance(model_config.get("effective_params"), dict)
        and not _metadata_value_missing(model_config.get("search_policy"))
        and not _metadata_value_missing(model_config.get("search_repeat"))
    )


def strip_internal_search_artifact_metadata(model_config: dict) -> dict:
    """Return the persisted parameter view without compatibility-only labels."""
    return {
        key: value
        for key, value in dict(model_config).items()
        if key not in INTERNAL_SEARCH_ARTIFACT_KEYS
    }


def _metadata_value_missing(value) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _normalize_split_seeds(value) -> list[int]:
    if value is None or value == "":
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("split_seeds must be an ordered list")
    normalized = []
    seen_seeds = set()
    for item in value:
        seed = int(item)
        if seed in seen_seeds:
            raise ValueError("split_seeds contains a duplicate seed")
        seen_seeds.add(seed)
        normalized.append(seed)
    return normalized


def _stored_split_seeds(identity: dict) -> list[int]:
    stored_split_hashes = identity.get("split_sha256s")
    if not isinstance(stored_split_hashes, (list, tuple)):
        return []
    seeds = []
    for item in stored_split_hashes:
        if not isinstance(item, dict) or "seed" not in item:
            return []
        seeds.append(int(item["seed"]))
    return _normalize_split_seeds(seeds)


_TASK_DEFINITIONS = {
    "forecasting": "future-time-conditioned",
    "imputation": "held-out-observed-cell",
    "outlier_detection": "positive-label-is-outlier",
    "change_detection": "positive-label-is-changed-state",
}


def task_definition_for_task(task: str) -> str | None:
    return _TASK_DEFINITIONS.get(str(task).strip().lower())


def validate_task_definition(
    model_config: dict,
    *,
    task: str,
    context: str = "param artifact",
) -> str | None:
    expected = task_definition_for_task(task)
    if expected is None:
        return None
    identity = model_config.get("artifact_identity") if isinstance(model_config, dict) else None
    actual = (
        str(identity.get("task_definition") or "").strip().lower()
        if isinstance(identity, dict)
        else ""
    )
    if not actual:
        raise ValueError(
            f"{context} predates the current {str(task).strip().lower()} task definition. "
            "Run a fresh parameter search for this task with the current scripts."
        )
    if actual != expected:
        raise ValueError(
            f"{context} task_definition mismatch: "
            f"found={actual!r}, expected={expected!r}."
        )
    return expected


def build_param_artifact_identity(
    *,
    data_name: str,
    model_name: str,
    task: str,
    target_source: str | None = None,
    forecast_horizon: int | None = None,
    dataset_source: str | None = None,
    split_seeds=None,
    task_semantics: dict | None = None,
) -> dict:
    task_name = str(task).strip().lower()
    if task_name == "classification":
        normalized_target_source = "label"
        normalized_forecast_horizon = None
    else:
        normalized_target_source = str(target_source or "").strip().lower()
        normalized_forecast_horizon = (
            int(forecast_horizon)
            if task_name == "forecasting" and forecast_horizon is not None
            else None
        )
    identity = {
        "data_name": str(data_name),
        "model_name": _canonical_model_name(model_name),
        "task": task_name,
        "target_source": normalized_target_source,
        "forecast_horizon": normalized_forecast_horizon,
        "dataset_source": str(dataset_source or "").strip().lower(),
        "split_seeds": _normalize_split_seeds(split_seeds),
    }
    task_definition = task_definition_for_task(task_name)
    if task_definition is not None:
        identity["task_definition"] = task_definition
    method_definition = _method_definition_for_model(model_name)
    if method_definition is not None:
        identity["method_definition"] = method_definition
    if task_name != "classification" and task_semantics is not None:
        identity["task_semantics"] = copy.deepcopy(task_semantics)
    return identity


def build_param_task_semantics(
    *,
    task: str,
    output_dim: int,
    target,
    target_mask=None,
    forecasting_schedule: dict | None = None,
    split_records=None,
) -> dict:
    task_name = str(task).strip().lower()
    if task_name == "classification":
        raise ValueError("classification parameter identity does not use task_semantics")
    target_array = np.asarray(target)
    semantics = {
        "output_dim": int(output_dim),
        "target_shape": [int(value) for value in target_array.shape],
        "target_dtype": str(target_array.dtype),
        "target": _json_safe_semantic_values(
            target_array,
            field="target",
            allow_nan=True,
        ),
        "search_splits": [],
    }
    if target_mask is not None:
        mask_array = np.asarray(target_mask)
        semantics.update(
            target_mask_shape=[int(value) for value in mask_array.shape],
            target_mask_dtype=str(mask_array.dtype),
            target_mask=_json_safe_semantic_values(mask_array, field="target_mask"),
        )
    if forecasting_schedule is not None:
        future = np.asarray(forecasting_schedule["future_timestamps"], dtype=np.float64)
        origin = np.asarray(forecasting_schedule["forecast_origin"], dtype=np.float64)
        semantics.update(
            future_timestamps=_json_safe_semantic_values(
                future,
                field="future_timestamps",
            ),
            forecast_origin=_json_safe_semantic_values(origin, field="forecast_origin"),
        )
    for record in split_records or []:
        semantics["search_splits"].append(
            {
                "seed": int(record["seed"]),
                "train_indices": np.asarray(record["train_indices"], dtype=np.int64).tolist(),
                "valid_indices": np.asarray(record["valid_indices"], dtype=np.int64).tolist(),
                "test_indices": np.asarray(record["test_indices"], dtype=np.int64).tolist(),
            }
        )
    return semantics


def _json_safe_semantic_values(values, *, field: str, allow_nan: bool = False):
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{field} must contain real-valued data")
    if np.issubdtype(array.dtype, np.floating):
        if np.isinf(array).any() or (not allow_nan and np.isnan(array).any()):
            raise ValueError(f"{field} must contain finite values")
        if np.isnan(array).any():
            payload = array.astype(object)
            payload[np.isnan(array)] = None
            return payload.tolist()
    return array.tolist()


def validate_param_artifact_identity(
    model_config: dict,
    *,
    data_name: str,
    model_name: str,
    task: str,
    target_source: str | None = None,
    forecast_horizon: int | None = None,
    dataset_source: str | None = None,
    split_seeds=None,
    task_semantics: dict | None = None,
    context: str = "param artifact",
) -> None:
    identity = model_config.get("artifact_identity") if isinstance(model_config, dict) else None
    if not isinstance(identity, dict):
        raise ValueError(f"{context} is missing required artifact_identity metadata.")

    expected = build_param_artifact_identity(
        data_name=data_name,
        model_name=model_name,
        task=task,
        target_source=target_source,
        forecast_horizon=forecast_horizon,
        dataset_source=dataset_source,
        split_seeds=split_seeds,
        task_semantics=task_semantics,
    )
    actual_task = str(identity.get("task", "")).strip().lower()
    actual_target_source = identity.get("target_source")
    if actual_task == "classification" and actual_target_source in {None, ""}:
        actual_target_source = "label"
    actual_forecast_horizon = identity.get("forecast_horizon")
    if actual_task != "forecasting" or actual_forecast_horizon in {None, ""}:
        actual_forecast_horizon = None
    else:
        try:
            actual_forecast_horizon = int(actual_forecast_horizon)
        except (TypeError, ValueError):
            actual_forecast_horizon = str(actual_forecast_horizon)
    explicit_split_seeds = not _metadata_value_missing(identity.get("split_seeds"))
    if actual_task != "classification" and not explicit_split_seeds:
        raise ValueError(
            f"{context} predates the current non-classification task semantics. "
            "Regenerate this task-specific parameter artifact with the current scripts."
        )
    actual_split_seeds = (
        _normalize_split_seeds(identity.get("split_seeds"))
        if explicit_split_seeds
        else _stored_split_seeds(identity)
    )
    actual = {
        "data_name": str(identity.get("data_name", "")),
        "model_name": _canonical_model_name(str(identity.get("model_name", ""))),
        "task": actual_task,
        "task_definition": str(identity.get("task_definition") or "").strip().lower(),
        "target_source": str(actual_target_source or "").strip().lower(),
        "forecast_horizon": actual_forecast_horizon,
        "dataset_source": str(identity.get("dataset_source") or "").strip().lower(),
        "split_seeds": actual_split_seeds,
        "method_definition": str(identity.get("method_definition") or "").strip().lower(),
        "task_semantics": copy.deepcopy(identity.get("task_semantics")),
    }
    compared_fields = ["data_name", "model_name", "task", "target_source"]
    if expected.get("method_definition"):
        compared_fields.append("method_definition")
    if actual_task == "classification":
        if actual["dataset_source"]:
            compared_fields.append("dataset_source")
        if actual["split_seeds"]:
            compared_fields.append("split_seeds")
    else:
        compared_fields.extend(("forecast_horizon", "dataset_source", "split_seeds"))
        if task_semantics is not None:
            compared_fields.append("task_semantics")
        if task_definition_for_task(actual_task) is not None:
            validate_task_definition(model_config, task=actual_task, context=context)
            compared_fields.append("task_definition")
    mismatches = [
        key for key in compared_fields if actual.get(key) != expected.get(key)
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: found={actual.get(key)!r}, expected={expected[key]!r}"
            for key in mismatches
        )
        raise ValueError(f"{context} artifact identity mismatch ({details}).")


_PARAM_ARTIFACT_INTEGRITY_SENTINELS = (
    "artifact_identity",
    "search_contract",
)


def is_pre_integrity_param_artifact(model_config: dict | None) -> bool:
    """Return True for parameter artifacts that lack integrity metadata."""
    if not isinstance(model_config, dict) or not model_config:
        return False
    return all(
        _metadata_value_missing(model_config.get(field))
        for field in _PARAM_ARTIFACT_INTEGRITY_SENTINELS
    )


def validate_param_artifact_metadata(
    model_config: dict | None,
    *,
    context: str = "param artifact",
    allow_compatibility_search_profile: bool = False,
    require_lr: bool = True,
) -> str:
    """Validate saved parameter artifacts before reuse or evaluation."""
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError(f"{context} is empty or not a metadata dictionary.")

    raw_profile = model_config.get("search_profile")
    compatibility_profile_missing = _metadata_value_missing(raw_profile)
    semantic_artifact = compatibility_profile_missing and _is_semantic_param_artifact(model_config)
    if semantic_artifact:
        profile = DEFAULT_SEARCH_PROFILE
    elif compatibility_profile_missing:
        if not allow_compatibility_search_profile:
            raise ValueError(
                f"{context} is missing required field 'search_profile'. "
                "Regenerate params with scripts/param_search.py. Compatibility search "
                "profiles require explicit opt-in."
            )
        profile = REPRODUCTION_SEARCH_PROFILE
    else:
        profile = normalize_search_profile(raw_profile)

    required_fields = [
        "artifact_identity",
        "search_policy",
        "search_repeat",
        "search_algorithm",
        "search_algorithm_seed",
        "search_budget_class",
        "search_max_num_epochs",
        "search_num_samples",
        "effective_params",
    ]
    missing_fields = [
        field
        for field in required_fields
        if _metadata_value_missing(model_config.get(field))
    ]
    pre_integrity_shape = is_pre_integrity_param_artifact(model_config)
    if missing_fields and not (
        allow_compatibility_search_profile and pre_integrity_shape
    ):
        missing = ", ".join(missing_fields)
        raise ValueError(
            f"{context} is missing required search detail field(s): {missing}. "
            "Regenerate params with scripts/param_search.py."
        )

    if require_lr and _metadata_value_missing(model_config.get("lr")):
        raise ValueError(
            f"{context} is missing required field 'lr'. "
            "Refusing silent learning-rate fallback for saved params; regenerate "
            "the artifact with scripts/param_search.py."
        )
    return profile


def _resolve_config_search_profile(config: dict, search_profile: str | None) -> str:
    raw = config.get("search_profile")
    if raw not in {None, ""}:
        return normalize_search_profile(raw)
    if search_profile is not None:
        return normalize_search_profile(search_profile)
    return REPRODUCTION_SEARCH_PROFILE


def _canonical_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    try:
        from torch_ists.config.alias_policy import normalize_flow_alias, normalize_tsde_method_alias

        normalized = normalize_tsde_method_alias(normalize_flow_alias(normalized))
    except Exception:
        pass
    return normalized


def _method_definition_for_model(model_name: str) -> str | None:
    normalized = _canonical_model_name(model_name)
    if normalized.startswith("flowpath-"):
        return "flowpath-input-x-learned-control"
    if normalized.startswith("fusioncde-"):
        return "fusioncde-raw-fixed-control-fusion"
    return None


def _is_flow_model_name(model_name: str) -> bool:
    normalized = _canonical_model_name(model_name)
    try:
        from torch_ists.config.alias_policy import is_flow_family_name

        return bool(is_flow_family_name(normalized))
    except Exception:
        return normalized.startswith(_FLOW_PREFIXES)


def _is_flowpath_model_name(model_name: str) -> bool:
    return _canonical_model_name(model_name).startswith("flowpath-")


def resolve_param_control_value_transform(
    model_name: str,
    model_config: dict,
    requested_transform: str,
) -> tuple[str, bool, dict | None]:
    """Resolve transform use, including the corrected FlowPath metadata interpretation."""
    from torch_ists.handlers import resolve_model_control_value_transform
    from torch_ists.preprocessing import normalize_control_value_transform

    effective, consumes_coefficients = resolve_model_control_value_transform(
        model_name,
        requested_transform,
    )
    stored = normalize_control_value_transform(
        model_config.get("control_value_transform", "identity")
    )
    if stored == effective:
        return effective, bool(consumes_coefficients), None

    identity = model_config.get("artifact_identity")
    metadata = model_config.get("control_value_transform_metadata")
    method_definition = (
        str(identity.get("method_definition") or "").strip().lower()
        if isinstance(identity, dict)
        else ""
    )
    corrected_flowpath = (
        not consumes_coefficients
        and stored == "standardize"
        and effective == "identity"
        and isinstance(identity, dict)
        and _canonical_model_name(identity.get("model_name", ""))
        == _canonical_model_name(model_name)
        and method_definition == "flowpath-input-x-learned-control"
        and _method_definition_for_model(model_name) == method_definition
        and isinstance(metadata, dict)
        and normalize_control_value_transform(metadata.get("method")) == "standardize"
        and metadata.get("fit_scope") == "train_observed"
        and normalize_control_value_transform(metadata.get("requested_transform"))
        == "standardize"
        and normalize_control_value_transform(metadata.get("effective_transform"))
        == "standardize"
        and metadata.get("consumes_coefficients") is True
    )
    if corrected_flowpath:
        return effective, False, {
            "stored_transform": stored,
            "effective_transform": effective,
            "basis": "coefficient_invariant_method",
            "method_definition": method_definition,
        }

    raise ValueError(
        "Parameter artifact control-value transform mismatch: "
        f"stored={stored!r}, requested={effective!r}. "
        "Run parameter search with the same --control-value-transform setting."
    )


def _normalize_flow_backend_for_output(value: object) -> str:
    backend = str(value).strip().lower()
    aliases = {
        "default": "native",
        "auto": "auto",
        "native": "native",
        "stribor": "stribor",
        "nfe": "stribor",
        "nfe-stibor": "stribor",
        "nfe-stribor": "stribor",
    }
    normalized = aliases.get(backend)
    if normalized is None:
        raise ValueError(
            f"Unsupported flow backend: {value!r}. Allowed backends: ['auto', 'native', 'stribor']"
        )
    return normalized


def prepare_flow_backend_config(
    model_name: str,
    model_config: dict,
    *,
    require_recorded: bool = False,
) -> dict:
    """Resolve a flow backend once and freeze it into an execution config."""
    config = dict(model_config)
    if not _is_flow_model_name(model_name):
        return config

    from torch_ists.module.flow import (
        FlowBackendSelection,
        assert_effective_backend_match,
        freeze_model_params,
        resolve_flow_backend,
        resolve_flow_backend_once,
    )

    model_params = dict(config.get("model_params") or {})
    recorded_request = config.get("requested_backend")
    recorded_effective = config.get("effective_backend")
    if (recorded_request is None) != (recorded_effective is None):
        raise ValueError(
            "Flow parameter artifacts must record requested_backend and "
            "effective_backend together."
        )

    if recorded_request is None and recorded_effective is None:
        configured_backend = model_params.get("flow_backend")
        normalized_configured = (
            None
            if configured_backend is None
            else resolve_flow_backend(configured_backend)
        )
        if require_recorded and normalized_configured not in {"stribor", "native"}:
            raise ValueError(
                "Flow execution requires a recorded backend selection. Regenerate "
                "the parameter artifact or provide an explicitly recorded native "
                "or stribor backend."
            )
        request = normalized_configured or "native"
        selection = resolve_flow_backend_once(request)
    else:
        expected = FlowBackendSelection(
            str(recorded_request),
            str(recorded_effective),
            config.get("fallback_reason"),
        )
        if expected.requested_backend == "auto":
            # Parameter search resolves ``auto`` once. Final execution consumes
            # that recorded numerical backend instead of probing a new host.
            selection = expected
        else:
            selection = resolve_flow_backend_once(expected.requested_backend)
            assert_effective_backend_match(config, selection)
            if selection != expected:
                raise ValueError(
                    "Parameter-search and model-run flow backends differ: "
                    f"params={expected.effective_backend!r}, "
                    f"run={selection.effective_backend!r}"
                )

    configured_backend = model_params.get("flow_backend")
    if configured_backend is not None:
        normalized_configured = resolve_flow_backend(configured_backend)
        if normalized_configured != "auto" and normalized_configured != selection.effective_backend:
            raise ValueError(
                "Parameter model_params.flow_backend differs from the resolved "
                f"effective backend: params={normalized_configured!r}, "
                f"run={selection.effective_backend!r}"
            )

    config["model_params"] = freeze_model_params(model_params, selection)
    config.update(selection.as_manifest_fields())
    if selection.fallback_reason is None:
        config.pop("fallback_reason", None)
    return config


def assert_flow_backend_execution_context(model_name: str, model_config: dict) -> None:
    """Fail before a trial starts when its optional runtime differs from the driver."""
    if not _is_flow_model_name(model_name):
        return
    from torch_ists.module.flow import (
        FlowBackendSelection,
        assert_backend_agreement,
        resolve_flow_backend_once,
    )

    if model_config.get("requested_backend") is None or model_config.get("effective_backend") is None:
        raise ValueError("Flow execution context is missing its frozen backend selection.")
    expected = FlowBackendSelection(
        str(model_config["requested_backend"]),
        str(model_config["effective_backend"]),
        model_config.get("fallback_reason"),
    )
    observed = resolve_flow_backend_once(expected.requested_backend)
    assert_backend_agreement(expected, observed)


def _default_flow_depth_for_model(model_name: str) -> int:
    normalized = _canonical_model_name(model_name)
    if normalized.startswith("neuralflow-"):
        return 3
    if normalized.startswith("dualdynamics-"):
        return 2
    if normalized.startswith(("flowpath-", "fusioncde-")):
        return 1
    return 2


def _is_model_native_profile(profile: str) -> bool:
    return normalize_search_profile(profile) == MODEL_NATIVE_SEARCH_PROFILE


def _ordered_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _default_method_for_axis(model_name: str) -> str | None:
    try:
        from torch_ists.config import default_method_for_model

        method = default_method_for_model(model_name)
        return None if method is None else str(method)
    except Exception:
        return None


def _native_extra_param_space(normalized_model: str, profile: str, grid: dict) -> tuple[dict, dict]:
    defaults = dict(_MODEL_NATIVE_EXTRA_PARAM_DEFAULTS.get(normalized_model, {}))
    if not _is_model_native_profile(profile):
        return {}, defaults
    values = {
        key: list(value)
        for key, value in _MODEL_NATIVE_EXTRA_PARAM_CANDIDATES.get(normalized_model, {}).items()
    }
    if _is_flow_model_name(normalized_model):
        values["flow_depth"] = list(grid["flow_depth_values"])
        defaults["flow_depth"] = int(_default_flow_depth_for_model(normalized_model))
    return values, defaults


def _resolve_native_axis_metadata(normalized_model: str, profile: str) -> dict:
    native_profile = _is_model_native_profile(profile)
    width_axis = "hidden_dim"
    depth_axis = "num_hidden_layers"
    native_axes = ["hidden_dim"]
    fixed_axes = []
    excluded_axes = []

    if normalized_model in _DEPTH_COLLAPSED_MODELS:
        fixed_axes.append("depth_level")
        if normalized_model == "mtan":
            excluded_axes.append("model_params.num_stack")
    else:
        native_axes.append("num_hidden_layers")

    if _is_flow_model_name(normalized_model):
        if _is_flowpath_model_name(normalized_model):
            depth_axis = "num_hidden_layers"
            native_axes = ["hidden_dim", "num_hidden_layers"]
            fixed_axes.append("flow_depth=1")
        else:
            depth_axis = "head_depth"
            native_axes = ["hidden_dim", "head_depth"]
            if native_profile:
                native_axes.append("flow_depth")
            else:
                fixed_axes.append("flow_depth")
    elif normalized_model in _SAND_MODELS:
        width_axis = "model_params.d_model"
        depth_axis = "model_params.n_layers"
        native_axes = ["model_params.d_model", "model_params.n_layers", "model_params.n_heads", "model_params.factor"]
    elif normalized_model in _MIAM_MODELS:
        depth_axis = "model_params.num_stack"
        native_axes = ["hidden_dim", "model_params.num_stack", "model_params.n_heads", "model_params.n_iter"]
    elif normalized_model in _MTAN_MODELS:
        native_axes = ["hidden_dim", "model_params.n_heads", "model_params.embed_time"]
    elif normalized_model in _TRANSFORMER_MODELS:
        depth_axis = "num_layers"
        native_axes = ["hidden_dim", "num_layers", "model_params.n_heads"]
    elif normalized_model in _BASIC_SEQUENCE_MODELS:
        depth_axis = "num_layers"
        native_axes = ["hidden_dim", "num_layers"]
    elif normalized_model in _NEURALRDE_DEPTH:
        fixed_axes.append(f"model_params.rde_depth={_NEURALRDE_DEPTH[normalized_model]}")
        fixed_axes.append(
            f"model_params.logsignature_window_length={DEFAULT_NEURALRDE_LOGSIGNATURE_WINDOW_LENGTH}"
        )
    elif normalized_model in _CDE_CONTROL_BY_MODEL:
        fixed_axes.append(f"model_params.control={_CDE_CONTROL_BY_MODEL[normalized_model]}")
    elif normalized_model in _SDE_MODELS:
        fixed_axes.append("model_params.drift_diffusion=model_registry")

    default_method = _default_method_for_axis(normalized_model)
    if default_method and (normalized_model in _ODE_SOLVER_MODELS or normalized_model in _CDE_SOLVER_MODELS or normalized_model in _SDE_MODELS or _is_flow_model_name(normalized_model)):
        fixed_axes.append(f"model_params.method={default_method}")

    return {
        "width_axis": width_axis,
        "depth_axis": depth_axis,
        "native_axes": _ordered_unique(native_axes),
        "fixed_axes": _ordered_unique(fixed_axes),
        "excluded_axes": _ordered_unique(excluded_axes),
    }


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
    contract = describe_search_profile(profile)
    grid = _SEARCH_POLICY_GRIDS[profile]
    normalized_model = _canonical_model_name(model_name)
    depth_values = list(grid["depth_level_values"])
    depth_collapsed = normalized_model in _DEPTH_COLLAPSED_MODELS
    if depth_collapsed:
        depth_values = [1]
    sampled_axes = ["lr", "hidden_dim"]
    if not depth_collapsed or not _is_model_native_profile(profile):
        sampled_axes.append("depth_level")
    extra_param_values, default_extra_params = _native_extra_param_space(normalized_model, profile, grid)
    sampled_axes.extend(extra_param_values)
    native_meta = _resolve_native_axis_metadata(normalized_model, profile)
    candidate_axes = ["lr", native_meta["width_axis"]]
    if not depth_collapsed or not _is_model_native_profile(profile):
        candidate_axes.append(native_meta["depth_axis"])
    candidate_axes.extend(f"model_params.{key}" if key not in {"flow_depth"} else key for key in extra_param_values)
    space = {
        "policy": MODEL_SPECIFIC_SEARCH_POLICY_NAME,
        "search_profile": profile,
        **contract,
        "model_name": normalized_model,
        "hidden_dim_values": list(grid["hidden_dim_values"]),
        "depth_level_values": depth_values,
        "default_hidden_dim": int(grid["default_hidden_dim"]),
        "default_depth_level": int(depth_values[0] if depth_values else grid["default_depth_level"]),
        "lr_bounds": list(grid["lr_bounds"]),
        "sampled_axes": _ordered_unique(sampled_axes),
        "candidate_axes": _ordered_unique(candidate_axes),
        "extra_param_values": extra_param_values,
        "default_extra_params": default_extra_params,
        "depth_collapsed": bool(depth_collapsed),
        **native_meta,
    }
    if _is_flow_model_name(normalized_model):
        space["default_flow_depth"] = int(_default_flow_depth_for_model(normalized_model))
        if _is_model_native_profile(profile):
            space["flow_depth_values"] = list(grid["flow_depth_values"])
            if "flow_depth" not in space["sampled_axes"]:
                space["sampled_axes"].append("flow_depth")
    return space


def _coerce_positive_int(value, *, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _coerce_int_choice(value, *, default: int, allowed: list[int] | None = None) -> int:
    coerced = _coerce_positive_int(value, default=default)
    if allowed and coerced not in allowed:
        return int(default)
    return int(coerced)


def _require_positive_int(value, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field} must be a positive integer, got {resolved}")
    return resolved


def _require_dropout(value, *, field: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be in [0, 1), got {value!r}") from exc
    if not math.isfinite(resolved) or not 0.0 <= resolved < 1.0:
        raise ValueError(f"{field} must be in [0, 1), got {resolved}")
    return resolved


def _require_positive_float(value, *, field: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite value, got {value!r}") from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{field} must be a positive finite value, got {resolved}")
    return resolved


def _require_nonnegative_float(value, *, field: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative finite value, got {value!r}") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{field} must be a non-negative finite value, got {resolved}")
    return resolved


def _resolve_attention_architecture(
    model_name: str,
    *,
    hidden_dim: int,
    config: dict,
    model_params: dict,
) -> tuple[dict, dict]:
    normalized = _canonical_model_name(model_name)
    params = dict(model_params)
    identity: dict[str, int | float] = {}

    if normalized in _TRANSFORMER_MODELS:
        default_width = 4 * int(hidden_dim)
        width = _require_positive_int(
            config.get("dim_feedforward", params.get("dim_feedforward", default_width)),
            field="dim_feedforward",
        )
        params["dim_feedforward"] = width
        identity["dim_feedforward"] = width
    elif normalized in _SAND_MODELS:
        d_model = _require_positive_int(params.get("d_model", hidden_dim), field="d_model")
        width = _require_positive_int(
            config.get("dim_feedforward", params.get("dim_feedforward", 4 * d_model)),
            field="dim_feedforward",
        )
        params["dim_feedforward"] = width
        identity["dim_feedforward"] = width
    elif normalized in _MIAM_MODELS:
        width = _require_positive_int(
            config.get("dim_feedforward", params.get("dim_feedforward", hidden_dim)),
            field="dim_feedforward",
        )
        top_level_dropout = _require_dropout(config.get("dropout", 0.1), field="dropout")
        defaults = {
            "attention_dropout": top_level_dropout,
            "ffn_dropout": top_level_dropout,
            "residual_dropout": top_level_dropout,
        }
        params["dim_feedforward"] = width
        identity["dim_feedforward"] = width
        for field, default in defaults.items():
            value = _require_dropout(
                config.get(field, params.get(field, default)),
                field=field,
            )
            params[field] = value
            identity[field] = value
    elif normalized in _MTAN_MODELS:
        default_head = int(hidden_dim)
        head = _require_positive_int(
            config.get("latent_head_dim", params.get("latent_head_dim", default_head)),
            field="latent_head_dim",
        )
        time_scale = _require_positive_float(
            config.get("time_scale", params.get("time_scale", 48.0)),
            field="time_scale",
        )
        params["latent_head_dim"] = head
        params["time_scale"] = time_scale
        identity.update(latent_head_dim=head, time_scale=time_scale)

    return params, identity


def _resolve_fixed_construction_architecture(
    model_name: str,
    *,
    config: dict,
    model_params: dict,
) -> tuple[dict, dict]:
    normalized = _canonical_model_name(model_name)
    params = dict(model_params)
    identity: dict[str, int | float] = {}

    if normalized == "cnn":
        kernel_size = _require_positive_int(
            config.get("kernel_size", params.get("kernel_size", 3)),
            field="kernel_size",
        )
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        params["kernel_size"] = kernel_size
        identity["kernel_size"] = kernel_size
    elif normalized == "leap":
        width = _require_positive_int(
            config.get(
                "leap_intermediate_width",
                params.get("leap_intermediate_width", 128),
            ),
            field="leap_intermediate_width",
        )
        params["leap_intermediate_width"] = width
        identity["leap_intermediate_width"] = width
    elif normalized == "exit":
        for field in ("kinetic_energy_coef", "jacobian_norm2_coef"):
            value = _require_nonnegative_float(
                config.get(field, params.get(field, 1.0)),
                field=field,
            )
            params[field] = value
            identity[field] = value
        div_samples = _require_positive_int(
            config.get("div_samples", params.get("div_samples", 1)),
            field="div_samples",
        )
        params["div_samples"] = div_samples
        identity["div_samples"] = div_samples

    return params, identity


_FIXED_CONSTRUCTION_FIELDS = {
    "cnn": frozenset({"kernel_size"}),
    "leap": frozenset({"leap_intermediate_width"}),
    "exit": frozenset({"kinetic_energy_coef", "jacobian_norm2_coef", "div_samples"}),
}


_ATTENTION_ARCHITECTURE_FIELDS = {
    "transformer": frozenset({"dim_feedforward"}),
    "transformer-mask": frozenset({"dim_feedforward"}),
    "sand": frozenset({"dim_feedforward"}),
    "miam": frozenset(
        {
            "dim_feedforward",
            "attention_dropout",
            "ffn_dropout",
            "residual_dropout",
        }
    ),
    "mtan": frozenset({"latent_head_dim", "time_scale"}),
}


def normalize_attention_model_parameters_for_comparison(
    model_name: str,
    model_parameters: dict,
) -> dict:
    """Return a non-mutating comparison view for attention output lineage."""
    normalized = _canonical_model_name(model_name)
    required = _ATTENTION_ARCHITECTURE_FIELDS.get(normalized)
    result = copy.deepcopy(model_parameters)
    if required is None:
        return result

    model_params = dict(result.get("model_params") or {})
    present = required.intersection(model_params)
    if not present:
        raise ValueError(
            "fieldless attention output artifact is ambiguous; provide all "
            "resolved architecture fields in a read-only compatibility view"
        )
    if present and present != required:
        missing = ", ".join(sorted(required - present))
        raise ValueError(
            "incomplete attention architecture fields in stored output "
            f"model parameters: missing {missing}"
        )
    hidden_dim = _require_positive_int(result.get("hidden_dim"), field="hidden_dim")
    model_params, _ = _resolve_attention_architecture(
        normalized,
        hidden_dim=hidden_dim,
        config={},
        model_params=model_params,
    )
    result["model_params"] = model_params
    return result


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
    for key, default_value in space.get("default_extra_params", {}).items():
        if key == "flow_depth":
            continue
        allowed = space.get("extra_param_values", {}).get(key)
        value = config.get(key, model_params.get(key, default_value))
        if isinstance(default_value, int):
            value = _coerce_int_choice(value, default=int(default_value), allowed=allowed)
        model_params[key] = value

    if _is_flowpath_model_name(normalized_model):
        flow_depth = 1
        num_layers = 1
        num_hidden_layers = depth_level
        head_depth = depth_level
        model_params.setdefault("flow_backend", "native")
    elif _is_flow_model_name(normalized_model):
        if _is_model_native_profile(profile):
            raw_flow_depth = config.get(
                "flow_depth",
                config.get("flow_num_layers", space.get("default_flow_depth", _default_flow_depth_for_model(normalized_model))),
            )
            flow_depth = _coerce_positive_int(
                raw_flow_depth,
                default=space.get("default_flow_depth", _default_flow_depth_for_model(normalized_model)),
            )
        else:
            flow_depth = _default_flow_depth_for_model(normalized_model)
        num_layers = 1
        num_hidden_layers = depth_level
        head_depth = depth_level
        model_params.setdefault("flow_backend", "native")
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

    if _is_flow_model_name(normalized_model):
        model_params["flow_backend"] = _normalize_flow_backend_for_output(
            model_params["flow_backend"]
        )

    required_attention_fields = _ATTENTION_ARCHITECTURE_FIELDS.get(normalized_model)
    required_integrity_fields = required_attention_fields or _FIXED_CONSTRUCTION_FIELDS.get(normalized_model)
    stored_effective = config.get("effective_params")
    stored_effective_model_params = (
        dict(stored_effective.get("model_params") or {})
        if isinstance(stored_effective, dict)
        else {}
    )
    if required_integrity_fields is not None:
        for field in required_integrity_fields:
            if field not in stored_effective_model_params:
                continue
            effective_value = stored_effective_model_params[field]
            if field in model_params and model_params[field] != effective_value:
                raise ValueError(
                    "Parameter artifact architecture identity mismatch: "
                    f"model_params.{field}={model_params[field]!r}, "
                    f"effective_params.model_params.{field}={effective_value!r}."
                )
            model_params[field] = effective_value

    stored_architecture_identity = config.get("architecture_identity")
    if "architecture_identity" in config and not isinstance(
        stored_architecture_identity,
        dict,
    ):
        raise ValueError("architecture_identity must be a mapping when present")
    if isinstance(stored_architecture_identity, dict):
        for key, value in stored_architecture_identity.items():
            if key in model_params and model_params[key] != value:
                raise ValueError(
                    "Parameter artifact architecture identity mismatch: "
                    f"model_params.{key}={model_params[key]!r}, "
                    f"architecture_identity.{key}={value!r}."
                )
            model_params[key] = value
    semantic_without_identity = (
        not isinstance(stored_architecture_identity, dict)
        and _is_semantic_param_artifact(config)
    )
    if semantic_without_identity and required_attention_fields is not None:
        required_fields = required_attention_fields
        present_fields = {
            field
            for field in required_fields
            if config.get(field, model_params.get(field)) is not None
        }
        if not present_fields:
            raise ValueError(
                "fieldless attention parameter artifact is ambiguous; provide all "
                "resolved architecture fields in a read-only compatibility view"
            )
        if present_fields != required_fields:
            missing = ", ".join(sorted(required_fields - present_fields))
            raise ValueError(
                "incomplete attention architecture fields in stored parameter "
                f"artifact: missing {missing}"
            )
    model_params, architecture_identity = _resolve_attention_architecture(
        normalized_model,
        hidden_dim=hidden_dim,
        config=config,
        model_params=model_params,
    )
    model_params, fixed_identity = _resolve_fixed_construction_architecture(
        normalized_model,
        config=config,
        model_params=model_params,
    )
    architecture_identity.update(fixed_identity)
    if isinstance(stored_architecture_identity, dict):
        if architecture_identity != stored_architecture_identity:
            raise ValueError(
                "Parameter artifact architecture identity mismatch: "
                f"stored={stored_architecture_identity!r}, "
                f"resolved={architecture_identity!r}."
            )

    saved = dict(config)
    saved["search_profile"] = profile
    saved.update(describe_search_profile(profile))
    saved["search_policy"] = str(config.get("search_policy", STANDARD_SEARCH_POLICY))
    saved["search_repeat"] = str(config.get("search_repeat", SINGLE_SEARCH_REPEAT))
    saved["search_algorithm_policy"] = MODEL_SPECIFIC_SEARCH_POLICY_NAME
    saved["search_grid_version"] = profile
    saved["hidden_dim"] = int(hidden_dim)
    saved["depth_level"] = int(depth_level)
    saved["num_layers"] = int(num_layers)
    saved["num_hidden_layers"] = int(num_hidden_layers)
    saved["head_depth"] = int(head_depth)
    saved["flow_num_layers"] = int(flow_depth)
    saved["flow_depth"] = int(flow_depth)
    saved["model_params"] = model_params
    if architecture_identity:
        saved["architecture_identity"] = architecture_identity
    for key, value in space.get("default_extra_params", {}).items():
        if key != "flow_depth" and key in model_params:
            saved[key] = model_params[key]
    saved["architecture_axes"] = {
        "policy": MODEL_SPECIFIC_SEARCH_POLICY_NAME,
        "search_algorithm_policy": MODEL_SPECIFIC_SEARCH_POLICY_NAME,
        "search_grid_version": profile,
        "width_axis": space["width_axis"],
        "depth_axis": space["depth_axis"],
        "hidden_dim_values": list(space["hidden_dim_values"]),
        "depth_level_values": list(space["depth_level_values"]),
        "flow_depth_values": list(space.get("flow_depth_values", [])),
        "sampled_axes": list(space["sampled_axes"]),
        "candidate_axes": list(space["candidate_axes"]),
        "extra_param_values": {
            key: list(value)
            for key, value in space.get("extra_param_values", {}).items()
        },
        "default_extra_params": dict(space.get("default_extra_params", {})),
        "native_axes": list(space["native_axes"]),
        "fixed_axes": list(space["fixed_axes"]),
        "excluded_axes": list(space["excluded_axes"]),
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
    stored_effective = model_config.get("effective_params")
    required_effective_fields = {
        "hidden_dim",
        "num_layers",
        "num_hidden_layers",
        "head_depth",
        "flow_num_layers",
        "flow_depth",
    }
    if isinstance(stored_effective, dict) and required_effective_fields.issubset(stored_effective):
        effective = dict(stored_effective)
    else:
        effective = resolve_effective_param_config(
            model_name,
            model_config,
            search_profile=infer_param_search_profile(model_config),
        )
    hidden_dim = int(effective["hidden_dim"])
    model_params = dict(effective.get("model_params") or {})
    stored_architecture_identity = model_config.get("architecture_identity")
    if "architecture_identity" in model_config and not isinstance(
        stored_architecture_identity,
        dict,
    ):
        raise ValueError("architecture_identity must be a mapping when present")
    normalized_model = _canonical_model_name(model_name)
    required_attention_fields = _ATTENTION_ARCHITECTURE_FIELDS.get(normalized_model)
    required_integrity_fields = required_attention_fields or _FIXED_CONSTRUCTION_FIELDS.get(normalized_model)
    top_level_model_params = dict(model_config.get("model_params") or {})
    if required_integrity_fields is not None:
        for field in required_integrity_fields:
            if field not in top_level_model_params:
                continue
            if field in model_params and model_params[field] != top_level_model_params[field]:
                raise ValueError(
                    "Parameter artifact architecture identity mismatch: "
                    f"model_params.{field}={top_level_model_params[field]!r}, "
                    f"effective_params.model_params.{field}={model_params[field]!r}."
                )
            model_params[field] = top_level_model_params[field]
    if isinstance(stored_architecture_identity, dict):
        for key, value in stored_architecture_identity.items():
            if key in model_params and model_params[key] != value:
                raise ValueError(
                    "Parameter artifact architecture identity mismatch: "
                    f"model_params.{key}={model_params[key]!r}, "
                    f"architecture_identity.{key}={value!r}."
                )
            model_params[key] = value
        model_params, resolved_identity = _resolve_attention_architecture(
            model_name,
            hidden_dim=hidden_dim,
            config=stored_architecture_identity,
            model_params=model_params,
        )
        model_params, fixed_identity = _resolve_fixed_construction_architecture(
            model_name,
            config=stored_architecture_identity,
            model_params=model_params,
        )
        resolved_identity.update(fixed_identity)
        if resolved_identity != stored_architecture_identity:
            raise ValueError(
                "Parameter artifact architecture identity mismatch: "
                f"stored={stored_architecture_identity!r}, "
                f"resolved={resolved_identity!r}."
            )
    elif _is_semantic_param_artifact(model_config) or isinstance(stored_effective, dict):
        if required_attention_fields is not None:
            present_fields = {
                field
                for field in required_attention_fields
                if model_params.get(field) is not None
            }
            if not present_fields:
                raise ValueError(
                    "fieldless attention parameter artifact is ambiguous; provide all "
                    "resolved architecture fields in a read-only compatibility view"
                )
            if present_fields != required_attention_fields:
                missing = ", ".join(sorted(required_attention_fields - present_fields))
                raise ValueError(
                    "incomplete attention architecture fields in stored parameter "
                    f"artifact: missing {missing}"
                )
        model_params, _ = _resolve_attention_architecture(
            model_name,
            hidden_dim=hidden_dim,
            config={},
            model_params=model_params,
        )
        model_params, _ = _resolve_fixed_construction_architecture(
            model_name,
            config={},
            model_params=model_params,
        )
    return {
        "hidden_dim": hidden_dim,
        "hidden_hidden_dim": hidden_dim,
        "num_layers": int(effective["num_layers"]),
        "num_hidden_layers": int(effective["num_hidden_layers"]),
        "head_depth": int(effective["head_depth"]),
        "flow_num_layers": int(effective["flow_num_layers"]),
        "flow_depth": int(effective["flow_depth"]),
        "model_params": model_params,
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
        fallback_upper = max(upper, min(train_n, max(upper + 8, upper * 2)))
        candidates = [
            candidate
            for candidate in range(upper + 1, fallback_upper + 1)
            if train_n % candidate != 1
        ]
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
        "memory_cap_respected": bool(batch_size <= memory_cap),
        "singleton_remainder_unresolved": bool(last_batch_size == 1 and train_n > batch_size),
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
    require_config: bool = False,
) -> dict:
    """Resolve optimizer LR without coupling fallback behavior to batch size."""
    if config_lr is None:
        if require_config:
            raise ValueError(
                "Saved param artifact is missing required field 'lr'. "
                "Refusing silent learning-rate fallback for model execution."
            )
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


def default_method_for_model(model_name: str) -> str | None:
    try:
        from torch_ists.config import default_method_for_model as runtime_default_method_for_model

        return runtime_default_method_for_model(model_name)
    except Exception:
        normalized = str(model_name).strip().lower()
        if normalized == "neuralflow" or normalized.startswith("neuralflow-"):
            return None
        if normalized == "latentsde":
            return "srk"
        if normalized in {
            "naive_cde",
            "neural_ode",
            "neural_ode_t",
            "neural_sde",
            "neural_sde_t",
            "neural_lsde",
            "neural_lnsde",
            "neural_gsde",
            "neural_physgsde",
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
        if normalized in {
            "neuralcde",
            "neuralcde-l",
            "neuralcde-r",
            "neuralcde-c",
            "neuralcde-h",
            "ancde",
            "exit",
            "leap",
        }:
            return "rk4"
        if normalized.startswith("neuraltsde_") or normalized.startswith("neuraldsde_"):
            return "srk"
        return "euler"
