#!/usr/bin/env python
"""Shared runtime/bootstrap helpers for script entrypoints."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import sys
from pathlib import Path

MODEL_SCOPE_VARIANT_ENV = "TORCH_ISTS_MODEL_SCOPE_VARIANT"
BATCH_SIZE_POLICY_NAME = "dataset-derived-fixed"
LEARNING_RATE_POLICY_NAME = "fixed-lr-fallback"
WEIGHT_DECAY_POLICY_NAME = "lr-proportional-weight-decay"
LEGACY_ICLR2024_SEARCH_PROFILE = "legacy-iclr2024"
LEGACY_EXPANDED_SEARCH_PROFILE = "legacy-expanded"
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

_SEARCH_PROFILE_ALIASES = {
    "default": MODEL_NATIVE_SEARCH_PROFILE,
    "model-native": MODEL_NATIVE_SEARCH_PROFILE,
    "model-native-v1": MODEL_NATIVE_SEARCH_PROFILE,
    "model-specific": MODEL_NATIVE_SEARCH_PROFILE,
    MODEL_NATIVE_SEARCH_PROFILE: MODEL_NATIVE_SEARCH_PROFILE,
    "legacy": LEGACY_EXPANDED_SEARCH_PROFILE,
    "expanded": LEGACY_EXPANDED_SEARCH_PROFILE,
    "legacy-expanded": LEGACY_EXPANDED_SEARCH_PROFILE,
    "legacy-expanded-v1": LEGACY_EXPANDED_SEARCH_PROFILE,
    LEGACY_EXPANDED_SEARCH_PROFILE: LEGACY_EXPANDED_SEARCH_PROFILE,
    "iclr2024": LEGACY_ICLR2024_SEARCH_PROFILE,
    "legacy-iclr2024": LEGACY_ICLR2024_SEARCH_PROFILE,
    "legacy-iclr2024-v0": LEGACY_ICLR2024_SEARCH_PROFILE,
    LEGACY_ICLR2024_SEARCH_PROFILE: LEGACY_ICLR2024_SEARCH_PROFILE,
}

_SEARCH_PROFILE_GRIDS = {
    MODEL_NATIVE_SEARCH_PROFILE: {
        "hidden_dim_values": list(DEFAULT_MODEL_NATIVE_HIDDEN_DIM_VALUES),
        "depth_level_values": [1, 2, 3, 4, 5],
        "flow_depth_values": [1, 2, 3],
        "default_hidden_dim": 32,
        "default_depth_level": 1,
        "lr_bounds": [1e-4, 1e-2],
    },
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
_FLOW_PREFIXES = ("neuralflow-", "dualdynamics-", "flowpath-")

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


def build_runtime_provenance(runtime_baseline_id: str | None = None) -> dict:
    """Return portable runtime metadata without recording local filesystem paths."""
    try:
        import torch_ists

        package_version = str(getattr(torch_ists, "__version__", "unknown"))
        package_file = Path(getattr(torch_ists, "__file__", "")).resolve()
        runtime_root = package_file.parents[1] if package_file else Path(__file__).resolve().parents[1]
    except Exception:
        package_version = "unknown"
        runtime_root = Path(__file__).resolve().parents[1]

    status = _read_json_file(runtime_root / "PACKAGE_STATUS.json") or {}
    mirror_commit_sha = (
        os.environ.get("TORCH_ISTS_MIRROR_COMMIT_SHA")
        or status.get("mirror_commit_sha")
        or (_read_public_mirror_head(runtime_root) if (runtime_root / "PACKAGE_STATUS.json").exists() else None)
    )
    archive_checksum = os.environ.get("TORCH_ISTS_ARCHIVE_SHA256") or status.get("mirror_archive_checksum")

    return {
        "package_version": package_version,
        "mirror_commit_sha": None if mirror_commit_sha is None else str(mirror_commit_sha),
        "mirror_archive_checksum": None if archive_checksum is None else str(archive_checksum),
        "runtime_baseline_id": None if runtime_baseline_id is None else str(runtime_baseline_id),
        "wheel_or_source_checkout": "source_checkout" if (runtime_root / "scripts").exists() else "installed_package",
    }


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
    """Ensure packaged runtime resolution for script execution."""
    caller_path = Path(caller_file).resolve()
    source_repo_root = caller_path.parents[2]
    source_release_root = source_repo_root / "release"
    public_checkout_root = caller_path.parents[1]
    env_runtime_root = os.environ.get("TORCH_ISTS_RUNTIME_ROOT")
    if env_runtime_root:
        runtime_root = Path(env_runtime_root).resolve()
        repo_root = source_repo_root
    elif (source_release_root / "torch_ists" / "__init__.py").exists():
        runtime_root = source_release_root.resolve()
        repo_root = source_repo_root
    elif (public_checkout_root / "torch_ists" / "__init__.py").exists():
        runtime_root = public_checkout_root.resolve()
        repo_root = public_checkout_root.resolve()
    else:
        runtime_root = source_release_root.resolve()
        repo_root = source_repo_root
    scripts_root = runtime_root / "scripts"

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


def apply_model_scope_variant(variant: str | None) -> str:
    normalized = "" if variant is None else str(variant).strip().lower()
    if normalized in {"", "active"}:
        os.environ.pop(MODEL_SCOPE_VARIANT_ENV, None)
        return "active"
    if normalized == "tsde_internal":
        os.environ[MODEL_SCOPE_VARIANT_ENV] = "tsde_internal"
        return "tsde_internal"
    if normalized == "dsde_internal":
        os.environ[MODEL_SCOPE_VARIANT_ENV] = "dsde_internal"
        return "dsde_internal"
    if normalized == "physgsde_internal":
        os.environ[MODEL_SCOPE_VARIANT_ENV] = "physgsde_internal"
        return "physgsde_internal"
    if normalized == "ode_source_internal":
        os.environ[MODEL_SCOPE_VARIANT_ENV] = "ode_source_internal"
        return "ode_source_internal"
    raise ValueError(
        "Unsupported model scope variant. "
        "Expected one of {'active','tsde_internal','dsde_internal','physgsde_internal','ode_source_internal'}."
    )


def normalize_search_profile(raw: str | None) -> str:
    normalized = DEFAULT_SEARCH_PROFILE if raw is None else str(raw).strip().lower()
    try:
        return _SEARCH_PROFILE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(_SEARCH_PROFILE_GRIDS))
        raise ValueError(f"Unknown search profile: {raw!r}. Allowed: {allowed}.") from exc


def describe_search_profile(raw: str | None) -> dict:
    """Return public search details for a saved artifact profile."""
    profile = normalize_search_profile(raw)
    if profile == LEGACY_ICLR2024_SEARCH_PROFILE:
        return {
            "search_contract": HISTORICAL_SEARCH_CONTRACT_NAME,
            "search_family": "historical_reproducibility",
            "search_profile_visibility": COMPATIBILITY_SEARCH_PROFILE_VISIBILITY,
            "public_search_name": "historical reproducibility",
        }
    if profile == LEGACY_EXPANDED_SEARCH_PROFILE:
        return {
            "search_contract": DEFAULT_SEARCH_CONTRACT_NAME,
            "search_family": "internal_reproducibility",
            "search_profile_visibility": COMPATIBILITY_SEARCH_PROFILE_VISIBILITY,
            "public_search_name": "expanded internal search",
        }
    return {
        "search_contract": DEFAULT_SEARCH_CONTRACT_NAME,
        "search_family": "default",
        "search_profile_visibility": PUBLIC_SEARCH_PROFILE_VISIBILITY,
        "public_search_name": "default model-specific search",
    }


def infer_param_search_profile(model_config: dict | None) -> str:
    """Infer profile for saved params; missing profile is no longer implicit."""
    if not model_config:
        raise ValueError(
            "Param artifact is empty or missing; cannot infer search_profile. "
            "Regenerate params with scripts/param_search.py."
        )
    raw = model_config.get("search_profile")
    if raw in {None, ""}:
        raise ValueError(
            "Param artifact is missing required field 'search_profile'. "
            "Regenerate params with the current default search, or pass the hidden "
            "compatibility flag only for intentional legacy artifact audits."
        )
    return normalize_search_profile(raw)


def _metadata_value_missing(value) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def compute_param_config_hash(model_config: dict) -> str:
    """Hash a saved parameter config, excluding the stored hash field itself."""
    if not isinstance(model_config, dict):
        raise ValueError("param config must be a metadata dictionary.")
    payload = {key: value for key, value in model_config.items() if key != "param_hash"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_param_artifact_identity(
    *,
    data_name: str,
    model_name: str,
    task: str,
) -> dict:
    return {
        "data_name": str(data_name),
        "model_name": _canonical_model_name(model_name),
        "task": str(task).strip().lower(),
    }


def validate_param_artifact_identity(
    model_config: dict,
    *,
    data_name: str,
    model_name: str,
    task: str,
    context: str = "param artifact",
) -> None:
    identity = model_config.get("artifact_identity") if isinstance(model_config, dict) else None
    if not isinstance(identity, dict):
        raise ValueError(f"{context} is missing required artifact_identity metadata.")

    expected = build_param_artifact_identity(data_name=data_name, model_name=model_name, task=task)
    actual = {
        "data_name": str(identity.get("data_name", "")),
        "model_name": _canonical_model_name(str(identity.get("model_name", ""))),
        "task": str(identity.get("task", "")).strip().lower(),
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if str(actual.get(key)) != str(expected_value)
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: found={actual.get(key)!r}, expected={expected[key]!r}"
            for key in mismatches
        )
        raise ValueError(f"{context} artifact identity mismatch ({details}).")


def validate_param_artifact_metadata(
    model_config: dict | None,
    *,
    context: str = "param artifact",
    allow_legacy_search_profile: bool = False,
    require_lr: bool = True,
) -> str:
    """Validate saved parameter artifacts before reuse or evaluation."""
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError(f"{context} is empty or not a metadata dictionary.")

    raw_profile = model_config.get("search_profile")
    legacy_profile_missing = _metadata_value_missing(raw_profile)
    if legacy_profile_missing:
        if not allow_legacy_search_profile:
            raise ValueError(
                f"{context} is missing required field 'search_profile'. "
                "Regenerate params with scripts/param_search.py. Hidden legacy "
                "profile compatibility must be explicitly opted in for audits."
            )
        profile = LEGACY_ICLR2024_SEARCH_PROFILE
    else:
        profile = normalize_search_profile(raw_profile)

    required_fields = [
        "artifact_identity",
        "param_hash",
        "search_contract",
        "search_family",
        "public_search_name",
        "search_profile_visibility",
        "search_policy",
        "search_repeat",
        "search_algorithm_policy",
        "search_algorithm",
        "search_algorithm_seed",
        "search_budget_class",
        "search_execution_backend",
        "search_max_num_epochs",
        "search_num_samples",
        "architecture_axes",
        "effective_params",
    ]
    missing_fields = [
        field
        for field in required_fields
        if _metadata_value_missing(model_config.get(field))
    ]
    if missing_fields and not (allow_legacy_search_profile and profile == LEGACY_ICLR2024_SEARCH_PROFILE):
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
    if not (allow_legacy_search_profile and profile == LEGACY_ICLR2024_SEARCH_PROFILE):
        expected_hash = compute_param_config_hash(model_config)
        if str(model_config.get("param_hash")) != expected_hash:
            raise ValueError(
                f"{context} param_hash mismatch. "
                "The saved parameter artifact may be stale or corrupted; regenerate it with scripts/param_search.py."
            )
    return profile


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
        from torch_ists.config.alias_policy import normalize_flow_alias, normalize_tsde_method_alias

        normalized = normalize_tsde_method_alias(normalize_flow_alias(normalized))
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


def _is_model_native_profile(profile: str) -> bool:
    return normalize_search_profile(profile) == MODEL_NATIVE_SEARCH_PROFILE


def _ordered_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _default_method_for_axis(model_name: str) -> str | None:
    try:
        from torch_ists.config import default_method_for_model

        return str(default_method_for_model(model_name))
    except Exception:
        return None


def _native_extra_param_space(normalized_model: str, profile: str, grid: dict) -> tuple[dict, dict]:
    if not _is_model_native_profile(profile):
        return {}, {}
    values = {
        key: list(value)
        for key, value in _MODEL_NATIVE_EXTRA_PARAM_CANDIDATES.get(normalized_model, {}).items()
    }
    defaults = dict(_MODEL_NATIVE_EXTRA_PARAM_DEFAULTS.get(normalized_model, {}))
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
        depth_axis = "head_depth"
        native_axes = ["hidden_dim", "head_depth"]
        if native_profile:
            native_axes.append("flow_depth")
        else:
            fixed_axes.append("flow_depth")
        fixed_axes.append("model_params.flow_backend=native")
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
    grid = _SEARCH_PROFILE_GRIDS[profile]
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

    if _is_flow_model_name(normalized_model):
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

    saved = dict(config)
    saved["search_profile"] = profile
    saved.update(describe_search_profile(profile))
    saved["search_policy"] = str(config.get("search_policy", STANDARD_SEARCH_POLICY))
    saved["search_repeat"] = str(config.get("search_repeat", SINGLE_SEARCH_REPEAT))
    saved["search_algorithm_policy"] = MODEL_SPECIFIC_SEARCH_POLICY_NAME
    saved["search_grid_version"] = profile
    saved["legacy_grid_version"] = (
        profile if profile in {LEGACY_ICLR2024_SEARCH_PROFILE, LEGACY_EXPANDED_SEARCH_PROFILE} else None
    )
    saved["hidden_dim"] = int(hidden_dim)
    saved["depth_level"] = int(depth_level)
    saved["num_layers"] = int(num_layers)
    saved["num_hidden_layers"] = int(num_hidden_layers)
    saved["head_depth"] = int(head_depth)
    saved["flow_num_layers"] = int(flow_depth)
    saved["flow_depth"] = int(flow_depth)
    saved["model_params"] = model_params
    for key, value in space.get("default_extra_params", {}).items():
        if key != "flow_depth" and key in model_params:
            saved[key] = model_params[key]
    saved["architecture_axes"] = {
        "policy": MODEL_SPECIFIC_SEARCH_POLICY_NAME,
        "search_algorithm_policy": MODEL_SPECIFIC_SEARCH_POLICY_NAME,
        "search_grid_version": profile,
        "legacy_grid_version": saved["legacy_grid_version"],
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


def default_method_for_model(model_name: str) -> str:
    try:
        from torch_ists.config import default_method_for_model as runtime_default_method_for_model

        return runtime_default_method_for_model(model_name)
    except Exception:
        normalized = str(model_name).strip().lower()
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
