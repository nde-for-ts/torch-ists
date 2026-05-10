#!/usr/bin/env python
"""Shared bootstrap helpers for packaged scripts."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

MODEL_SCOPE_VARIANT_ENV = "TORCH_ISTS_MODEL_SCOPE_VARIANT"
__all__ = ["configure_runtime_paths", "apply_model_scope_variant"]


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
        if normalized in {"neuralcde", "neuralcde-l", "neuralcde-r", "neuralcde-c", "neuralcde-h"}:
            return "rk4"
        return "euler"
