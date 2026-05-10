"""Utilities for loading the packaged model scope."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import List

PACKAGED_ACTIVE_REGISTRY_FILENAME = "active_models" + ".yaml"


def _normalize_model_name(raw: str) -> str:
    return raw.strip().strip("`").strip()


def _parse_registry_text(text: str) -> List[str]:
    models: List[str] = []
    seen = set()
    in_models = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("models:"):
            in_models = True
            continue
        if in_models and stripped.startswith("- "):
            model = _normalize_model_name(stripped[2:])
            if not model:
                continue
            if model in seen:
                raise RuntimeError(
                    f"Model registry is corrupted: duplicate entry '{model}'."
                )
            seen.add(model)
            models.append(model)
            continue
        if in_models and ":" in stripped and not stripped.startswith("- "):
            break

    if not in_models:
        raise RuntimeError(
            "Model registry is corrupted: missing required section."
        )
    if not models:
        raise RuntimeError("Model registry is corrupted: model list is empty.")
    return models


def _load_packaged_registry_text() -> str:
    files_api = getattr(resources, "files", None)
    if files_api is not None:
        resource = files_api(__package__).joinpath(PACKAGED_ACTIVE_REGISTRY_FILENAME)
        if not resource.is_file():
            raise RuntimeError(
                "Packaged model registry missing. Reinstall the package."
            )
        return resource.read_text(encoding="utf-8")
    try:
        return resources.read_text(
            __package__,
            PACKAGED_ACTIVE_REGISTRY_FILENAME,
            encoding="utf-8",
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Packaged model registry missing. Reinstall the package."
        ) from exc


@lru_cache(maxsize=1)
def _load_active_models_cached() -> tuple[str, ...]:
    return tuple(_parse_registry_text(_load_packaged_registry_text()))


def load_active_models() -> List[str]:
    return list(_load_active_models_cached())
