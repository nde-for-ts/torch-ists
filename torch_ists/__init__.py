"""Runtime package for irregular time series modeling."""

from __future__ import annotations

from importlib import import_module

from .data_loader import DataLoader, DatasetBundle
from .preprocessing import MISSINGNESS_EXPERIMENTAL, apply_missingness, preprocess
from .split_utils import get_split_path, load_splits, make_stratified_splits, save_splits
from .seed_utils import SeedContext, get_rng_state, set_global_seed, set_rng_state

__all__ = [
    "DataLoader",
    "DatasetBundle",
    "MISSINGNESS_EXPERIMENTAL",
    "apply_missingness",
    "preprocess",
    "get_split_path",
    "load_splits",
    "make_stratified_splits",
    "save_splits",
    "benchmark_loader",
    "set_global_seed",
    "get_rng_state",
    "set_rng_state",
    "SeedContext",
    "ists_classifier",
    "ists_model",
    "ists_layer",
]

__version__ = "0.5.0"

_LAZY_EXPORTS = {
    "ists_layer": (".layer", "ists_layer"),
    "ists_classifier": (".model", "ists_classifier"),
    "ists_model": (".model", "ists_model"),
    "benchmark_loader": (".benchmark_loader", None),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
