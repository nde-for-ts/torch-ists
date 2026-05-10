"""Stable SDE model configs for the runtime package.

Descriptions are intentionally concise and provenance-gated.
"""

from typing import Dict


STABLE_SDE_CONFIGS: Dict[str, Dict] = {
    "neural_lsde": {
        "drift": {
            "control": True,
            "latent": True,
            "time": False,
            "geometric": False,
        },
        "diffusion": {
            "network": "nonlinear",
            "input": "t",
            "type": "additive",
        },
        "description": "ICLR 2024 stable SDE core model (neural_lsde)",
        "legacy_code": "neuralsde_2_16",
        "paper": "ICLR 2024",
        "handler": "SDEHandler",
    },
    "neural_lnsde": {
        "drift": {
            "control": True,
            "latent": True,
            "time": True,
            "geometric": False,
        },
        "diffusion": {
            "network": "nonlinear",
            "input": "t",
            "type": "multiplicative",
        },
        "description": "ICLR 2024 stable SDE core model (neural_lnsde)",
        "legacy_code": "neuralsde_4_17",
        "paper": "ICLR 2024",
        "handler": "SDEHandler",
    },
    "neural_gsde": {
        "drift": {
            "control": True,
            "latent": True,
            "time": True,
            "geometric": True,
        },
        "diffusion": {
            "network": "nonlinear",
            "input": "t",
            "type": "multiplicative",
        },
        "description": "ICLR 2024 stable SDE core model (neural_gsde)",
        "legacy_code": "neuralsde_6_17",
        "paper": "ICLR 2024",
        "handler": "SDEHandler",
    },
}


STABLE_SDE_ALIASES: Dict[str, str] = {}
for _model_name, _cfg in STABLE_SDE_CONFIGS.items():
    _legacy_code = str(_cfg.get("legacy_code", "")).strip()
    if not _legacy_code:
        continue
    _existing = STABLE_SDE_ALIASES.get(_legacy_code)
    if _existing is not None and _existing != _model_name:
        raise RuntimeError(
            f"duplicate stable SDE legacy_code detected: {_legacy_code} -> {_existing}, {_model_name}"
        )
    STABLE_SDE_ALIASES[_legacy_code] = _model_name


STABLE_SDE_LEGACY_ALIASES = STABLE_SDE_ALIASES

__all__ = ["STABLE_SDE_CONFIGS", "STABLE_SDE_ALIASES", "STABLE_SDE_LEGACY_ALIASES"]
