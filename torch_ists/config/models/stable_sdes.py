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
        "paper": "ICLR 2024",
        "handler": "SDEHandler",
    },
}


# Numeric SDE encodings are implementation details, not public API names.
STABLE_SDE_ALIASES: Dict[str, str] = {}
STABLE_SDE_LEGACY_ALIASES = STABLE_SDE_ALIASES

__all__ = ["STABLE_SDE_CONFIGS", "STABLE_SDE_ALIASES", "STABLE_SDE_LEGACY_ALIASES"]
