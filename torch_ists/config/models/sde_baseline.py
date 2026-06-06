"""Continuous-time baseline model configs for the runtime package."""

from typing import Dict


SDE_BASELINE_MODELS: Dict[str, Dict] = {
    "naive_cde": {
        "drift": {"control": True, "latent": False, "time": False, "geometric": False},
        "diffusion": {"network": "none", "input": None, "type": None},
        "description": "Control-only ODE via SDE framework",
        "handler": "SDEHandler",
    },
    "neural_ode": {
        "drift": {"control": False, "latent": True, "time": False, "geometric": False},
        "diffusion": {"network": "none", "input": None, "type": None},
        "description": "Neural ODE",
        "handler": "SDEHandler",
    },
    "neural_ode_t": {
        "drift": {"control": False, "latent": True, "time": True, "geometric": False},
        "diffusion": {"network": "none", "input": None, "type": None},
        "description": "Neural ODE with time embedding",
        "handler": "SDEHandler",
    },
    "neural_sde": {
        "drift": {"control": False, "latent": True, "time": False, "geometric": False},
        "diffusion": {"network": "nonlinear", "input": "ty", "type": "additive"},
        "description": "Neural SDE",
        "handler": "SDEHandler",
    },
    "neural_sde_t": {
        "drift": {"control": False, "latent": True, "time": True, "geometric": False},
        "diffusion": {"network": "nonlinear", "input": "ty", "type": "additive"},
        "description": "Neural SDE with time embedding",
        "handler": "SDEHandler",
    },
}


__all__ = ["SDE_BASELINE_MODELS"]
