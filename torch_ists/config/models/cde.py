"""CDE model configs for the runtime package."""

from typing import Dict


CDE_MODELS: Dict[str, Dict] = {
    "neuralcde": {"description": "NeuralCDE (natural)", "handler": "CDEHandler"},
    "neuralcde-l": {"description": "NeuralCDE (linear)", "handler": "CDEHandler"},
    "neuralcde-r": {"description": "NeuralCDE (rectilinear)", "handler": "CDEHandler"},
    "neuralcde-c": {"description": "NeuralCDE (cubic)", "handler": "CDEHandler"},
    "neuralcde-h": {"description": "NeuralCDE (hermite)", "handler": "CDEHandler"},
    "neuralrde-1": {"description": "Neural RDE (depth 1)", "handler": "CDEHandler"},
    "neuralrde-2": {"description": "Neural RDE (depth 2)", "handler": "CDEHandler"},
    "neuralrde-3": {"description": "Neural RDE (depth 3)", "handler": "CDEHandler"},
    "ancde": {"description": "ANCDE", "handler": "CDEHandler"},
    "exit": {"description": "EXIT", "handler": "CDEHandler"},
    "leap": {"description": "LEAP", "handler": "CDEHandler"},
}


__all__ = ["CDE_MODELS"]
