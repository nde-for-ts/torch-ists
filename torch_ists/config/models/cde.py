"""CDE model configs for the runtime package."""

from typing import Dict


CDE_MODELS: Dict[str, Dict] = {
    "neuralcde": {"description": "NeuralCDE (natural)", "handler": "CDEHandler"},
    "neuralcde-l": {"description": "NeuralCDE (linear)", "handler": "CDEHandler"},
    "neuralcde-r": {"description": "NeuralCDE (rectilinear)", "handler": "CDEHandler"},
    "neuralcde-c": {"description": "NeuralCDE (cubic)", "handler": "CDEHandler"},
    "neuralcde-h": {"description": "NeuralCDE (hermite)", "handler": "CDEHandler"},
}


__all__ = ["CDE_MODELS"]
