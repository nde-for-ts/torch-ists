"""Flow handler model configs for packaged runtime."""

from typing import Dict


FLOW_MODELS: Dict[str, Dict] = {
    "neuralflow-r": {"description": "NeuralFlow (rectilinear)", "handler": "FlowHandler"},
    "neuralflow-g": {"description": "NeuralFlow (gaussian)", "handler": "FlowHandler"},
    "neuralflow-c": {"description": "NeuralFlow (cubic)", "handler": "FlowHandler"},
    "flowpath-r": {"description": "FlowPath (rectilinear)", "handler": "FlowHandler"},
    "flowpath-g": {"description": "FlowPath (gaussian)", "handler": "FlowHandler"},
    "flowpath-c": {"description": "FlowPath (cubic)", "handler": "FlowHandler"},
    "dualdynamics-r": {"description": "DualDynamics (rectilinear)", "handler": "FlowHandler"},
    "dualdynamics-g": {"description": "DualDynamics (gaussian)", "handler": "FlowHandler"},
    "dualdynamics-c": {"description": "DualDynamics (cubic)", "handler": "FlowHandler"},
}


__all__ = ["FLOW_MODELS"]
