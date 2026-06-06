"""Flow handler model configs for packaged runtime."""

from typing import Dict


FLOW_MODELS: Dict[str, Dict] = {
    "neuralflow-r": {
        "description": "NeuralFlow (residual flow)",
        "handler": "FlowHandler",
        "method_family": "neuralflow",
        "flow_operator": "residual",
    },
    "neuralflow-g": {
        "description": "NeuralFlow (GRU-style flow)",
        "handler": "FlowHandler",
        "method_family": "neuralflow",
        "flow_operator": "gru_style",
    },
    "neuralflow-c": {
        "description": "NeuralFlow (coupling-style flow)",
        "handler": "FlowHandler",
        "method_family": "neuralflow",
        "flow_operator": "coupling_style",
    },
    "flowpath-r": {
        "description": "FlowPath (advanced CDE control path, residual flow operator)",
        "handler": "FlowHandler",
        "method_family": "advanced_cde_control_path",
        "flow_operator": "residual",
    },
    "flowpath-g": {
        "description": "FlowPath (advanced CDE control path, GRU-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "advanced_cde_control_path",
        "flow_operator": "gru_style",
    },
    "flowpath-c": {
        "description": "FlowPath (advanced CDE control path, coupling-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "advanced_cde_control_path",
        "flow_operator": "coupling_style",
    },
    "dualdynamics-r": {
        "description": "DualDynamics (advanced CDE control path, residual flow operator)",
        "handler": "FlowHandler",
        "method_family": "advanced_cde_control_path",
        "flow_operator": "residual",
    },
    "dualdynamics-g": {
        "description": "DualDynamics (advanced CDE control path, GRU-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "advanced_cde_control_path",
        "flow_operator": "gru_style",
    },
    "dualdynamics-c": {
        "description": "DualDynamics (advanced CDE control path, coupling-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "advanced_cde_control_path",
        "flow_operator": "coupling_style",
    },
}


__all__ = ["FLOW_MODELS"]
