"""Flow handler model configs for packaged runtime."""

from typing import Dict


FLOW_MODELS: Dict[str, Dict] = {
    "neuralflow-r": {
        "description": "NeuralFlow (residual flow)",
        "handler": "FlowHandler",
        "method_family": "neuralflow",
        "flow_operator": "residual",
        "flow_backend": "auto",
    },
    "neuralflow-g": {
        "description": "NeuralFlow (GRU-style flow)",
        "handler": "FlowHandler",
        "method_family": "neuralflow",
        "flow_operator": "gru_style",
        "flow_backend": "auto",
    },
    "neuralflow-c": {
        "description": "NeuralFlow (coupling-style flow)",
        "handler": "FlowHandler",
        "method_family": "neuralflow",
        "flow_operator": "coupling_style",
        "flow_backend": "auto",
    },
    "flowpath-r": {
        "description": "FlowPath (paper-aligned learned path with residual flow operator)",
        "handler": "FlowHandler",
        "method_family": "flowpath",
        "flow_operator": "residual",
        "flow_backend": "auto",
        "coefficient_input": "accepted_but_unused",
    },
    "flowpath-g": {
        "description": "FlowPath (paper-aligned learned path with GRU-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "flowpath",
        "flow_operator": "gru_style",
        "flow_backend": "auto",
        "coefficient_input": "accepted_but_unused",
    },
    "flowpath-c": {
        "description": "FlowPath (paper-aligned learned path with coupling-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "flowpath",
        "flow_operator": "coupling_style",
        "flow_backend": "auto",
        "coefficient_input": "accepted_but_unused",
    },
    "fusioncde-r": {
        "description": "FusionCDE (coefficient-augmented residual flow and CDE control path)",
        "handler": "FlowHandler",
        "method_family": "fusioncde",
        "flow_operator": "residual",
        "flow_backend": "auto",
    },
    "fusioncde-g": {
        "description": "FusionCDE (coefficient-augmented GRU-style flow and CDE control path)",
        "handler": "FlowHandler",
        "method_family": "fusioncde",
        "flow_operator": "gru_style",
        "flow_backend": "auto",
    },
    "fusioncde-c": {
        "description": "FusionCDE (coefficient-augmented coupling flow and CDE control path)",
        "handler": "FlowHandler",
        "method_family": "fusioncde",
        "flow_operator": "coupling_style",
        "flow_backend": "auto",
    },
    "dualdynamics-r": {
        "description": "DualDynamics (CDE control path with residual flow operator)",
        "handler": "FlowHandler",
        "method_family": "flow_backed_cde_control_path",
        "flow_operator": "residual",
        "flow_backend": "auto",
    },
    "dualdynamics-g": {
        "description": "DualDynamics (CDE control path with GRU-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "flow_backed_cde_control_path",
        "flow_operator": "gru_style",
        "flow_backend": "auto",
    },
    "dualdynamics-c": {
        "description": "DualDynamics (CDE control path with coupling-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "flow_backed_cde_control_path",
        "flow_operator": "coupling_style",
        "flow_backend": "auto",
    },
}


__all__ = ["FLOW_MODELS"]
