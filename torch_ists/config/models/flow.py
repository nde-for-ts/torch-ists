"""Flow handler model configs for packaged runtime."""

from typing import Dict


FLOW_MODELS: Dict[str, Dict] = {
    "neuralflow-r": {
        "description": "NeuralFlow (residual flow)",
        "handler": "FlowHandler",
        "method_family": "neuralflow",
        "flow_operator": "residual",
        "flow_backend": "native",
    },
    "neuralflow-g": {
        "description": "NeuralFlow (GRU-style flow)",
        "handler": "FlowHandler",
        "method_family": "neuralflow",
        "flow_operator": "gru_style",
        "flow_backend": "native",
    },
    "neuralflow-c": {
        "description": "NeuralFlow (coupling-style flow)",
        "handler": "FlowHandler",
        "method_family": "neuralflow",
        "flow_operator": "coupling_style",
        "flow_backend": "native",
    },
    "flowpath-r": {
        "description": "FlowPath (paper-aligned learned path with residual flow operator)",
        "handler": "FlowHandler",
        "method_family": "flowpath",
        "flow_operator": "residual",
        "flow_backend": "native",
        "coefficient_input": "accepted_but_unused",
    },
    "flowpath-g": {
        "description": "FlowPath (paper-aligned learned path with GRU-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "flowpath",
        "flow_operator": "gru_style",
        "flow_backend": "native",
        "coefficient_input": "accepted_but_unused",
    },
    "flowpath-c": {
        "description": "FlowPath (paper-aligned learned path with coupling-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "flowpath",
        "flow_operator": "coupling_style",
        "flow_backend": "native",
        "coefficient_input": "accepted_but_unused",
    },
    "fusioncde-r": {
        "description": "FusionCDE (coefficient-augmented residual flow and CDE control path)",
        "handler": "FlowHandler",
        "method_family": "fusioncde",
        "flow_operator": "residual",
        "flow_backend": "native",
    },
    "fusioncde-g": {
        "description": "FusionCDE (coefficient-augmented GRU-style flow and CDE control path)",
        "handler": "FlowHandler",
        "method_family": "fusioncde",
        "flow_operator": "gru_style",
        "flow_backend": "native",
    },
    "fusioncde-c": {
        "description": "FusionCDE (coefficient-augmented coupling flow and CDE control path)",
        "handler": "FlowHandler",
        "method_family": "fusioncde",
        "flow_operator": "coupling_style",
        "flow_backend": "native",
    },
    "dualdynamics-r": {
        "description": "DualDynamics (CDE control path with residual flow operator)",
        "handler": "FlowHandler",
        "method_family": "flow_backed_cde_control_path",
        "flow_operator": "residual",
        "flow_backend": "native",
    },
    "dualdynamics-g": {
        "description": "DualDynamics (CDE control path with GRU-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "flow_backed_cde_control_path",
        "flow_operator": "gru_style",
        "flow_backend": "native",
    },
    "dualdynamics-c": {
        "description": "DualDynamics (CDE control path with coupling-style flow operator)",
        "handler": "FlowHandler",
        "method_family": "flow_backed_cde_control_path",
        "flow_operator": "coupling_style",
        "flow_backend": "native",
    },
}


__all__ = ["FLOW_MODELS"]
