"""Compatibility shim for split packaged flow modules.

Canonical modules:
- operators: ``torch_ists.module.flow.operators*``
- neuralflow: ``torch_ists.module.flow.neuralflow``
- flowpath: ``torch_ists.module.flow.flowpath``
- dualdynamics: ``torch_ists.module.flow.dualdynamics``

This shim is kept for one transition cycle to preserve import compatibility.
"""

from .dualdynamics import DualDynamics
from .flowpath import FlowPath
from .neuralflow import NeuralFlow
from .operators_native import CouplingFlow, GRUFlow, ResNetFlow

__all__ = [
    "ResNetFlow",
    "GRUFlow",
    "CouplingFlow",
    "NeuralFlow",
    "FlowPath",
    "DualDynamics",
]
