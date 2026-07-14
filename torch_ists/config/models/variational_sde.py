"""Variational SDE model configs for the packaged runtime."""

from typing import Dict


VARIATIONAL_SDE_MODELS: Dict[str, Dict] = {
    "latentsde": {
        "description": "Latent SDE with KL regularization",
        "handler": "VariationalSDEHandler",
    }
}


__all__ = ["VARIATIONAL_SDE_MODELS"]
