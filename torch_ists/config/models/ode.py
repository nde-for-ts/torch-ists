"""ODE model configs for the runtime package."""

from typing import Dict


ODE_MODELS: Dict[str, Dict] = {
    "gru-dt": {"description": "GRU with time deltas", "handler": "ODEHandler"},
    "gru-d": {"description": "GRU-D", "handler": "ODEHandler"},
    "gru-ode": {"description": "GRU-ODE", "handler": "ODEHandler"},
    "ode-rnn-spline": {"description": "ODE-RNN (spline runtime)", "handler": "ODEHandler"},
    "ode-lstm": {"description": "ODE-LSTM", "handler": "ODELSTMHandler"},
}

ODE_SOURCE_MODELS: Dict[str, Dict] = {
    "ode-rnn": {"description": "ODE-RNN (reference implementation)", "handler": "SourceODEHandler"},
    "latentode": {"description": "Latent ODE (reference implementation)", "handler": "SourceODEHandler"},
    "gru-ode-bayes": {
        "description": "GRU-ODE-Bayes (reference implementation)",
        "handler": "SourceODEHandler",
    },
}

# Historical import name retained for older callers.
ODE_EXPERIMENT_CONFIGS = ODE_SOURCE_MODELS

__all__ = ["ODE_MODELS", "ODE_SOURCE_MODELS", "ODE_EXPERIMENT_CONFIGS"]
