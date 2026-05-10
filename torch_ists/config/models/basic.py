"""Basic model configs for the runtime package."""

from typing import Dict


BASIC_MODELS: Dict[str, Dict] = {
    "cnn": {"description": "Convolutional Neural Network", "handler": "BasicHandler"},
    "rnn": {"description": "Recurrent Neural Network", "handler": "BasicHandler"},
    "lstm": {"description": "Long Short-Term Memory", "handler": "BasicHandler"},
    "gru": {"description": "Gated Recurrent Unit", "handler": "BasicHandler"},
    "gru-simple": {"description": "Simplified GRU", "handler": "BasicHandler"},
    "bilstm": {"description": "Bidirectional LSTM", "handler": "BasicHandler"},
    "tlstm": {"description": "Time-Aware LSTM", "handler": "BasicHandler"},
    "plstm": {"description": "Phased LSTM", "handler": "BasicHandler"},
    "tglstm": {"description": "Time-Gated LSTM", "handler": "BasicHandler"},
}


__all__ = ["BASIC_MODELS"]
