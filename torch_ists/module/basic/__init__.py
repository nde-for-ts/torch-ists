"""Basic model backbones.

Optional imports remain model-scoped in the packaged runtime.
"""

__all__ = []

try:
    from .cnn import CNNBackbone
    __all__.append("CNNBackbone")
except Exception:  # pragma: no cover - model-scoped optional import
    CNNBackbone = None

try:
    from .rnn import RNNBackbone
    __all__.append("RNNBackbone")
except Exception:  # pragma: no cover - model-scoped optional import
    RNNBackbone = None

try:
    from .transformer import TransformerBackbone
    __all__.append("TransformerBackbone")
except Exception:  # pragma: no cover - model-scoped optional import
    TransformerBackbone = None

try:
    from .plstm import PLSTM
    __all__.append("PLSTM")
except Exception:  # pragma: no cover - model-scoped optional import
    PLSTM = None

try:
    from .tlstm import TLSTM
    __all__.append("TLSTM")
except Exception:  # pragma: no cover - model-scoped optional import
    TLSTM = None

try:
    from .tglstm import TGLSTM
    __all__.append("TGLSTM")
except Exception:  # pragma: no cover - model-scoped optional import
    TGLSTM = None
