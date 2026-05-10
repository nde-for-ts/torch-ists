"""CDE method modules for the runtime package.

Core NeuralCDE paths are available in the packaged runtime.
"""

import sys as _sys

from . import _controldiffeq as controldiffeq
from .neuralcde_variants import FinalTanh, FinalTanhT, NeuralCDE

_sys.modules.setdefault(__name__ + ".controldiffeq", controldiffeq)

__all__ = ["controldiffeq", "FinalTanh", "FinalTanhT", "NeuralCDE"]
