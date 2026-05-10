"""CDE method modules for the runtime package.

Core NeuralCDE paths are always available. Optional ANCDE, EXIT, and LEAP
extensions may be unavailable if their dependencies are not installed.
"""

import sys as _sys

from . import _controldiffeq as controldiffeq
from .neuralcde_variants import FinalTanh, FinalTanhT, NeuralCDE

_sys.modules.setdefault(__name__ + ".controldiffeq", controldiffeq)

__all__ = ["controldiffeq", "FinalTanh", "FinalTanhT", "NeuralCDE"]

try:
    from .ancde.ancde_model import ANCDE
    from .ancde.vector_fields import FinalTanh as ANCDEFinalTanh
    from .ancde.vector_fields import FinalTanh_ff6 as ANCDEFinalTanhFF6

    __all__.extend(["ANCDE", "ANCDEFinalTanh", "ANCDEFinalTanhFF6"])
except ModuleNotFoundError:
    pass

try:
    from .exit.exit_model import NeuralCDE_IDEA4
    from .exit.vector_fields import FinalTanh_g as EXITFinalTanhG
    from .exit.vector_fields import ODEFunc_f as EXITODEFuncF

    __all__.extend(["NeuralCDE_IDEA4", "EXITFinalTanhG", "EXITODEFuncF"])
except ModuleNotFoundError:
    pass

try:
    from .leap.leap_model import NeuralCDE_Learnable
    from .leap.vector_fields import Mapping_f as LEAPMappingF
    from .leap.vector_fields import ODEFunc_ff as LEAPODEFuncFF

    __all__.extend(["NeuralCDE_Learnable", "LEAPODEFuncFF", "LEAPMappingF"])
except ModuleNotFoundError:
    pass
