"""Runtime model configuration schema: types, defaults, validation, conversion, helpers."""

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict, Union


# ── Types ────────────────────────────────────────────────────────────────────

class DriftConfig(TypedDict, total=False):
    control: bool
    latent: bool
    time: bool
    geometric: bool


DiffusionNetwork = Literal["none", "constant", "diagonal", "special", "linear", "nonlinear"]
DiffusionInput = Literal["none", "t", "y", "ty"]
DiffusionType = Literal[
    "additive",
    "multiplicative",
    "scalar",
    "diag",
    "sqrt",
    "cubic",
    "sigmoid",
    "relu",
    "mixed",
]


class DiffusionConfig(TypedDict, total=False):
    network: DiffusionNetwork
    input: DiffusionInput
    type: DiffusionType


class SolverConfig(TypedDict, total=False):
    method: str
    dt: Union[float, Literal["auto"]]
    adaptive: bool
    rtol: Optional[float]
    atol: Optional[float]
    adjoint: Optional[bool]
    noise_type: Literal["scalar", "diagonal", "general"]


class DataConfig(TypedDict, total=False):
    interpolation: Literal["natural", "linear", "rectilinear", "cubic", "hermite"]
    use_intensity: bool


class ExtendedConfig(TypedDict, total=False):
    initial: Literal["zero", "random", "data", "control"]
    delay: int
    attention: int
    flow_input: Literal["n", "x", "y", "z"]
    flow_type: Literal["n", "r", "g", "c"]
    constraint: Literal["none", "soft", "proj", "correction", "spherical", "full"]


class ModelConfig(TypedDict, total=False):
    drift: DriftConfig
    diffusion: DiffusionConfig
    solver: SolverConfig
    data: DataConfig
    extended: ExtendedConfig


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_DRIFT: DriftConfig = {
    "control": False,
    "latent": True,
    "time": False,
    "geometric": False,
}

DEFAULT_DIFFUSION: DiffusionConfig = {
    "network": "none",
    "input": "none",
    "type": "additive",
}

# Reference defaults for schema/config callers only.
# ODEHandler and CDEHandler do not auto-merge these into runtime models; they
# resolve the method and let dt/tolerance behavior fall through to model- or
# handler-specific logic.
DEFAULT_SOLVER_ODE: SolverConfig = {
    "method": "euler",
    "dt": 1e-3,
    "adaptive": False,
    "rtol": 1e-4,
    "atol": 1e-5,
    "adjoint": False,
}

DEFAULT_SOLVER_CDE: SolverConfig = {
    "method": "euler",
    "dt": 1e-3,
    "adaptive": False,
    "rtol": None,
    "atol": None,
    "adjoint": False,
}

DEFAULT_SOLVER_SDE: SolverConfig = {
    "method": "srk",
    "dt": "auto",
    "adaptive": False,
    "rtol": None,
    "atol": None,
    "adjoint": None,
    "noise_type": "scalar",
}

DEFAULT_DATA: DataConfig = {
    "interpolation": "natural",
    "use_intensity": True,
}

DEFAULT_EXTENDED: ExtendedConfig = {
    "initial": "control",
    "delay": 0,
    "attention": 0,
    "flow_input": "z",
    "flow_type": "n",
    "constraint": "correction",
}

DIFFUSION_NETWORK_VALUES = {"none", "constant", "diagonal", "special", "linear", "nonlinear"}
DIFFUSION_INPUT_VALUES = {"none", "t", "y", "ty"}
DIFFUSION_TYPE_VALUES = {
    "additive",
    "multiplicative",
    "scalar",
    "diag",
    "sqrt",
    "cubic",
    "sigmoid",
    "relu",
    "mixed",
}
DIFFUSION_TO_NUMERIC: Dict[Tuple[str, Optional[str], Optional[str]], int] = {
    ("none", None, None): 0,
    ("constant", None, "scalar"): 1,
    ("constant", "t", "scalar"): 2,
    ("constant", "y", "scalar"): 3,
    ("diagonal", None, "diag"): 4,
    ("diagonal", "t", "diag"): 5,
    ("diagonal", "y", "diag"): 6,
    ("special", "y", "sqrt"): 7,
    ("special", "y", "cubic"): 8,
    ("special", "y", "sigmoid"): 9,
    ("special", "y", "relu"): 10,
    ("special", "ty", "mixed"): 11,
    ("linear", "t", "additive"): 12,
    ("linear", "t", "multiplicative"): 13,
    ("linear", "ty", "additive"): 14,
    ("linear", "ty", "multiplicative"): 15,
    ("nonlinear", "t", "additive"): 16,
    ("nonlinear", "t", "multiplicative"): 17,
    ("nonlinear", "ty", "additive"): 18,
    ("nonlinear", "ty", "multiplicative"): 19,
}
NUMERIC_TO_DIFFUSION = {value: key for key, value in DIFFUSION_TO_NUMERIC.items()}


# ── Validation ───────────────────────────────────────────────────────────────

def validate_drift_config(drift: Dict[str, Any]) -> bool:
    if not isinstance(drift, dict):
        return False
    required_keys = {"control", "latent", "time", "geometric"}
    if not required_keys.issubset(drift.keys()):
        return False
    for key in required_keys:
        if not isinstance(drift[key], bool):
            return False
    if drift.get("geometric", False) and not drift.get("time", False):
        return False
    return True


def validate_diffusion_config(diffusion: Dict[str, Any]) -> bool:
    if not isinstance(diffusion, dict):
        return False
    if "network" not in diffusion:
        return False
    if diffusion["network"] not in DIFFUSION_NETWORK_VALUES:
        return False
    if diffusion["network"] == "none":
        return True
    if "input" in diffusion:
        if diffusion["input"] not in DIFFUSION_INPUT_VALUES and diffusion["input"] is not None:
            return False
    if "type" in diffusion:
        if diffusion["type"] not in DIFFUSION_TYPE_VALUES and diffusion["type"] is not None:
            return False
    return True


def validate_solver_config(solver: Dict[str, Any]) -> bool:
    if not isinstance(solver, dict):
        return False
    if "method" not in solver:
        return False
    if not isinstance(solver["method"], str):
        return False
    if "dt" in solver:
        dt = solver["dt"]
        if dt != "auto" and not isinstance(dt, (int, float)):
            return False
        if isinstance(dt, (int, float)) and dt <= 0:
            return False
    for bool_field in ["adaptive", "adjoint"]:
        if bool_field in solver and solver[bool_field] is not None and not isinstance(solver[bool_field], bool):
            return False
    for tol_field in ["rtol", "atol"]:
        if tol_field in solver:
            if solver[tol_field] is None:
                continue
            if not isinstance(solver[tol_field], (int, float)):
                return False
            if solver[tol_field] <= 0:
                return False
    if "noise_type" in solver:
        if solver["noise_type"] not in {"scalar", "diagonal", "general"}:
            return False
    return True


def validate_data_config(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    if "interpolation" in data:
        if data["interpolation"] not in {"natural", "linear", "rectilinear", "cubic", "hermite"}:
            return False
    if "use_intensity" in data:
        if not isinstance(data["use_intensity"], bool):
            return False
    return True


def validate_extended_config(extended: Dict[str, Any]) -> bool:
    if not isinstance(extended, dict):
        return False
    if "initial" in extended:
        if extended["initial"] not in {"zero", "random", "data", "control"}:
            return False
    if "delay" in extended:
        if not isinstance(extended["delay"], int):
            return False
        if not (0 <= extended["delay"] <= 4):
            return False
    if "attention" in extended:
        if not isinstance(extended["attention"], int):
            return False
        if not (0 <= extended["attention"] <= 11):
            return False
    if "flow_input" in extended:
        if extended["flow_input"] not in {"n", "x", "y", "z"}:
            return False
    if "flow_type" in extended:
        if extended["flow_type"] not in {"n", "r", "g", "c"}:
            return False
    if "constraint" in extended:
        if extended["constraint"] not in {"none", "soft", "proj", "correction", "spherical", "full"}:
            return False
    return True


def validate_model_config(config: Dict[str, Any]) -> bool:
    if not isinstance(config, dict):
        return False
    if "drift" in config:
        if not validate_drift_config(config["drift"]):
            return False
    if "diffusion" in config:
        if not validate_diffusion_config(config["diffusion"]):
            return False
    if "solver" in config:
        if not validate_solver_config(config["solver"]):
            return False
    if "data" in config:
        if not validate_data_config(config["data"]):
            return False
    if "extended" in config:
        if not validate_extended_config(config["extended"]):
            return False
    return True


# ── Conversion ───────────────────────────────────────────────────────────────

def resolve_drift_to_numeric(drift: DriftConfig) -> int:
    c = drift.get("control", False)
    l = drift.get("latent", True)
    t = drift.get("time", False)
    g = drift.get("geometric", False)

    if c and not l:
        return 0
    if not c and l and not t and not g:
        return 1
    if c and l and not t and not g:
        return 2
    if not c and l and t and not g:
        return 3
    if c and l and t and not g:
        return 4
    if not c and l and t and g:
        return 5
    if c and l and t and g:
        return 6
    return 2


def _normalize_diffusion_input(value: Any) -> Optional[str]:
    if value in (None, "none"):
        return None
    return str(value)


def _normalize_diffusion_key(
    diffusion: DiffusionConfig,
) -> Tuple[str, Optional[str], Optional[str]]:
    net = str(diffusion.get("network", "none"))
    if net == "none":
        return ("none", None, None)
    inp = _normalize_diffusion_input(diffusion.get("input"))
    typ = diffusion.get("type")
    normalized_type = None if typ is None else str(typ)
    return (net, inp, normalized_type)


def resolve_diffusion_to_numeric(diffusion: DiffusionConfig) -> int:
    key = _normalize_diffusion_key(diffusion)
    return DIFFUSION_TO_NUMERIC.get(key, 0)


def input_option_to_drift(input_option: int) -> DriftConfig:
    mapping = {
        0: {"control": True, "latent": False, "time": False, "geometric": False},
        1: {"control": False, "latent": True, "time": False, "geometric": False},
        2: {"control": True, "latent": True, "time": False, "geometric": False},
        3: {"control": False, "latent": True, "time": True, "geometric": False},
        4: {"control": True, "latent": True, "time": True, "geometric": False},
        5: {"control": False, "latent": True, "time": True, "geometric": True},
        6: {"control": True, "latent": True, "time": True, "geometric": True},
    }
    return mapping.get(input_option, mapping[2])


def noise_option_to_diffusion(noise_option: int) -> DiffusionConfig:
    net, inp, typ = NUMERIC_TO_DIFFUSION.get(noise_option, NUMERIC_TO_DIFFUSION[0])
    if net == "none":
        return {"network": "none"}
    result: DiffusionConfig = {"network": net}
    result["input"] = "none" if inp is None else inp
    if typ is not None:
        result["type"] = typ
    return result


# ── Helpers ──────────────────────────────────────────────────────────────────

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_default_config(model_type: str = "sde") -> ModelConfig:
    solver_defaults = {
        "ode": DEFAULT_SOLVER_ODE,
        "cde": DEFAULT_SOLVER_CDE,
        "sde": DEFAULT_SOLVER_SDE,
    }
    return {
        "drift": deepcopy(DEFAULT_DRIFT),
        "diffusion": deepcopy(DEFAULT_DIFFUSION),
        "solver": deepcopy(solver_defaults.get(model_type, DEFAULT_SOLVER_SDE)),
        "data": deepcopy(DEFAULT_DATA),
        "extended": deepcopy(DEFAULT_EXTENDED),
    }


__all__ = [
    "DriftConfig", "DiffusionConfig", "SolverConfig", "DataConfig",
    "ExtendedConfig", "ModelConfig",
    "DEFAULT_DRIFT", "DEFAULT_DIFFUSION", "DEFAULT_SOLVER_ODE",
    "DEFAULT_SOLVER_CDE", "DEFAULT_SOLVER_SDE", "DEFAULT_DATA", "DEFAULT_EXTENDED",
    "validate_drift_config", "validate_diffusion_config", "validate_solver_config",
    "validate_data_config", "validate_extended_config", "validate_model_config",
    "resolve_drift_to_numeric", "resolve_diffusion_to_numeric",
    "input_option_to_drift", "noise_option_to_diffusion",
    "merge_configs", "get_default_config",
]
