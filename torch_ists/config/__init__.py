"""Public configuration interface for the packaged runtime."""

from .models import (
    MODEL_REGISTRY,
    MODEL_LIST,
    STABLE_SDE_CONFIGS,
    STABLE_SDE_MODELS,
    BASIC_MODELS,
    ODE_MODELS,
    ODE_SOURCE_MODELS,
    CDE_MODELS,
    FLOW_MODELS,
    SDE_BASELINE_MODELS,
    VARIATIONAL_SDE_MODELS,
    ATTENTION_MODELS,
    BASELINE_MODEL_LIST,
)

from .schema import (
    DriftConfig,
    DiffusionConfig,
    SolverConfig,
    DataConfig,
    ExtendedConfig,
    ModelConfig,
    DEFAULT_DRIFT,
    DEFAULT_DIFFUSION,
    DEFAULT_SOLVER_ODE,
    DEFAULT_SOLVER_CDE,
    DEFAULT_SOLVER_SDE,
    DEFAULT_DATA,
    DEFAULT_EXTENDED,
    validate_drift_config,
    validate_diffusion_config,
    validate_solver_config,
    validate_data_config,
    validate_extended_config,
    validate_model_config,
    input_option_to_drift,
    noise_option_to_diffusion,
    resolve_drift_to_numeric,
    resolve_diffusion_to_numeric,
    merge_configs,
    get_default_config,
)

from .model_data_policy import (
    get_data_config,
    has_auxiliary_loss,
    resolve_layer_depth,
)
from .model_scope import (
    get_active_model_list,
    ACTIVE_MODELS,
)
from .dataset_catalog import (
    get_default_selected_datasets,
)


_SOURCE_BASED_DEFAULT_METHODS = {
    "naive_cde": "srk",
    "neural_ode": "srk",
    "neural_ode_t": "srk",
    "neural_sde": "srk",
    "neural_sde_t": "srk",
    "neural_lsde": "srk",
    "neural_lnsde": "srk",
    "neural_gsde": "srk",
    "ode-rnn-spline": "rk4",
    "ode-rnn": "euler",
    "latentode": "dopri5",
    "gru-ode-bayes": "euler",
    "ode-lstm": "dopri5",
    "neuralcde": "rk4",
    "neuralcde-l": "rk4",
    "neuralcde-r": "rk4",
    "neuralcde-c": "rk4",
    "neuralcde-h": "rk4",
    "flowpath-r": "rk4",
    "flowpath-g": "rk4",
    "flowpath-c": "rk4",
    "dualdynamics-r": "rk4",
    "dualdynamics-g": "rk4",
    "dualdynamics-c": "rk4",
}


def default_method_for_model(model_name: str) -> str:
    """Return the packaged-runtime default method for supported families."""
    normalized = str(model_name).strip().lower()
    return _SOURCE_BASED_DEFAULT_METHODS.get(normalized, "euler")


__all__ = [
    "MODEL_REGISTRY",
    "MODEL_LIST",
    "resolve_drift_to_numeric",
    "resolve_diffusion_to_numeric",
    "merge_configs",
    "get_default_config",
    "get_data_config",
    "has_auxiliary_loss",
    "resolve_layer_depth",
    "get_active_model_list",
    "get_default_selected_datasets",
    "default_method_for_model",
    "ODE_SOURCE_MODELS",
    "FLOW_MODELS",
]
