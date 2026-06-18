from pathlib import Path
import inspect
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from _common import (
    build_model_kwargs_from_config,
    describe_search_profile,
    infer_param_search_profile,
    normalize_search_profile,
    resolve_effective_param_config,
    resolve_model_search_space,
    validate_param_artifact_metadata,
)
from param_search import (
    _build_param_artifact_name as _build_search_param_artifact_name,
    _derive_trial_seed,
    tune_model,
)
from model_run import (
    _build_param_artifact_name as _build_run_param_artifact_name,
    _infer_cli_search_repeat as _infer_run_cli_search_repeat,
    _resolve_param_artifact_path,
)
from torch_ists import ists_classifier
from torch_ists.config import get_active_model_list


def _complete_param_config(**overrides):
    cfg = {
        "search_profile": "model-native-v1",
        "search_contract": "model-specific-default-v1",
        "search_family": "default",
        "public_search_name": "default model-specific search",
        "search_profile_visibility": "internal",
        "search_policy": "default",
        "search_repeat": "single",
        "search_algorithm_policy": "model-native-effective-params-v1",
        "architecture_axes": {"candidate_axes": ["lr", "hidden_dim", "num_layers"]},
        "effective_params": {"hidden_dim": 64, "num_layers": 2},
        "lr": 1e-3,
    }
    cfg.update(overrides)
    return cfg


def test_search_profile_aliases_and_grids():
    assert normalize_search_profile("default") == "model-native-v1"
    assert normalize_search_profile("model-specific") == "model-native-v1"
    assert normalize_search_profile("legacy") == "legacy-expanded-v1"
    assert normalize_search_profile("iclr2024") == "legacy-iclr2024-v0"
    assert describe_search_profile("default") == {
        "search_contract": "model-specific-default-v1",
        "search_family": "default",
        "search_profile_visibility": "internal",
        "public_search_name": "default model-specific search",
    }
    assert describe_search_profile("legacy") == {
        "search_contract": "model-specific-default-v1",
        "search_family": "internal_reproducibility",
        "search_profile_visibility": "internal",
        "public_search_name": "expanded internal search",
    }
    assert describe_search_profile("iclr2024") == {
        "search_contract": "historical-reproducibility-v1",
        "search_family": "historical_reproducibility",
        "search_profile_visibility": "internal",
        "public_search_name": "historical reproducibility",
    }

    native = resolve_model_search_space("rnn", "default")
    assert native["search_profile"] == "model-native-v1"
    assert native["hidden_dim_values"] == [32, 64, 96, 128, 192, 256]
    assert native["depth_level_values"] == [1, 2, 3, 4, 5]
    assert native["candidate_axes"] == ["lr", "hidden_dim", "num_layers"]

    expanded = resolve_model_search_space("rnn", "legacy-expanded-v1")
    assert expanded["hidden_dim_values"] == [32, 64, 96, 128, 192, 256]
    assert expanded["depth_level_values"] == [1, 2, 3, 4, 5]

    legacy = resolve_model_search_space("rnn", "legacy-iclr2024-v0")
    assert legacy["hidden_dim_values"] == [16, 32, 64, 128]
    assert legacy["depth_level_values"] == [1, 2, 3, 4]


def test_public_active_models_resolve_effective_config():
    models = get_active_model_list()
    assert len(models) == 50
    for model_name in models:
        space = resolve_model_search_space(model_name, "default")
        assert space["search_profile"] == "model-native-v1"
        assert space["hidden_dim_values"]
        assert space["depth_level_values"]
        assert space["candidate_axes"]

        cfg = resolve_effective_param_config(
            model_name,
            {
                "model_name": model_name,
                "lr": 1e-3,
                "hidden_dim": 64,
                "depth_level": space["depth_level_values"][-1],
            },
            search_profile="default",
        )
        assert cfg["search_policy"] == "default"
        assert cfg["search_repeat"] == "single"
        assert cfg["search_algorithm_policy"] == "model-native-effective-params-v1"
        assert cfg["search_contract"] == "model-specific-default-v1"
        assert cfg["search_family"] == "default"
        assert cfg["search_profile_visibility"] == "internal"
        assert cfg["search_grid_version"] == "model-native-v1"
        assert cfg["legacy_grid_version"] is None
        assert cfg["architecture_axes"]["width_axis"] in cfg["architecture_axes"]["native_axes"]
        assert cfg["architecture_axes"]["search_algorithm_policy"] == "model-native-effective-params-v1"
        assert cfg["architecture_axes"]["candidate_axes"]
        assert cfg["effective_params"]["hidden_dim"] == 64
        assert cfg["effective_params"]["depth_level"] in space["depth_level_values"]


def test_model_native_candidate_axes_are_valid_for_all_active_models():
    for model_name in get_active_model_list():
        space = resolve_model_search_space(model_name, "default")
        axes = set(space["candidate_axes"])
        native_axes = set(space["native_axes"])
        fixed_axes = set(space["fixed_axes"])
        excluded_axes = set(space["excluded_axes"])

        assert "lr" in axes
        assert native_axes
        assert not native_axes.intersection(excluded_axes)
        assert "legacy-expanded-v1" not in space["search_profile"]

        if "model_params.n_heads" in axes:
            n_heads_values = space["extra_param_values"]["n_heads"]
            for hidden_dim in space["hidden_dim_values"]:
                for n_heads in n_heads_values:
                    assert hidden_dim % n_heads == 0

        if "model_params.embed_time" in axes:
            embed_time_values = space["extra_param_values"]["embed_time"]
            n_heads_values = space["extra_param_values"]["n_heads"]
            for embed_time in embed_time_values:
                for n_heads in n_heads_values:
                    assert embed_time % n_heads == 0

        if model_name.startswith(("neuralflow-", "dualdynamics-", "flowpath-")):
            assert "flow_depth" in axes
            assert space["extra_param_values"]["flow_depth"] == [1, 2, 3]
            assert "model_params.flow_backend=native" in fixed_axes


def test_missing_profile_fails_fast_without_explicit_legacy_opt_in():
    legacy_cfg = {"hidden_dim": 32, "num_layers": 1, "lr": 1e-3}
    with pytest.raises(ValueError, match="search_profile"):
        infer_param_search_profile(legacy_cfg)
    with pytest.raises(ValueError, match="search_profile"):
        validate_param_artifact_metadata(legacy_cfg)
    assert (
        validate_param_artifact_metadata(
            legacy_cfg,
            allow_legacy_search_profile=True,
        )
        == "legacy-iclr2024-v0"
    )


def test_param_artifact_metadata_requires_contract_fields_and_lr():
    with pytest.raises(ValueError, match="search-contract field"):
        validate_param_artifact_metadata(_complete_param_config(search_contract=None))
    with pytest.raises(ValueError, match="'lr'"):
        validate_param_artifact_metadata(_complete_param_config(lr=None))
    assert validate_param_artifact_metadata(_complete_param_config()) == "model-native-v1"


def test_ray_trial_init_seed_is_trial_specific_but_stable():
    assert "trial_seed" in inspect.signature(tune_model).parameters
    assert _derive_trial_seed(42, "trial-a") == _derive_trial_seed(42, "trial-a")
    assert _derive_trial_seed(42, "trial-a") != _derive_trial_seed(42, "trial-b")
    assert _derive_trial_seed(_derive_trial_seed(42, "trial-a"), 0) != _derive_trial_seed(
        _derive_trial_seed(42, "trial-a"),
        1,
    )


def test_cli_search_axes_artifact_names():
    assert _build_search_param_artifact_name(
        "BasicMotions",
        "cnn",
        search_policy="default",
        search_repeat="single",
    ) == "BasicMotions_cnn"
    assert _build_search_param_artifact_name(
        "BasicMotions",
        "cnn",
        search_policy="default",
        search_repeat="aggregated",
    ) == "BasicMotions_cnn_aggregated"
    assert _build_search_param_artifact_name(
        "BasicMotions",
        "cnn",
        search_policy="constrained",
        search_repeat="single",
    ) == "BasicMotions_cnn_constrained"
    assert _build_search_param_artifact_name(
        "BasicMotions",
        "cnn",
        search_policy="constrained",
        search_repeat="aggregated",
    ) == "BasicMotions_cnn_constrained_aggregated"
    assert _build_run_param_artifact_name(
        "BasicMotions",
        "cnn",
        param_policy="default",
        param_repeat="aggregated",
    ) == "BasicMotions_cnn_aggregated"


def test_param_artifact_names_are_missingness_independent():
    search_params = inspect.signature(_build_search_param_artifact_name).parameters
    run_params = inspect.signature(_build_run_param_artifact_name).parameters
    assert "missing_rate" not in search_params
    assert "missing_rates" not in search_params
    assert "missing_rate" not in run_params
    assert "missing_rates" not in run_params

    names = [
        _build_search_param_artifact_name(
            "BasicMotions",
            "cnn",
            search_policy="default",
            search_repeat="single",
        ),
        _build_search_param_artifact_name(
            "BasicMotions",
            "cnn",
            search_policy="default",
            search_repeat="aggregated",
        ),
        _build_search_param_artifact_name(
            "BasicMotions",
            "cnn",
            search_policy="constrained",
            search_repeat="single",
        ),
        _build_search_param_artifact_name(
            "BasicMotions",
            "cnn",
            search_policy="constrained",
            search_repeat="aggregated",
        ),
        _build_run_param_artifact_name(
            "BasicMotions",
            "cnn",
            param_policy="default",
            param_repeat="single",
        ),
        _build_run_param_artifact_name(
            "BasicMotions",
            "cnn",
            param_policy="constrained",
            param_repeat="aggregated",
        ),
    ]
    for artifact_name in names:
        assert "missing" not in artifact_name.lower()
        assert "_0.0" not in artifact_name
        assert "_0.3" not in artifact_name


def test_aggregated_param_artifact_compatibility_fallback(tmp_path):
    param_root = tmp_path / "params"
    dataset_dir = param_root / "BasicMotions"
    dataset_dir.mkdir(parents=True)

    canonical = dataset_dir / "BasicMotions_cnn_aggregated"
    canonical.write_bytes(b"new")
    param_name, param_path = _resolve_param_artifact_path(
        str(param_root),
        "BasicMotions",
        "cnn",
        param_policy="default",
        param_repeat="aggregated",
    )
    assert param_name == "BasicMotions_cnn_aggregated"
    assert Path(param_path) == canonical

    canonical.unlink()
    legacy = dataset_dir / "BasicMotions_cnn_cv"
    legacy.write_bytes(b"old")
    param_name, param_path = _resolve_param_artifact_path(
        str(param_root),
        "BasicMotions",
        "cnn",
        param_policy="default",
        param_repeat="aggregated",
    )
    assert param_name == "BasicMotions_cnn_cv"
    assert Path(param_path) == legacy
    assert _infer_run_cli_search_repeat({}, "BasicMotions_cnn_aggregated") == "aggregated"
    assert _infer_run_cli_search_repeat({}, "BasicMotions_cnn_cv") == "aggregated"


def test_public_docs_track_cli_policy_repeat_surface():
    docs = (REPO_ROOT / "docs" / "scripts_reference.md").read_text()
    assert "--search-policy" in docs
    assert "--search-repeat" in docs
    assert "--param-policy" in docs
    assert "--param-repeat" in docs
    assert "| `--missing-rate` |" not in docs
    assert "--search-mode" not in docs
    assert "--param-mode" not in docs
    param_doc_lines = [line for line in docs.splitlines() if "param-root" in line or "Params" in line or "Param file" in line]
    assert param_doc_lines
    assert all("<missing_rate>" not in line for line in param_doc_lines)
    assert all("missing-rate suffix" not in line.lower() for line in param_doc_lines)

    param_help = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / "param_search.py"), "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    run_help = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / "model_run.py"), "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert "--search-policy" in param_help
    assert "{default,constrained}" in param_help
    assert "standard=model-specific" not in param_help
    assert "--search-repeat" in param_help
    assert "--missing-rate" not in param_help
    assert "--search-mode" not in param_help
    assert "--param-policy" in run_help
    assert "{default,constrained}" in run_help
    assert "standard=model-specific" not in run_help
    assert "--param-repeat" in run_help
    assert "--param-mode" not in run_help

    bad_param_search = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / "param_search.py"), "--missing-rate", "0.3"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert bad_param_search.returncode == 2
    assert "clean regular data only" in bad_param_search.stderr


def test_depth_collapsed_models_do_not_search_noop_depth():
    for model_name in ["gru-dt", "gru-d", "gru-ode", "gru-ode-bayes", "mtan"]:
        space = resolve_model_search_space(model_name, "default")
        assert space["depth_level_values"] == [1]
        assert "depth_level" not in space["sampled_axes"]
        cfg = resolve_effective_param_config(
            model_name,
            {"hidden_dim": 64, "depth_level": 5},
            search_profile="default",
        )
        assert cfg["depth_level"] == 1
        assert cfg["effective_params"]["depth_level"] == 1


def test_attention_specific_effective_params():
    sand = resolve_effective_param_config(
        "sand",
        {"hidden_dim": 96, "depth_level": 4},
        search_profile="legacy-expanded-v1",
    )
    assert sand["num_layers"] == 1
    assert sand["model_params"]["n_layers"] == 4
    assert sand["model_params"]["d_model"] == 96

    miam = resolve_effective_param_config(
        "miam",
        {"hidden_dim": 64, "depth_level": 3},
        search_profile="legacy-expanded-v1",
    )
    assert miam["num_layers"] == 1
    assert miam["model_params"]["num_stack"] == 3

    mtan_kwargs = build_model_kwargs_from_config(
        "mtan",
        {"hidden_dim": 64, "depth_level": 5, "search_profile": "legacy-expanded-v1"},
    )
    assert mtan_kwargs["num_layers"] == 1
    assert mtan_kwargs["num_hidden_layers"] == 1


def test_model_native_extra_params_materialize_for_attention_and_flow():
    transformer = resolve_effective_param_config(
        "transformer",
        {"hidden_dim": 64, "depth_level": 3, "n_heads": 8},
        search_profile="default",
    )
    assert transformer["num_layers"] == 3
    assert transformer["model_params"]["n_heads"] == 8
    assert "model_params.n_heads" in transformer["architecture_axes"]["candidate_axes"]

    sand = resolve_effective_param_config(
        "sand",
        {"hidden_dim": 96, "depth_level": 4, "n_heads": 2, "factor": 8},
        search_profile="default",
    )
    assert sand["model_params"]["d_model"] == 96
    assert sand["model_params"]["n_layers"] == 4
    assert sand["model_params"]["n_heads"] == 2
    assert sand["model_params"]["factor"] == 8

    miam = resolve_effective_param_config(
        "miam",
        {"hidden_dim": 64, "depth_level": 3, "n_heads": 4, "n_iter": 2},
        search_profile="default",
    )
    assert miam["model_params"]["num_stack"] == 3
    assert miam["model_params"]["n_heads"] == 4
    assert miam["model_params"]["n_iter"] == 2

    mtan = resolve_effective_param_config(
        "mtan",
        {"hidden_dim": 64, "depth_level": 5, "n_heads": 4, "embed_time": 32},
        search_profile="default",
    )
    assert mtan["depth_level"] == 1
    assert mtan["model_params"]["n_heads"] == 4
    assert mtan["model_params"]["embed_time"] == 32
    assert "model_params.num_stack" in mtan["architecture_axes"]["excluded_axes"]

    flow = resolve_effective_param_config(
        "flowpath-r",
        {"hidden_dim": 128, "depth_level": 5, "flow_depth": 3},
        search_profile="default",
    )
    assert flow["head_depth"] == 5
    assert flow["flow_depth"] == 3
    assert flow["flow_num_layers"] == 3
    assert flow["model_params"]["flow_backend"] == "native"
    assert "flow_depth" in flow["architecture_axes"]["candidate_axes"]


def test_attention_constructor_axes_are_meaningful():
    sand_kwargs = build_model_kwargs_from_config(
        "sand",
        {"hidden_dim": 96, "depth_level": 4, "search_profile": "legacy-expanded-v1"},
    )
    sand = ists_classifier(
        model_name="sand",
        input_dim=6,
        seq_len=12,
        num_class=3,
        device="cpu",
        **sand_kwargs,
    )
    sand_backbone = sand.seq_layer.seq_layer
    assert len(sand_backbone.blocks) == 4
    assert sand_backbone.input_embedding.out_channels == 96

    miam_kwargs = build_model_kwargs_from_config(
        "miam",
        {"hidden_dim": 64, "depth_level": 3, "search_profile": "legacy-expanded-v1"},
    )
    miam = ists_classifier(
        model_name="miam",
        input_dim=6,
        seq_len=12,
        num_class=3,
        device="cpu",
        **miam_kwargs,
    )
    miam_backbone = miam.seq_layer.seq_layer
    assert miam_backbone.obs_encoding_block.n == 3
    assert len(miam_backbone.obs_encoding_block.layers) == 3


def test_flow_effective_params_preserve_head_and_family_flow_depth():
    expected_flow_depth = {
        "neuralflow-r": 3,
        "dualdynamics-r": 2,
        "flowpath-r": 1,
    }
    for model_name, flow_depth in expected_flow_depth.items():
        cfg = resolve_effective_param_config(
            model_name,
            {"hidden_dim": 128, "depth_level": 5},
            search_profile="legacy-expanded-v1",
        )
        assert cfg["num_layers"] == 1
        assert cfg["num_hidden_layers"] == 5
        assert cfg["head_depth"] == 5
        assert cfg["flow_depth"] == flow_depth
        assert cfg["effective_params"]["flow_depth"] == flow_depth

    stale = resolve_effective_param_config(
        "flowpath-r",
        {"hidden_dim": 128, "depth_level": 5, "flow_depth": 99, "flow_num_layers": 99},
        search_profile="legacy-expanded-v1",
    )
    assert stale["flow_depth"] == 1
    assert stale["flow_num_layers"] == 1
