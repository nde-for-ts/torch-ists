from pathlib import Path
import sys

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
)
from torch_ists import ists_classifier
from torch_ists.config import get_active_model_list


def test_search_profile_aliases_and_grids():
    assert normalize_search_profile("default") == "legacy-expanded-v1"
    assert normalize_search_profile("iclr2024") == "legacy-iclr2024-v0"
    assert describe_search_profile("default") == {
        "search_contract": "model-specific-default-v1",
        "search_family": "default",
        "search_profile_visibility": "internal",
        "public_search_name": "default model-specific search",
    }
    assert describe_search_profile("iclr2024") == {
        "search_contract": "historical-reproducibility-v1",
        "search_family": "historical_reproducibility",
        "search_profile_visibility": "internal",
        "public_search_name": "historical reproducibility",
    }

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
        space = resolve_model_search_space(model_name, "legacy-expanded-v1")
        assert space["search_profile"] == "legacy-expanded-v1"
        assert space["hidden_dim_values"]
        assert space["depth_level_values"]

        cfg = resolve_effective_param_config(
            model_name,
            {
                "model_name": model_name,
                "lr": 1e-3,
                "hidden_dim": 64,
                "depth_level": space["depth_level_values"][-1],
            },
            search_profile="legacy-expanded-v1",
        )
        assert cfg["search_policy"] == "model-specific-effective-params-v1"
        assert cfg["search_contract"] == "model-specific-default-v1"
        assert cfg["search_family"] == "default"
        assert cfg["search_profile_visibility"] == "internal"
        assert cfg["legacy_grid_version"] == "legacy-expanded-v1"
        assert cfg["architecture_axes"]["width_axis"] == "hidden_dim"
        assert cfg["architecture_axes"]["depth_axis"] == "depth_level"
        assert cfg["effective_params"]["hidden_dim"] == 64
        assert cfg["effective_params"]["depth_level"] in space["depth_level_values"]


def test_missing_profile_infers_historical_grid():
    assert infer_param_search_profile({"hidden_dim": 32, "num_layers": 1}) == "legacy-iclr2024-v0"


def test_depth_collapsed_models_do_not_search_noop_depth():
    for model_name in ["gru-dt", "gru-d", "gru-ode", "gru-ode-bayes", "mtan"]:
        space = resolve_model_search_space(model_name, "legacy-expanded-v1")
        assert space["depth_level_values"] == [1]
        cfg = resolve_effective_param_config(
            model_name,
            {"hidden_dim": 64, "depth_level": 5},
            search_profile="legacy-expanded-v1",
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
