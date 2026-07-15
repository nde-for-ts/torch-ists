from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import torch_ists
from torch_ists import ists_classifier
from torch_ists.config import get_active_model_list, get_data_config
from torch_ists.preprocessing import apply_missingness
from torch_ists.split_utils import make_stratified_splits


def test_package_imports_and_version():
    package_root = Path(__file__).resolve().parents[1]
    package_status = json.loads(
        (package_root / "PACKAGE_STATUS.json").read_text(
            encoding="utf-8"
        )
    )
    version = package_status["package_version"]
    pyproject = (package_root / "pyproject.toml").read_text(encoding="utf-8")
    readme = (package_root / "README.md").read_text(encoding="utf-8")

    assert torch_ists.__version__ == version
    assert f'version = "{version}"' in pyproject
    assert f"version-{version}-blue" in readme
    assert get_active_model_list() == package_status["included_models"]
    assert "interpolation" in get_data_config("cnn")


def test_semantic_artifact_identity_does_not_require_content_hashes():
    package_root = Path(__file__).resolve().parents[1]
    common_path = package_root / "scripts" / "_common.py"
    spec = importlib.util.spec_from_file_location("public_script_common", common_path)
    assert spec is not None and spec.loader is not None
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)

    identity = common.build_param_artifact_identity(
        data_name="BasicMotions",
        model_name="cnn",
        task="classification",
        target_source="label",
        dataset_source="default",
        split_seeds=[0],
    )
    assert not any("hash" in key.lower() for key in identity)
    common.validate_param_artifact_identity(
        {"artifact_identity": identity},
        data_name="BasicMotions",
        model_name="cnn",
        task="classification",
        target_source="label",
        dataset_source="default",
        split_seeds=[0],
    )

    legacy_identity = dict(identity)
    legacy_identity.pop("split_seeds")
    legacy_identity["split_sha256s"] = [{"seed": 0, "sha256": "optional-provenance"}]
    common.validate_param_artifact_identity(
        {"artifact_identity": legacy_identity},
        data_name="BasicMotions",
        model_name="cnn",
        task="classification",
        target_source="label",
        dataset_source="default",
        split_seeds=[0],
    )


def test_source_distribution_manifest_is_complete():
    package_root = Path(__file__).resolve().parents[1]
    manifest = (package_root / "MANIFEST.in").read_text(encoding="utf-8")
    required = (
        "include PACKAGE_STATUS.json",
        "recursive-include docs *.md",
        "recursive-include scripts *.py",
    )
    assert all(entry in manifest for entry in required)
    flow_root = package_root / "torch_ists" / "module" / "flow"
    assert (flow_root / "operators_nfe.py").is_file()
    assert (flow_root / "nfe_reference.py").is_file()
    assert (flow_root / "NFE_PROVENANCE.json").is_file()
    assert not (flow_root / "_neural_flows").exists()
    assert not (flow_root / "_nfe_guard.py").exists()


def test_basic_classifier_forward_cpu():
    model = ists_classifier(
        model_name="cnn",
        input_dim=3,
        seq_len=8,
        hidden_dim=8,
        num_class=2,
    )
    model.eval()
    seq = torch.zeros(4, 3, 8, 3)

    with torch.no_grad():
        logits = model(seq, None)

    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_source_ode_registry_names_are_canonical_and_distinct():
    from torch_ists.config.models.ode import ODE_MODELS, ODE_SOURCE_MODELS

    active_models = get_active_model_list()
    source_names = {"ode-rnn", "latentode", "gru-ode-bayes", "njode"}
    related_runtime_names = {"gru-ode", "ode-rnn-spline"}
    internal_class_names = {
        "sourceodernnbackbone",
        "sourcelatentodebackbone",
        "sourcegruodebayesbackbone",
    }

    assert source_names == set(ODE_SOURCE_MODELS)
    assert related_runtime_names.issubset(ODE_MODELS)
    assert source_names | related_runtime_names <= set(active_models)
    assert all(active_models.count(name) == 1 for name in source_names | related_runtime_names)
    assert internal_class_names.isdisjoint(active_models)


def test_inherited_runtime_configuration_guards():
    from torch_ists.config import default_method_for_model
    from torch_ists.module.attention.miam import AttentionMIAMModel

    for model_name in ("neuralflow", "neuralflow-r", "neuralflow-g", "neuralflow-c"):
        assert default_method_for_model(model_name) is None
    with pytest.raises(ValueError, match="hidden_dim >= 2"):
        AttentionMIAMModel(input_dim=3, hidden_dim=1, seq_len=8)


def test_ito_sde_rejects_heun_configuration():
    pytest.importorskip("torchcde")
    pytest.importorskip("torchsde")
    from torch_ists import ists_layer

    with pytest.raises(ValueError, match="Unsupported solver method"):
        ists_layer(
            model_name="neural_sde",
            input_dim=3,
            hidden_dim=8,
            seq_len=8,
            method="heun",
            device="cpu",
        )


def test_ode_rnn_uses_batched_fixed_grid_evolution(monkeypatch):
    pytest.importorskip("torchdiffeq")
    from torch_ists.module.ode import source_variants

    def reject_scalar_dispatch(*_args, **_kwargs):
        raise AssertionError("ODE-RNN fixed-grid execution must remain batched")

    monkeypatch.setattr(
        source_variants,
        "_integrate_scalar_interval",
        reject_scalar_dispatch,
    )
    model = source_variants.SourceODERNNBackbone(
        input_dim=2,
        hidden_dim=5,
        output_dim=5,
        num_layers=1,
    )
    values = torch.randn(4, 3, 2)
    mask = torch.ones_like(values)
    sample_times = torch.tensor(
        [[0.0, 0.25, 1.0], [0.0, 0.5, 1.0], [0.0, 0.75, 1.0], [0.0, 1.0, 2.0]]
    )

    stream, summary = model(
        values,
        mask,
        sample_times,
        torch.full((4,), 2, dtype=torch.long),
        method="euler",
        options={"step_size": 0.1},
    )

    assert stream.shape == (4, 3, 5)
    assert summary.shape == (4, 5)
    assert torch.isfinite(stream).all()
    assert torch.isfinite(summary).all()


def test_latent_ode_ignores_padding_only_timestamps():
    pytest.importorskip("torchdiffeq")
    from torch_ists.module.ode.source_variants import SourceLatentODEBackbone

    torch.manual_seed(11)
    model = SourceLatentODEBackbone(
        input_dim=2,
        hidden_dim=5,
        output_dim=5,
        num_layers=1,
    )
    model.eval()
    values = torch.randn(2, 5, 2)
    mask = torch.ones_like(values)
    mask[0, 3:] = 0.0
    values[0, 3:] = float("nan")
    final_index = torch.tensor([2, 4])
    base_times = torch.tensor(
        [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]]
    )
    changed_times = base_times.clone()
    changed_times[0, 3:] = torch.tensor([10.0, 20.0])

    first = model(
        values,
        mask,
        base_times,
        final_index,
        method="euler",
        options={"step_size": 0.1},
    )
    second = model(
        values,
        mask,
        changed_times,
        final_index,
        method="euler",
        options={"step_size": 0.1},
    )

    assert torch.allclose(first[0][0, :3], second[0][0, :3])
    assert torch.allclose(first[1], second[1])
    assert torch.allclose(first[2], second[2])
    assert torch.allclose(first[0][0, 3:], first[0][0, 2:3].expand(2, -1))


def test_latent_sde_uses_each_sample_terminal_kl_and_holds_tail(monkeypatch):
    pytest.importorskip("torchcde")
    pytest.importorskip("torchsde")
    from torch_ists.module.sde.latentsde import LatentSDE

    model = LatentSDE(
        input_channels=2,
        hidden_channels=3,
        hidden_hidden_channels=5,
        num_hidden_layers=1,
        kl=True,
        kl_weight=1.0,
    )

    def fake_integrate(*, y0, times, **_kwargs):
        result = y0.unsqueeze(0).repeat(times.numel(), 1, 1)
        for step_index in range(times.numel()):
            result[step_index, :, :-1] = float(step_index)
            result[step_index, :, -1] = 10.0 * step_index
        return result

    monkeypatch.setattr(model, "_integrate", fake_integrate)
    values = torch.full((2, 4, 2), float("nan"))
    mask = torch.zeros_like(values)
    delta = torch.zeros_like(values)
    times = torch.tensor(
        [[0.0, 1.0, float("nan"), float("nan")], [0.0, 1.0, 2.0, 3.0]]
    )

    _out, path, loss = model(
        values,
        mask,
        delta,
        times,
        torch.tensor([1, 3]),
        method="euler",
        dt=0.1,
        adjoint=False,
        eps=torch.zeros(2, 3),
    )

    assert loss.item() == pytest.approx(20.0)
    assert torch.allclose(path[0, 2:], path[0, 1:2].expand(2, -1))
    assert torch.allclose(path[1, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_distributed_flow_backend_defaults_native_and_supports_pinned_nfe():
    from torch_ists.module.flow import resolve_flow_backend
    from torch_ists.module.flow.nfe_reference import (
        NFEInstallationError,
        PINNED_NFE_COMMIT,
        validate_nfe_installation,
    )
    from torch_ists.module.flow.operators import build_flow_operator
    from torch_ists.module.flow.operators_native import CouplingFlow, GRUFlow, ResNetFlow

    flow_root = Path(torch_ists.__file__).resolve().parent / "module" / "flow"
    assert resolve_flow_backend(None) == "native"
    assert resolve_flow_backend("default") == "native"
    assert resolve_flow_backend("nfe") == "nfe"
    assert not (flow_root / "_nfe_guard.py").exists()
    assert not (flow_root / "_neural_flows").exists()
    assert (flow_root / "operators_nfe.py").is_file()
    provenance = json.loads((flow_root / "NFE_PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["source_commit"] == PINNED_NFE_COMMIT
    assert provenance["source_code_bundled"] is False

    validation = validate_nfe_installation()
    if validation["valid"]:
        reference_flow = build_flow_operator(
            flow_option="r",
            backend="nfe",
            dim=3,
            n_layers=2,
            hidden_dims=[5],
        )
        assert reference_flow.backend == "nfe"
    else:
        with pytest.raises(NFEInstallationError, match="pinned external checkout"):
            build_flow_operator(
                flow_option="r",
                backend="nfe",
                dim=3,
                n_layers=2,
                hidden_dims=[5],
            )

    x = torch.randn(2, 4, 3)
    zero_time = torch.zeros(2, 4, 1)
    positive_time = torch.ones(2, 4, 1)
    for flow_class in (ResNetFlow, GRUFlow, CouplingFlow):
        flow = flow_class(dim=3, n_layers=2, hidden_dims=[5]).eval()
        at_zero = flow(x, zero_time)
        transformed = flow(x, positive_time)
        recovered = flow.inverse(transformed, positive_time)
        assert torch.equal(at_zero, x)
        assert torch.allclose(recovered, x, atol=2e-4, rtol=2e-4)


def test_native_flow_time_axis_validation_is_fail_closed():
    from torch_ists.module.flow.operators_native import ResNetFlow

    flow = ResNetFlow(dim=3, n_layers=1, hidden_dims=[5])
    evaluation_times = torch.zeros(2, 4, 1)
    assert flow(torch.randn(2, 1, 3), evaluation_times).shape == (2, 4, 3)
    with pytest.raises(ValueError, match="time dimension must be one or match"):
        flow(torch.randn(2, 2, 3), evaluation_times)


@pytest.mark.parametrize("model_name", ["tlstm", "plstm", "tglstm"])
def test_time_aware_eval_dropout_is_deterministic_cpu(model_name: str):
    model = ists_classifier(
        model_name=model_name,
        input_dim=3,
        seq_len=8,
        hidden_dim=8,
        num_class=2,
        dropout=0.4,
    )
    model.eval()

    seq = torch.randn(4, 3, 8, 3)
    seq[:, 2, :, :] = torch.rand(4, 8, 3)

    with torch.no_grad():
        logits_a = model(seq, None)
        logits_b = model(seq, None)

    assert torch.allclose(logits_a, logits_b)


def test_tlstm_dropout_argument_does_not_flip_bidirectional():
    from torch_ists.module.basic.tlstm import TLSTM

    layer = TLSTM(input_size=3, hidden_size=8, num_layers=1, dropout=0.4)
    cell = layer.layers[0]

    assert cell.bidirectional is False
    assert cell.dropout == pytest.approx(0.4)
    assert isinstance(cell.dropout_layer, torch.nn.Dropout)


def test_missingness_and_split_helpers_smoke():
    X = np.arange(20 * 3 * 8, dtype=np.float32).reshape(20, 3, 8)
    y = np.asarray([0, 1] * 10)

    train_idx, valid_idx, test_idx = make_stratified_splits(y, seed=0)
    assert len(train_idx) + len(valid_idx) + len(test_idx) == len(y)

    masked, mask = apply_missingness(X, rate=0.2, mechanism="MCAR", seed=0)
    assert masked.shape == X.shape
    assert mask.shape == X.shape
    assert np.isnan(masked[mask == 0]).any()


def test_probabilistic_missingness_keeps_sequence_floor():
    X = np.full((64, 2, 45), np.nan, dtype=np.float32)
    X[:, :, :2] = 1.0

    for mechanism in ("MAR", "MNAR"):
        X_missing, mask = apply_missingness(
            X,
            rate=0.95,
            mechanism=mechanism,
            seed=123,
        )
        observed_steps = (mask > 0).any(axis=1).sum(axis=1)
        assert np.all(observed_steps >= 1)
        assert np.isfinite(X_missing[mask > 0]).all()


def test_missingness_floor_provenance_stats():
    X = np.full((64, 2, 45), np.nan, dtype=np.float32)
    X[:, :, 0] = 1.0

    _, mask, stats = apply_missingness(
        X,
        rate=0.9,
        mechanism="MAR",
        seed=0,
        return_stats=True,
    )

    assert stats["n_sequences"] == 64
    assert stats["fully_masked_before_floor"] > 0
    assert np.all((mask > 0).any(axis=1).sum(axis=1) >= 1)


def test_param_search_uses_repeat_axis_for_policy_branching():
    script = Path(__file__).resolve().parents[1] / "scripts" / "param_search.py"
    source = script.read_text()

    assert "def _search_mode_from_repeat" not in source
    assert "args.search_mode =" not in source
    assert "search_mode=" not in source
    assert "search_mode ==" not in source
    assert "Unknown search_mode" not in source
    assert "search_repeat == SINGLE_SEARCH_REPEAT" in source
    assert "search_repeat == AGGREGATED_SEARCH_REPEAT" in source
