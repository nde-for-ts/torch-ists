from __future__ import annotations

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
    assert isinstance(torch_ists.__version__, str)
    assert torch_ists.__version__
    assert "cnn" in get_active_model_list()
    assert "interpolation" in get_data_config("cnn")


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


def test_param_search_uses_repeat_axis_for_internal_branching():
    script = Path(__file__).resolve().parents[1] / "scripts" / "param_search.py"
    source = script.read_text()

    assert "def _search_mode_from_repeat" not in source
    assert "args.search_mode =" not in source
    assert "search_mode=" not in source
    assert "search_mode ==" not in source
    assert "Unknown search_mode" not in source
    assert "search_repeat == SINGLE_SEARCH_REPEAT" in source
    assert "search_repeat == AGGREGATED_SEARCH_REPEAT" in source
