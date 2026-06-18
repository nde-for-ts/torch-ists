from __future__ import annotations

import numpy as np
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


def test_missingness_and_split_helpers_smoke():
    X = np.arange(20 * 3 * 8, dtype=np.float32).reshape(20, 3, 8)
    y = np.asarray([0, 1] * 10)

    train_idx, valid_idx, test_idx = make_stratified_splits(y, seed=0)
    assert len(train_idx) + len(valid_idx) + len(test_idx) == len(y)

    masked, mask = apply_missingness(X, rate=0.2, mechanism="MCAR", seed=0)
    assert masked.shape == X.shape
    assert mask.shape == X.shape
    assert np.isnan(masked[mask == 0]).any()
