import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from torch_ists import ists_classifier
from _training_runtime import evaluate, ists_dataset, train


STRICT_FLOW_MODELS = (
    "neuralflow-r",
    "neuralflow-g",
    "neuralflow-c",
    "flowpath-r",
    "flowpath-g",
    "flowpath-c",
)


def _independent_channel_missing_seq():
    x = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
        dtype=torch.float32,
    )
    mask = torch.ones_like(x)
    mask[:, 1, 0] = 0.0
    mask[:, 2, 1] = 0.0

    x = x.clone()
    x[:, 1, 0] = float("nan")
    x[:, 2, 1] = float("nan")

    delta = torch.zeros_like(x)
    return torch.stack([x, mask, delta], dim=1)


def _flowpath_coeffs(seq):
    torchcde = pytest.importorskip("torchcde")
    n, _, length, _ = seq.shape
    times = torch.linspace(0, 1, length)
    seq_ts = times.view(1, length, 1).repeat(n, 1, 1)
    control = torch.cat([seq_ts, seq[:, 0]], dim=-1)
    return torchcde.hermite_cubic_coefficients_with_backward_differences(control, times)


def _normalize_with_train_stats(x_missing, train_idx):
    normalized_channels = []
    for values, train_values in zip(x_missing.unbind(dim=-1), x_missing[train_idx].unbind(dim=-1)):
        observed = train_values.masked_select(~torch.isnan(train_values))
        normalized_channels.append((values - observed.mean()) / (observed.std() + 1e-5))
    return torch.stack(normalized_channels, dim=-1)


def _non_square_real_nan_seq():
    n, length, dim = 6, 7, 5
    values = torch.arange(n * length * dim, dtype=torch.float32).reshape(n, length, dim)
    values = values / values.nan_to_num().abs().max().clamp_min(1.0)
    mask = torch.ones_like(values)

    # Independent per-channel real missingness, matching custom NPZ/real-data input
    # before any synthetic missingness injection.
    for sample in range(n):
        values[sample, (sample + 1) % length, sample % dim] = float("nan")
        values[sample, (sample + 3) % length, (sample + 2) % dim] = float("nan")
    mask[torch.isnan(values)] = 0.0

    train_idx = torch.tensor([0, 1, 2, 3])
    normalized = _normalize_with_train_stats(values, train_idx)
    assert torch.isfinite(normalized[mask <= 0]).sum().item() == 0

    delta = torch.zeros_like(values)
    return torch.stack([normalized, mask, delta], dim=1)


@pytest.mark.parametrize("model_name", STRICT_FLOW_MODELS)
def test_strict_flow_accepts_independent_channel_raw_missing_mask(model_name):
    seq = _independent_channel_missing_seq()
    coeffs = _flowpath_coeffs(seq) if model_name.startswith("flowpath-") else None
    model = ists_classifier(
        model_name=model_name,
        input_dim=2,
        seq_len=4,
        hidden_dim=8,
        num_class=2,
    )
    model.eval()

    with torch.no_grad():
        logits = model(seq, coeffs)

    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_name", STRICT_FLOW_MODELS)
def test_strict_flow_accepts_script_style_real_nan_after_normalization(model_name):
    seq = _non_square_real_nan_seq()
    coeffs = _flowpath_coeffs(seq) if model_name.startswith("flowpath-") else None
    model = ists_classifier(
        model_name=model_name,
        input_dim=5,
        seq_len=7,
        hidden_dim=8,
        num_class=2,
    )
    model.eval()

    with torch.no_grad():
        logits = model(seq, coeffs)

    assert logits.shape == (6, 2)
    assert torch.isfinite(logits).all()


def test_training_runtime_preserves_raw_real_nan_for_strict_flow():
    seq = _non_square_real_nan_seq()
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    coeffs = torch.zeros_like(seq[:, 0])
    split = torch.arange(seq.shape[0])
    dataset = ists_dataset(
        labels,
        seq[:, 0],
        seq[:, 1],
        seq[:, 2],
        coeffs,
        split,
        task="classification",
    )
    loader = DataLoader(dataset, batch_size=3, shuffle=False, drop_last=False)
    model = ists_classifier(
        model_name="neuralflow-r",
        input_dim=5,
        seq_len=7,
        hidden_dim=8,
        num_class=2,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = train(
        "neuralflow-r",
        model,
        optimizer,
        criterion,
        loader,
        interpolate="natural",
        use_intensity=False,
        device=torch.device("cpu"),
    )
    valid_loss = evaluate(
        "neuralflow-r",
        model,
        criterion,
        loader,
        interpolate="natural",
        use_intensity=False,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(valid_loss))
