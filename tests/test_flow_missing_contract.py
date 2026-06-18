import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from torch_ists import ists_classifier


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
