"""Transformer backbone for attention handler runtime.

Canonical location for transformer after Basic -> Attention migration.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerBackbone(nn.Module):
    """Simple Transformer model for sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int = 100,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.transformer_in = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.seq_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hn = self.transformer_in(x)
        hn = self.positional_encoding(hn)
        hn = self.seq_layer(hn)
        out = self.transformer_out(hn)
        return out


__all__ = ["PositionalEncoding", "TransformerBackbone"]

