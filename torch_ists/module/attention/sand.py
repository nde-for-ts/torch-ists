"""SAnD attention backbone adapted from khirotaka/SAnD (MIT)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for SAnD."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = math.sqrt(self.d_model) * x
        return x + self.pe[:, : x.size(1), :]


class SAnDEncoderBlock(nn.Module):
    """Single encoder block for SAnD."""

    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class DenseInterpolation(nn.Module):
    """Dense interpolation for SAnD."""

    def __init__(self, seq_len: int, factor: int):
        super().__init__()
        self.seq_len = seq_len
        self.factor = factor
        t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
        m = torch.arange(factor, dtype=torch.float32).unsqueeze(1)
        s = factor * t / float(seq_len)
        w = torch.pow(1.0 - torch.abs(s - m) / float(factor), 2.0)
        self.register_buffer("w", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, D] -> [N, factor, D]
        return torch.einsum("fl,nld->nfd", self.w, x)


class ClassificationModule(nn.Module):
    """Classification module for SAnD."""

    def __init__(self, d_model: int, factor: int, n_class: int):
        super().__init__()
        self.linear = nn.Linear(d_model * factor, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


class AttentionSAnDModel(nn.Module):
    """SAnD (Simply Attend and Diagnose) model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        n_heads: int = 4,
        factor: int = 16,
        n_layers: int = 1,
        d_model: int = 32,
        dropout: float = 0.1,
        model_name: str = "sand",
    ):
        super().__init__()
        self.model_name = model_name
        self.seq_len = seq_len

        self.input_embedding = nn.Conv1d(input_dim, d_model, 1)
        self.positional_encoding = PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([SAnDEncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.dense_interpolation = DenseInterpolation(seq_len, factor)
        self.clf = ClassificationModule(d_model, factor, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [N, L, D] -> [N, D, L] -> [N, L, d_model]
        x = self.input_embedding(x.transpose(1, 2)).transpose(1, 2)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)

        x = self.dense_interpolation(x)
        hn = self.clf(x)
        out = hn.unsqueeze(1).repeat(1, self.seq_len, 1)
        out = self.embedding(self.dropout(out))
        return out, hn


__all__ = [
    "PositionalEncoding",
    "SAnDEncoderBlock",
    "DenseInterpolation",
    "ClassificationModule",
    "AttentionSAnDModel",
]
