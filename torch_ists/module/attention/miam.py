"""MIAM attention backbone adapted from YurimALee/MIAM.

See the packaged third-party notices for source and redistribution status.
"""

from __future__ import annotations

import math
from copy import deepcopy
from numbers import Integral
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field} must be a positive integer, got {resolved}")
    return resolved


def get_clones(module, n):
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


class MIAMNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias


class MIAMMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, tp, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1)

        if tp == 1:
            k = k.transpose(2, 3)
            q = q.transpose(2, 3)
            v = v.transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        scores = F.softmax(scores, dim=-1)
        if mask is not None:
            has_unmasked_key = (~mask).any(dim=-1, keepdim=True)
            scores = torch.where(has_unmasked_key, scores, torch.zeros_like(scores))
        scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        if tp == 1:
            output = output.transpose(2, 3)

        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(concat)


class MIAMFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


class MIAMEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        max_length,
        heads,
        d_ff,
        dropout=0,
        *,
        attention_dropout=0.2,
        ffn_dropout=0.1,
        residual_dropout=None,
    ):
        super().__init__()
        del max_length
        if residual_dropout is None:
            residual_dropout = dropout
        self.norm_q = MIAMNorm(d_model)
        self.norm_k = MIAMNorm(d_model)
        self.norm_q_attn = MIAMNorm(d_model)
        self.attn = MIAMMultiHeadAttention(heads, d_model, dropout=attention_dropout)
        self.ff = MIAMFeedForward(d_model, d_ff, dropout=ffn_dropout)
        self.dropout_1 = nn.Dropout(residual_dropout)
        self.dropout_2 = nn.Dropout(residual_dropout)

    def forward(self, q, k, mask):
        q2 = self.norm_q(q)
        k2 = self.norm_k(k)
        q = q + self.dropout_1(self.attn(q2, k2, k2, 0, mask))
        q2 = self.norm_q_attn(q)
        q = q + self.dropout_2(self.ff(q2))
        return q, k


class MIAMEncodingBlock(nn.Module):
    def __init__(
        self,
        d_model,
        max_length,
        num_heads,
        d_ff,
        num_stack,
        *,
        attention_dropout=0.2,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    ):
        super().__init__()
        self.n = num_stack
        layer = MIAMEncoderLayer(
            d_model,
            max_length,
            num_heads,
            d_ff,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        )
        self.layers = get_clones(layer, num_stack)
        self.norm = MIAMNorm(d_model)

    def forward(self, q, k, attn_mask):
        for i in range(self.n):
            q, k = self.layers[i](q, k, attn_mask)
        return self.norm(q)


class MIAMEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Linear(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class MIAMPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def get_sinusoid_encoding_table(self, max_seq_len, d_model, t_set):
        def cal_angle(position, hid_idx):
            return position / np.power(max_seq_len, 2 * (hid_idx // 2) / d_model)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_model)]

        t_set = t_set.detach().cpu().numpy()
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in t_set])
        sinusoid_table[:, 0::2, :] = np.sin(sinusoid_table[:, 0::2, :])
        sinusoid_table[:, 1::2, :] = np.cos(sinusoid_table[:, 1::2, :])
        return torch.FloatTensor(sinusoid_table)

    def time_encoding(self, t):
        pe = self.get_sinusoid_encoding_table(self.max_seq_len, self.d_model, t)
        return pe.permute(0, 2, 1)

    def forward(self, x, m, delta, t):
        x = x * math.sqrt(self.d_model)
        m = m * math.sqrt(self.d_model)
        delta = delta * math.sqrt(self.d_model)
        pos = self.time_encoding(t)
        x = x + Variable(pos, requires_grad=False).to(x.device)
        m = m + Variable(pos, requires_grad=False).to(m.device)
        delta = delta + Variable(pos, requires_grad=False).to(delta.device)
        return x, m, delta


class AttentionMIAMModel(nn.Module):
    """Multi-view Integration Attention Model (MIAM)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        num_stack: int = 2,
        n_heads: int = 1,
        n_iter: int = 1,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
        attention_dropout: float | None = None,
        ffn_dropout: float | None = None,
        residual_dropout: float | None = None,
        model_name: str = "miam",
    ):
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("MIAM requires hidden_dim >= 2 for feature normalization.")
        self.model_name = model_name
        self.n_iter = n_iter

        d_model = hidden_dim
        d_ff = _require_positive_int(
            dim_feedforward if dim_feedforward is not None else hidden_dim,
            field="dim_feedforward",
        )
        self.dim_feedforward = d_ff
        self.attention_dropout = float(
            dropout if attention_dropout is None else attention_dropout
        )
        self.ffn_dropout = float(dropout if ffn_dropout is None else ffn_dropout)
        self.residual_dropout = float(
            dropout if residual_dropout is None else residual_dropout
        )
        for name, value in (
            ("attention_dropout", self.attention_dropout),
            ("ffn_dropout", self.ffn_dropout),
            ("residual_dropout", self.residual_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")

        self.obs_embed = MIAMEmbedder(input_dim, d_model)
        self.mask_embed = MIAMEmbedder(input_dim, d_model)
        self.deltas_embed = MIAMEmbedder(input_dim, d_model)

        self.pe = MIAMPositionalEncoder(d_model, seq_len)

        block_kwargs = {
            "attention_dropout": self.attention_dropout,
            "ffn_dropout": self.ffn_dropout,
            "residual_dropout": self.residual_dropout,
        }
        self.obs_encoding_block = MIAMEncodingBlock(
            d_model, seq_len, n_heads, d_ff, num_stack, **block_kwargs
        )
        self.mask_encoding_block = MIAMEncodingBlock(
            d_model, seq_len, n_heads, d_ff, num_stack, **block_kwargs
        )
        self.deltas_encoding_block = MIAMEncodingBlock(
            d_model, seq_len, n_heads, d_ff, num_stack, **block_kwargs
        )
        self.comb_encoding_block = MIAMEncodingBlock(
            d_model, seq_len, n_heads, d_ff, num_stack, **block_kwargs
        )
        self.missing_comb_encoding_block = MIAMEncodingBlock(
            d_model, seq_len, n_heads, d_ff, num_stack, **block_kwargs
        )

        obs_embed_weight = self.obs_embed.embed.weight
        n, v = obs_embed_weight.size()
        self.decoder = nn.Linear(n, v, bias=False)
        self.decoder.weight.data = obs_embed_weight.transpose(1, 0)
        self.decoder_bias = nn.Parameter(torch.zeros(v))

        self.hidden = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1.0 / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x: torch.Tensor, seq: torch.Tensor, seq_ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _var_num = x.size()

        mask = seq[:, 1, :, :].permute(0, 2, 1)
        deltas = seq[:, 2, :, :].permute(0, 2, 1)

        attn_mask = deltas.data.eq(0)[:, :, 0]
        attn_mask[:, 0] = 0
        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)

        x_z = self.obs_embed(x)
        m = self.mask_embed(mask)
        d = self.deltas_embed(deltas)

        x_z, m, d = self.pe(x_z, m, d, seq_ts)

        x_z = self.obs_encoding_block(x_z, x_z, attn_mask)
        m = self.mask_encoding_block(m, m, attn_mask)
        d = self.deltas_encoding_block(d, d, attn_mask)
        missing_comb_z = self.missing_comb_encoding_block(d, m, attn_mask)

        for _ in range(self.n_iter):
            comb_z = self.comb_encoding_block(missing_comb_z, x_z, attn_mask)
            x_z = self.obs_encoding_block(comb_z, x_z, attn_mask)
            missing_comb_z = self.missing_comb_encoding_block(missing_comb_z, missing_comb_z, attn_mask)

        x_mskd = self.obs_embed(x)
        m_mskd = self.mask_embed(mask)
        d_mskd = self.deltas_embed(deltas)
        x_mskd, m_mskd, d_mskd = self.pe(x_mskd, m_mskd, d_mskd, seq_ts)
        x_d = self.obs_encoding_block(x_mskd, x_mskd, attn_mask)
        x_d = self.obs_encoding_block(x_z, x_d, attn_mask)
        x_final = x_d + x_z

        x_dd = self.decoder(x_final) + self.decoder_bias

        hn = self.hidden(x_dd)
        out = self.embedding(self.dropout(hn))

        final_hn = hn[:, -1, :]
        return out, final_hn


__all__ = [
    "MIAMNorm",
    "MIAMMultiHeadAttention",
    "MIAMFeedForward",
    "MIAMEncoderLayer",
    "MIAMEncodingBlock",
    "MIAMEmbedder",
    "MIAMPositionalEncoder",
    "AttentionMIAMModel",
]
