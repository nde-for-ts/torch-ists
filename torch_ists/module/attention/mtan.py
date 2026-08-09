"""mTAN attention backbone adapted from reml-lab/mTAN (MIT)."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTimeAttention(nn.Module):
    """Multi-time attention module for mTAN."""

    def __init__(
        self,
        input_dim: int,
        nhidden: int,
        embed_time: int = 16,
        num_heads: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        del device
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.nhidden = nhidden
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time),
                nn.Linear(embed_time, embed_time),
                nn.Linear(input_dim * num_heads, nhidden),
            ]
        )

    def attention(self, query, key, value, mask=None, dropout=None):
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            valid = mask.unsqueeze(-3) != 0
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        p_attn = F.softmax(scores, dim=-2)
        if mask is not None:
            has_observation = valid.any(dim=-2, keepdim=True)
            p_attn = torch.where(has_observation, p_attn, torch.zeros_like(p_attn))
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        batch, _seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [
            l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class MTANEncoder(nn.Module):
    """Encoder for mTAN."""

    def __init__(
        self,
        input_dim: int,
        query: torch.Tensor,
        latent_dim: int = 2,
        nhidden: int = 16,
        embed_time: int = 16,
        num_heads: int = 1,
        learn_emb: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        del device
        self.embed_time = embed_time
        self.dim = input_dim
        self.register_buffer("query", query.clone().detach())
        self.learn_emb = learn_emb
        self.att = MultiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2),
        )
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.query.device).unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = 48.0 * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=pos.device).float() * -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps):
        mask = x[:, :, self.dim :]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.to(time_steps.device).unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.to(time_steps.device).unsqueeze(0))
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        return self.hiddens_to_z0(out)


class AttentionMTANModel(nn.Module):
    """mTAN (Multi-Time Attention Networks) model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        num_stack: int = 2,
        n_heads: int = 1,
        embed_time: int = 16,
        dropout: float = 0.1,
        device: str = "cpu",
        model_name: str = "mtan",
    ):
        super().__init__()
        del num_stack
        self.model_name = model_name
        self.hidden_dim = hidden_dim

        self.encoder = MTANEncoder(
            input_dim=input_dim,
            query=torch.linspace(0, 1.0, seq_len),
            latent_dim=hidden_dim,
            nhidden=hidden_dim,
            embed_time=embed_time,
            num_heads=n_heads,
            learn_emb=False,
            device=device,
        )

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, seq: torch.Tensor, seq_ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = seq[:, 1, :, :].permute(0, 2, 1)
        x_with_mask = torch.cat([x, mask], dim=2)
        out = self.encoder(x_with_mask, seq_ts)

        qz0_mean = out[:, :, : self.hidden_dim]
        qz0_logvar = out[:, :, self.hidden_dim :]
        epsilon = torch.randn_like(qz0_mean)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean

        hn = z0[:, -1, :]
        out = self.embedding(self.dropout(z0))
        return out, hn


__all__ = ["MultiTimeAttention", "MTANEncoder", "AttentionMTANModel"]
