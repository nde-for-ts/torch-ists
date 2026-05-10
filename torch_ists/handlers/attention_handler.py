"""AttentionHandler for attention-family models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base import BaseHandler


class AttentionHandler(BaseHandler):
    """Handler for attention-based time series models.

    Supports 5 models:
    - transformer: vanilla transformer backbone
    - transformer-mask: transformer backbone with padding-mask-aware attention
    - miam: Missing Information Attention Model
    - mtan: Multi-Time Attention Network
    - sand: Simply Attend and Diagnose
    """

    name: str = "attention"
    ATTENTION_MODELS = {"transformer", "transformer-mask", "miam", "mtan", "sand"}

    def can_handle(self, model_name: str) -> bool:
        return model_name in self.ATTENTION_MODELS

    @property
    def returns_loss(self) -> bool:
        return False

    def create_model(self, model_name: str, config: Dict[str, Any], device: str) -> nn.Module:
        if not self.can_handle(model_name):
            raise ValueError(
                "Unknown model: "
                f"{model_name}. AttentionHandler supports: {sorted(self.ATTENTION_MODELS)}"
            )

        input_dim = int(config.get("input_dim", 10))
        hidden_dim = int(config.get("hidden_dim", 64))
        seq_len = int(config.get("seq_len", 100))
        dropout = float(config.get("dropout", 0.1))

        if model_name in {"transformer", "transformer-mask"}:
            from ..module.attention import TransformerBackbone

            model = TransformerBackbone(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_heads=int(config.get("n_heads", 4)),
                seq_len=seq_len,
                num_layers=int(config.get("num_layers", 2)),
                dropout=dropout,
            )
        elif model_name == "miam":
            from ..module.attention import AttentionMIAMModel

            model = AttentionMIAMModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                num_stack=int(config.get("num_stack", 2)),
                n_heads=int(config.get("n_heads", 1)),
                n_iter=int(config.get("n_iter", 1)),
                dropout=dropout,
                model_name=model_name,
            )
        elif model_name == "mtan":
            from ..module.attention import AttentionMTANModel

            model = AttentionMTANModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                num_stack=int(config.get("num_stack", 2)),
                n_heads=int(config.get("n_heads", 1)),
                embed_time=int(config.get("embed_time", 16)),
                dropout=dropout,
                device=device,
                model_name=model_name,
            )
        else:  # sand
            from ..module.attention import AttentionSAnDModel

            model = AttentionSAnDModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                n_heads=int(config.get("n_heads", 4)),
                factor=int(config.get("factor", 16)),
                n_layers=int(config.get("n_layers", 1)),
                d_model=int(config.get("d_model", 32)),
                dropout=dropout,
                model_name=model_name,
            )

        model.model_name = model_name
        model.to(device)
        return model

    def forward(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        coeffs: torch.Tensor | None,
        times: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del coeffs
        model_name = kwargs.pop("model_name", getattr(model, "model_name", ""))

        data = self.preprocess_input(seq, times)
        x = torch.nan_to_num(data["x"], nan=0.0)
        seq_ts = data["seq_ts"]

        if model_name in {"transformer", "transformer-mask", "sand"}:
            if model_name == "transformer-mask":
                attention_mask = data["seq_mask"] <= 0
                result = model(x, src_key_padding_mask=attention_mask)
            else:
                result = model(x)
            if isinstance(result, tuple) and len(result) == 2:
                out, hn = result
            else:
                out = result
                hn = out[:, -1, :] if out.dim() == 3 else out
            return out, hn

        if model_name in {"miam", "mtan"}:
            out, hn = model(x, seq, seq_ts)
            return out, hn

        raise ValueError(f"Unknown attention model in forward: {model_name}")
