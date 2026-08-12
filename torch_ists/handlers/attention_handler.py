"""AttentionHandler for attention-family models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base import BaseHandler
from ..preprocessing import apply_imputation


class AttentionHandler(BaseHandler):
    """Handler for attention-based time series models.

    Supports 5 models:
    - transformer: vanilla transformer backbone
    - transformer-mask: transformer backbone with padding-mask-aware attention
    - sand: Simply Attend and Diagnose
    - mtan: Multi-Time Attention Network
    - miam: Multi-view Integration Attention Model
    """

    name: str = "attention"
    ATTENTION_MODELS = {"transformer", "transformer-mask", "sand", "mtan", "miam"}

    def can_handle(self, model_name: str) -> bool:
        return model_name in self.ATTENTION_MODELS

    @property
    def returns_loss(self) -> bool:
        return False

    @staticmethod
    def _validate_head_width(*, width_name: str, width: int, n_heads: int) -> None:
        if n_heads <= 0:
            raise ValueError(f"n_heads must be a positive integer, got {n_heads}")
        if width <= 0 or width % n_heads != 0:
            raise ValueError(
                f"{width_name}={width} must be divisible by n_heads={n_heads}"
            )

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

            raw_dim_feedforward = config.get("dim_feedforward")
            n_heads = int(config.get("n_heads", 4))
            self._validate_head_width(
                width_name="hidden_dim",
                width=hidden_dim,
                n_heads=n_heads,
            )
            model = TransformerBackbone(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                seq_len=seq_len,
                num_layers=int(config.get("num_layers", 2)),
                dropout=dropout,
                dim_feedforward=(
                    None if raw_dim_feedforward is None else int(raw_dim_feedforward)
                ),
            )
        elif model_name == "miam":
            from ..module.attention import AttentionMIAMModel

            n_heads = int(config.get("n_heads", 1))
            self._validate_head_width(
                width_name="hidden_dim",
                width=hidden_dim,
                n_heads=n_heads,
            )
            model = AttentionMIAMModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                num_stack=int(config.get("num_stack", 2)),
                n_heads=n_heads,
                n_iter=int(config.get("n_iter", 1)),
                dropout=dropout,
                dim_feedforward=(
                    None
                    if config.get("dim_feedforward") is None
                    else int(config["dim_feedforward"])
                ),
                attention_dropout=config.get("attention_dropout"),
                ffn_dropout=config.get("ffn_dropout"),
                residual_dropout=config.get("residual_dropout"),
                model_name=model_name,
            )
        elif model_name == "mtan":
            from ..module.attention import AttentionMTANModel

            n_heads = int(config.get("n_heads", 1))
            embed_time = int(config.get("embed_time", 16))
            self._validate_head_width(
                width_name="embed_time",
                width=embed_time,
                n_heads=n_heads,
            )
            model = AttentionMTANModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                num_stack=int(config.get("num_stack", 2)),
                n_heads=n_heads,
                embed_time=embed_time,
                latent_head_dim=(
                    None
                    if config.get("latent_head_dim") is None
                    else int(config["latent_head_dim"])
                ),
                time_scale=float(config.get("time_scale", 48.0)),
                dropout=dropout,
                device=device,
                model_name=model_name,
            )
        else:  # sand
            from ..module.attention import AttentionSAnDModel

            n_heads = int(config.get("n_heads", 4))
            d_model = int(config.get("d_model", 32))
            self._validate_head_width(
                width_name="d_model",
                width=d_model,
                n_heads=n_heads,
            )
            model = AttentionSAnDModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                n_heads=n_heads,
                factor=int(config.get("factor", 16)),
                n_layers=int(config.get("n_layers", 1)),
                d_model=d_model,
                dim_feedforward=(
                    None
                    if config.get("dim_feedforward") is None
                    else int(config["dim_feedforward"])
                ),
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
        strategy = kwargs.get("imputation_strategy", "zero")

        data = self.preprocess_input(
            seq,
            times,
            sample_times=kwargs.get("sample_times"),
        )
        x = apply_imputation(data["x"], strategy)
        seq_ts = data["seq_ts"]

        if model_name in {"transformer", "transformer-mask", "sand"}:
            if model_name == "transformer-mask":
                attention_mask = data["seq_mask"] <= 0
                result = model(x, src_key_padding_mask=attention_mask)
            else:
                # Known limitation: `transformer` and `sand` consume zero-filled
                # NaN positions without a padding mask. See public docs
                # `docs/model_reference.md` §2 Attention Family. The
                # unmasked rows preserve the baseline numerical path.
                result = model(x)
            if isinstance(result, tuple) and len(result) == 2:
                out, hn = result
                if hn.dim() == 3:
                    hn = self.gather_output_at_index(hn, data["final_index"])
                elif hn.dim() != 2 and out.dim() == 3:
                    hn = self.gather_output_at_index(out, data["final_index"])
            else:
                out = result
                hn = self.gather_output_at_index(out, data["final_index"]) if out.dim() == 3 else out
            return out, hn

        if model_name in {"mtan", "miam"}:
            out, hn = model(x, seq, seq_ts)
            if out.dim() == 3:
                hn = self.gather_output_at_index(out, data["final_index"])
            elif hn.dim() == 3:
                hn = self.gather_output_at_index(hn, data["final_index"])
            return out, hn

        raise ValueError(f"Unknown attention model in forward: {model_name}")
