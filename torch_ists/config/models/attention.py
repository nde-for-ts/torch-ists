"""Attention handler model configs for the packaged runtime."""

from typing import Dict


ATTENTION_MODELS: Dict[str, Dict] = {
    "transformer": {"description": "Transformer encoder", "handler": "AttentionHandler"},
    "transformer-mask": {
        "description": "Transformer encoder with padding-mask-aware attention",
        "handler": "AttentionHandler",
    },
    "miam": {"description": "Multi-view Integration Attention Model", "handler": "AttentionHandler"},
    "mtan": {"description": "Multi-Time Attention Network", "handler": "AttentionHandler"},
    "sand": {"description": "Self-Attention Network", "handler": "AttentionHandler"},
}


__all__ = ["ATTENTION_MODELS"]
