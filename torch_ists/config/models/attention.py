"""Attention handler model configs for release runtime."""

from typing import Dict


ATTENTION_MODELS: Dict[str, Dict] = {
    "transformer": {"description": "Transformer encoder", "handler": "AttentionHandler"},
    "miam": {"description": "Multi-time Attention Network", "handler": "AttentionHandler"},
    "mtan": {"description": "mTAN", "handler": "AttentionHandler"},
    "sand": {"description": "Self-Attention Network", "handler": "AttentionHandler"},
}


__all__ = ["ATTENTION_MODELS"]
