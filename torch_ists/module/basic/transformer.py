"""Compatibility shim for transformer backbone.

Transformer runtime ownership moved to ``module/attention``.
This shim preserves import compatibility for existing references to
``module.basic.transformer``.
"""

from ..attention.transformer import PositionalEncoding, TransformerBackbone

__all__ = ["PositionalEncoding", "TransformerBackbone"]

