"""CNN backbone for basic models."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """Simple CNN model for sequence classification.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of Conv1d layers (default: 2)
        dropout: Dropout probability
        kernel_size: Conv1d kernel size (default: 3)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        padding = kernel_size // 2
        self.cnn_in = nn.Linear(input_dim, hidden_dim)
        self.cnn_layer = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
            for _ in range(num_layers)
        ])
        self.cnn_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout  # Store as float

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [N, L, D] -> [N, L, H].

        Handler extracts hn = out[:, -1, :] from returned out.

        Args:
            x: Input tensor [N, L, D]

        Returns:
            out: Output tensor [N, L, H]
        """
        hn = self.cnn_in(x)  # No activation
        for cnn in self.cnn_layer:
            hn = cnn(hn.permute(0, 2, 1))
            hn = nn.functional.dropout(hn.permute(0, 2, 1), p=self.dropout, training=self.training)
            hn = nn.functional.relu(hn)  # ReLU after Conv
        out = self.cnn_out(hn)
        return out  # Handler extracts hn
