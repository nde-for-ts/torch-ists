"""RNN backbone for BasicHandler (7 models support).

Supports:
- rnn: nn.RNN
- lstm: nn.LSTM
- gru: nn.GRU
- gru-simple: nn.GRU with concat_input=True
- bilstm: nn.LSTM with bidirectional=True
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


class RNNBackbone(nn.Module):
    """RNN backbone supporting multiple cell types.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden state dimension
        cell: RNN cell type ('rnn', 'lstm', 'gru')
        num_layers: Number of RNN layers
        bidirectional: Whether to use bidirectional RNN
        dropout: Dropout rate (applied between RNN layers)
        concat_input: Whether to concatenate all 3 input channels (for gru-simple)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        cell: str = 'rnn',
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        concat_input: bool = False,
    ):
        super().__init__()
        self.cell = cell
        self.bidirectional = bool(bidirectional)
        self.hidden_dim = int(hidden_dim)
        self.concat_input = concat_input

        # Adjust input dimension for concat mode (gru-simple)
        actual_input = input_dim * 3 if concat_input else input_dim

        # Create RNN layer based on cell type
        if cell == 'rnn':
            self.seq_layer = nn.RNN(
                input_size=actual_input,
                hidden_size=hidden_dim,
                num_layers=max(int(num_layers), 1),
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif cell == 'lstm':
            self.seq_layer = nn.LSTM(
                input_size=actual_input,
                hidden_size=hidden_dim,
                num_layers=max(int(num_layers), 1),
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif cell == 'gru':
            self.seq_layer = nn.GRU(
                input_size=actual_input,
                hidden_size=hidden_dim,
                num_layers=max(int(num_layers), 1),
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown cell type: {cell}. Supported: rnn, lstm, gru")

        # Output projection (bidirectional → hidden_dim, unidirectional → hidden_dim)
        out_dim = hidden_dim * (2 if self.bidirectional else 1)
        self.out_proj = nn.Linear(out_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [N, L, D]

        Returns:
            out: Output tensor [N, L, H]
            hn_last: Last hidden state [N, H]
        """
        # RNN forward: out [N, L, H*directions], hn [num_directions*num_layers, N, H]
        out, hn = self.seq_layer(x)

        # Project output to hidden_dim
        out = self.out_proj(out)  # [N, L, H]

        # Extract last hidden state
        # LSTM returns (h, c), extract h first
        if isinstance(hn, tuple):
            hn = hn[0]  # Take h, discard c
        
        if self.bidirectional:
            # Concatenate forward and backward last layers
            # hn: [num_directions * num_layers, N, H]
            # Take last 2 layers (forward + backward)
            hn_last = hn[-2:].transpose(0, 1).reshape(x.shape[0], -1)  # [N, 2*H]
            hn_last = self.out_proj(hn_last.unsqueeze(1)).squeeze(1)  # [N, H]
        else:
            # Take last layer
            hn_last = self.out_proj(hn[-1])  # [N, H]

        return out, hn_last


__all__ = ['RNNBackbone']
