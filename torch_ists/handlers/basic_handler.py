"""BasicHandler for 9 basic models.

This handler supports 9 models:
- cnn: CNNBackbone
- rnn: RNNBackbone (cell='rnn')
- lstm: RNNBackbone (cell='lstm')
- gru: RNNBackbone (cell='gru')
- gru-simple: RNNBackbone (cell='gru', concat_input=True)
- bilstm: RNNBackbone (cell='lstm', bidirectional=True)
- plstm: PLSTM (time-aware)
- tlstm: TLSTM (time-aware)
- tglstm: TGLSTM (time-aware, sequence-first input contract)

Transformer-family models are handled by ``AttentionHandler``.

Reference:
    - module/basic/*.py (model implementations)
"""

from typing import Any, Dict
import torch.nn as nn

from .base import BaseHandler
from ..preprocessing import apply_imputation


class BasicHandler(BaseHandler):
    """Handler for 9 basic models.

    Supported models:
    - cnn: CNNBackbone
    - rnn, lstm, gru, gru-simple, bilstm: RNNBackbone with different configs
    - plstm, tlstm, tglstm: time-aware RNN backbones

    All basic models return (out, hn) tuple.
    """

    name: str = "basic"

    # BasicHandler-9 model configurations
    BASIC_MODELS = {
        # CNN (1 model)
        'cnn': {'type': 'cnn'},

        # RNN family (5 models)
        'rnn': {'type': 'rnn', 'cell': 'rnn', 'bidirectional': False, 'concat_input': False},
        'lstm': {'type': 'rnn', 'cell': 'lstm', 'bidirectional': False, 'concat_input': False},
        'gru': {'type': 'rnn', 'cell': 'gru', 'bidirectional': False, 'concat_input': False},
        'gru-simple': {'type': 'rnn', 'cell': 'gru', 'bidirectional': False, 'concat_input': True},
        'bilstm': {'type': 'rnn', 'cell': 'lstm', 'bidirectional': True, 'concat_input': False},
        'plstm': {'type': 'time_aware', 'cell': 'plstm', 'bidirectional': False},
        'tlstm': {'type': 'time_aware', 'cell': 'tlstm', 'bidirectional': False},
        'tglstm': {'type': 'time_aware', 'cell': 'tglstm', 'bidirectional': False},
    }

    def can_handle(self, model_name: str) -> bool:
        """Check if this handler can process the given model.

        Args:
            model_name: Model name to check

        Returns:
            bool: True if model is in BasicHandler-9 set
        """
        return model_name in self.BASIC_MODELS

    def create_model(
        self,
        model_name: str,
        config: Dict[str, Any],
        device: str
    ) -> nn.Module:
        """Create a BasicHandler-9 model instance.

        Args:
            model_name: Model name (one of 9 supported models)
            config: Configuration dict with keys:
                - input_dim: int
                - hidden_dim: int
                - num_layers: int
                - dropout: float
                - bidirectional: bool (optional, from BASIC_MODELS)
            device: Target device

        Returns:
            nn.Module: Created model instance

        Raises:
            ValueError: If model_name is not in BasicHandler-9 set
        """
        if not self.can_handle(model_name):
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"BasicHandler supports: {list(self.BASIC_MODELS.keys())}"
            )

        # Extract common parameters
        input_dim = int(config['input_dim'])
        hidden_dim = int(config['hidden_dim'])
        num_layers = int(config['num_layers'])
        dropout = float(config['dropout'])

        model_spec = self.BASIC_MODELS[model_name]
        model_type = model_spec['type']

        # Create model based on type
        if model_type == 'cnn':
            from ..module.basic.cnn import CNNBackbone
            return CNNBackbone(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        elif model_type == 'rnn':
            from ..module.basic.rnn import RNNBackbone
            cell = model_spec['cell']
            bidirectional = model_spec.get('bidirectional', False)
            concat_input = model_spec.get('concat_input', False)
            return RNNBackbone(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                cell=cell,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                concat_input=concat_input,
            )

        elif model_type == 'time_aware':
            cell = model_spec.get('cell', model_name)
            bidirectional = bool(model_spec.get('bidirectional', False))

            if cell == 'plstm':
                from ..module.basic import PLSTM

                return PLSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout,
                )
            if cell == 'tlstm':
                from ..module.basic import TLSTM

                return TLSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout,
                )
            if cell == 'tglstm':
                from ..module.basic import TGLSTM

                return TGLSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=False,
                    bidirectional=bidirectional,
                    dropout=dropout,
                )
            raise ValueError(f"Unsupported time-aware cell: {cell}")

        else:
            # This should never happen due to BASIC_MODELS validation
            raise ValueError(f"Unsupported model type: {model_type}")

    @property
    def returns_loss(self) -> bool:
        """Basic models don't return auxiliary loss.

        Returns:
            bool: Always False
        """
        return False

    @property
    def uses_filled(self) -> bool:
        """Basic models require NaN → 0 conversion (X_filled).

        BasicHandler models (including time-aware plstm/tlstm/tglstm)
        cannot handle NaN values. Handler forward pass converts X_missing
        to X_filled before model input.

        Returns:
            bool: Always True (basic models need filled values)
        """
        return True

    def forward(
        self,
        model,  # type: ignore
        seq,  # type: ignore
        coeffs,  # type: ignore
        times,  # type: ignore
        **kwargs  # type: ignore
    ):
        """Execute forward pass for BasicHandler models.

        Args:
            model: Model instance
            seq: Input sequence [N, 3, D, L] containing [data, mask, delta]
            coeffs: Interpolation coefficients (unused for basic models)
            times: Time vector [L]
            **kwargs: Additional parameters (e.g., model_name)

        Returns:
            tuple: (out, hn) where out is [N, L, H] and hn is [N, H]
        """
        import torch

        # Get model name from model attribute or kwargs
        model_name = getattr(model, 'model_name', kwargs.get('model_name', ''))
        # Value-channel imputation strategy (default "zero"; uses_filled = True).
        strategy = kwargs.get('imputation_strategy', 'zero')

        # Preprocess input
        data = self.preprocess_input(seq, times)
        x = data['x']  # [N, L, D]

        # Impute missing values before the naive backbone (value channel).
        x = apply_imputation(x, strategy)

        # Route to appropriate forward helper based on model_name
        if model_name == 'cnn':
            return self._forward_cnn(model, x)
        elif model_name == 'gru-simple':
            return self._forward_gru_simple(model, seq, strategy)
        elif model_name in ('plstm', 'tlstm'):
            return self._forward_time_aware(model, x, data['seq_delta'])
        elif model_name == 'tglstm':
            return self._forward_tglstm(model, x, data['seq_delta'])
        else:  # rnn, lstm, gru, bilstm
            return self._forward_rnn(model, x)

    # ══════════════════════════════════════════════════════════════════
    # Forward Pass Helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_hidden(hn, bidirectional=False):
        """Normalize hidden state to [N, H] shape.

        PyTorch RNNs return hn with shape [num_directions * num_layers, N, H].
        This method extracts the last layer's hidden state.

        Args:
            hn: Hidden state tensor, either [N, H] or [num_directions*num_layers, N, H]
                or tuple (h, c) for LSTM
            bidirectional: Whether the model is bidirectional

        Returns:
            torch.Tensor: Hidden state with shape [N, H] (or [N, 2*H] for bidirectional)
        """
        import torch

        # Handle LSTM tuple (h, c)
        if isinstance(hn, tuple):
            hn = hn[0]  # Take h, discard c

        # Extract last layer
        if hn.dim() == 3:
            if bidirectional:
                # Bidirectional: concat forward and backward from last layer
                hn = torch.cat([hn[-2, :, :], hn[-1, :, :]], dim=-1)
            else:
                # Unidirectional: take last layer
                hn = hn[-1, :, :]

        return hn

    def _forward_cnn(self, model, x):
        """Forward pass for CNN models.

        Args:
            model: CNNBackbone instance
            x: [N, L, D] Input tensor

        Returns:
            tuple: (out, hn) where out is [N, L, H] and hn is [N, H]
        """
        out = model(x)  # CNNBackbone returns out only
        hn = out[:, -1, :]  # Handler extracts hn
        return out, hn

    def _forward_rnn(self, model, x):
        """Forward pass for RNN/LSTM/GRU/BiLSTM models.

        Args:
            model: RNNBackbone instance
            x: [N, L, D] Input tensor

        Returns:
            tuple: (out, hn) where out is [N, L, H] and hn is [N, H]
        """
        out, hn = model(x)
        bidirectional = getattr(model, 'bidirectional', False)
        hn = self._normalize_hidden(hn, bidirectional)
        return out, hn

    def _forward_gru_simple(self, model, seq, strategy="zero"):
        """Forward pass for gru-simple (concatenated input).

        gru-simple concatenates all 3 channels [N, 3, D, L] → [N, L, 3*D]

        Args:
            model: RNNBackbone instance with concat_input=True
            seq: [N, 3, D, L] Raw input sequence
            strategy: Imputation strategy for the value channel (default "zero").

        Returns:
            tuple: (out, hn) where out is [N, L, H] and hn is [N, H]
        """
        import torch

        # Concatenate all 3 channels: [N, 3, D, L] -> [N, L, 3*D]
        # Permute to [N, L, D, 3], then reshape to [N, L, 3*D]
        x_concat = seq.permute(0, 3, 2, 1).reshape(seq.shape[0], seq.shape[3], -1)

        # Impute missing values (only the value channel carries NaN; mask/delta
        # have none, so the strategy is a no-op on those columns).
        x_concat = apply_imputation(x_concat, strategy)

        # Forward pass
        out, hn = model(x_concat)
        hn = self._normalize_hidden(hn)
        return out, hn

    def _forward_time_aware(self, model, x, seq_delta):
        """Forward pass for TLSTM/PLSTM models.

        Args:
            model: TLSTM/PLSTM backbone
            x: [N, L, D] Input tensor
            seq_delta: [N, L] Delta time tensor

        Returns:
            tuple: (out, hn) where out is [N, L, H] and hn is [N, H]
        """
        import torch

        seq_delta = apply_imputation(seq_delta, "zero")
        out, (hn, _cn) = model(x, seq_delta)
        hn = self._normalize_hidden(hn)
        return out, hn

    def _forward_tglstm(self, model, x, seq_delta):
        """Forward pass for TGLSTM with sequence-first input contract.

        TGLSTM expects [L, N, D] input and [L, N, 1] time input.
        """
        import torch

        seq_delta = apply_imputation(seq_delta, "zero")
        out, (hn, _cn) = model(
            x.permute(1, 0, 2),
            seq_delta.unsqueeze(-1).permute(1, 0, 2),
        )
        hn = self._normalize_hidden(hn)
        return out, hn


__all__ = ['BasicHandler']
