"""Base Handler ABC for torch_ists runtime.

Extended from minimal implementation to support Handler pattern forward delegation.

Implementation scope:
- Abstract methods: can_handle(), create_model(), forward()
- Properties: returns_loss, uses_filled
- Utility methods: preprocess_input(), get_final_index()

Reference:
    - HandlerRegistry routing principle
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Set, Tuple, Union
import torch
import torch.nn as nn


class BaseHandler(ABC):
    """Abstract base class for model handlers.

    Each handler is responsible for:
    1. Identifying which models it can handle (can_handle)
    2. Creating model instances (create_model)
    3. Executing forward passes with preprocessing (forward)
    4. Declaring if models return auxiliary loss (returns_loss)
    5. Declaring if models need NaN→0 conversion (uses_filled)

    Attributes:
        name: Handler name for identification
    """

    name: str = "base"

    @abstractmethod
    def can_handle(self, model_name: str) -> bool:
        """Check if this handler can process the given model.

        Args:
            model_name: Model name (e.g., 'cnn', 'rnn', 'transformer')

        Returns:
            bool: True if this handler can process the model
        """
        pass

    @abstractmethod
    def create_model(
        self,
        model_name: str,
        config: Dict[str, Any],
        device: str
    ) -> nn.Module:
        """Create a model instance.

        Args:
            model_name: Model name
            config: Model configuration dict (input_dim, hidden_dim, etc.)
            device: Target device ('cpu' or 'cuda')

        Returns:
            nn.Module: Created model instance
        """
        pass

    @abstractmethod
    def forward(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        coeffs: Optional[torch.Tensor],
        times: torch.Tensor,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Execute forward pass with preprocessing.

        Args:
            model: Model instance
            seq: Input sequence [N, 3, D, L] containing [data, mask, delta]
            coeffs: Interpolation coefficients (optional)
            times: Time vector [L]
            **kwargs: Additional parameters

        Returns:
            tuple: (out, hn) or (out, hn, loss)
                - out: [N, L, H] Output sequence
                - hn: [N, H] Last hidden state
                - loss: Auxiliary loss (optional, for some models)
        """
        pass

    @property
    @abstractmethod
    def returns_loss(self) -> bool:
        """Whether models in this handler return auxiliary loss.

        Returns:
            bool: True if models return (output, loss), False if just output
        """
        pass

    @property
    def uses_filled(self) -> bool:
        """Whether models need NaN → 0 conversion (X_filled).

        Default: False (models can handle NaN, use X_missing)
        Override to True for models requiring filled values.

        Returns:
            bool: True if models need NaN → 0 conversion
        """
        return False

    def consumes_coefficients(self, model_name: str) -> bool:
        """Whether ``model_name`` uses interpolation coefficients at forward time."""
        del model_name
        return False

    # ══════════════════════════════════════════════════════════════════
    # Common Utility Methods
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _coerce_mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _normalize_solver_method(
        method: Any,
        alias_map: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        if method is None:
            return None
        normalized = str(method).strip().lower()
        if not normalized:
            return None
        aliases = alias_map or {}
        return aliases.get(normalized, normalized)

    def resolve_solver_config(
        self,
        *,
        direct: Optional[Dict[str, Any]],
        model_params: Optional[Dict[str, Any]],
        model: Optional[nn.Module],
        default_method: str,
        allowed_methods: Set[str],
        alias_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        direct_map = self._coerce_mapping(direct)
        params_map = self._coerce_mapping(model_params)
        model_solver_kwargs = self._coerce_mapping(getattr(model, "solver_kwargs", {}))

        method = direct_map.get("method")
        if method is None:
            method = params_map.get("method")
        if method is None:
            method = getattr(model, "solver_method", None)
        if method is None:
            method = default_method
        method = self._normalize_solver_method(method, alias_map=alias_map)
        if method not in allowed_methods:
            allowed = ", ".join(sorted(allowed_methods))
            raise ValueError(
                f"Unsupported solver method '{method}'. Allowed methods: {allowed}"
            )

        options = direct_map.get("solver_options")
        if options is None:
            options = direct_map.get("options")
        if options is None:
            options = params_map.get("solver_options")
        if options is None:
            options = params_map.get("options")
        if options is None:
            options = getattr(model, "solver_options", None)
        if options is not None and not isinstance(options, Mapping):
            raise ValueError(
                f"solver_options/options must be a mapping, got {type(options).__name__}"
            )
        resolved_options: Dict[str, Any] = dict(options or {})

        step_size = direct_map.get("step_size")
        if step_size is None:
            step_size = direct_map.get("dt")
        if step_size is None:
            step_size = params_map.get("step_size")
        if step_size is None:
            step_size = params_map.get("dt")
        if step_size is None:
            step_size = resolved_options.get("step_size")
        if step_size is None:
            step_size = model_solver_kwargs.get("step_size")
        if step_size is None:
            step_size = model_solver_kwargs.get("dt")

        dt = direct_map.get("dt")
        if dt is None:
            dt = params_map.get("dt")
        if dt is None:
            dt = model_solver_kwargs.get("dt")
        if dt is None:
            dt = step_size

        rtol = direct_map.get("rtol")
        if rtol is None:
            rtol = params_map.get("rtol")
        if rtol is None:
            rtol = model_solver_kwargs.get("rtol")

        atol = direct_map.get("atol")
        if atol is None:
            atol = params_map.get("atol")
        if atol is None:
            atol = model_solver_kwargs.get("atol")

        adjoint = direct_map.get("adjoint")
        if adjoint is None:
            adjoint = params_map.get("adjoint")
        if adjoint is None:
            adjoint = model_solver_kwargs.get("adjoint")

        return {
            "method": method,
            "options": resolved_options,
            "step_size": step_size,
            "dt": dt,
            "rtol": rtol,
            "atol": atol,
            "adjoint": adjoint,
        }

    def preprocess_input(
        self,
        seq: torch.Tensor,
        times: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Preprocess input data for forward pass.

        This method extracts and transforms the input sequence into
        the format expected by most models.

        Reference: torch_ists/_layer.py L536-547

        Args:
            seq: [N, 3, D, L] Input sequence containing:
                 - seq[:, 0, :, :]: Observed data
                 - seq[:, 1, :, :]: Mask (1=observed, 0=missing)
                 - seq[:, 2, :, :]: Time delta since last observation
            times: [L] Time vector

        Returns:
            dict: Preprocessed data with keys:
                - x: [N, L, D] Data tensor
                - seq_mask: [N, L] Observation mask
                - seq_delta: [N, L] Time deltas
                - seq_ts: [N, L] Timestamps (repeated times)
                - final_index: [N] Last valid index for each sample
        """
        N = seq.shape[0]
        L = seq.shape[3]

        # Extract data: [N, D, L] -> [N, L, D]
        x = seq[:, 0, :, :].permute(0, 2, 1)

        # Collapse per-channel observation metadata to a sequence-level view.
        # Older code used channel 0, but public task heads use any-channel
        # last-observed readout; keeping these aligned avoids silent truncation
        # on asynchronous multivariate inputs.
        mask_nld = seq[:, 1, :, :].permute(0, 2, 1)
        delta_nld = seq[:, 2, :, :].permute(0, 2, 1)
        seq_mask = (mask_nld > 0).any(dim=-1).to(seq.dtype)
        seq_delta = self.reduce_sequence_delta(delta_nld)

        # Create timestamps: [L] -> [N, L]
        seq_ts = times.unsqueeze(0).repeat(N, 1)

        # Use the last observed timestep. This preserves ragged NaN-padding
        # boundaries while avoiding the old count-1 compression bug under
        # pre-existing missingness.
        final_index = self.get_final_index(seq_mask, L)

        return {
            'x': x,
            'seq_mask': seq_mask,
            'seq_delta': seq_delta,
            'seq_ts': seq_ts,
            'final_index': final_index,
        }

    def get_final_index(
        self,
        seq_mask: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Compute the last observed sequence index for each sample.

        This uses observed positions, not observed-count indexing. That keeps
        ragged tail-NaN padding and scattered missingness consistent.

        Args:
            seq_mask: [N, L] Observation mask (1=observed, 0=missing)
            seq_len: Optional sequence length

        Returns:
            torch.Tensor: [N] Last observed sequence index for each sample
        """
        resolved_len = int(seq_len) if seq_len is not None else int(seq_mask.shape[1])
        if resolved_len <= 0:
            return torch.zeros((int(seq_mask.shape[0]),), dtype=torch.long, device=seq_mask.device)

        effective_len = min(resolved_len, int(seq_mask.shape[1]))
        if effective_len <= 0:
            return torch.zeros((int(seq_mask.shape[0]),), dtype=torch.long, device=seq_mask.device)

        observed = seq_mask[:, :effective_len] > 0
        time_index = torch.arange(effective_len, dtype=torch.long, device=seq_mask.device).unsqueeze(0)
        missing_index = torch.full_like(time_index, -1)
        final_index = torch.where(observed, time_index, missing_index).amax(dim=1)
        return torch.clamp(final_index, min=0)

    @staticmethod
    def reduce_sequence_delta(delta: torch.Tensor) -> torch.Tensor:
        """Collapse channel-wise deltas to a sequence-level scalar delta."""
        if delta.dim() != 3:
            return delta
        finite_delta = torch.nan_to_num(
            delta,
            nan=float("inf"),
            posinf=float("inf"),
            neginf=0.0,
        ).clamp_min(0.0)
        seq_delta = finite_delta.amin(dim=-1)
        return torch.where(torch.isfinite(seq_delta), seq_delta, torch.zeros_like(seq_delta))

    @staticmethod
    def gather_output_at_index(
        out: torch.Tensor,
        final_index: torch.Tensor,
    ) -> torch.Tensor:
        """Gather a batch of sequence outputs at last observed indices."""
        if out.dim() != 3:
            raise ValueError(f"Expected [N, L, H] output, got shape {tuple(out.shape)}")
        if final_index.dim() != 1:
            final_index = final_index.reshape(-1)
        if final_index.shape[0] != out.shape[0]:
            raise ValueError(
                "final_index batch size does not match output batch size: "
                f"{final_index.shape[0]} != {out.shape[0]}"
            )
        index = final_index.to(device=out.device, dtype=torch.long)
        index = torch.clamp(index, min=0, max=out.shape[1] - 1)
        batch = torch.arange(out.shape[0], device=out.device)
        return out[batch, index, :]


__all__ = ['BaseHandler']
