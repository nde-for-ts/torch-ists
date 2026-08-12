"""Standalone task-model wrapper.

Public API:
    ists_model   - nn.Module for all tasks (classification, regression, etc.)
    ists_classifier - convenience factory that returns ists_model(task="classification")

Behavior notes:
- ``use_intensity=None`` defers to ``ists_layer`` model policy resolution.
- Auxiliary-loss backbones stay logits-first; ``return_aux_loss=True`` opt-in exposes the scalar.
- Non-finite predictions hard-fail with ``RuntimeError`` instead of being silently zero-filled.
- Classification heads multiply the final ``Linear`` gradients by ``head_grad_scale`` (default 100).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Literal, Optional

import torch
import torch.nn as nn

from .config.runtime_fault_policy import raise_for_nonfinite_tensor
from .layer import ists_layer
from .task_registry import normalize_task


def _normalize_task(task: str) -> str:
    return normalize_task(task)


class ists_model(nn.Module):
    def __init__(
        self,
        model_name: str = "cnn",
        num_class: int = 2,
        input_dim: int = 3,
        seq_len: int = 128,
        hidden_dim: int = 32,
        hidden_hidden_dim: int = 32,
        num_layers: int = 1,
        num_hidden_layers: int = 1,
        head_depth: Optional[int] = None,
        flow_depth: Optional[int] = None,
        flow_num_layers: Optional[int] = None,
        bidirectional: bool = False,
        dropout: float = 0.1,
        use_intensity: Optional[bool] = None,
        method: Optional[str] = None,
        file: Optional[str] = None,
        device: str = "cpu",
        model_params: Optional[dict] = None,
        imputation_strategy: str = "zero",
        task: Literal[
            "classification",
            "regression",
            "forecasting",
            "imputation",
            "outlier_detection",
            "change_detection",
        ] = "classification",
        output_dim: int = 1,
        forecast_horizon: Optional[int] = None,
        head_grad_scale: float = 100.0,
    ):
        super().__init__()

        self.task = _normalize_task(task)
        head_grad_scale = float(head_grad_scale)
        if not math.isfinite(head_grad_scale) or head_grad_scale <= 0:
            raise ValueError("head_grad_scale must be a positive finite number")
        self.head_grad_scale = head_grad_scale
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = max(1, int(output_dim))
        self.forecast_horizon: Optional[int] = None
        if self.task == "forecasting":
            if forecast_horizon is None or int(forecast_horizon) <= 0:
                raise ValueError("forecasting requires a positive forecast_horizon")
            self.forecast_horizon = int(forecast_horizon)

        self.seq_layer = ists_layer(
            model_name=model_name,
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            hidden_hidden_dim=hidden_hidden_dim,
            num_layers=num_layers,
            num_hidden_layers=num_hidden_layers,
            head_depth=head_depth,
            flow_depth=flow_depth,
            flow_num_layers=flow_num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            use_intensity=use_intensity,
            method=method,
            file=file,
            device=device,
            model_params=model_params,
            imputation_strategy=imputation_strategy,
        )

        if self.task == "classification":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_class),
            )
        elif self.task == "regression":
            self.head = nn.Linear(hidden_dim, self.output_dim)
        elif self.task == "forecasting":
            self.future_time_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
            ).to(self.seq_layer.device)
            self.head = nn.Linear(hidden_dim, self.output_dim)
        elif self.task == "imputation":
            self.head = nn.Linear(hidden_dim, self.input_dim)
        elif self.task == "outlier_detection":
            self.head = nn.Linear(hidden_dim, 1)
        elif self.task == "change_detection":
            self.head = nn.Linear(hidden_dim, 1)
        else:  # pragma: no cover - guarded by _normalize_task
            raise NotImplementedError(f"Unsupported task head: {self.task!r}")

        self.head = self.head.to(self.seq_layer.device)

        # Xavier init on head (all tasks) – matches ICLR 2024 protocol
        self.head.apply(self._init_weights)

        # Gradient scaling on the classifier head output layer only (default 100x);
        # head_grad_scale=1.0 leaves the gradients unscaled.
        if self.task == "classification" and self.head_grad_scale != 1.0:
            scale = self.head_grad_scale
            self.head[-1].weight.register_hook(lambda grad: scale * grad)
            self.head[-1].bias.register_hook(lambda grad: scale * grad)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    @staticmethod
    def _map_checkpoint_keys(state_dict):
        """Remap checkpoint keys from older layouts to the current layout.

        Handles three checkpoint layouts:
        - fc.* / seq_layer.*  (historical classifier checkpoints)
        - _model.head.* / _model.seq_layer.*  (pre-v1.1 ists_classifier wrapper)
        - head.* / seq_layer.*  (current format, pass-through)
        """
        mapped = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("fc."):
                new_key = f"head.{key[len('fc.'):]}"
                mapped.setdefault(new_key, value)
            elif key.startswith("_model.head."):
                new_key = f"head.{key[len('_model.head.'):]}"
                mapped.setdefault(new_key, value)
            elif key.startswith("_model.seq_layer."):
                new_key = f"seq_layer.{key[len('_model.seq_layer.'):]}"
                mapped.setdefault(new_key, value)
            else:
                mapped.setdefault(key, value)
        return mapped

    @staticmethod
    def _last_observed_readout(out: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Gather the output state at the last observed timestep per sample."""
        if out.dim() != 3 or seq.dim() != 4 or seq.shape[1] < 2:
            return out[:, -1, :] if out.dim() == 3 else out

        time_steps = int(out.shape[1])
        if time_steps <= 0:
            raise ValueError("Backbone output must contain at least one timestep.")

        seq_mask = seq[:, 1, :time_steps, :]
        observed = seq_mask > 0
        if observed.dim() == 3:
            observed = observed.any(dim=-1)
        if observed.shape[1] != time_steps:
            time_steps = int(observed.shape[1])

        time_index = torch.arange(time_steps, device=observed.device).unsqueeze(0)
        missing_index = torch.full_like(time_index, -1)
        final_index = torch.where(observed, time_index, missing_index).amax(dim=1).clamp(min=0)
        final_index = final_index.to(device=out.device, dtype=torch.long)
        gather_index = final_index.view(-1, 1, 1).expand(-1, 1, out.size(-1))
        return out[:, :time_steps, :].gather(dim=1, index=gather_index).squeeze(1)

    def load_state_dict(self, state_dict, strict: bool = True):
        remapped = self._map_checkpoint_keys(state_dict)
        return super().load_state_dict(remapped, strict=strict)

    def forward(
        self,
        seq: torch.Tensor,
        coeffs: Optional[torch.Tensor],
        *,
        return_aux_loss: bool = False,
        sample_times: Optional[torch.Tensor] = None,
        future_timestamps: Optional[torch.Tensor] = None,
        forecast_origin: Optional[torch.Tensor] = None,
    ):
        # Unpack the backbone output and optional auxiliary loss.
        result = self.seq_layer(seq, coeffs, sample_times=sample_times)
        if isinstance(result, tuple) and len(result) == 3:
            out, _hn, aux_loss = result
        elif isinstance(result, tuple) and len(result) == 2:
            out, _hn = result
            aux_loss = None
        else:
            raise ValueError(
                "ists_layer must return tuple(out, hn) or tuple(out, hn, loss), "
                f"got: {type(result).__name__}"
            )

        last_observed = self._last_observed_readout(out, seq)
        sample_summary = (
            _hn
            if self.seq_layer.model_name == "latentode" and _hn.dim() == 2
            else last_observed
        )

        if self.task == "classification":
            pred = self.head(sample_summary)
        elif self.task == "regression":
            pred = self.head(sample_summary)
        elif self.task == "forecasting":
            if future_timestamps is None or forecast_origin is None:
                raise ValueError(
                    "forecasting requires future_timestamps and forecast_origin"
                )
            future_timestamps = torch.as_tensor(
                future_timestamps,
                device=sample_summary.device,
                dtype=torch.float64,
            )
            forecast_origin = torch.as_tensor(
                forecast_origin,
                device=sample_summary.device,
                dtype=torch.float64,
            ).reshape(-1)
            expected_shape = (
                int(sample_summary.shape[0]),
                int(self.forecast_horizon),
            )
            if tuple(future_timestamps.shape) != expected_shape:
                raise ValueError(
                    "future_timestamps shape mismatch: "
                    f"expected {expected_shape}, got {tuple(future_timestamps.shape)}"
                )
            if tuple(forecast_origin.shape) != (expected_shape[0],):
                raise ValueError(
                    "forecast_origin shape mismatch: "
                    f"expected {(expected_shape[0],)}, got {tuple(forecast_origin.shape)}"
                )
            if not torch.isfinite(future_timestamps).all() or not torch.isfinite(
                forecast_origin
            ).all():
                raise ValueError("future_timestamps and forecast_origin must be finite")
            elapsed = future_timestamps - forecast_origin.unsqueeze(1)
            if not torch.all(elapsed > 0):
                raise ValueError("future_timestamps must be later than forecast_origin")
            if elapsed.shape[1] > 1 and not torch.all(
                future_timestamps[:, 1:] > future_timestamps[:, :-1]
            ):
                raise ValueError("future_timestamps must be strictly increasing")
            elapsed_features = torch.log1p(elapsed).to(dtype=sample_summary.dtype)
            time_features = self.future_time_encoder(elapsed_features.unsqueeze(-1))
            pred = self.head(sample_summary.unsqueeze(1) + time_features)
        elif self.task == "imputation":
            pred = self.head(out)
        elif self.task == "outlier_detection":
            pred = self.head(sample_summary)
        elif self.task == "change_detection":
            pred = self.head(out)
        else:  # pragma: no cover - guarded by _normalize_task
            raise NotImplementedError(f"Unsupported task forward: {self.task!r}")

        raise_for_nonfinite_tensor(
            pred,
            tensor_name=f"{self.task} prediction",
            model_name=self.seq_layer.model_name,
        )
        if return_aux_loss:
            if aux_loss is None:
                aux_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            return pred, aux_loss
        return pred


def ists_classifier(
    model_name: str = "cnn",
    num_class: int = 2,
    input_dim: int = 3,
    seq_len: int = 128,
    hidden_dim: int = 32,
    hidden_hidden_dim: int = 32,
    num_layers: int = 1,
    num_hidden_layers: int = 1,
    head_depth: Optional[int] = None,
    flow_depth: Optional[int] = None,
    flow_num_layers: Optional[int] = None,
    bidirectional: bool = False,
    dropout: float = 0.1,
    use_intensity: Optional[bool] = None,
    method: Optional[str] = None,
    file: Optional[str] = None,
    device: str = "cpu",
    model_params: Optional[dict] = None,
    imputation_strategy: str = "zero",
    head_grad_scale: float = 100.0,
) -> ists_model:
    """Classification convenience constructor.

    ``use_intensity=None`` resolves from model policy, and non-finite backbone/logit tensors
    raise ``RuntimeError``. ``head_grad_scale`` multiplies the gradients of the final head
    ``Linear`` layer (default ``100.0``; ``1.0`` leaves them unscaled).
    """
    return ists_model(
        model_name=model_name,
        num_class=num_class,
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        hidden_hidden_dim=hidden_hidden_dim,
        num_layers=num_layers,
        num_hidden_layers=num_hidden_layers,
        head_depth=head_depth,
        flow_depth=flow_depth,
        flow_num_layers=flow_num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        use_intensity=use_intensity,
        method=method,
        file=file,
        device=device,
        model_params=model_params,
        imputation_strategy=imputation_strategy,
        task="classification",
        head_grad_scale=head_grad_scale,
    )
