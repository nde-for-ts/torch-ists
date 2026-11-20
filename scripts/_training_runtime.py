"""Training runtime helpers for packaged script execution."""

from __future__ import annotations

import copy
import logging
import math
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

SUPPORTED_TASKS = frozenset(
    {
        "classification",
        "regression",
        "forecasting",
        "imputation",
        "outlier_detection",
        "change_detection",
    }
)
LOGGER = logging.getLogger("torch_ists.script_runtime")
MASKED_SUPERVISION_TASKS = frozenset(
    {"forecasting", "imputation", "change_detection"}
)


def _load_flow_policy_helpers():
    try:
        from torch_ists.config.alias_policy import is_flow_strict_raw_name
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Failed to import torch_ists.config.alias_policy. "
            "Set TORCH_ISTS_RUNTIME_ROOT to a package root that contains torch_ists/."
        ) from exc
    return is_flow_strict_raw_name


_is_flow_strict_raw_name = _load_flow_policy_helpers()


def _resolve_task(task: str) -> str:
    resolved = str(task).strip().lower()
    if resolved not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task: {task!r}. "
            f"Supported tasks are {sorted(SUPPORTED_TASKS)}."
        )
    return resolved


def _expand_mask_or_delta(feature: torch.Tensor, x_missing: torch.Tensor) -> torch.Tensor:
    """Expand [N,L] or [N,L,D] tensor to [N,L,D] for sequence stacking."""
    if feature.dim() == 3:
        if feature.shape == x_missing.shape:
            return feature
        if feature.shape[:2] == x_missing.shape[:2] and feature.shape[-1] == 1:
            return feature.repeat((1, 1, x_missing.shape[-1]))
        feature = feature[..., 0]
    return feature.unsqueeze(-1).repeat((1, 1, x_missing.shape[-1]))


def compute_train_missing_summary(
    x_missing: torch.Tensor, train_idx
) -> Dict[str, float]:
    """Summarize NaN-based missingness on the training split."""
    train_values = x_missing[train_idx, ...]
    missing_count = int(torch.isnan(train_values).sum().item())
    total_count = int(train_values.numel())
    missing_ratio = 0.0 if total_count <= 0 else float(missing_count / float(total_count))
    return {
        "missing_count": int(missing_count),
        "total_count": int(total_count),
        "missing_ratio": float(missing_ratio),
    }


def compute_train_normalization_stats(x_missing: torch.Tensor, train_idx):
    """Compute finite train-only per-channel normalization stats for [N,L,D]."""
    from torch_ists.preprocessing import compute_train_normalization_stats as package_compute

    return package_compute(x_missing, train_idx)


def apply_train_normalization_stats(x_missing: torch.Tensor, stats):
    """Apply per-channel stats while preserving NaNs at missing positions."""
    from torch_ists.preprocessing import apply_train_normalization_stats as package_apply

    return package_apply(x_missing, stats)


def normalize_with_train_stats(x_missing: torch.Tensor, train_idx):
    from torch_ists.preprocessing import normalize_with_train_stats as package_normalize

    return package_normalize(x_missing, train_idx)


def enforce_train_missing_gate(
    missing_rate: float,
    summary: Dict[str, float],
    task: str,
    supervised_missing_count: Optional[int] = None,
) -> None:
    """Enforce train missing-data prerequisites for non-zero missing-rate runs."""
    if float(missing_rate) <= 0.0:
        return
    missing_count = int(summary.get("missing_count", 0))
    if missing_count <= 0:
        raise ValueError(
            "train missing gate failed: missing_rate > 0 but train missing_count == 0"
        )
    task_name = _resolve_task(task)
    if task_name == "imputation" and supervised_missing_count is not None:
        if int(supervised_missing_count) <= 0:
            raise ValueError(
                "imputation supervised missing gate failed: "
                "missing_rate > 0 but supervised_missing_count == 0"
            )


def _align_prediction_target(
    pred: torch.Tensor, target: torch.Tensor, task: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if task == "regression":
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        if pred.dim() != 2:
            raise ValueError(f"regression expects 2D predictions [N,1], got shape {tuple(pred.shape)}")
        if target.dim() != 2:
            raise ValueError(f"regression expects 2D targets [N,1], got shape {tuple(target.shape)}")
        if tuple(pred.shape) != tuple(target.shape):
            raise ValueError(
                f"regression prediction/target shape mismatch: "
                f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        return pred, target

    if task in {"forecasting", "imputation", "change_detection"}:
        if pred.dim() != 3:
            raise ValueError(f"{task} expects 3D predictions [N,L,D], got shape {tuple(pred.shape)}")
        if target.dim() != 3:
            raise ValueError(f"{task} expects 3D targets [N,L,D], got shape {tuple(target.shape)}")
        if tuple(pred.shape) != tuple(target.shape):
            raise ValueError(
                f"{task} prediction/target shape mismatch: "
                f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        return pred, target

    if task == "outlier_detection":
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        if pred.dim() != 2:
            raise ValueError(
                f"outlier_detection expects 2D predictions [N,1], got shape {tuple(pred.shape)}"
            )
        if target.dim() != 2:
            raise ValueError(
                f"outlier_detection expects 2D targets [N,1], got shape {tuple(target.shape)}"
            )
        if tuple(pred.shape) != tuple(target.shape):
            raise ValueError(
                "outlier_detection prediction/target shape mismatch: "
                f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        if pred.shape[1] != 1:
            raise ValueError(f"outlier_detection second dim must be 1, got {pred.shape[1]}")
        return pred, target

    return pred, target


def _is_flow_model_name(model_name: str) -> bool:
    return _is_flow_strict_raw_name(str(model_name))


def _resolve_raw_missing_policy(model_name: str, model) -> bool:
    runtime_model_name = getattr(getattr(model, "seq_layer", None), "model_name", model_name)
    return _is_flow_strict_raw_name(str(model_name)) or _is_flow_strict_raw_name(str(runtime_model_name))


class ists_dataset(Dataset):
    def __init__(
        self,
        y,
        x_missing,
        x_mask,
        x_delta,
        coeffs,
        split,
        task: str = "classification",
        targets=None,
        target_masks=None,
        sample_times=None,
        future_timestamps=None,
        forecast_origin=None,
    ):
        self.task = _resolve_task(task)
        self.Y = None if y is None else y[split]
        self.X_missing = x_missing[split, ...]
        self.X_mask = x_mask[split, ...]
        self.X_delta = x_delta[split, ...]
        self.Coeffs = coeffs[split, ...]
        self.SampleTimes = None if sample_times is None else sample_times[split, ...]
        self.FutureTimestamps = (
            None if future_timestamps is None else future_timestamps[split, ...]
        )
        self.ForecastOrigin = (
            None if forecast_origin is None else forecast_origin[split, ...]
        )
        if targets is None:
            if self.Y is None:
                raise ValueError("classification labels are required when task targets are absent")
            self.Targets = self.Y
        else:
            self.Targets = targets[split, ...]
        self.TargetMasks = None if target_masks is None else target_masks[split, ...]

    def __len__(self) -> int:
        return len(self.X_missing)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "target": self.Targets[idx],
            "x_missing": self.X_missing[idx],
            "x_mask": self.X_mask[idx],
            "x_delta": self.X_delta[idx],
            "coeffs": self.Coeffs[idx],
        }
        if self.Y is not None:
            sample["label"] = self.Y[idx]
        if self.TargetMasks is not None:
            sample["target_mask"] = self.TargetMasks[idx]
        if self.SampleTimes is not None:
            sample["sample_times"] = self.SampleTimes[idx]
        if self.FutureTimestamps is not None:
            sample["future_timestamps"] = self.FutureTimestamps[idx]
        if self.ForecastOrigin is not None:
            sample["forecast_origin"] = self.ForecastOrigin[idx]
        return sample


def _align_target_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 1 and target.dim() == 2:
        mask = mask.unsqueeze(-1)
    if mask.dim() == 2 and target.dim() == 3:
        mask = mask.unsqueeze(-1)
    if mask.dim() != target.dim():
        raise ValueError(
            f"target_mask rank mismatch: mask={tuple(mask.shape)} target={tuple(target.shape)}"
        )
    if tuple(mask.shape) != tuple(target.shape):
        raise ValueError(
            f"Target mask shape mismatch: mask={tuple(mask.shape)} target={tuple(target.shape)}"
        )
    return mask


def _forward_with_optional_aux(
    model,
    seq: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    request_aux: bool,
    sample_times: Optional[torch.Tensor] = None,
    future_timestamps: Optional[torch.Tensor] = None,
    forecast_origin: Optional[torch.Tensor] = None,
):
    runtime_kwargs = {}
    if sample_times is not None:
        runtime_kwargs["sample_times"] = sample_times
    if future_timestamps is not None:
        runtime_kwargs["future_timestamps"] = future_timestamps
    if forecast_origin is not None:
        runtime_kwargs["forecast_origin"] = forecast_origin
    if request_aux:
        try:
            return model(seq, coeffs, return_aux_loss=True, **runtime_kwargs)
        except TypeError as exc:
            # Fall back only when the model rejects the keyword itself; any other
            # TypeError raised inside the model is a real error and propagates.
            if "return_aux_loss" not in str(exc):
                raise
    return model(seq, coeffs, **runtime_kwargs)


def _split_prediction_and_aux_loss(output):
    aux_loss = None
    if isinstance(output, tuple):
        pred = output[0]
        if len(output) > 1:
            aux_loss = output[1]
    else:
        pred = output
    return pred, aux_loss


def _masked_task_loss_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    task: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _align_target_mask(target_mask, target)
    if not torch.isfinite(mask).all():
        raise ValueError(f"{task} target_mask must be finite")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{task} target_mask must be binary 0/1")
    weight = mask.to(dtype=target.dtype)
    if task in {"forecasting", "imputation"}:
        safe_target = torch.where(mask > 0, target, pred.detach())
        element_loss = (pred - safe_target) ** 2
    elif task == "change_detection":
        element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction="none",
        )
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"masked supervision is unsupported for task={task!r}")
    return (element_loss * weight).sum(), weight.sum()


def train(
    model_name,
    model,
    optimizer,
    criterion,
    train_iter,
    interpolate,
    use_intensity,
    device,
    task: str = "classification",
):
    del interpolate, use_intensity
    task = _resolve_task(task)
    model.train()
    criterion.train()
    total_loss = 0.0
    total_weight = 0
    masked_loss_sum = 0.0
    masked_weight = 0.0
    masked_aux_sum = 0.0
    masked_aux_weight = 0
    use_raw_missing = _resolve_raw_missing_policy(str(model_name), model)

    for batch in train_iter:
        y = batch["label"].long().to(device) if task == "classification" else None
        mask = _expand_mask_or_delta(batch["x_mask"], batch["x_missing"])
        delta = _expand_mask_or_delta(batch["x_delta"], batch["x_missing"])
        x_values = batch["x_missing"] if use_raw_missing else torch.nan_to_num(batch["x_missing"], 0.0)
        seq = torch.stack([x_values, mask, delta], dim=1).to(device)

        optimizer.zero_grad()
        output = _forward_with_optional_aux(
            model,
            seq,
            batch["coeffs"].to(device),
            request_aux=True,
            sample_times=(
                None if "sample_times" not in batch else batch["sample_times"].to(device)
            ),
            future_timestamps=(
                None
                if "future_timestamps" not in batch
                else batch["future_timestamps"].to(device)
            ),
            forecast_origin=(
                None
                if "forecast_origin" not in batch
                else batch["forecast_origin"].to(device)
            ),
        )

        if task == "classification":
            assert y is not None
            logits, aux_loss = _split_prediction_and_aux_loss(output)
            loss = criterion(logits, y)
            if aux_loss is not None:
                loss = loss + aux_loss
            batch_size = int(y.shape[0])
        else:
            pred, aux_loss = _split_prediction_and_aux_loss(output)
            target = batch["target"].to(device).float()
            pred, target = _align_prediction_target(pred, target, task)
            target_mask = batch.get("target_mask")
            if task in MASKED_SUPERVISION_TASKS:
                if target_mask is None:
                    raise ValueError(f"{task} requires target_mask")
                numerator, denom = _masked_task_loss_components(
                    pred,
                    target,
                    target_mask.to(device).float(),
                    task,
                )
                denom_value = float(denom.item())
                if denom_value <= 0.0:
                    continue
                loss = numerator / denom
                masked_loss_sum += float(numerator.detach().item())
                masked_weight += denom_value
            else:
                loss = criterion(pred, target)
            if aux_loss is not None:
                loss = loss + aux_loss
            batch_size = int(target.shape[0])
            if task in MASKED_SUPERVISION_TASKS and aux_loss is not None:
                masked_aux_sum += float(aux_loss.detach().item()) * batch_size
                masked_aux_weight += batch_size

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if task not in MASKED_SUPERVISION_TASKS:
            total_loss += float(loss.item()) * batch_size
            total_weight += batch_size

    if task in MASKED_SUPERVISION_TASKS:
        if masked_weight <= 0.0:
            raise ValueError(f"{task} target_mask has zero supervised entries in split")
        aux_mean = masked_aux_sum / masked_aux_weight if masked_aux_weight > 0 else 0.0
        return masked_loss_sum / masked_weight + aux_mean
    if total_weight <= 0:
        raise ValueError("training split is empty")
    return total_loss / total_weight


def evaluate(
    model_name,
    model,
    criterion,
    val_iter,
    interpolate,
    use_intensity,
    device,
    task: str = "classification",
):
    del interpolate, use_intensity
    task = _resolve_task(task)
    model.eval()
    criterion.eval()
    total_loss = 0.0
    total_weight = 0
    masked_loss_sum = 0.0
    masked_weight = 0.0
    masked_aux_sum = 0.0
    masked_aux_weight = 0
    use_raw_missing = _resolve_raw_missing_policy(str(model_name), model)

    with torch.no_grad():
        for batch in val_iter:
            y = batch["label"].long().to(device) if task == "classification" else None
            mask = _expand_mask_or_delta(batch["x_mask"], batch["x_missing"])
            delta = _expand_mask_or_delta(batch["x_delta"], batch["x_missing"])
            x_values = batch["x_missing"] if use_raw_missing else torch.nan_to_num(batch["x_missing"], 0.0)
            seq = torch.stack([x_values, mask, delta], dim=1).to(device)

            output = _forward_with_optional_aux(
                model,
                seq,
                batch["coeffs"].to(device),
                request_aux=True,
                sample_times=(
                    None if "sample_times" not in batch else batch["sample_times"].to(device)
                ),
                future_timestamps=(
                    None
                    if "future_timestamps" not in batch
                    else batch["future_timestamps"].to(device)
                ),
                forecast_origin=(
                    None
                    if "forecast_origin" not in batch
                    else batch["forecast_origin"].to(device)
                ),
            )
            if task == "classification":
                assert y is not None
                logits, aux_loss = _split_prediction_and_aux_loss(output)
                loss = criterion(logits, y)
                if aux_loss is not None:
                    loss = loss + aux_loss
                batch_size = int(y.shape[0])
            else:
                pred, aux_loss = _split_prediction_and_aux_loss(output)
                target = batch["target"].to(device).float()
                pred, target = _align_prediction_target(pred, target, task)
                target_mask = batch.get("target_mask")
                if task in MASKED_SUPERVISION_TASKS:
                    if target_mask is None:
                        raise ValueError(f"{task} requires target_mask")
                    numerator, denom = _masked_task_loss_components(
                        pred,
                        target,
                        target_mask.to(device).float(),
                        task,
                    )
                    denom_value = float(denom.item())
                    if denom_value <= 0.0:
                        continue
                    loss = numerator / denom
                    masked_loss_sum += float(numerator.item())
                    masked_weight += denom_value
                else:
                    loss = criterion(pred, target)
                if aux_loss is not None:
                    loss = loss + aux_loss
                batch_size = int(target.shape[0])
                if task in MASKED_SUPERVISION_TASKS and aux_loss is not None:
                    masked_aux_sum += float(aux_loss.item()) * batch_size
                    masked_aux_weight += batch_size

            if task not in MASKED_SUPERVISION_TASKS:
                total_loss += float(loss.item()) * batch_size
                total_weight += batch_size

    if task in MASKED_SUPERVISION_TASKS:
        if masked_weight <= 0.0:
            raise ValueError(f"{task} target_mask has zero supervised entries in split")
        aux_mean = masked_aux_sum / masked_aux_weight if masked_aux_weight > 0 else 0.0
        return masked_loss_sum / masked_weight + aux_mean
    if total_weight <= 0:
        raise ValueError("evaluation split is empty")
    return total_loss / total_weight


def run_train_valid_loop(
    *,
    model_name,
    model,
    optimizer,
    criterion,
    train_batch,
    valid_batch,
    interpolate,
    use_intensity,
    device,
    epochs: int,
    task: str = "classification",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.5,
    early_stop_warmup: int = 22,
    early_stop_patience: int = 10,
    epoch_reporter=None,
    log_prefix: str = "",
):
    """Run the shared train/validation outer loop used by search and final runs."""
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(scheduler_step_size),
        gamma=float(scheduler_gamma),
    )
    best_loss = math.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    patient = 0
    epochs_completed = 0
    stopped_early = False

    for e in range(int(epochs)):
        train_loss = train(
            model_name,
            model,
            optimizer,
            criterion,
            train_batch,
            interpolate,
            use_intensity,
            device,
            task=task,
        )
        valid_loss = evaluate(
            model_name,
            model,
            criterion,
            valid_batch,
            interpolate,
            use_intensity,
            device,
            task=task,
        )
        epochs_completed = e + 1
        valid_loss_value = float(valid_loss)

        if epochs_completed == 1 or epochs_completed % 10 == 0:
            LOGGER.info(
                "%sEpoch=%d train=%.6f valid=%.6f",
                f"{log_prefix} " if log_prefix else "",
                e,
                float(train_loss),
                valid_loss_value,
            )

        if not math.isfinite(valid_loss_value):
            raise RuntimeError(
                f"Validation loss became non-finite for model={model_name} "
                f"epoch={e}: {valid_loss!r}."
            )

        if valid_loss_value < best_loss:
            best_loss = valid_loss_value
            best_model_wts = copy.deepcopy(model.state_dict())
            patient = 0
        else:
            patient += 1

        if epoch_reporter is not None:
            epoch_reporter(best_loss, epoch=epochs_completed)

        if epochs_completed >= int(early_stop_warmup) and patient > int(early_stop_patience):
            stopped_early = True
            break

        scheduler.step()

    return {
        "best_loss": float(best_loss),
        "best_model_wts": best_model_wts,
        "epochs_completed": int(epochs_completed),
        "stopped_early": bool(stopped_early),
        "scheduler_step_size": int(scheduler_step_size),
        "scheduler_gamma": float(scheduler_gamma),
        "early_stop_warmup": int(early_stop_warmup),
        "early_stop_patience": int(early_stop_patience),
    }
