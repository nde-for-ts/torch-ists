"""Training runtime helpers for packaged script execution."""

from __future__ import annotations

import copy
import logging
import math
import sys
from pathlib import Path
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
    train_values = x_missing[train_idx, ...]
    stats = []
    fallback_channels: list[int] = []
    for channel_idx, train_channel in enumerate(train_values.unbind(dim=-1)):
        observed = train_channel.masked_select(torch.isfinite(train_channel))
        if observed.numel() == 0:
            mean = torch.zeros((), dtype=x_missing.dtype, device=x_missing.device)
            std = torch.ones((), dtype=x_missing.dtype, device=x_missing.device)
            fallback_channels.append(int(channel_idx))
        else:
            mean = observed.mean()
            if observed.numel() < 2:
                std = torch.ones((), dtype=x_missing.dtype, device=x_missing.device)
                fallback_channels.append(int(channel_idx))
            else:
                std = observed.std()
                if (not bool(torch.isfinite(std).item())) or float(std.item()) <= 0.0:
                    std = torch.ones((), dtype=x_missing.dtype, device=x_missing.device)
                    fallback_channels.append(int(channel_idx))
        stats.append((mean, std))
    summary = {
        "normalization_fallback_channel_count": int(len(fallback_channels)),
        "normalization_fallback_channels": fallback_channels,
    }
    return stats, summary


def apply_train_normalization_stats(x_missing: torch.Tensor, stats):
    """Apply per-channel stats while preserving NaNs at missing positions."""
    normalized_channels = []
    for Xi, (mean, std) in zip(x_missing.unbind(dim=-1), stats):
        normalized_channels.append((Xi - mean) / (std + 1e-5))
    normalized = torch.stack(normalized_channels, dim=-1)
    finite_observed = normalized[torch.isfinite(x_missing)]
    if finite_observed.numel() > 0 and not bool(torch.isfinite(finite_observed).all().item()):
        raise ValueError("normalization produced non-finite values at observed positions")
    return normalized


def normalize_with_train_stats(x_missing: torch.Tensor, train_idx):
    stats, summary = compute_train_normalization_stats(x_missing, train_idx)
    return apply_train_normalization_stats(x_missing, stats), summary


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


def _load_flow_policy_helpers():
    try:
        from torch_ists.config.alias_policy import is_flow_strict_raw_name
        return is_flow_strict_raw_name
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        deployment_root = repo_root / "deployment"
        for path in (deployment_root, repo_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        from torch_ists.config.alias_policy import is_flow_strict_raw_name
        return is_flow_strict_raw_name


_is_flow_strict_raw_name = _load_flow_policy_helpers()


def _is_flow_model_name(model_name: str) -> bool:
    return _is_flow_strict_raw_name(str(model_name))


def _resolve_raw_missing_policy(model_name: str, model) -> bool:
    runtime_model_name = getattr(getattr(model, "seq_layer", None), "model_name", model_name)
    return _is_flow_strict_raw_name(str(model_name)) or _is_flow_strict_raw_name(str(runtime_model_name))


def _forward_with_optional_aux(model, seq: torch.Tensor, coeffs: torch.Tensor, *, request_aux: bool):
    if request_aux:
        try:
            return model(seq, coeffs, return_aux_loss=True)
        except TypeError:
            return model(seq, coeffs)
    return model(seq, coeffs)


def _split_prediction_and_aux_loss(output):
    aux_loss = None
    if isinstance(output, tuple):
        pred = output[0]
        if len(output) > 1:
            aux_loss = output[1]
    else:
        pred = output
    return pred, aux_loss


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
    ):
        self.task = _resolve_task(task)
        self.Y = y[split]
        self.X_missing = x_missing[split, ...]
        self.X_mask = x_mask[split, ...]
        self.X_delta = x_delta[split, ...]
        self.Coeffs = coeffs[split, ...]
        self.Targets = self.Y if targets is None else targets[split, ...]
        self.TargetMasks = None if target_masks is None else target_masks[split, ...]

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "label": self.Y[idx],
            "target": self.Targets[idx],
            "x_missing": self.X_missing[idx],
            "x_mask": self.X_mask[idx],
            "x_delta": self.X_delta[idx],
            "coeffs": self.Coeffs[idx],
        }
        if self.TargetMasks is not None:
            sample["target_mask"] = self.TargetMasks[idx]
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
    use_raw_missing = _resolve_raw_missing_policy(str(model_name), model)

    for batch in train_iter:
        y = batch["label"].long().to(device)
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
        )

        if task == "classification":
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
            if task == "imputation" and target_mask is not None:
                mask = _align_target_mask(target_mask.to(device).float(), target)
                weight = (mask > 0.5).to(target.dtype)
                denom = weight.sum()
                if float(denom.item()) <= 0.0:
                    raise ValueError(
                        "imputation target_mask has zero supervised entries in current batch"
                    )
                loss = ((pred - target) ** 2 * weight).sum() / denom
            else:
                loss = criterion(pred, target)
            if aux_loss is not None:
                loss = loss + aux_loss
            batch_size = int(target.shape[0])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += float(loss.item()) * batch_size

    size = len(train_iter.dataset)
    return total_loss / size


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
    use_raw_missing = _resolve_raw_missing_policy(str(model_name), model)

    with torch.no_grad():
        for batch in val_iter:
            y = batch["label"].long().to(device)
            mask = _expand_mask_or_delta(batch["x_mask"], batch["x_missing"])
            delta = _expand_mask_or_delta(batch["x_delta"], batch["x_missing"])
            x_values = batch["x_missing"] if use_raw_missing else torch.nan_to_num(batch["x_missing"], 0.0)
            seq = torch.stack([x_values, mask, delta], dim=1).to(device)

            output = _forward_with_optional_aux(
                model,
                seq,
                batch["coeffs"].to(device),
                request_aux=True,
            )
            if task == "classification":
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
                if task == "imputation" and target_mask is not None:
                    mask = _align_target_mask(target_mask.to(device).float(), target)
                    weight = (mask > 0.5).to(target.dtype)
                    denom = weight.sum()
                    if float(denom.item()) <= 0.0:
                        raise ValueError(
                            "imputation target_mask has zero supervised entries in current batch"
                        )
                    loss = ((pred - target) ** 2 * weight).sum() / denom
                else:
                    loss = criterion(pred, target)
                if aux_loss is not None:
                    loss = loss + aux_loss
                batch_size = int(target.shape[0])

            total_loss += float(loss.item()) * batch_size

    size = len(val_iter.dataset)
    return total_loss / size


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
