"""Training runtime helpers for packaged script execution."""

from __future__ import annotations

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
        dim = min(pred.shape[-1], target.shape[-1])
        return pred[..., :dim], target[..., :dim]

    if task in {"forecasting", "imputation", "change_detection"}:
        if pred.dim() != 3:
            raise ValueError(f"{task} expects 3D predictions [N,L,D], got shape {tuple(pred.shape)}")
        if target.dim() != 3:
            raise ValueError(f"{task} expects 3D targets [N,L,D], got shape {tuple(target.shape)}")
        length = min(pred.shape[1], target.shape[1])
        dim = min(pred.shape[2], target.shape[2])
        return pred[:, :length, :dim], target[:, :length, :dim]

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
        return pred[:, :1], target[:, :1]

    return pred, target


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
    if mask.dim() == 3:
        length = min(mask.shape[1], target.shape[1])
        dim = min(mask.shape[2], target.shape[2])
        return mask[:, :length, :dim]
    dim = min(mask.shape[-1], target.shape[-1])
    return mask[..., :dim]


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
