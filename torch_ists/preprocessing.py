"""Self-contained preprocessing utilities for torch_ists."""

from typing import Any, List, Optional, Tuple
import warnings

import numpy as np

MISSINGNESS_EXPERIMENTAL = True
SUPPORTED_MISSINGNESS_MECHANISMS = {
    "MCAR",
    "MAR",
    "MNAR",
    "BLOCK",
    "BURST",
    "CHANNEL_DROP",
}
SUPPORTED_IMPUTATION_STRATEGIES = ("zero", "last_observed", "channel_mean")
SUPPORTED_CONTROL_VALUE_TRANSFORMS = ("identity", "standardize")
CONTROL_VALUE_STANDARDIZE_EPSILON = 1e-5
SUPPORTED_CONTROL_TIME_SOURCES = ("auto", "index", "physical")


def normalize_control_value_transform(method: Optional[str] = None) -> str:
    """Return the canonical value transform used to construct controls."""
    normalized = "identity" if method is None else str(method).strip().lower()
    if normalized == "train_standardize":
        normalized = "standardize"
    if normalized not in SUPPORTED_CONTROL_VALUE_TRANSFORMS:
        raise ValueError(
            "control_value_transform must be one of "
            f"{SUPPORTED_CONTROL_VALUE_TRANSFORMS}, got {method!r}"
        )
    return normalized


def normalize_control_time_source(source: Optional[str] = None) -> str:
    """Return the canonical source used for the control time channel."""
    normalized = "auto" if source is None else str(source).strip().lower()
    if normalized not in SUPPORTED_CONTROL_TIME_SOURCES:
        raise ValueError(
            "control_time_source must be one of "
            f"{SUPPORTED_CONTROL_TIME_SOURCES}, got {source!r}"
        )
    return normalized


def fit_control_time_transform(
    timestamps,
    train_indices,
    *,
    source: str = "auto",
    sequence_length: Optional[int] = None,
) -> dict:
    """Fit the train-only unit used by the physical control time channel."""
    resolved = normalize_control_time_source(source)
    if resolved == "auto":
        effective = "physical" if timestamps is not None else "index"
    else:
        effective = resolved

    if effective == "index":
        return {
            "requested_source": resolved,
            "effective_source": "index",
            "fit_scope": "none",
            "scale": 1.0,
        }
    if timestamps is None:
        raise ValueError("physical control time requires timestamps")

    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    if timestamps_np.ndim != 2:
        raise ValueError("physical control timestamps must have shape [N,L]")
    if sequence_length is not None and int(timestamps_np.shape[1]) != int(sequence_length):
        raise ValueError("physical control timestamps do not match the sequence length")
    if not np.isfinite(timestamps_np).all():
        raise ValueError("physical control timestamps must be finite")
    if train_indices is None:
        raise ValueError("physical control time requires explicit training indices")
    indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("physical control time requires non-empty training indices")
    if int(indices.min()) < 0 or int(indices.max()) >= int(timestamps_np.shape[0]):
        raise ValueError("control-time training indices are out of bounds")

    train_times = timestamps_np[indices]
    positive_gaps = np.diff(train_times, axis=1)
    positive_gaps = positive_gaps[np.isfinite(positive_gaps) & (positive_gaps > 0.0)]
    if positive_gaps.size == 0:
        raise ValueError("physical control time requires a positive training timestamp gap")
    scale = float(np.median(positive_gaps))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("physical control-time scale must be finite and positive")
    return {
        "requested_source": resolved,
        "effective_source": "physical",
        "fit_scope": "train_positive_gaps",
        "scale": scale,
    }


def apply_control_time_transform(timestamps, state: dict, *, sequence_length: int):
    """Construct a finite ``[N,L]`` control-time feature from a fitted state."""
    import torch

    effective = str(state.get("effective_source", "index")).strip().lower()
    if effective == "index":
        if timestamps is not None:
            sample_count = int(np.asarray(timestamps).shape[0])
        else:
            sample_count = int(state.get("sample_count", 1))
        index_grid = torch.linspace(0.0, 1.0, int(sequence_length), dtype=torch.float32)
        return index_grid.view(1, -1).expand(sample_count, -1).clone()
    if effective != "physical":
        raise ValueError(f"unsupported effective control-time source: {effective!r}")
    if timestamps is None:
        raise ValueError("physical control time requires timestamps")

    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    if timestamps_np.ndim != 2 or int(timestamps_np.shape[1]) != int(sequence_length):
        raise ValueError(f"physical control timestamps must have shape [N,{sequence_length}]")
    if not np.isfinite(timestamps_np).all():
        raise ValueError("physical control timestamps must be finite")
    scale = float(state.get("scale", 0.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("physical control-time state has an invalid scale")

    # Subtract in float64 before the float32 cast so large absolute origins do
    # not collapse small, valid gaps.
    elapsed = (timestamps_np - timestamps_np[:, :1]) / scale
    if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed, axis=1) <= 0.0):
        raise ValueError("physical control timestamps must be strictly increasing")
    return torch.from_numpy(elapsed.astype(np.float32, copy=False))


def control_time_transform_metadata(state: dict) -> dict:
    """Return artifact-safe control-time semantics without fitted units."""
    effective = str(state.get("effective_source", "index")).strip().lower()
    return {
        "requested_source": normalize_control_time_source(state.get("requested_source", effective)),
        "effective_source": effective,
        "fit_scope": "train_positive_gaps" if effective == "physical" else "none",
    }


def compute_train_normalization_stats(x_missing, train_indices):
    """Compute finite train-only per-channel statistics for ``[N,L,D]`` values."""
    import torch

    values = torch.as_tensor(x_missing)
    indices = torch.as_tensor(train_indices, dtype=torch.long, device=values.device).reshape(-1)
    train_values = values.index_select(0, indices)
    stats = []
    fallback_channels: list[int] = []
    observed_counts: list[int] = []
    for channel_idx, train_channel in enumerate(train_values.unbind(dim=-1)):
        observed = train_channel.masked_select(torch.isfinite(train_channel))
        observed_counts.append(int(observed.numel()))
        if observed.numel() == 0:
            mean = torch.zeros((), dtype=values.dtype, device=values.device)
            scale = torch.ones((), dtype=values.dtype, device=values.device)
            fallback_channels.append(int(channel_idx))
        else:
            mean = observed.mean()
            if observed.numel() < 2:
                scale = torch.ones((), dtype=values.dtype, device=values.device)
                fallback_channels.append(int(channel_idx))
            else:
                scale = observed.std()
                if (not bool(torch.isfinite(scale).item())) or float(scale.item()) <= 0.0:
                    scale = torch.ones((), dtype=values.dtype, device=values.device)
                    fallback_channels.append(int(channel_idx))
        stats.append((mean, scale))
    summary = {
        "normalization_fallback_channel_count": int(len(fallback_channels)),
        "normalization_fallback_channels": fallback_channels,
        "normalization_observed_counts": observed_counts,
    }
    return stats, summary


def apply_train_normalization_stats(x_missing, stats):
    """Apply channel statistics while preserving NaNs at missing positions."""
    import torch

    values = torch.as_tensor(x_missing)
    if len(stats) != int(values.shape[-1]):
        raise ValueError("normalization stats width does not match input channels")
    channels = [
        (channel - mean) / (scale + CONTROL_VALUE_STANDARDIZE_EPSILON)
        for channel, (mean, scale) in zip(values.unbind(dim=-1), stats)
    ]
    normalized = torch.stack(channels, dim=-1)
    observed = torch.isfinite(values)
    if bool(observed.any().item()) and not bool(torch.isfinite(normalized[observed]).all().item()):
        raise ValueError("normalization produced non-finite values at observed positions")
    return normalized


def normalize_with_train_stats(x_missing, train_indices):
    stats, summary = compute_train_normalization_stats(x_missing, train_indices)
    return apply_train_normalization_stats(x_missing, stats), summary


def fit_control_value_transform(x_missing, train_indices, method: str = "identity") -> dict:
    """Fit a channel-wise transform from finite training values in ``[N,L,D]``."""
    import torch

    resolved = normalize_control_value_transform(method)
    values = torch.as_tensor(x_missing)
    if values.ndim != 3 or not torch.is_floating_point(values):
        raise ValueError("control values must be a floating tensor with shape [N,L,D]")
    if resolved == "identity":
        return {
            "method": "identity",
            "fit_scope": "none",
            "mean": None,
            "scale": None,
            "fallback_channels": [],
            "fallback_channel_count": 0,
            "observed_counts": [],
        }
    if train_indices is None:
        raise ValueError("standardize requires explicit training indices")
    indices = torch.as_tensor(train_indices, dtype=torch.long, device=values.device).reshape(-1)
    if indices.numel() <= 0:
        raise ValueError("standardize requires non-empty training indices")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(values.shape[0]):
        raise ValueError("control-value training indices are out of bounds")

    stats, summary = compute_train_normalization_stats(values, indices)
    return {
        "method": "standardize",
        "fit_scope": "train_observed",
        "mean": [float(mean.item()) for mean, _scale in stats],
        "scale": [float(scale.item()) for _mean, scale in stats],
        "fallback_channels": list(summary["normalization_fallback_channels"]),
        "fallback_channel_count": int(summary["normalization_fallback_channel_count"]),
        "observed_counts": list(summary["normalization_observed_counts"]),
    }


def apply_control_value_transform(x_missing, state: dict):
    """Apply a fitted transform while preserving NaNs at missing positions."""
    import torch

    values = torch.as_tensor(x_missing)
    method = normalize_control_value_transform(state.get("method"))
    if method == "identity":
        return values
    mean_values = state.get("mean")
    scale_values = state.get("scale")
    if mean_values is None or scale_values is None:
        raise ValueError("standardize control-value state is missing mean or scale")
    mean = torch.as_tensor(mean_values, dtype=values.dtype, device=values.device)
    scale = torch.as_tensor(scale_values, dtype=values.dtype, device=values.device)
    if mean.ndim != 1 or scale.ndim != 1 or mean.numel() != values.shape[-1] or scale.numel() != values.shape[-1]:
        raise ValueError("control-value state width does not match input channels")
    return apply_train_normalization_stats(values, list(zip(mean.unbind(), scale.unbind())))


def control_value_transform_metadata(state: dict) -> dict:
    """Return artifact-safe semantic metadata without fitted statistics."""
    method = normalize_control_value_transform(state.get("method"))
    return {
        "method": method,
        "fit_scope": "none" if method == "identity" else "train_observed",
        "fallback_channel_count": int(state.get("fallback_channel_count", 0)),
        "fallback_channels": [int(value) for value in state.get("fallback_channels", [])],
    }


def _require_torchcde():
    try:
        import torchcde
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != "torchcde":
            raise
        raise ImportError(
            "preprocess() requires optional dependency 'torchcde' to build interpolation "
            "coefficients. Install `torchcde` directly or, from this source checkout, run "
            "`python -m pip install -e \".[cde]\"` or "
            "`python -m pip install -e \".[all]\"`."
        ) from exc
    return torchcde


def _impute_last_observed(x, time_dim: int = -2):
    """Last-observation-carried-forward along ``time_dim``.

    Gaps before the first observation in a channel become ``0.0``. Fully vectorized
    (no Python loop over time): a cumulative max of the observed time indices gives,
    for each step, the index of the most recent observation, which is then gathered.
    """
    import torch

    length = x.shape[time_dim]
    index_shape = [1] * x.dim()
    index_shape[time_dim] = length
    time_index = torch.arange(length, device=x.device).reshape(index_shape).expand_as(x)
    observed = ~torch.isnan(x)
    last_index = torch.where(observed, time_index, torch.zeros_like(time_index))
    last_index = torch.cummax(last_index, dim=time_dim).values
    filled = torch.nan_to_num(x, nan=0.0)
    return torch.gather(filled, time_dim, last_index)


def _impute_channel_mean(x, time_dim: int = -2):
    """Per-channel, mask-aware mean over ``time_dim``.

    Each missing position is filled with the mean of the observed values in the
    same channel; channels with no observation become ``0.0``.
    """
    import torch

    isnan = torch.isnan(x)
    observed = (~isnan).to(x.dtype)
    filled = torch.nan_to_num(x, nan=0.0)
    count = observed.sum(dim=time_dim, keepdim=True).clamp_min(1.0)
    mean = (filled * observed).sum(dim=time_dim, keepdim=True) / count
    return torch.where(isnan, mean.expand_as(x), x)


def apply_imputation(x, strategy: str = "zero", *, time_dim: int = -2):
    """Impute missing (NaN) values before a naive backbone forward pass.

    Assumes the packaged value layout with the time axis at ``time_dim`` (default
    ``-2``) and channels at the last axis (the ``[N, L, D]`` value stack).

    Strategies:
    - ``"zero"`` (default): replace NaN with 0.0. With training-only z-scoring
      applied upstream, this matches per-channel mean imputation in raw value
      space; without it, it is literal zero-fill. Numerically identical to the
      prior ``torch.nan_to_num(x, nan=0.0)`` call.
    - ``"last_observed"``: last-observation-carried-forward along time; gaps
      before the first observation become 0.
    - ``"channel_mean"``: per-channel mask-aware mean over time; all-missing
      channels become 0.
    """
    import torch

    if strategy == "zero":
        return torch.nan_to_num(x, nan=0.0)
    if strategy == "last_observed":
        return _impute_last_observed(x, time_dim=time_dim)
    if strategy == "channel_mean":
        return _impute_channel_mean(x, time_dim=time_dim)
    raise ValueError(
        f"Unknown imputation strategy: {strategy!r}. Supported: {SUPPORTED_IMPUTATION_STRATEGIES}."
    )


def _validate_inputs(X: np.ndarray, rate: float) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must be [N, D, L], got shape {X.shape}")
    if not (0.0 <= rate < 1.0):
        raise ValueError(f"rate must be in [0, 1), got {rate}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_nanmean_scalar(x: np.ndarray, default: float = 0.0) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        value = np.nanmean(x)
    return float(value) if np.isfinite(value) else float(default)


def _safe_nanstd_scalar(x: np.ndarray, default: float = 1.0) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        value = np.nanstd(x)
    return float(value) if np.isfinite(value) and float(value) > 0.0 else float(default)


def _safe_nanmean_array(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        value = np.nanmean(x, axis=axis, keepdims=keepdims)
    return np.asarray(value, dtype=np.float32)


def _build_mar_probability(
    X: np.ndarray,
    rate: float,
    reference_X: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Feature-dependent missingness probability.
    reference = X if reference_X is None else np.asarray(reference_X, dtype=np.float32)
    feature_mean = _safe_nanmean_array(X, axis=2, keepdims=True)
    reference_feature_mean = _safe_nanmean_array(reference, axis=2, keepdims=True)
    centered = feature_mean - _safe_nanmean_scalar(reference_feature_mean)
    reference_centered = reference_feature_mean - _safe_nanmean_scalar(reference_feature_mean)
    reference_scale = _safe_nanstd_scalar(reference_feature_mean) + 1e-8
    logits = centered / reference_scale
    reference_logits = reference_centered / reference_scale
    probs = _sigmoid(logits)
    reference_probs = _sigmoid(reference_logits)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.99, neginf=0.0)
    reference_probs = np.nan_to_num(reference_probs, nan=0.0, posinf=0.99, neginf=0.0)
    probs = probs / (_safe_nanmean_scalar(reference_probs, default=1.0) + 1e-8) * rate
    return np.clip(probs, 0.0, 0.99)


def _build_mnar_probability(
    X: np.ndarray,
    rate: float,
    reference_X: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Value-dependent missingness probability.
    reference = X if reference_X is None else np.asarray(reference_X, dtype=np.float32)
    reference_mean = _safe_nanmean_scalar(reference)
    reference_scale = _safe_nanstd_scalar(reference) + 1e-8
    norm = (X - reference_mean) / reference_scale
    reference_norm = (reference - reference_mean) / reference_scale
    probs = _sigmoid(norm)
    reference_probs = _sigmoid(reference_norm)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.99, neginf=0.0)
    reference_probs = np.nan_to_num(reference_probs, nan=0.0, posinf=0.99, neginf=0.0)
    probs = probs / (_safe_nanmean_scalar(reference_probs, default=1.0) + 1e-8) * rate
    return np.clip(probs, 0.0, 0.99)


def _enforce_sequence_observation_floor(
    existing_missing: np.ndarray,
    sampled_missing: np.ndarray,
    rng: np.random.Generator,
    *,
    channel_sync: bool,
) -> Tuple[np.ndarray, List[int]]:
    """Keep synthetic missingness from removing a sample's last observed value.

    Returns the possibly modified mask plus the sequence indices that were
    emptied by sampling before the floor restored one observed step.
    """
    observed_before = ~existing_missing
    observed_after = ~(existing_missing | sampled_missing)
    has_native_observation = observed_before.any(axis=(1, 2))
    emptied_by_sampling = has_native_observation & ~observed_after.any(axis=(1, 2))
    fully_masked_before_floor_indices = np.flatnonzero(emptied_by_sampling).astype(int).tolist()
    if not np.any(emptied_by_sampling):
        return sampled_missing, fully_masked_before_floor_indices

    sampled_missing = sampled_missing.copy()
    for sample_idx in np.flatnonzero(emptied_by_sampling):
        candidate_channels, candidate_times = np.where(observed_before[sample_idx])
        if candidate_times.size == 0:
            continue
        choice = int(rng.integers(candidate_times.size))
        channel = int(candidate_channels[choice])
        time_idx = int(candidate_times[choice])
        if channel_sync:
            sampled_missing[sample_idx, :, time_idx] = False
        else:
            sampled_missing[sample_idx, channel, time_idx] = False
    return sampled_missing, fully_masked_before_floor_indices


def _sample_block_indices(rng: np.random.Generator, length: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if count >= length:
        return np.arange(length, dtype=np.int64)
    start = int(rng.integers(0, length - count + 1))
    return np.arange(start, start + count, dtype=np.int64)


def _sample_burst_indices(rng: np.random.Generator, length: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if count >= length:
        return np.arange(length, dtype=np.int64)

    selected = np.zeros(length, dtype=bool)
    max_burst = max(1, min(count, int(round(length * 0.05)) or 1))
    attempts = 0
    while int(selected.sum()) < count and attempts < max(32, length * 8):
        attempts += 1
        remaining = count - int(selected.sum())
        burst_len = int(rng.integers(1, min(max_burst, remaining) + 1))
        start = int(rng.integers(0, length - burst_len + 1))
        candidates = np.arange(start, start + burst_len, dtype=np.int64)
        new_candidates = candidates[~selected[candidates]]
        if new_candidates.size == 0:
            continue
        selected[new_candidates[:remaining]] = True

    remaining = count - int(selected.sum())
    if remaining > 0:
        fill = rng.choice(np.flatnonzero(~selected), size=remaining, replace=False)
        selected[fill] = True

    return np.flatnonzero(selected).astype(np.int64)


def _compute_delta(mask_observed_ndl: np.ndarray) -> np.ndarray:
    """Compute index-based delta tensor with shape [N, D, L].

    Note:
        Delta is index-based (time-step count since previous observation),
        not timestamp-aware.
    """
    observed = mask_observed_ndl.astype(bool)
    n, d, l = observed.shape
    delta = np.zeros((n, d, l), dtype=np.float32)

    running = np.zeros((n, d), dtype=np.float32)
    for t in range(1, l):
        running = np.where(observed[:, :, t - 1], 1.0, running + 1.0)
        delta[:, :, t] = running
    return delta


def _validate_timestamps_for_preprocess(timestamps: Any, X_np: np.ndarray) -> np.ndarray:
    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    expected_shape = (int(X_np.shape[0]), int(X_np.shape[2]))
    if timestamps_np.ndim != 2:
        raise ValueError(f"timestamps must be 2D [N, L], got shape {timestamps_np.shape}")
    if tuple(timestamps_np.shape) != expected_shape:
        raise ValueError(f"timestamps shape mismatch: expected {expected_shape}, got {timestamps_np.shape}")
    for row in timestamps_np:
        finite_row = row[np.isfinite(row)]
        if finite_row.size > 1 and np.any(np.diff(finite_row) <= 0):
            raise ValueError("timestamps must be strictly increasing within each sample")
    return timestamps_np


def _resolve_timestamp_delta_scale(
    timestamps_nl: np.ndarray,
    timestamp_delta_scale: Optional[float],
) -> float:
    if timestamp_delta_scale is not None:
        scale = float(timestamp_delta_scale)
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError(f"timestamp_delta_scale must be finite and positive, got {timestamp_delta_scale}")
        return scale
    if timestamps_nl.shape[1] < 2:
        return 1.0
    left = timestamps_nl[:, :-1]
    right = timestamps_nl[:, 1:]
    diffs = right - left
    valid = np.isfinite(left) & np.isfinite(right) & (diffs > 0)
    positive_diffs = diffs[valid]
    if positive_diffs.size == 0:
        return 1.0
    return float(np.median(positive_diffs))


def _compute_timestamp_delta(
    mask_observed_ndl: np.ndarray,
    timestamps_nl: np.ndarray,
    *,
    timestamp_delta_scale: float,
    ) -> np.ndarray:
    """Compute elapsed time since the last observation, or sequence start."""
    observed = mask_observed_ndl.astype(bool)
    timestamps = np.asarray(timestamps_nl, dtype=np.float64)
    n, d, l = observed.shape
    delta = np.zeros((n, d, l), dtype=np.float32)

    for i in range(n):
        times_i = timestamps[i]
        finite_i = np.isfinite(times_i)
        finite_indices = np.flatnonzero(finite_i)
        sequence_start_time = (
            float(times_i[int(finite_indices[0])]) if finite_indices.size else None
        )
        for j in range(d):
            last_observed_time = sequence_start_time
            for t in range(l):
                if finite_i[t] and last_observed_time is not None:
                    elapsed = max(float(times_i[t] - last_observed_time), 0.0)
                    delta[i, j, t] = elapsed / float(timestamp_delta_scale)
                if observed[i, j, t] and finite_i[t]:
                    last_observed_time = float(times_i[t])
    return delta


def _common_timestamp_grid(timestamps_nl: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if timestamps_nl is None:
        return None
    timestamps = np.asarray(timestamps_nl, dtype=np.float32)
    if timestamps.ndim != 2 or timestamps.shape[0] == 0:
        return None
    if not np.isfinite(timestamps).all():
        return None
    reference = timestamps[0]
    if not all(np.allclose(row, reference, atol=1e-6, rtol=0.0) for row in timestamps[1:]):
        return None
    if not np.all(np.diff(reference) > 0):
        return None
    time_min = float(reference.min())
    time_max = float(reference.max())
    if time_max <= time_min:
        return None
    return ((reference - time_min) / (time_max - time_min)).astype(np.float32)


def _finalize_preprocess_outputs(
    X_missing_np: np.ndarray,
    mask_ndl: np.ndarray,
    delta_ndl: np.ndarray,
    *,
    interpolate: str,
    use_intensity: bool,
    time_grid_np: Optional[np.ndarray] = None,
    control_time_feature=None,
    control_time_transform_state: Optional[dict] = None,
    return_control_time_transform: bool = False,
    control_value_transform: str = "identity",
    control_value_train_indices: Optional[Any] = None,
    control_value_transform_state: Optional[dict] = None,
    return_control_value_transform: bool = False,
):
    import torch

    torchcde = _require_torchcde()
    from .module.cde import controldiffeq

    X_missing = torch.from_numpy(X_missing_np).to(torch.float32).permute(0, 2, 1)  # [N, L, D]
    X_mask = torch.from_numpy(mask_ndl).to(torch.float32).permute(0, 2, 1)         # [N, L, D]
    X_delta = torch.from_numpy(delta_ndl).to(torch.float32).permute(0, 2, 1)       # [N, L, D]

    seq_len = X_missing.shape[1]
    if time_grid_np is None:
        times = torch.linspace(0, 1, seq_len, dtype=X_missing.dtype)
    else:
        times = torch.as_tensor(time_grid_np, dtype=X_missing.dtype)
        if times.ndim != 1 or times.numel() != seq_len:
            raise ValueError(f"time grid must have shape [{seq_len}], got {tuple(times.shape)}")

    intensity = (~torch.isnan(X_missing)).to(X_missing.dtype).cumsum(dim=1)
    if control_time_feature is None:
        time_feat = times.view(1, seq_len, 1).expand(X_missing.shape[0], seq_len, 1)
    else:
        time_feat = torch.as_tensor(control_time_feature, dtype=X_missing.dtype)
        if tuple(time_feat.shape) != (int(X_missing.shape[0]), int(seq_len)):
            raise ValueError(
                "control-time feature must have shape "
                f"[{int(X_missing.shape[0])},{int(seq_len)}], got {tuple(time_feat.shape)}"
            )
        time_feat = time_feat.unsqueeze(-1)

    resolved_transform = normalize_control_value_transform(control_value_transform)
    if control_value_transform_state is None:
        transform_state = fit_control_value_transform(
            X_missing,
            control_value_train_indices,
            method=resolved_transform,
        )
    else:
        transform_state = dict(control_value_transform_state)
        state_method = normalize_control_value_transform(transform_state.get("method"))
        if state_method != resolved_transform:
            raise ValueError(
                "control-value transform state/method mismatch: "
                f"state={state_method!r}, requested={resolved_transform!r}"
            )
    control_values = apply_control_value_transform(X_missing, transform_state)

    values_t = torch.cat([time_feat, control_values], dim=-1)
    values_ti = torch.cat([time_feat, intensity, control_values], dim=-1)

    target = values_ti if use_intensity else values_t
    if interpolate == "natural":
        coeff_tuple = controldiffeq.natural_cubic_spline_coeffs(times, target)
        coeffs = torch.cat(coeff_tuple, dim=-1)
    elif interpolate == "linear":
        coeffs = torchcde.linear_interpolation_coeffs(target, times)
    elif interpolate == "rectilinear":
        rect_times = torch.linspace(0, 1, seq_len * 2 - 1, dtype=times.dtype, device=times.device)
        coeffs = torchcde.linear_interpolation_coeffs(target, rect_times, rectilinear=0)
    elif interpolate == "cubic":
        coeffs = torchcde.natural_cubic_coeffs(target, times)
    elif interpolate == "hermite":
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(target, times)
    else:
        raise ValueError(
            f"Unsupported interpolation: {interpolate}. "
            "Supported values are {'natural', 'linear', 'rectilinear', 'cubic', 'hermite'}."
        )

    outputs = (X_missing, X_mask, X_delta, coeffs)
    if return_control_value_transform:
        outputs = (*outputs, transform_state)
    if return_control_time_transform:
        if control_time_transform_state is None:
            raise ValueError("control-time transform state is required when returning it")
        outputs = (*outputs, dict(control_time_transform_state))
    return outputs


def apply_missingness_compat(
    X: np.ndarray,
    rate: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply random per-sample missingness using the alternative (compatibility) removal scheme."""
    import torch

    _validate_inputs(X, rate)

    X = np.asarray(X, dtype=np.float32)
    n, d, l = X.shape
    X_missing = X.copy()
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    if rate > 0.0:
        num_remove = int(l * rate)
        if num_remove > 0:
            for i in range(n):
                removed = (
                    torch.randperm(l, generator=generator)[:num_remove]
                    .sort()
                    .values
                    .cpu()
                    .numpy()
                )
                X_missing[i, :, removed] = np.nan

    first_channel_mask = (~np.isnan(X_missing[:, :1, :])).astype(np.float32)
    mask_ndl = np.repeat(first_channel_mask, d, axis=1)
    delta_first_channel = _compute_delta(first_channel_mask)
    delta_ndl = np.repeat(delta_first_channel, d, axis=1)
    return X_missing, mask_ndl, delta_ndl


def apply_missingness(
    X: np.ndarray,
    rate: float,
    mechanism: str = "MCAR",
    seed: int = 0,
    channel_sync: bool = False,
    reference_X: Optional[np.ndarray] = None,
    return_stats: bool = False,
) -> tuple:
    """Apply synthetic missingness to [N, D, L] input data.

    Mechanism guidance:
    - MCAR: default/recommended packaged baseline.
    - MAR/MNAR: advanced missingness modes intended mainly for comparative analysis.
    - BLOCK: one contiguous temporal gap per sample/channel (or per sample if channel_sync=True).
    - BURST: several short contiguous temporal gaps per sample/channel (or per sample if channel_sync=True).
    - CHANNEL_DROP: one seed-stable global channel subset; channel_sync is not an axis.

    Returns:
        X_missing: [N, D, L], with missing values as NaN.
        mask_observed: [N, D, L], 1.0 observed / 0.0 missing.
        stats (only when return_stats=True): mechanism-specific provenance.
    """
    _validate_inputs(X, rate)

    mechanism = mechanism.upper()
    if mechanism not in SUPPORTED_MISSINGNESS_MECHANISMS:
        raise ValueError(f"Unsupported mechanism: {mechanism}")
    if mechanism == "CHANNEL_DROP":
        if channel_sync:
            raise ValueError("CHANNEL_DROP does not define channel_sync; use False")
        from .channel_drop import apply_channel_drop

        X_missing, mask_observed, stats = apply_channel_drop(X, rate, seed=seed)
        stats = dict(stats)
        stats["all_missing_sequence_count"] = int(
            np.count_nonzero(~(mask_observed > 0).any(axis=(1, 2)))
        )
        if return_stats:
            return X_missing, mask_observed, stats
        return X_missing, mask_observed

    X = np.asarray(X, dtype=np.float32)
    n, d, l = X.shape
    existing_missing = np.isnan(X)
    rng = np.random.default_rng(seed)

    sampled_missing = np.zeros((n, d, l), dtype=bool)

    if rate > 0.0 and mechanism == "MCAR":
        num_remove = int(l * rate)
        if num_remove > 0:
            if channel_sync:
                for i in range(n):
                    removed = rng.choice(l, size=num_remove, replace=False)
                    sampled_missing[i, :, removed] = True
            else:
                for i in range(n):
                    for j in range(d):
                        removed = rng.choice(l, size=num_remove, replace=False)
                        sampled_missing[i, j, removed] = True
    elif rate > 0.0 and mechanism in {"BLOCK", "BURST"}:
        num_remove = int(l * rate)
        sampler = _sample_block_indices if mechanism == "BLOCK" else _sample_burst_indices
        if num_remove > 0:
            if channel_sync:
                for i in range(n):
                    removed = sampler(rng, l, num_remove)
                    sampled_missing[i, :, removed] = True
            else:
                for i in range(n):
                    for j in range(d):
                        removed = sampler(rng, l, num_remove)
                        sampled_missing[i, j, removed] = True
    elif rate > 0.0:
        if mechanism == "MAR":
            prob = _build_mar_probability(X, rate, reference_X=reference_X)
        else:
            prob = _build_mnar_probability(X, rate, reference_X=reference_X)

        if channel_sync:
            # Aggregate channel-level probabilities to time-level probabilities.
            prob_time = np.clip(np.nanmean(prob, axis=1), 0.0, 0.99)  # [N, L]
            sampled_time = rng.random((n, l), dtype=np.float32) < prob_time
            sampled_missing = np.repeat(sampled_time[:, None, :], d, axis=1)
        else:
            sampled_missing = rng.random((n, d, l), dtype=np.float32) < prob.astype(np.float32)

    sampled_missing, fully_masked_before_floor_indices = _enforce_sequence_observation_floor(
        existing_missing,
        sampled_missing,
        rng,
        channel_sync=channel_sync,
    )
    missing = existing_missing | sampled_missing
    X_missing = X.copy()
    X_missing[missing] = np.nan
    mask_observed = (~np.isnan(X_missing)).astype(np.float32)

    if return_stats:
        stats = {
            "fully_masked_before_floor": int(len(fully_masked_before_floor_indices)),
            "fully_masked_before_floor_indices": fully_masked_before_floor_indices,
            "n_sequences": int(n),
        }
        return X_missing, mask_observed, stats
    return X_missing, mask_observed


def preprocess(
    X: Any,
    missing_rate: Optional[float] = None,
    interpolate: Optional[str] = None,
    use_intensity: Optional[bool] = None,
    seed: int = 0,
    channel_sync: bool = False,
    mechanism: str = "MCAR",
    model_name: Optional[str] = None,
    timestamps: Optional[Any] = None,
    timestamp_delta_scale: Optional[float] = None,
    missingness_reference: Optional[Any] = None,
    control_value_transform: str = "identity",
    control_value_train_indices: Optional[Any] = None,
    control_value_transform_state: Optional[dict] = None,
    return_control_value_transform: bool = False,
    control_time_source: str = "auto",
    control_time_train_indices: Optional[Any] = None,
    control_time_transform_state: Optional[dict] = None,
    return_control_time_transform: bool = False,
):
    """Preprocess data and build interpolation features.

    Args:
        X: Input data [N, D, L] as numpy array or torch tensor.
        missing_rate: Missing rate in [0, 1). If None or 0, no additional injection.
        interpolate: One of {"natural", "linear", "rectilinear", "cubic", "hermite"}.
            Defaults to "cubic" if neither this nor `model_name` is set.
        use_intensity: Whether to include intensity channels. If None, derived
            from `model_name` (or defaults to True for backward compatibility).
        seed: RNG seed for missingness.
        channel_sync: If True, synchronize missing indices across channels.
        mechanism: Missing mechanism {"MCAR", "MAR", "MNAR", "BLOCK", "BURST",
            "CHANNEL_DROP"}.
            MCAR is default/recommended. MAR/MNAR are value/probability-oriented modes;
            BLOCK/BURST are temporal-gap robustness modes.
        model_name: Optional model name. When set, `interpolate` and
            `use_intensity` default to the model's expected configuration via
            `get_data_config(model_name)`. Explicit kwargs still override.
        timestamps: Optional per-sample timestamp grid [N, L]. When provided,
            `X_delta` uses elapsed timestamp distance instead of index distance.
        timestamp_delta_scale: Positive scale used to express timestamp deltas in
            index-step equivalent units. If omitted, the dataset median positive
            adjacent timestamp gap is used.
        missingness_reference: Optional [N_ref, D, L] array used as the reference
            distribution for MAR/MNAR synthetic masks. Script entrypoints pass the
            training split here to avoid test-to-train mask-stat leakage.
        control_value_transform: ``identity`` (default) or train-observed
            ``standardize`` for value channels before coefficient construction.
            ``train_standardize`` is accepted as an input alias.
        control_value_train_indices: Training sample indices used to fit
            ``standardize`` after missingness is applied.
        control_value_transform_state: Optional previously fitted state. Robustness
            evaluation uses this to keep training statistics frozen.
        return_control_value_transform: If true, append the fitted state to the
            usual four return values.
        control_time_source: ``auto`` (physical when timestamps are supplied),
            ``physical``, or ``index``. Physical time is made relative to each
            sample origin and expressed in a train-fitted median-gap unit.
        control_time_train_indices: Training sample indices used to fit the
            physical time unit.
        control_time_transform_state: Optional previously fitted state to reuse.
        return_control_time_transform: If true, append the control-time state.

    Returns:
        X_missing: torch.Tensor [N, L, D]
        X_mask: torch.Tensor [N, L, D]
        X_delta: torch.Tensor [N, L, D]
        coeffs: torch.Tensor interpolation coefficients
    """
    import torch

    resolved_interpolate = interpolate
    resolved_use_intensity = use_intensity
    if model_name is not None:
        from .config.model_data_policy import get_data_config

        data_config = get_data_config(model_name)
        if resolved_interpolate is None:
            resolved_interpolate = data_config.get("interpolation", "cubic")
        if resolved_use_intensity is None:
            resolved_use_intensity = bool(data_config.get("use_intensity", False))

    if resolved_interpolate is None:
        resolved_interpolate = "cubic"
    if resolved_use_intensity is None:
        resolved_use_intensity = True

    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy().astype(np.float32)
    else:
        X_np = np.asarray(X, dtype=np.float32)
    reference_np = None if missingness_reference is None else np.asarray(missingness_reference, dtype=np.float32)

    rate = 0.0 if missing_rate is None else float(missing_rate)
    _validate_inputs(X_np, rate)
    timestamps_np = None if timestamps is None else _validate_timestamps_for_preprocess(timestamps, X_np)
    requested_time_source = normalize_control_time_source(control_time_source)
    if control_time_transform_state is None:
        control_time_state = fit_control_time_transform(
            timestamps_np,
            control_time_train_indices,
            source=requested_time_source,
            sequence_length=int(X_np.shape[2]),
        )
    else:
        control_time_state = dict(control_time_transform_state)
        state_requested = normalize_control_time_source(
            control_time_state.get("requested_source", control_time_state.get("effective_source"))
        )
        state_effective = str(control_time_state.get("effective_source", "")).strip().lower()
        requested_effective = (
            "physical" if requested_time_source == "auto" and timestamps_np is not None
            else "index" if requested_time_source == "auto"
            else requested_time_source
        )
        if state_effective != requested_effective:
            raise ValueError(
                "control-time transform state/source mismatch: "
                f"state={state_effective!r}, requested={requested_effective!r}"
            )
        control_time_state["requested_source"] = state_requested
    control_time_state["sample_count"] = int(X_np.shape[0])
    control_time_feature = apply_control_time_transform(
        timestamps_np,
        control_time_state,
        sequence_length=int(X_np.shape[2]),
    )

    if rate > 0.0:
        X_missing_np, mask_ndl = apply_missingness(
            X_np,
            rate=rate,
            mechanism=mechanism,
            seed=seed,
            channel_sync=channel_sync,
            reference_X=reference_np,
        )
    else:
        X_missing_np = X_np.copy()
        mask_ndl = (~np.isnan(X_missing_np)).astype(np.float32)

    if timestamps_np is not None:
        resolved_timestamp_delta_scale = _resolve_timestamp_delta_scale(timestamps_np, timestamp_delta_scale)
        delta_ndl = _compute_timestamp_delta(
            mask_ndl,
            timestamps_np,
            timestamp_delta_scale=resolved_timestamp_delta_scale,
        )
    else:
        delta_ndl = _compute_delta(mask_ndl)
    return _finalize_preprocess_outputs(
        X_missing_np,
        mask_ndl,
        delta_ndl,
        interpolate=resolved_interpolate,
        use_intensity=resolved_use_intensity,
        time_grid_np=None,
        control_time_feature=control_time_feature,
        control_time_transform_state=control_time_state,
        return_control_time_transform=return_control_time_transform,
        control_value_transform=control_value_transform,
        control_value_train_indices=control_value_train_indices,
        control_value_transform_state=control_value_transform_state,
        return_control_value_transform=return_control_value_transform,
    )


def preprocess_compat(
    X: Any,
    missing_rate: Optional[float] = None,
    interpolate: str = "natural",
    use_intensity: bool = True,
    seed: int = 0,
):
    """Preprocess helper for the alternative (compatibility) missingness scheme."""
    import torch

    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy().astype(np.float32)
    else:
        X_np = np.asarray(X, dtype=np.float32)

    rate = 0.0 if missing_rate is None else float(missing_rate)
    _validate_inputs(X_np, rate)

    if rate > 0.0:
        X_missing_np, mask_ndl, delta_ndl = apply_missingness_compat(
            X_np,
            rate=rate,
            seed=seed,
        )
    else:
        X_missing_np = X_np.copy()
        mask_ndl = (~np.isnan(X_missing_np)).astype(np.float32)
        delta_ndl = _compute_delta(mask_ndl)

    return _finalize_preprocess_outputs(
        X_missing_np,
        mask_ndl,
        delta_ndl,
        interpolate=interpolate,
        use_intensity=use_intensity,
    )


apply_missingness_refactoring_compat = apply_missingness_compat
preprocess_refactoring_compat = preprocess_compat
