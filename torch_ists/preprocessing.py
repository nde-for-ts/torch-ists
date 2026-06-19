"""Self-contained preprocessing utilities for torch_ists."""

from typing import Any, Optional, Tuple
import warnings

import numpy as np

MISSINGNESS_EXPERIMENTAL = True
SUPPORTED_MISSINGNESS_MECHANISMS = {"MCAR", "MAR", "MNAR", "BLOCK", "BURST"}
SUPPORTED_IMPUTATION_STRATEGIES = ("zero", "last_observed", "channel_mean")


def _require_torchcde():
    try:
        import torchcde
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != "torchcde":
            raise
        raise ImportError(
            "preprocess() requires optional dependency 'torchcde' to build interpolation "
            "coefficients. Install `torchcde` directly or use `pip install \"torch-ists[cde]\"` "
            "or `pip install \"torch-ists[all]\"`."
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
    timestamps_np = np.asarray(timestamps, dtype=np.float32)
    expected_shape = (int(X_np.shape[0]), int(X_np.shape[2]))
    if timestamps_np.ndim != 2:
        raise ValueError(f"timestamps must be 2D [N, L], got shape {timestamps_np.shape}")
    if tuple(timestamps_np.shape) != expected_shape:
        raise ValueError(f"timestamps shape mismatch: expected {expected_shape}, got {timestamps_np.shape}")
    for row in timestamps_np:
        finite_row = row[np.isfinite(row)]
        if finite_row.size > 1 and np.any(np.diff(finite_row) < 0):
            raise ValueError("timestamps must be non-decreasing within each sample")
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
    """Compute scale-normalized elapsed-time delta tensor with shape [N, D, L]."""
    observed = mask_observed_ndl.astype(bool)
    timestamps = np.asarray(timestamps_nl, dtype=np.float32)
    n, d, l = observed.shape
    delta = np.zeros((n, d, l), dtype=np.float32)

    for i in range(n):
        times_i = timestamps[i]
        finite_i = np.isfinite(times_i)
        for j in range(d):
            last_observed_time = None
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
    time_feat = times.view(1, seq_len, 1).expand(X_missing.shape[0], seq_len, 1)

    values_t = torch.cat([time_feat, X_missing], dim=-1)
    values_ti = torch.cat([time_feat, intensity, X_missing], dim=-1)

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

    return X_missing, X_mask, X_delta, coeffs


def apply_missingness_refactoring_compat(
    X: np.ndarray,
    rate: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce root-refactoring missingness semantics for parity diagnosis only."""
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply synthetic missingness to [N, D, L] input data.

    Mechanism guidance:
    - MCAR: default/recommended packaged baseline.
    - MAR/MNAR: advanced missingness modes intended mainly for comparative analysis.
    - BLOCK: one contiguous temporal gap per sample/channel (or per sample if channel_sync=True).
    - BURST: several short contiguous temporal gaps per sample/channel (or per sample if channel_sync=True).

    Returns:
        X_missing: [N, D, L], with missing values as NaN.
        mask_observed: [N, D, L], 1.0 observed / 0.0 missing.
    """
    _validate_inputs(X, rate)

    mechanism = mechanism.upper()
    if mechanism not in SUPPORTED_MISSINGNESS_MECHANISMS:
        raise ValueError(f"Unsupported mechanism: {mechanism}")

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

    missing = existing_missing | sampled_missing
    X_missing = X.copy()
    X_missing[missing] = np.nan
    mask_observed = (~np.isnan(X_missing)).astype(np.float32)

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
        mechanism: Missing mechanism {"MCAR", "MAR", "MNAR", "BLOCK", "BURST"}.
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
    )


def preprocess_refactoring_compat(
    X: Any,
    missing_rate: Optional[float] = None,
    interpolate: str = "natural",
    use_intensity: bool = True,
    seed: int = 0,
):
    """Parity-only preprocess helper that matches root-refactoring semantics."""
    import torch

    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy().astype(np.float32)
    else:
        X_np = np.asarray(X, dtype=np.float32)

    rate = 0.0 if missing_rate is None else float(missing_rate)
    _validate_inputs(X_np, rate)

    if rate > 0.0:
        X_missing_np, mask_ndl, delta_ndl = apply_missingness_refactoring_compat(
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


apply_missingness_compat = apply_missingness_refactoring_compat
preprocess_compat = preprocess_refactoring_compat
