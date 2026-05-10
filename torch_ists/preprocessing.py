"""Self-contained preprocessing utilities for torch_ists."""

from typing import Any, Optional, Tuple

import numpy as np

MISSINGNESS_EXPERIMENTAL = True


def _validate_inputs(X: np.ndarray, rate: float) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must be [N, D, L], got shape {X.shape}")
    if not (0.0 <= rate < 1.0):
        raise ValueError(f"rate must be in [0, 1), got {rate}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_mar_probability(X: np.ndarray, rate: float) -> np.ndarray:
    # Feature-dependent missingness probability.
    feature_mean = np.nanmean(X, axis=2, keepdims=True)
    centered = feature_mean - np.nanmean(feature_mean)
    logits = centered / (np.nanstd(centered) + 1e-8)
    probs = _sigmoid(logits)
    probs = probs / (np.nanmean(probs) + 1e-8) * rate
    return np.clip(probs, 0.0, 0.99)


def _build_mnar_probability(X: np.ndarray, rate: float) -> np.ndarray:
    # Value-dependent missingness probability.
    norm = (X - np.nanmean(X)) / (np.nanstd(X) + 1e-8)
    probs = _sigmoid(norm)
    probs = probs / (np.nanmean(probs) + 1e-8) * rate
    return np.clip(probs, 0.0, 0.99)


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


def _finalize_preprocess_outputs(
    X_missing_np: np.ndarray,
    mask_ndl: np.ndarray,
    delta_ndl: np.ndarray,
    *,
    interpolate: str,
    use_intensity: bool,
):
    import torch
    import torchcde

    from .module.cde import controldiffeq

    X_missing = torch.from_numpy(X_missing_np).to(torch.float32).permute(0, 2, 1)  # [N, L, D]
    X_mask = torch.from_numpy(mask_ndl).to(torch.float32).permute(0, 2, 1)         # [N, L, D]
    X_delta = torch.from_numpy(delta_ndl).to(torch.float32).permute(0, 2, 1)       # [N, L, D]

    seq_len = X_missing.shape[1]
    times = torch.linspace(0, 1, seq_len, dtype=X_missing.dtype)

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


def apply_missingness_compat(
    X: np.ndarray,
    rate: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce reference missingness semantics for compatibility checks only."""
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply synthetic missingness to [N, D, L] input data.

    Mechanism guidance:
    - MCAR: default/recommended production baseline.
    - MAR/MNAR: pseudo-level experimental implementations for analysis only.
      Keep `MISSINGNESS_EXPERIMENTAL=True` until theoretical validation is complete.

    Returns:
        X_missing: [N, D, L], with missing values as NaN.
        mask_observed: [N, D, L], 1.0 observed / 0.0 missing.
    """
    _validate_inputs(X, rate)

    mechanism = mechanism.upper()
    if mechanism not in {"MCAR", "MAR", "MNAR"}:
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
    elif rate > 0.0:
        if mechanism == "MAR":
            prob = _build_mar_probability(X, rate)
        else:
            prob = _build_mnar_probability(X, rate)

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
):
    """Preprocess data and build interpolation features.

    Args:
        X: Input data [N, D, L] as numpy array or torch tensor.
        missing_rate: Missing rate in [0, 1). If None or 0, no additional injection.
        interpolate: One of {"natural", "linear", "rectilinear", "cubic", "hermite"}.
            Defaults to "natural" if neither this nor `model_name` is set.
        use_intensity: Whether to include intensity channels. If None, derived
            from `model_name` (or defaults to True for backward compatibility).
        seed: RNG seed for missingness.
        channel_sync: If True, synchronize missing indices across channels.
        mechanism: Missing mechanism {"MCAR", "MAR", "MNAR"}.
            MCAR is default/recommended. MAR/MNAR are pseudo-level experimental modes.
        model_name: Optional model name. When set, `interpolate` and
            `use_intensity` default to the model's expected configuration via
            `get_data_config(model_name)`. Explicit kwargs still override.

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
            resolved_interpolate = data_config.get("interpolation", "natural")
        if resolved_use_intensity is None:
            resolved_use_intensity = bool(data_config.get("use_intensity", False))

    if resolved_interpolate is None:
        resolved_interpolate = "natural"
    if resolved_use_intensity is None:
        resolved_use_intensity = True

    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy().astype(np.float32)
    else:
        X_np = np.asarray(X, dtype=np.float32)

    rate = 0.0 if missing_rate is None else float(missing_rate)
    _validate_inputs(X_np, rate)

    if rate > 0.0:
        X_missing_np, mask_ndl = apply_missingness(
            X_np,
            rate=rate,
            mechanism=mechanism,
            seed=seed,
            channel_sync=channel_sync,
        )
    else:
        X_missing_np = X_np.copy()
        mask_ndl = (~np.isnan(X_missing_np)).astype(np.float32)

    delta_ndl = _compute_delta(mask_ndl)
    return _finalize_preprocess_outputs(
        X_missing_np,
        mask_ndl,
        delta_ndl,
        interpolate=resolved_interpolate,
        use_intensity=resolved_use_intensity,
    )


def preprocess_compat(
    X: Any,
    missing_rate: Optional[float] = None,
    interpolate: str = "natural",
    use_intensity: bool = True,
    seed: int = 0,
):
    """Compatibility preprocess helper that matches reference semantics."""
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
