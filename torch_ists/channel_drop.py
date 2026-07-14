"""Deterministic channel-failure perturbation for robustness evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


UNIVARIATE_EXCLUSION_REASON = "univariate-channel-drop-not-interpretable"


def build_channel_drop_plan(
    *,
    channel_count: int,
    rate: float,
    seed: int,
) -> dict[str, Any]:
    """Return the exact seed-bound channel-drop provenance plan."""
    channel_count = int(channel_count)
    rate_value = float(rate)
    if channel_count < 1:
        raise ValueError("channel_count must be positive")
    if not (0.0 <= rate_value < 1.0):
        raise ValueError(f"rate must be in [0, 1), got {rate}")
    requested_count = int(np.floor(channel_count * rate_value + 0.5))
    if channel_count == 1 and rate_value > 0.0:
        return {
            "channel_count": 1,
            "requested_channel_drop_count": requested_count,
            "dropped_channel_count": 0,
            "dropped_channel_indices": [],
            "channel_permutation": [0],
            "minimum_one_channel_forced": False,
            "maximum_one_channel_retained": bool(requested_count >= 1),
            "channel_count_quantized": True,
            "realized_channel_drop_rate": 0.0,
            "channel_drop_applicable": False,
            "analysis_exclusion_reason": UNIVARIATE_EXCLUSION_REASON,
        }
    dropped_count = (
        0 if rate_value == 0.0 else min(channel_count - 1, max(1, requested_count))
    )
    channel_order = np.random.default_rng(int(seed)).permutation(channel_count).astype(np.int64)
    dropped = np.sort(channel_order[:dropped_count])
    realized_rate = float(dropped_count / channel_count)
    return {
        "channel_count": channel_count,
        "requested_channel_drop_count": requested_count,
        "dropped_channel_count": dropped_count,
        "dropped_channel_indices": dropped.tolist(),
        "channel_permutation": channel_order.tolist(),
        "minimum_one_channel_forced": bool(rate_value > 0.0 and requested_count < 1),
        "maximum_one_channel_retained": bool(rate_value > 0.0 and requested_count >= channel_count),
        "channel_count_quantized": bool(
            rate_value > 0.0 and not np.isclose(realized_rate, rate_value)
        ),
        "realized_channel_drop_rate": realized_rate,
        "channel_drop_applicable": True,
        "analysis_exclusion_reason": None,
    }


def apply_channel_drop(
    X: np.ndarray,
    rate: float,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Drop a seed-stable, rate-nested channel subset from ``[N, D, L]`` data.

    Positive multivariate rates use nearest-integer channel counts, with at
    least one channel dropped and one retained. Positive univariate rates
    return explicit unchanged placeholders for later analysis exclusion.
    """
    values = np.asarray(X, dtype=np.float32)
    rate_value = float(rate)
    if values.ndim != 3:
        raise ValueError(f"X must be [N, D, L], got shape {values.shape}")
    if values.shape[1] < 1:
        raise ValueError("X must contain at least one channel")
    if not (0.0 <= rate_value < 1.0):
        raise ValueError(f"rate must be in [0, 1), got {rate}")

    n_sequences, channel_count, _ = values.shape
    observed_before = ~np.isnan(values)
    observed_value_count = int(np.count_nonzero(observed_before))
    plan = build_channel_drop_plan(
        channel_count=channel_count,
        rate=rate_value,
        seed=seed,
    )

    if plan["channel_drop_applicable"] is False:
        output = values.copy()
        return output, observed_before.astype(np.float32), {
            "mechanism": "CHANNEL_DROP",
            "n_sequences": int(n_sequences),
            **plan,
            "observed_value_count": observed_value_count,
            "realized_missing_count": 0,
            "realized_missing_rate": 0.0,
        }

    dropped = np.asarray(plan["dropped_channel_indices"], dtype=np.int64)
    output = values.copy()
    output[:, dropped, :] = np.nan

    synthetic_missing = observed_before & np.isnan(output)
    realized_missing_count = int(np.count_nonzero(synthetic_missing))
    stats: dict[str, Any] = {
        "mechanism": "CHANNEL_DROP",
        "n_sequences": int(n_sequences),
        **plan,
        "observed_value_count": observed_value_count,
        "realized_missing_count": realized_missing_count,
        "realized_missing_rate": (
            float(realized_missing_count / observed_value_count) if observed_value_count else 0.0
        ),
    }
    return output, (~np.isnan(output)).astype(np.float32), stats
