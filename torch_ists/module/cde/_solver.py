from __future__ import annotations

from typing import Any

import torch


_FIXED_GRID_METHODS = {"euler", "midpoint", "rk4"}


def apply_advanced_cde_solver_defaults(
    times: torch.Tensor,
    kwargs: dict[str, Any],
    *,
    default_method: str = "rk4",
) -> dict[str, Any]:
    """Apply the package runtime defaults for promoted advanced-CDE models.

    The ANCDE/EXIT/LEAP low-level modules preserve upstream-style integer-grid
    fallbacks, including rk4 step_size=1.0 branches. Package
    preprocessing builds coefficients on normalized [0, 1] times, so the
    active wrappers prefill a data-driven fixed-grid step size before those
    provenance-only fallbacks can fire.
    """

    if "method" not in kwargs:
        kwargs["method"] = default_method

    method = kwargs.get("method")
    if method not in _FIXED_GRID_METHODS:
        return kwargs

    options = kwargs.get("options")
    if options is None:
        options = {}
        kwargs["options"] = options

    if "step_size" in options or "grid_constructor" in options:
        return kwargs

    if times.numel() < 2:
        return kwargs

    time_diffs = times[1:] - times[:-1]
    options["step_size"] = time_diffs.min().item()
    return kwargs
