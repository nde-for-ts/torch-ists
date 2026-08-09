from __future__ import annotations

from typing import Any

import torch


_FIXED_GRID_METHODS = {"euler", "midpoint", "rk4"}


def _control_knot_grid_constructor(control_knots: torch.Tensor):
    """Build a torchdiffeq fixed grid containing every control knot."""
    frozen_knots = control_knots.detach().clone()

    def construct_grid(_func, _y0, requested_times):
        knots = frozen_knots.to(device=requested_times.device, dtype=requested_times.dtype)
        start = requested_times[0]
        end = requested_times[-1]
        control_interior = knots[(knots > start) & (knots < end)]
        requested_interior = requested_times[1:-1]
        return torch.unique(
            torch.cat(
                [
                    start.reshape(1),
                    requested_interior,
                    control_interior,
                    end.reshape(1),
                ]
            ),
            sorted=True,
        )

    return construct_grid


def apply_control_aware_cde_solver_defaults(
    times: torch.Tensor,
    control_knots: torch.Tensor,
    kwargs: dict[str, Any],
    *,
    piecewise_linear: bool,
    default_method: str = "rk4",
) -> dict[str, Any]:
    """Derive solver scheduling from the interpolation object's actual knots.

    Explicit caller options always win. Fixed-grid methods receive a grid that
    visits all control knots, including rectilinear expansion knots. Adaptive
    methods receive interior discontinuity times only for piecewise-linear
    controls.
    """
    if "method" not in kwargs:
        kwargs["method"] = default_method
    method = kwargs.get("method")
    options = kwargs.get("options")
    if options is None:
        options = {}

    if method in _FIXED_GRID_METHODS:
        if "step_size" not in options and "grid_constructor" not in options:
            options["grid_constructor"] = _control_knot_grid_constructor(control_knots)
            kwargs["options"] = options
        return kwargs

    if piecewise_linear and "jump_t" not in options:
        interior = control_knots[1:-1]
        if interior.numel() > 0:
            options["jump_t"] = interior
            kwargs["options"] = options
    return kwargs


def apply_advanced_cde_solver_defaults(
    times: torch.Tensor,
    kwargs: dict[str, Any],
    *,
    default_method: str = "rk4",
) -> dict[str, Any]:
    """Apply the package runtime defaults for packaged advanced-CDE models.

    The ANCDE/EXIT/LEAP low-level modules preserve upstream-style integer-grid
    fallbacks, including rk4 step_size=1.0 branches. Package
    preprocessing builds coefficients on normalized [0, 1] times, so the
    active wrappers install the shared knot-containing fixed grid before
    those compatibility fallbacks can fire.
    """

    return apply_control_aware_cde_solver_defaults(
        times,
        times,
        kwargs,
        piecewise_linear=False,
        default_method=default_method,
    )
