"""RNG utilities for runtime."""

import random
from typing import Any, Dict, Optional

import numpy as np
import torch

DETERMINISTIC_MODES = {"off", "warn", "strict"}


def _to_uint8_list(array_like: Any) -> list:
    """Convert array-like to list of uint8 (for RNG state serialization)."""
    arr = np.asarray(array_like, dtype=np.uint8)
    return arr.tolist()


def _to_uint8_tensor(data: Any) -> torch.Tensor:
    """Convert data to CPU uint8 tensor for torch RNG state APIs."""
    return torch.tensor(_to_uint8_list(data), dtype=torch.uint8, device="cpu")


def _resolve_deterministic_mode(
    deterministic_mode: Optional[str],
    deterministic: Optional[str],
) -> str:
    if deterministic_mode is not None and deterministic is not None:
        if deterministic_mode != deterministic:
            raise ValueError(
                "deterministic_mode and deterministic are both set but mismatch: "
                f"{deterministic_mode} != {deterministic}"
            )
    mode = deterministic_mode if deterministic_mode is not None else deterministic
    mode = "off" if mode is None else mode
    if mode not in DETERMINISTIC_MODES:
        raise ValueError(
            f"deterministic mode must be one of {DETERMINISTIC_MODES}, got {mode}"
        )
    return mode


def _get_backend_deterministic_state() -> Dict[str, Any]:
    """Snapshot backend flags that affect deterministic execution."""
    state = {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled"):
        state["deterministic_algorithms_warn_only"] = bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
    return state


def _set_backend_deterministic_state(state: Dict[str, Any]) -> None:
    """Restore backend flags that affect deterministic execution."""
    torch.backends.cudnn.deterministic = bool(state.get("cudnn_deterministic", False))
    torch.backends.cudnn.benchmark = bool(state.get("cudnn_benchmark", False))
    deterministic_algorithms = bool(state.get("deterministic_algorithms", False))
    warn_only = bool(state.get("deterministic_algorithms_warn_only", False))
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    else:
        torch.use_deterministic_algorithms(False)


def set_global_seed(
    seed: int,
    deterministic_mode: Optional[str] = None,
    include_cuda: bool = True,
    deterministic: Optional[str] = None,
    warn_only: bool = True,
    set_torch: bool = True,
    set_np: bool = True,
    set_random: bool = True,
) -> Dict[str, Any]:
    """Set global RNG seeds and return execution metadata."""
    mode = _resolve_deterministic_mode(deterministic_mode, deterministic)

    if set_random:
        random.seed(seed)
    if set_np:
        np.random.seed(seed)

    torch_available = bool(set_torch)
    cuda_available = bool(torch.cuda.is_available()) if torch_available else False
    deterministic_enabled = False

    if torch_available:
        torch.manual_seed(seed)
        if include_cuda and cuda_available:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if mode == "warn":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic_enabled = True
        elif mode == "strict":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=False)
            deterministic_enabled = True
        elif mode == "off":
            # Keep backend flags untouched in off mode for backward compatibility.
            deterministic_enabled = False

    return {
        "seed": int(seed),
        "mode": str(mode),
        "deterministic_mode": str(mode),
        "torch_available": bool(torch_available),
        "cuda_available": bool(cuda_available),
        "include_cuda": bool(include_cuda),
        "deterministic_enabled": bool(deterministic_enabled),
        "warn_only": bool(warn_only),
    }


def get_rng_state() -> Dict[str, Any]:
    """Get current RNG states from all sources."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": _to_uint8_list(torch.get_rng_state()),
    }

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        cuda_states = []
        for i in range(torch.cuda.device_count()):
            cuda_state = torch.cuda.get_rng_state(device=i)
            cuda_states.append(_to_uint8_list(cuda_state))
        state["torch_cuda"] = cuda_states
    else:
        state["torch_cuda"] = []

    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states to all sources."""
    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np.random.set_state(state["numpy"])

    if "torch" in state:
        torch_state = _to_uint8_tensor(state["torch"])
        torch.set_rng_state(torch_state)

    if "torch_cuda" in state and torch.cuda.is_available():
        for i, cuda_state_list in enumerate(state["torch_cuda"]):
            if i < torch.cuda.device_count():
                cuda_state = _to_uint8_tensor(cuda_state_list)
                torch.cuda.set_rng_state(cuda_state, device=i)


class SeedContext:
    """Context manager for temporary seed changes."""

    def __init__(
        self,
        seed: int,
        deterministic_mode: Optional[str] = None,
        include_cuda: bool = True,
        deterministic: Optional[str] = None,
    ):
        self.seed = seed
        self.mode = _resolve_deterministic_mode(deterministic_mode, deterministic)
        self.include_cuda = bool(include_cuda)
        self.saved_state = None
        self.saved_backend_state = None

    def __enter__(self):
        self.saved_state = get_rng_state()
        self.saved_backend_state = _get_backend_deterministic_state()
        set_global_seed(
            self.seed,
            deterministic_mode=self.mode,
            include_cuda=self.include_cuda,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.saved_state is not None:
            set_rng_state(self.saved_state)
        if self.saved_backend_state is not None:
            _set_backend_deterministic_state(self.saved_backend_state)
        return False


__all__ = [
    "SeedContext",
    "get_rng_state",
    "set_global_seed",
    "set_rng_state",
]
