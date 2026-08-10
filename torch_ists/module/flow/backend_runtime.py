"""Run-level flow-backend selection and lineage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .operators import resolve_flow_backend
from .operators_stribor import StriborBackendUnavailableError, require_stribor


AUTO_FALLBACK_REASON = "optional_backend_unavailable"


@dataclass(frozen=True)
class FlowBackendSelection:
    requested_backend: str
    effective_backend: str
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        requested = resolve_flow_backend(self.requested_backend)
        effective = resolve_flow_backend(self.effective_backend)
        if effective == "auto":
            raise ValueError("effective_backend must be 'stribor' or 'native', not 'auto'")
        if requested != "auto" and requested != effective:
            raise ValueError("explicit flow backend requests cannot change effective backend")
        if self.fallback_reason is not None and not (
            requested == "auto" and effective == "native"
        ):
            raise ValueError("fallback_reason is valid only for auto-to-native selection")
        object.__setattr__(self, "requested_backend", requested)
        object.__setattr__(self, "effective_backend", effective)

    def as_manifest_fields(self) -> dict[str, str]:
        fields = {
            "requested_backend": self.requested_backend,
            "effective_backend": self.effective_backend,
        }
        if self.fallback_reason is not None:
            fields["fallback_reason"] = self.fallback_reason
        return fields


def resolve_flow_backend_once(requested: object | None) -> FlowBackendSelection:
    """Resolve availability before a search/run and return a frozen selection."""
    normalized = resolve_flow_backend(requested)
    if normalized == "native":
        return FlowBackendSelection("native", "native")
    try:
        require_stribor()
    except StriborBackendUnavailableError:
        if normalized == "stribor":
            raise
        return FlowBackendSelection("auto", "native", AUTO_FALLBACK_REASON)
    return FlowBackendSelection(normalized, "stribor")


def freeze_model_params(
    model_params: Mapping[str, object] | None,
    selection: FlowBackendSelection,
) -> dict[str, object]:
    """Pin construction to the effective backend; keep request at run level."""
    frozen = dict(model_params or {})
    frozen["flow_backend"] = selection.effective_backend
    return frozen


def assert_backend_agreement(
    expected: FlowBackendSelection,
    observed: FlowBackendSelection,
) -> None:
    if expected != observed:
        raise RuntimeError(
            "Flow backend selection differs between execution contexts: "
            f"expected={expected.as_manifest_fields()}, "
            f"observed={observed.as_manifest_fields()}"
        )


def assert_effective_backend_match(
    parameter_payload: Mapping[str, object],
    selection: FlowBackendSelection,
) -> None:
    recorded = parameter_payload.get("effective_backend")
    if recorded is None:
        raise ValueError(
            "Parameter artifact is missing effective_backend; flow execution "
            "requires a recorded backend selection."
        )
    if str(recorded) != selection.effective_backend:
        raise ValueError(
            "Parameter-search and model-run flow backends differ: "
            f"params={recorded!r}, run={selection.effective_backend!r}"
        )


__all__ = [
    "AUTO_FALLBACK_REASON",
    "FlowBackendSelection",
    "resolve_flow_backend_once",
    "freeze_model_params",
    "assert_backend_agreement",
    "assert_effective_backend_match",
]
