"""Validation for the optional external Neural Flows reference backend."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_PROVENANCE_PATH = Path(__file__).with_name("NFE_PROVENANCE.json")
_PROVENANCE = json.loads(_PROVENANCE_PATH.read_text(encoding="utf-8"))

PINNED_NFE_REPO = str(_PROVENANCE["source_repo"])
PINNED_NFE_COMMIT = str(_PROVENANCE["source_commit"])
PINNED_NFE_VERSION = str(_PROVENANCE["source_distribution"]).split("==", 1)[1]
PINNED_STRIBOR_VERSION = str(_PROVENANCE["dependencies"]["stribor"])
EXPECTED_NFE_FILES = {
    str(path): str(digest) for path, digest in _PROVENANCE["files"].items()
}

NFE_INSTALL_COMMAND = (
    'python -m pip install "stribor==0.1.0" "torchdiffeq>=0.2.2"\n'
    "python -m pip install --no-deps "
    '"git+https://github.com/mbilos/neural-flows-experiments.git'
    f"@{PINNED_NFE_COMMIT}\""
)


class NFEInstallationError(RuntimeError):
    """Raised when the requested NFE reference installation is not exact."""


def _normalize_repo_url(value: object) -> str:
    raw = str(value or "").strip()
    if raw.startswith("git+"):
        raw = raw[4:]
    parsed = urlsplit(raw)
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_direct_url(distribution: metadata.Distribution) -> dict[str, Any]:
    text = distribution.read_text("direct_url.json")
    if not text:
        raise ValueError("nfe installation has no PEP 610 direct_url.json metadata")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("nfe direct_url.json must contain a JSON object")
    return payload


def validate_nfe_installation() -> dict[str, Any]:
    """Validate the user-installed NFE checkout without importing its code."""

    errors: list[str] = []
    try:
        nfe_distribution = metadata.distribution("nfe")
    except metadata.PackageNotFoundError:
        nfe_distribution = None
        errors.append("nfe distribution is not installed")

    try:
        stribor_version = metadata.distribution("stribor").version
    except metadata.PackageNotFoundError:
        stribor_version = None
        errors.append("stribor distribution is not installed")
    if stribor_version is not None and str(stribor_version) != PINNED_STRIBOR_VERSION:
        errors.append(
            f"stribor version must be {PINNED_STRIBOR_VERSION}, found {stribor_version}"
        )

    try:
        metadata.distribution("torchdiffeq")
    except metadata.PackageNotFoundError:
        errors.append("torchdiffeq distribution is not installed")

    if nfe_distribution is not None:
        if str(nfe_distribution.version) != PINNED_NFE_VERSION:
            errors.append(
                f"nfe version must be {PINNED_NFE_VERSION}, found {nfe_distribution.version}"
            )
        try:
            direct_url = _read_direct_url(nfe_distribution)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
        else:
            actual_repo = _normalize_repo_url(direct_url.get("url"))
            expected_repo = _normalize_repo_url(PINNED_NFE_REPO)
            if actual_repo != expected_repo:
                errors.append(
                    f"nfe source repository must be {PINNED_NFE_REPO}, found {actual_repo or 'unknown'}"
                )
            vcs_info = direct_url.get("vcs_info")
            actual_vcs = vcs_info.get("vcs") if isinstance(vcs_info, dict) else None
            if actual_vcs != "git":
                errors.append(f"nfe source control must be git, found {actual_vcs or 'unknown'}")
            actual_commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
            if actual_commit != PINNED_NFE_COMMIT:
                errors.append(
                    f"nfe source commit must be {PINNED_NFE_COMMIT}, "
                    f"found {actual_commit or 'unknown'}"
                )

        for relative_path, expected_hash in EXPECTED_NFE_FILES.items():
            source_path = Path(nfe_distribution.locate_file(relative_path))
            if not source_path.is_file():
                errors.append(f"nfe source file is missing: {relative_path}")
                continue
            actual_hash = _sha256_file(source_path)
            if actual_hash != expected_hash:
                errors.append(f"nfe source file differs from the pinned commit: {relative_path}")

    return {
        "valid": not errors,
        "source_repo": PINNED_NFE_REPO,
        "source_commit": PINNED_NFE_COMMIT,
        "nfe_version": PINNED_NFE_VERSION,
        "stribor_version": PINNED_STRIBOR_VERSION,
        "errors": errors,
    }


def require_nfe_installation() -> dict[str, Any]:
    """Return validated provenance or raise with an actionable install command."""

    result = validate_nfe_installation()
    if result["valid"]:
        return result
    detail = "; ".join(result["errors"])
    raise NFEInstallationError(
        "The optional NFE reference backend requires the pinned external checkout. "
        f"Validation failed: {detail}. Install it with:\n{NFE_INSTALL_COMMAND}"
    )


__all__ = [
    "EXPECTED_NFE_FILES",
    "NFE_INSTALL_COMMAND",
    "NFEInstallationError",
    "PINNED_NFE_COMMIT",
    "PINNED_NFE_REPO",
    "PINNED_NFE_VERSION",
    "PINNED_STRIBOR_VERSION",
    "require_nfe_installation",
    "validate_nfe_installation",
]
