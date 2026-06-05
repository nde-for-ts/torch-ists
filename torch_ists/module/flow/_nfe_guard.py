"""NFE vendor provenance and runtime guard helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable


PINNED_NFE_REPO = "https://github.com/mbilos/neural-flows-experiments"
PINNED_NFE_COMMIT = "bd19f7c92461e83521e268c1a235ef845a3dd963"
REQUIRED_VENDOR_FILES = ("flow.py", "gru.py", "ode.py", "__init__.py")
PROVENANCE_FILE = "PROVENANCE.json"
NFE_ENABLE_ENV = "TORCH_ISTS_ENABLE_NFE_BACKEND"


def _vendor_root() -> Path:
    return Path(__file__).resolve().parent / "_vendor_nfe"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_files(files: Iterable[dict]) -> Dict[str, dict]:
    normalized: Dict[str, dict] = {}
    for item in files:
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        normalized[path] = item
    return normalized


def validate_nfe_vendor_provenance() -> dict:
    root = _vendor_root()
    provenance_path = root / PROVENANCE_FILE
    errors: list[str] = []
    warnings: list[str] = []

    if not root.exists():
        errors.append(f"vendor root missing: {root}")
        return {
            "status": "fail",
            "repo": PINNED_NFE_REPO,
            "commit": PINNED_NFE_COMMIT,
            "vendor_root": str(root),
            "errors": errors,
            "warnings": warnings,
        }

    if not provenance_path.exists():
        errors.append(f"missing provenance file: {provenance_path}")
        return {
            "status": "fail",
            "repo": PINNED_NFE_REPO,
            "commit": PINNED_NFE_COMMIT,
            "vendor_root": str(root),
            "errors": errors,
            "warnings": warnings,
        }

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid provenance json: {exc}")
        return {
            "status": "fail",
            "repo": PINNED_NFE_REPO,
            "commit": PINNED_NFE_COMMIT,
            "vendor_root": str(root),
            "errors": errors,
            "warnings": warnings,
        }

    repo = str(provenance.get("source_repo", "")).strip()
    commit = str(provenance.get("source_commit", "")).strip()
    if repo != PINNED_NFE_REPO:
        errors.append(f"repo mismatch: expected={PINNED_NFE_REPO}, found={repo}")
    if commit != PINNED_NFE_COMMIT:
        errors.append(f"commit mismatch: expected={PINNED_NFE_COMMIT}, found={commit}")

    files = _normalize_files(provenance.get("files", []))
    for rel_path in REQUIRED_VENDOR_FILES:
        file_path = root / rel_path
        if not file_path.exists():
            errors.append(f"missing vendor file: {file_path}")
            continue
        if rel_path not in files:
            errors.append(f"missing provenance entry for {rel_path}")
            continue
        actual_hash = _sha256(file_path)
        declared_hash = str(files[rel_path].get("sha256", "")).strip()
        if declared_hash != actual_hash:
            errors.append(
                f"hash mismatch for {rel_path}: declared={declared_hash}, actual={actual_hash}"
            )

    license_info = provenance.get("license", {})
    approved = bool(license_info.get("approved_for_public_package", False))
    if not approved:
        errors.append(
            "license guard not approved (license.approved_for_public_package is false)"
        )
    if not str(license_info.get("note", "")).strip():
        warnings.append("license note is empty")

    return {
        "status": "pass" if not errors else "fail",
        "repo": PINNED_NFE_REPO,
        "commit": PINNED_NFE_COMMIT,
        "vendor_root": str(root),
        "provenance_file": str(provenance_path),
        "errors": errors,
        "warnings": warnings,
        "license_approved": approved,
    }


def ensure_nfe_backend_enabled() -> None:
    if os.environ.get(NFE_ENABLE_ENV, "0") != "1":
        raise RuntimeError(
            f"nfe backend is guarded. Set {NFE_ENABLE_ENV}=1 and run "
            "scripts/validate_nfe_vendor_provenance.py before enabling."
        )
    validation = validate_nfe_vendor_provenance()
    if validation["status"] != "pass":
        errors = "; ".join(validation.get("errors", []))
        raise RuntimeError(f"nfe backend provenance validation failed: {errors}")
