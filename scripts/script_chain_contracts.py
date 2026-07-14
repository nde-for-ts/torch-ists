#!/usr/bin/env python
"""Utilities for JSON run logs and related artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


SUPPORTED_TASKS = frozenset(
    {
        "classification",
        "regression",
        "forecasting",
        "imputation",
        "outlier_detection",
        "change_detection",
    }
)
SUPPORTED_TARGET_SOURCES = frozenset({"dataset", "surrogate", "label"})


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def generate_run_id(prefix: str = "run") -> str:
    # Include microseconds to avoid same-second collisions in manifest paths.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{stamp}"


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_parent_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tmp_path_for(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return path.with_name(f".{path.name}.{os.getpid()}.{stamp}.tmp")


def atomic_pickle_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_parent_dir(path.parent)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        with tmp_path.open("w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_parent_dir(path.parent)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise



_ABSOLUTE_PATH_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|/)[^\"'\s,;]+"
)
_OMITTED_MANIFEST_KEYS = frozenset(
    {
        "architecture_axes",
        "capacity_config",
        "capacity_metadata",
        "capacity_policy",
        "capacity_target",
        "capacity_target_name",
        "legacy_grid_version",
        "public_search_name",
        "search_algorithm_policy",
        "search_contract",
        "search_family",
        "search_grid_version",
    }
)


def portable_artifact_reference(value: Any, *, root: bool = False) -> Any:
    """Return a useful artifact reference without recording a machine-local root."""
    if value in {None, ""}:
        return value
    text = str(value)
    windows_absolute = bool(re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"))
    path = Path(text)
    if not path.is_absolute() and not windows_absolute:
        return text.replace("\\", "/")
    if path.is_absolute():
        try:
            relative = path.resolve().relative_to(Path.cwd().resolve())
        except (OSError, ValueError):
            relative = None
        if relative is not None:
            return relative.as_posix()
    parts = [part for part in text.replace("\\", "/").split("/") if part and not part.endswith(":")]
    if not parts:
        return ""
    keep = 1 if root else min(3, len(parts))
    return "/".join(parts[-keep:])


def _portable_text(value: str) -> str:
    return _ABSOLUTE_PATH_TOKEN_RE.sub(
        lambda match: str(portable_artifact_reference(match.group(0))),
        value,
    )


def _public_manifest_projection(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        projected = {}
        for raw_key, raw_value in value.items():
            normalized_key = str(raw_key).lower()
            if (
                normalized_key in _OMITTED_MANIFEST_KEYS
                or "profile" in normalized_key
                or "backend" in normalized_key
                or (normalized_key.startswith("schema_") and normalized_key.endswith("version"))
            ):
                continue
            projected[raw_key] = _public_manifest_projection(raw_value, key=normalized_key)
        return projected
    if isinstance(value, list):
        return [_public_manifest_projection(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [_public_manifest_projection(item, key=key) for item in value]
    if isinstance(value, str):
        if key == "dataset_npz" or (key and key.endswith("_root")):
            return portable_artifact_reference(value, root=True)
        if key and key.endswith("_path"):
            return portable_artifact_reference(value)
        return _portable_text(value)
    return value

def write_manifest(
    *,
    stage: str,
    contract_dir: str | Path,
    run_id: str,
    payload: Dict[str, Any],
    existing_policy: str = "fail",
) -> Tuple[Path, Dict[str, Any]]:
    policy = str(existing_policy or "fail").strip().lower()
    if policy not in {"fail", "reuse", "overwrite"}:
        raise ValueError("existing_policy must be one of {'fail', 'reuse', 'overwrite'}")

    out_dir = Path(contract_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stage}.manifest.json"
    manifest_exists = out_path.exists()
    if manifest_exists and policy == "fail":
        raise FileExistsError(
            f"manifest already exists: {out_path}. "
            "Use --manifest-existing-policy reuse or overwrite to proceed intentionally."
        )

    body = {
        "stage": stage,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "manifest_existing_policy": policy,
        **payload,
    }
    if manifest_exists and policy == "reuse":
        body["resumed_from_run_id"] = run_id
        body["reused_manifest_path"] = str(out_path)
    elif manifest_exists and policy == "overwrite":
        body["overwrote_existing_manifest"] = True
    body = _public_manifest_projection(body)
    atomic_write_text(out_path, json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path, body


def load_manifest(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _require_fields(doc: Dict[str, Any], required: List[str]) -> List[str]:
    errors: List[str] = []
    for key in required:
        if key not in doc:
            errors.append(f"missing field: {key}")
    return errors


def _is_finite_number(value: Any) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def _validate_target_source(record: Dict[str, Any], path: str, errors: List[str], task: str) -> None:
    target_source = str(record.get("target_source", "")).strip().lower()
    if task == "classification":
        if target_source and target_source not in SUPPORTED_TARGET_SOURCES:
            errors.append(f"{path}.target_source unsupported: {target_source!r}")
        return
    if not target_source:
        errors.append(f"{path}.target_source missing for task={task}")
        return
    if target_source not in SUPPORTED_TARGET_SOURCES:
        errors.append(
            f"{path}.target_source unsupported: {target_source!r}. "
            f"Supported sources are {sorted(SUPPORTED_TARGET_SOURCES)}"
        )
        return
    if task == "change_detection" and target_source != "dataset":
        errors.append(f"{path}.target_source must be 'dataset' for task=change_detection")


def _validate_metric_values(metrics: Dict[str, Any], *, task: str, path: str, errors: List[str]) -> None:
    if task == "classification":
        has_current_metrics = all(metric in metrics for metric in ("accuracy", "f1", "wf1"))
        has_compat_metrics = all(metric in metrics for metric in ("accuracy", "f1_weighted"))
        if not (has_current_metrics or has_compat_metrics):
            errors.append(
                f"{path}.metrics missing required keys: "
                "expect (accuracy,f1,wf1) or compatibility tuple (accuracy,f1_weighted)"
            )
            return

        required_metrics = (
            ("accuracy", "f1", "wf1")
            if has_current_metrics
            else ("accuracy", "f1_weighted")
        )
        for metric in required_metrics:
            if not _is_finite_number(metrics.get(metric)):
                errors.append(f"{path}.metrics.{metric} must be finite")
                continue
            value = float(metrics[metric])
            if value < 0.0 or value > 1.0:
                errors.append(f"{path}.metrics.{metric} out of range [0, 1]")

        if has_current_metrics and "f1_weighted" in metrics:
            if not _is_finite_number(metrics["f1_weighted"]):
                errors.append(f"{path}.metrics.f1_weighted must be finite")
            else:
                wf1 = float(metrics["wf1"])
                f1w = float(metrics["f1_weighted"])
                if abs(wf1 - f1w) > 1e-12:
                    errors.append(f"{path}.metrics alias mismatch: wf1={wf1} != f1_weighted={f1w}")
        return

    if task == "outlier_detection":
        required_noncls = ("bce", "outlier_acc")
    elif task == "change_detection":
        required_noncls = ("bce", "change_acc")
    else:
        required_noncls = ("mse", "mae")

    for metric in required_noncls:
        if metric not in metrics:
            errors.append(f"{path}.metrics missing field for task={task}: {metric}")
            continue
        if not _is_finite_number(metrics[metric]):
            errors.append(f"{path}.metrics.{metric} must be finite")
            continue
        value = float(metrics[metric])
        if metric in {"outlier_acc", "change_acc"}:
            if value < 0.0 or value > 1.0:
                errors.append(f"{path}.metrics.{metric} out of range [0, 1]")
        elif value < 0.0:
            errors.append(f"{path}.metrics.{metric} must be >= 0")


def _validate_output_payload(record: Dict[str, Any], *, path: str, errors: List[str]) -> None:
    output_path = record.get("output_path")
    if not output_path:
        errors.append(f"{path}.output_path missing")
        return
    artifact = Path(str(output_path))
    if not artifact.exists() or artifact.stat().st_size <= 0:
        errors.append(f"{path} output artifact missing/empty: {artifact}")
        return
    try:
        with artifact.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        errors.append(f"{path} output artifact is not a readable pickle payload: {artifact} ({exc})")
        return
    if not isinstance(payload, dict):
        errors.append(f"{path} output payload must be an object")
        return

    for key in ("dataset", "model", "seed", "task"):
        if key in record and str(payload.get(key)) != str(record.get(key)):
            errors.append(f"{path} output payload mismatch for {key}: {payload.get(key)!r} != {record.get(key)!r}")
    if "missing_rate" in record and "train_missing_rate" in payload:
        if abs(float(payload["train_missing_rate"]) - float(record["missing_rate"])) > 1e-12:
            errors.append(
                f"{path} output payload train_missing_rate mismatch: "
                f"{payload['train_missing_rate']!r} != {record['missing_rate']!r}"
            )

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        errors.append(f"{path} output payload meta must be an object")
        return
    output_contract = meta.get("output_contract")
    if not isinstance(output_contract, dict):
        errors.append(f"{path} output payload missing output_contract")
        return
    record_regime = record.get("evaluation_regime") or record.get("mode")
    payload_regime = (
        payload.get("evaluation_regime")
        or payload.get("mode")
        or meta.get("evaluation_regime")
        or output_contract.get("evaluation_regime")
    )
    if record_regime and payload_regime and str(record_regime) != str(payload_regime):
        errors.append(
            f"{path} output payload evaluation_regime mismatch: "
            f"{payload_regime!r} != {record_regime!r}"
        )
    evaluation_regime = str(record_regime or payload_regime or "")

    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        errors.append(f"{path} output payload evaluations must be a non-empty array")
        return
    task = str(record.get("task", payload.get("task", "classification"))).strip().lower()
    for eval_idx, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, dict):
            errors.append(f"{path}.evaluations[{eval_idx}] must be an object")
            continue
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{path}.evaluations[{eval_idx}].metrics must be an object")
            continue
        _validate_metric_values(metrics, task=task, path=f"{path}.evaluations[{eval_idx}]", errors=errors)

    record_metrics = record.get("metrics")
    if isinstance(record_metrics, dict):
        reference_evaluation = evaluations[0]
        if evaluation_regime == "robustness":
            reference_evaluation = next(
                (
                    evaluation
                    for evaluation in evaluations
                    if abs(float(evaluation.get("test_missing_rate", 1.0))) < 1e-12
                    and str(evaluation.get("test_missing_type", "")).upper() == "STANDARD"
                ),
                evaluations[0],
            )
        payload_metrics = reference_evaluation.get("metrics", {})
        if isinstance(payload_metrics, dict):
            for metric, record_value in record_metrics.items():
                if metric not in payload_metrics:
                    continue
                if not (_is_finite_number(record_value) and _is_finite_number(payload_metrics[metric])):
                    continue
                if abs(float(record_value) - float(payload_metrics[metric])) > 1e-12:
                    errors.append(
                        f"{path}.metrics.{metric} mismatch with output payload: "
                        f"{record_value} != {payload_metrics[metric]}"
                    )

    if evaluation_regime == "robustness":
        anchors = [
            evaluation
            for evaluation in evaluations
            if abs(float(evaluation.get("test_missing_rate", 1.0))) < 1e-12
            and str(evaluation.get("test_missing_type", "")).upper() == "STANDARD"
        ]
        if len(anchors) != 1:
            errors.append(f"{path} robustness output must contain exactly one STANDARD clean anchor")
    elif len(evaluations) != 1:
        errors.append(f"{path} non-robustness output must contain exactly one evaluation")


def validate_split_manifest(doc: Dict[str, Any]) -> List[str]:
    errors = _require_fields(
        doc,
        [
            "stage",
            "run_id",
            "created_at",
            "status",
            "dataset",
            "seed",
            "dataset_source",
            "split_path",
            "counts",
        ],
    )
    if doc.get("stage") != "split":
        errors.append(f"invalid stage for split manifest: {doc.get('stage')!r}")
    reuse_existing = doc.get("reuse_existing")
    if reuse_existing is not None and not isinstance(reuse_existing, bool):
        errors.append("reuse_existing must be bool when provided")
    counts = doc.get("counts", {})
    if not isinstance(counts, dict):
        errors.append("counts must be an object")
    else:
        for key in ("train", "valid", "test"):
            if key not in counts:
                errors.append(f"counts missing key: {key}")
            elif not isinstance(counts[key], int):
                errors.append(f"counts.{key} must be int")
            elif counts[key] <= 0:
                errors.append(f"counts.{key} must be > 0")
    return errors


def validate_search_manifest(doc: Dict[str, Any]) -> List[str]:
    errors = _require_fields(
        doc,
        [
            "stage",
            "run_id",
            "created_at",
            "status",
            "seed",
            "missing_rate",
            "split_root",
            "param_root",
            "records",
        ],
    )
    if doc.get("stage") != "search":
        errors.append(f"invalid stage for search manifest: {doc.get('stage')!r}")
    if not _is_finite_number(doc.get("missing_rate")):
        errors.append("missing_rate must be a finite number")
    records = doc.get("records")
    if not isinstance(records, list):
        errors.append("records must be an array")
        return errors
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"records[{idx}] must be an object")
            continue
        record_required = [
            "dataset",
            "model",
            "seed",
            "missing_rate",
            "split_path",
            "param_path",
            "best_loss",
        ]
        for key in record_required:
            if key not in record:
                errors.append(f"records[{idx}] missing field: {key}")
        task = str(record.get("task", doc.get("task", "classification"))).strip().lower()
        if task in SUPPORTED_TASKS:
            _validate_target_source(record, f"records[{idx}]", errors, task)
        if "best_loss" in record and not _is_finite_number(record.get("best_loss")):
            errors.append(f"records[{idx}].best_loss must be finite")

    reused_records = doc.get("reused_records", [])
    if reused_records is None:
        reused_records = []
    if not isinstance(reused_records, list):
        errors.append("reused_records must be an array")
        return errors
    if str(doc.get("status", "")).strip().lower() == "pass" and not records and not reused_records:
        errors.append("pass manifest must contain at least one record or reused_record")
    for idx, record in enumerate(reused_records):
        if not isinstance(record, dict):
            errors.append(f"reused_records[{idx}] must be an object")
            continue
        record_required = [
            "dataset",
            "model",
            "seed",
            "missing_rate",
            "split_path",
            "param_path",
        ]
        for key in record_required:
            if key not in record:
                errors.append(f"reused_records[{idx}] missing field: {key}")
        task = str(record.get("task", doc.get("task", "classification"))).strip().lower()
        if task in SUPPORTED_TASKS:
            _validate_target_source(record, f"reused_records[{idx}]", errors, task)
    return errors


def validate_run_manifest(doc: Dict[str, Any]) -> List[str]:
    errors = _require_fields(
        doc,
        [
            "stage",
            "run_id",
            "created_at",
            "status",
            "seed",
            "split_root",
            "param_root",
            "out_root",
            "records",
        ],
    )
    if doc.get("stage") != "run":
        errors.append(f"invalid stage for run manifest: {doc.get('stage')!r}")
    records = doc.get("records")
    if not isinstance(records, list):
        errors.append("records must be an array")
        return errors
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"records[{idx}] must be an object")
            continue
        record_required = [
            "dataset",
            "model",
            "seed",
            "missing_rate",
            "split_path",
            "param_path",
            "output_path",
            "metrics",
        ]
        for key in record_required:
            if key not in record:
                errors.append(f"records[{idx}] missing field: {key}")
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append(f"records[{idx}].metrics must be an object")
            continue
        task = str(record.get("task", "classification")).strip().lower()
        if task not in SUPPORTED_TASKS:
            errors.append(
                f"records[{idx}].task unsupported: {task!r}. "
                f"Supported tasks are {sorted(SUPPORTED_TASKS)}"
            )
            continue
        _validate_target_source(record, f"records[{idx}]", errors, task)
        if task == "classification":
            has_current_metrics = all(metric in metrics for metric in ("accuracy", "f1", "wf1"))
            has_compat_metrics = all(metric in metrics for metric in ("accuracy", "f1_weighted"))
            if not (has_current_metrics or has_compat_metrics):
                errors.append(
                    f"records[{idx}].metrics missing required keys: "
                    "expect (accuracy,f1,wf1) or compatibility tuple (accuracy,f1_weighted)"
                )
                continue

            required_metrics = (
                ("accuracy", "f1", "wf1")
                if has_current_metrics
                else ("accuracy", "f1_weighted")
            )
            for metric in required_metrics:
                if not _is_finite_number(metrics[metric]):
                    errors.append(f"records[{idx}].metrics.{metric} must be finite")
                    continue
                value = float(metrics[metric])
                if value < 0.0 or value > 1.0:
                    errors.append(f"records[{idx}].metrics.{metric} out of range [0, 1]")

            if has_current_metrics and "f1_weighted" in metrics:
                if not _is_finite_number(metrics["f1_weighted"]):
                    errors.append(f"records[{idx}].metrics.f1_weighted must be finite")
                else:
                    wf1 = float(metrics["wf1"])
                    f1w = float(metrics["f1_weighted"])
                    if abs(wf1 - f1w) > 1e-12:
                        errors.append(
                            f"records[{idx}].metrics alias mismatch: wf1={wf1} != f1_weighted={f1w}"
                        )
        else:
            if task == "outlier_detection":
                required_noncls = ("bce", "outlier_acc")
            elif task == "change_detection":
                required_noncls = ("bce", "change_acc")
            else:
                required_noncls = ("mse", "mae")

            for metric in required_noncls:
                if metric not in metrics:
                    errors.append(f"records[{idx}].metrics missing field for task={task}: {metric}")
                    continue
                if not _is_finite_number(metrics[metric]):
                    errors.append(f"records[{idx}].metrics.{metric} must be finite")
                    continue
                value = float(metrics[metric])
                if metric in {"outlier_acc", "change_acc"}:
                    if value < 0.0 or value > 1.0:
                        errors.append(f"records[{idx}].metrics.{metric} out of range [0, 1]")
                elif value < 0.0:
                    errors.append(f"records[{idx}].metrics.{metric} must be >= 0")

            task_metrics = record.get("task_metrics")
            if task_metrics is not None:
                if not isinstance(task_metrics, dict):
                    errors.append(f"records[{idx}].task_metrics must be an object when provided")
                else:
                    for metric in required_noncls:
                        if metric in task_metrics and not _is_finite_number(task_metrics[metric]):
                            errors.append(f"records[{idx}].task_metrics.{metric} must be finite")
        _validate_output_payload(record, path=f"records[{idx}]", errors=errors)

    reused_records = doc.get("reused_records", [])
    if reused_records is None:
        reused_records = []
    if not isinstance(reused_records, list):
        errors.append("reused_records must be an array")
        return errors
    if str(doc.get("status", "")).strip().lower() == "pass" and not records and not reused_records:
        errors.append("pass manifest must contain at least one record or reused_record")
    for idx, record in enumerate(reused_records):
        if not isinstance(record, dict):
            errors.append(f"reused_records[{idx}] must be an object")
            continue
        record_required = [
            "dataset",
            "model",
            "seed",
            "missing_rate",
            "split_path",
            "param_path",
            "output_path",
        ]
        for key in record_required:
            if key not in record:
                errors.append(f"reused_records[{idx}] missing field: {key}")
        task = str(record.get("task", doc.get("task", "classification"))).strip().lower()
        if task in SUPPORTED_TASKS:
            _validate_target_source(record, f"reused_records[{idx}]", errors, task)
        _validate_output_payload(record, path=f"reused_records[{idx}]", errors=errors)
    return errors
