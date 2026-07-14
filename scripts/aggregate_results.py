#!/usr/bin/env python
"""Aggregate packaged script manifests into public-safe result tables."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


REDACTED_FIELD_FRAGMENTS = (
    "hash",
    "backend",
    "architecture_axes",
    "capacity_config",
    "profile",
    "public_search_name",
    "search_algorithm_policy",
    "search_contract",
    "search_family",
    "grid_version",
)
REDACTED_VALUE_FRAGMENTS = (
    "session_",
    "profile",
)
PUBLIC_ROBUSTNESS_RATES = [round(index / 10, 1) for index in range(10)]
PUBLIC_ROBUSTNESS_MECHANISMS = ["MCAR", "MAR", "MNAR", "BLOCK", "BURST", "CHANNEL_DROP"]
PUBLIC_ROBUSTNESS_SYNC_OPTIONS = [False, True]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return left == right


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _public_path(value: Any) -> Any:
    if value in {None, ""}:
        return value
    path = Path(str(value))
    parts = [part for part in path.parts if part not in {path.anchor, ""}]
    if not parts:
        return ""
    return "/".join(parts[-3:])


_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\\\/][^\"'\s,;]+")
_POSIX_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:[^/\"'\s,;]+/)*[^/\"'\s,;]+")
_HIDDEN_PATH_COMPONENT_RE = re.compile(r"(?:^|[\\/])\.[^\\/]+(?:[\\/]|$)")


def _contains_absolute_path(value: Any) -> bool:
    text = str(value)
    return bool(
        _WINDOWS_ABSOLUTE_PATH_RE.search(text)
        or _POSIX_ABSOLUTE_PATH_RE.search(text)
        or text.startswith("\\\\")
    )


def _resolve_output_path(value: Any, out_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [path, out_root / path]
    if path.parts and path.parts[0] == out_root.name:
        candidates.append(out_root / Path(*path.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _copy_runtime_baseline_id(row: dict[str, Any], *payloads: dict[str, Any] | None) -> None:
    for payload in payloads:
        if isinstance(payload, dict) and payload.get("runtime_baseline_id"):
            row["runtime_baseline_id"] = payload.get("runtime_baseline_id")
            return


def _flatten_timestamp_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    degenerate_rows = payload.get(
        "timestamp_degenerate_rows_count",
        payload.get("timestamp_degenerate_rows"),
    )
    return {
        "timestamp_mode": payload.get("timestamp_mode"),
        "timestamp_delta_scale_fallback": payload.get("timestamp_delta_scale_fallback"),
        "timestamp_degenerate_rows_count": degenerate_rows,
    }


def _copy_public_fields(row: dict[str, Any], payload: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if key.endswith("_path") or key.endswith("_root"):
                value = _public_path(value)
            row[key] = value


def _copy_metrics(row: dict[str, Any], metrics: dict[str, Any] | None) -> None:
    if not isinstance(metrics, dict):
        return
    for key, value in metrics.items():
        if key == "confusion_matrix":
            row[key] = json.dumps(value, separators=(",", ":"))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value


def _expected_axis_keys(axes: dict[str, Any]) -> list[tuple[float, str, bool | None, int]]:
    rates = [float(rate) for rate in axes.get("test_missing_rates", [])]
    mechanisms = [str(value).upper() for value in axes.get("test_missing_types", [])]
    sync_options = [bool(value) for value in axes.get("test_channel_sync_options", [])]
    seed = int(axes.get("missing_seed", 0))
    positive_rates = [rate for rate in rates if rate != 0.0]
    expected = [(0.0, "STANDARD", None, seed)]
    for rate in positive_rates:
        for mechanism in mechanisms:
            if mechanism == "CHANNEL_DROP":
                continue
            for channel_sync in sync_options:
                expected.append((rate, mechanism, channel_sync, seed))
    if "CHANNEL_DROP" in mechanisms:
        expected.extend((rate, "CHANNEL_DROP", None, seed) for rate in positive_rates)
    return expected


def _evaluation_axis_key(evaluation: dict[str, Any]) -> tuple[float, str, bool | None, int]:
    return (
        float(evaluation.get("test_missing_rate")),
        str(evaluation.get("test_missing_type", "")).upper(),
        evaluation.get("test_channel_sync"),
        int(evaluation.get("missing_seed")),
    )


def _validate_public_100_cell_axes(
    axes: dict[str, Any],
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    rates = [float(value) for value in axes.get("test_missing_rates", [])]
    mechanisms = [str(value).upper() for value in axes.get("test_missing_types", [])]
    sync_options = [bool(value) for value in axes.get("test_channel_sync_options", [])]
    if (
        rates != PUBLIC_ROBUSTNESS_RATES
        or mechanisms != PUBLIC_ROBUSTNESS_MECHANISMS
        or sync_options != PUBLIC_ROBUSTNESS_SYNC_OPTIONS
        or int(axes.get("expected_evaluations", -1)) != 100
        or int(axes.get("missing_seed", -1)) != int(payload.get("seed", -2))
    ):
        raise ValueError(f"{output_path}: public export requires canonical 100-cell axes")


def _validate_full_predictions(predictions: Any, output_path: Path, *, num_class: int | None) -> dict[str, Any]:
    if not isinstance(predictions, dict) or predictions.get("storage") != "full":
        raise ValueError(f"{output_path}: public 100-cell output requires full prediction storage")
    if predictions.get("prediction_identity_schema") != "sample_indices":
        raise ValueError(f"{output_path}: prediction identity schema is invalid")
    if predictions.get("sample_index_dtype") != "int64":
        raise ValueError(f"{output_path}: sample index dtype is invalid")
    indices = predictions.get("sample_indices")
    count = predictions.get("sample_index_count")
    if indices is None or count is None:
        raise ValueError(f"{output_path}: full prediction identity is incomplete")
    raw_indices = np.asarray(indices)
    if raw_indices.ndim != 1 or raw_indices.dtype != np.dtype("int64"):
        raise ValueError(f"{output_path}: stored sample indices must be one-dimensional int64")
    index_array = np.ascontiguousarray(raw_indices)
    if index_array.shape != (int(count),):
        raise ValueError(f"{output_path}: sample_index_count does not match stored indices")
    y_true = np.asarray(predictions.get("y_true"))
    y_pred = np.asarray(predictions.get("y_pred"))
    logits = np.asarray(predictions.get("logit_list"))
    if y_true.shape != (int(count),) or not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError(f"{output_path}: y_true does not match sample_index_count")
    if y_pred.shape != (int(count),) or not np.issubdtype(y_pred.dtype, np.integer):
        raise ValueError(f"{output_path}: y_pred does not match sample_index_count")
    resolved_num_class = int(num_class) if num_class is not None else (
        int(logits.shape[1]) if logits.ndim == 2 else 0
    )
    if (
        resolved_num_class < 2
        or logits.shape != (int(count), resolved_num_class)
        or not np.isfinite(logits).all()
    ):
        raise ValueError(f"{output_path}: logit_list is invalid")
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if np.any(values < 0) or np.any(values >= resolved_num_class):
            raise ValueError(f"{output_path}: {name} contains invalid class indices")
    return {
        "sample_index_count": int(count),
        "sample_indices": tuple(int(value) for value in index_array),
        "y_true": tuple(int(value) for value in y_true),
        "num_class": resolved_num_class,
    }


def _expected_channel_drop_plan(channel_count: int, rate: float, seed: int) -> dict[str, Any]:
    requested_count = int(np.floor(channel_count * rate + 0.5))
    if channel_count == 1 and rate > 0.0:
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
            "analysis_exclusion_reason": "univariate-channel-drop-not-interpretable",
        }
    dropped_count = min(channel_count - 1, max(1, requested_count))
    channel_order = np.random.default_rng(seed).permutation(channel_count).astype(np.int64)
    dropped = np.sort(channel_order[:dropped_count])
    realized_rate = float(dropped_count / channel_count)
    return {
        "channel_count": channel_count,
        "requested_channel_drop_count": requested_count,
        "dropped_channel_count": dropped_count,
        "dropped_channel_indices": dropped.tolist(),
        "channel_permutation": channel_order.tolist(),
        "minimum_one_channel_forced": bool(requested_count < 1),
        "maximum_one_channel_retained": bool(requested_count >= channel_count),
        "channel_count_quantized": bool(not np.isclose(realized_rate, rate)),
        "realized_channel_drop_rate": realized_rate,
        "channel_drop_applicable": True,
        "analysis_exclusion_reason": None,
    }


def _validate_channel_drop_rows(evaluations: list[dict[str, Any]], output_path: Path) -> None:
    required = (
        "channel_count",
        "requested_channel_drop_count",
        "dropped_channel_count",
        "dropped_channel_indices",
        "channel_permutation",
        "minimum_one_channel_forced",
        "maximum_one_channel_retained",
        "channel_count_quantized",
        "realized_channel_drop_rate",
        "channel_drop_applicable",
        "analysis_exclusion_reason",
        "all_missing_sequence_count",
    )
    for evaluation in evaluations:
        if str(evaluation.get("test_missing_type", "")).upper() != "CHANNEL_DROP":
            continue
        if any(key not in evaluation for key in required):
            raise ValueError(f"{output_path}: channel-drop provenance is incomplete")
        channel_count = int(evaluation["channel_count"])
        expected = _expected_channel_drop_plan(
            channel_count,
            float(evaluation.get("test_missing_rate")),
            int(evaluation.get("missing_seed")),
        )
        if any(not _values_match(evaluation.get(key), value) for key, value in expected.items()):
            raise ValueError(f"{output_path}: channel-drop provenance does not match the seed-bound plan")
        all_missing = int(evaluation["all_missing_sequence_count"])
        sample_count = int(evaluation.get("predictions", {}).get("sample_index_count", -1))
        if all_missing < 0 or all_missing > sample_count:
            raise ValueError(f"{output_path}: all-missing sequence count is invalid")


def _validate_robustness_output(
    payload: dict[str, Any],
    output_path: Path,
    *,
    public_export: bool,
) -> tuple[int, str] | None:
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError(f"{output_path}: evaluations must be a list")
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    regime = payload.get("evaluation_regime") or payload.get("mode") or meta.get("evaluation_regime")
    if not regime and evaluations:
        regime = evaluations[0].get("evaluation_regime") if isinstance(evaluations[0], dict) else None
    axes = meta.get("robustness_axes")
    if not isinstance(axes, dict):
        contract = meta.get("output_contract") if isinstance(meta.get("output_contract"), dict) else {}
        axes = contract.get("robustness_axes")
    is_multicell = str(regime).lower() == "robustness" and (
        len(evaluations) > 1 or isinstance(axes, dict)
    )
    if not is_multicell:
        return None

    if isinstance(axes, dict):
        expected_axes = _expected_axis_keys(axes)
        expected_count = int(axes.get("expected_evaluations", len(expected_axes)))
    elif len(evaluations) == 100:
        expected_count = 100
        expected_axes = _expected_axis_keys(
            {
                "missing_seed": payload.get("seed", 0),
                "test_missing_rates": PUBLIC_ROBUSTNESS_RATES,
                "test_missing_types": PUBLIC_ROBUSTNESS_MECHANISMS,
                "test_channel_sync_options": PUBLIC_ROBUSTNESS_SYNC_OPTIONS,
            }
        )
    else:
        expected_count = len(evaluations)
        expected_axes = []

    if public_export and (expected_count != 100 or len(evaluations) != 100):
        raise ValueError(f"{output_path}: public robustness export requires exactly 100 evaluations")
    if len(evaluations) != expected_count or (expected_axes and len(expected_axes) != expected_count):
        raise ValueError(f"{output_path}: robustness output requires exactly {expected_count} evaluations")
    if expected_axes:
        try:
            actual_axes = [_evaluation_axis_key(evaluation) for evaluation in evaluations]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{output_path}: robustness axis values are invalid") from exc
        if actual_axes != expected_axes or len(set(actual_axes)) != len(actual_axes):
            raise ValueError(f"{output_path}: robustness axis order or uniqueness is invalid")
    if public_export and expected_count == 100:
        if not isinstance(axes, dict):
            raise ValueError(f"{output_path}: public export requires canonical 100-cell axes")
        _validate_public_100_cell_axes(axes, payload, output_path)

    storages = {
        evaluation.get("predictions", {}).get("storage")
        for evaluation in evaluations
        if isinstance(evaluation, dict)
    }
    storage = next(iter(storages)) if len(storages) == 1 else "mixed"
    if expected_count == 100:
        if payload.get("task", "classification") != "classification":
            raise ValueError(f"{output_path}: public 100-cell output supports classification only")
        contract = meta.get("output_contract") if isinstance(meta.get("output_contract"), dict) else {}
        num_class = contract.get("num_class")
        reference_prediction_identity = None
        for evaluation in evaluations:
            identity = _validate_full_predictions(
                evaluation.get("predictions"),
                output_path,
                num_class=num_class,
            )
            if reference_prediction_identity is None:
                reference_prediction_identity = identity
                continue
            if (
                identity["sample_index_count"] != reference_prediction_identity["sample_index_count"]
                or identity["sample_indices"] != reference_prediction_identity["sample_indices"]
            ):
                raise ValueError(f"{output_path}: sample indices differ across evaluations")
            if identity["y_true"] != reference_prediction_identity["y_true"]:
                raise ValueError(f"{output_path}: y_true differs across evaluations")
            if identity["num_class"] != reference_prediction_identity["num_class"]:
                raise ValueError(f"{output_path}: logit_list class dimension differs across evaluations")
        _validate_channel_drop_rows(evaluations, output_path)
        storage = "full"
    return expected_count, str(storage)


def _payload_value(payload: dict[str, Any], record: dict[str, Any], key: str) -> Any:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    contract = meta.get("output_contract") if isinstance(meta.get("output_contract"), dict) else {}
    if payload.get(key) is not None:
        return payload.get(key)
    if contract.get(key) is not None:
        return contract.get(key)
    if meta.get(key) is not None:
        return meta.get(key)
    return record.get(key)


def _validate_record_payload_integrity(
    record: dict[str, Any],
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    identity_fields = (
        "dataset",
        "model",
        "seed",
        "task",
        "evaluation_regime",
        "target_source",
        "forecast_horizon",
    )
    for key in identity_fields:
        recorded = record.get(key)
        if recorded is None:
            continue
        payload_value = _payload_value(payload, record, key)
        if not _values_match(recorded, payload_value):
            raise ValueError(f"{output_path}: record {key} does not match the output payload")

    record_metrics = record.get("metrics")
    evaluations = payload.get("evaluations")
    representative_metrics = (
        evaluations[0].get("metrics")
        if isinstance(evaluations, list) and evaluations and isinstance(evaluations[0], dict)
        else None
    )
    if isinstance(record_metrics, dict):
        if not isinstance(representative_metrics, dict) or any(
            key not in representative_metrics or not _values_match(value, representative_metrics[key])
            for key, value in record_metrics.items()
        ):
            raise ValueError(f"{output_path}: record metrics do not match the output payload")


def _logical_output_key(payload: dict[str, Any], record: dict[str, Any]) -> tuple[Any, ...]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    contract = meta.get("output_contract") if isinstance(meta.get("output_contract"), dict) else {}
    axes = meta.get("robustness_axes")
    if not isinstance(axes, dict):
        axes = contract.get("robustness_axes")
    regime = _payload_value(payload, record, "evaluation_regime")
    if regime is None:
        regime = payload.get("mode", record.get("mode"))
    axes_identity = None
    if isinstance(axes, dict):
        axes_identity = (
            tuple(float(value) for value in axes.get("test_missing_rates", [])),
            tuple(str(value).upper() for value in axes.get("test_missing_types", [])),
            tuple(bool(value) for value in axes.get("test_channel_sync_options", [])),
            int(axes.get("missing_seed", 0)),
            int(axes.get("expected_evaluations", 0)),
        )
    return (
        _payload_value(payload, record, "dataset"),
        _payload_value(payload, record, "model"),
        _payload_value(payload, record, "seed"),
        _payload_value(payload, record, "task"),
        _payload_value(payload, record, "dataset_source"),
        _payload_value(payload, record, "target_source"),
        _payload_value(payload, record, "forecast_horizon"),
        regime,
        _payload_value(payload, record, "train_missing_rate")
        if _payload_value(payload, record, "train_missing_rate") is not None
        else record.get("missing_rate"),
        _payload_value(payload, record, "train_missing_type"),
        _payload_value(payload, record, "train_channel_sync"),
        axes_identity,
    )


def _evaluation_rows_from_output(
    *,
    record: dict[str, Any],
    manifest: dict[str, Any],
    output_path: Path,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        return rows
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    timestamp_summary = meta.get("timestamp_summary") or record.get("timestamp_summary")
    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, dict):
            continue
        row = {
            "run_id": manifest.get("run_id"),
            "source_stage": "run",
            "record_kind": record.get("record_kind", "executed"),
            "evaluation_index": index,
        }
        _copy_public_fields(
            row,
            {**record, **payload, **evaluation},
            [
                "dataset",
                "model",
                "seed",
                "task",
                "evaluation_regime",
                "mode",
                "train_missing_rate",
                "train_missing_type",
                "train_channel_sync",
                "test_missing_rate",
                "test_missing_type",
                "test_channel_sync",
                "requested_missing_rate",
                "realized_missing_rate",
                "realized_missing_count",
                "observed_value_count",
                "fully_masked_before_floor",
                "floor_n_sequences",
                "channel_count",
                "requested_channel_drop_count",
                "dropped_channel_count",
                "dropped_channel_indices",
                "minimum_one_channel_forced",
                "maximum_one_channel_retained",
                "channel_count_quantized",
                "realized_channel_drop_rate",
                "channel_drop_applicable",
                "analysis_exclusion_reason",
                "all_missing_sequence_count",
                "missing_seed",
                "output_path",
                "param_path",
                "split_path",
                "runtime_status",
                "runtime_status_reason",
                "cell_wall_clock_s",
            ],
        )
        _copy_metrics(row, evaluation.get("metrics"))
        if str(evaluation.get("test_missing_type", "")).upper() == "CHANNEL_DROP":
            row["analysis_excluded"] = bool(
                evaluation.get("analysis_exclusion_reason")
                or evaluation.get("channel_drop_applicable") is False
            )
        _copy_runtime_baseline_id(row, evaluation, payload, meta, record, manifest)
        row.update(_flatten_timestamp_summary(timestamp_summary))
        rows.append(row)
    return rows


def _run_record_row(record: dict[str, Any], manifest: dict[str, Any], *, kind: str) -> dict[str, Any]:
    row = {
        "run_id": manifest.get("run_id"),
        "stage": manifest.get("stage"),
        "record_kind": kind,
        "manifest_status": manifest.get("status"),
        "manifest_existing_policy": manifest.get("manifest_existing_policy"),
    }
    _copy_public_fields(
        row,
        record,
        [
            "dataset",
            "model",
            "seed",
            "task",
            "missing_rate",
            "output_path",
            "param_path",
            "split_path",
            "reason",
            "runtime_status",
            "runtime_status_reason",
            "cell_wall_clock_s",
        ],
    )
    _copy_metrics(row, record.get("metrics"))
    _copy_runtime_baseline_id(row, record, manifest)
    row.update(_flatten_timestamp_summary(record.get("timestamp_summary")))
    return row


def _search_record_row(record: dict[str, Any], manifest: dict[str, Any], *, kind: str) -> dict[str, Any]:
    row = {
        "run_id": manifest.get("run_id"),
        "stage": manifest.get("stage"),
        "record_kind": kind,
        "manifest_status": manifest.get("status"),
        "manifest_existing_policy": manifest.get("manifest_existing_policy"),
    }
    _copy_public_fields(
        row,
        record,
        [
            "dataset",
            "model",
            "task",
            "seed",
            "param_path",
            "split_path",
            "best_loss",
            "search_max_t_reached",
            "search_incomplete_fallback_used",
            "search_error_fallback_used",
            "search_selected_iteration",
            "search_selected_trial_status",
            "reason",
        ],
    )
    _copy_runtime_baseline_id(row, record, manifest)
    row.update(_flatten_timestamp_summary(record.get("timestamp_summary")))
    return row


def _is_public_safe_row(row: dict[str, Any]) -> bool:
    for key, value in row.items():
        key_text = str(key)
        value_text = str(value)
        if any(fragment in key_text for fragment in REDACTED_FIELD_FRAGMENTS):
            return False
        if any(fragment in value_text for fragment in REDACTED_VALUE_FRAGMENTS):
            return False
        if _HIDDEN_PATH_COMPONENT_RE.search(value_text):
            return False
        if _contains_absolute_path(value):
            return False
    return True


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate(
    *,
    evidence_root: str | Path,
    out_root: str | Path,
    output_dir: str | Path,
    public_export: bool = True,
) -> dict[str, Any]:
    evidence_root = Path(evidence_root)
    out_root = Path(out_root)
    output_dir = Path(output_dir)
    manifests = sorted(evidence_root.glob("**/*.manifest.json"))

    runs: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    search_cells: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    seen_output_artifacts: set[str] = set()
    logical_output_artifacts: dict[tuple[Any, ...], str] = {}
    robustness_signature: tuple[int, str] | None = None
    duplicate_output_references = 0

    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        stage = manifest.get("stage")
        records = manifest.get("records") if isinstance(manifest.get("records"), list) else []
        reused = manifest.get("reused_records") if isinstance(manifest.get("reused_records"), list) else []
        skipped = manifest.get("skipped") if isinstance(manifest.get("skipped"), list) else []

        if stage == "run":
            for kind, group in (("executed", records), ("reused", reused)):
                for record in group:
                    if not isinstance(record, dict):
                        continue
                    record = {**record, "record_kind": kind}
                    output_path = record.get("output_path")
                    if output_path:
                        resolved_path = _resolve_output_path(output_path, out_root)
                        if not resolved_path.exists():
                            raise ValueError(f"Output artifact {resolved_path} does not exist")
                        try:
                            payload = _read_pickle(resolved_path)
                        except Exception as exc:
                            raise ValueError(f"Output artifact {resolved_path} cannot be read: {exc}") from exc
                        if not isinstance(payload, dict):
                            raise ValueError(f"Output artifact {resolved_path} must contain an object")
                        _validate_record_payload_integrity(
                            record,
                            payload,
                            resolved_path,
                        )
                        artifact_id = str(resolved_path.resolve())
                        logical_key = _logical_output_key(payload, record)
                        previous = logical_output_artifacts.get(logical_key)
                        if previous is not None and previous != artifact_id:
                            raise ValueError(
                                f"Logical run {logical_key} has conflicting output artifacts: "
                                f"{previous} and {artifact_id}"
                            )
                        logical_output_artifacts[logical_key] = artifact_id
                        if artifact_id in seen_output_artifacts:
                            duplicate_output_references += 1
                            continue
                        seen_output_artifacts.add(artifact_id)
                        signature = _validate_robustness_output(
                            payload,
                            resolved_path,
                            public_export=public_export,
                        )
                        if signature is not None:
                            if robustness_signature is not None and robustness_signature != signature:
                                raise ValueError(
                                    "Aggregate contains mixed robustness evaluation counts or prediction storage"
                                )
                            robustness_signature = signature
                        runs.append(_run_record_row(record, manifest, kind=kind))
                        robustness_rows.extend(
                            _evaluation_rows_from_output(
                                record=record,
                                manifest=manifest,
                                output_path=resolved_path,
                                payload=payload,
                            )
                        )
                    else:
                        runs.append(_run_record_row(record, manifest, kind=kind))
            for record in skipped:
                if isinstance(record, dict):
                    exclusions.append(_run_record_row(record, manifest, kind="skipped"))
        elif stage == "search":
            for kind, group in (("executed", records), ("reused", reused)):
                for record in group:
                    if not isinstance(record, dict):
                        continue
                    row = _search_record_row(record, manifest, kind=kind)
                    search_cells.append(row)
                    if _as_bool(record.get("search_incomplete_fallback_used")) or _as_bool(
                        record.get("search_error_fallback_used")
                    ):
                        fallbacks.append(row)
            for record in skipped:
                if isinstance(record, dict):
                    exclusions.append(_search_record_row(record, manifest, kind="skipped"))
        elif stage == "split":
            for record in skipped:
                if isinstance(record, dict):
                    exclusions.append(
                        {
                            "run_id": manifest.get("run_id"),
                            "stage": stage,
                            "record_kind": "skipped",
                            "reason": record.get("reason"),
                        }
                    )

    if public_export:
        for table_name, rows in {
            "runs": runs,
            "robustness_rows": robustness_rows,
            "search_cells": search_cells,
            "exclusions": exclusions,
            "fallbacks": fallbacks,
        }.items():
            for row in rows:
                if not _is_public_safe_row(row):
                    raise ValueError(f"{table_name} contains non-public metadata: {row}")

    _write_csv(output_dir / "runs.csv", runs)
    _write_csv(output_dir / "robustness_rows.csv", robustness_rows)
    _write_csv(output_dir / "search_cells.csv", search_cells)
    _write_csv(output_dir / "exclusions.csv", exclusions)
    _write_csv(output_dir / "fallbacks.csv", fallbacks)

    summary = {
        "manifest_count": len(manifests),
        "run_rows": len(runs),
        "robustness_rows": len(robustness_rows),
        "search_rows": len(search_cells),
        "exclusion_rows": len(exclusions),
        "fallback_rows": len(fallbacks),
        "duplicate_output_references": duplicate_output_references,
        "public_export": bool(public_export),
        "out_root": _public_path(out_root) if public_export else str(out_root),
    }
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate torch-ists script outputs into canonical tables.")
    parser.add_argument("--evidence-root", default="evidence", help="Root containing *.manifest.json files.")
    parser.add_argument("--out-root", default="out", help="Output artifact root used by model_run.py.")
    parser.add_argument("--output-dir", default="aggregate", help="Directory for aggregate CSV/JSON files.")
    return parser.parse_args()


def main() -> int:
    args = parse_cli_args()
    summary = aggregate(
        evidence_root=args.evidence_root,
        out_root=args.out_root,
        output_dir=args.output_dir,
        public_export=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
