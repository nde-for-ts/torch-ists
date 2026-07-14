#!/usr/bin/env python
"""Aggregate packaged script manifests into public-safe result tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


REDACTED_FIELD_FRAGMENTS = (
    "hash",
    "backend",
    "architecture_axes",
    "capacity_config",
    "profile",
)
REDACTED_VALUE_FRAGMENTS = (
    "/home/",
    "/media/",
    ".claude",
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _validate_full_predictions(predictions: Any, output_path: Path) -> None:
    if not isinstance(predictions, dict) or predictions.get("storage") != "full":
        raise ValueError(f"{output_path}: public 100-cell output requires full prediction storage")
    indices = predictions.get("sample_indices")
    count = predictions.get("sample_index_count")
    if indices is None or count is None:
        raise ValueError(f"{output_path}: full prediction identity is incomplete")
    index_array = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
    if len(index_array) != int(count):
        raise ValueError(f"{output_path}: sample_index_count does not match stored indices")
    if predictions.get("sample_index_hash") != hashlib.sha256(index_array.tobytes()).hexdigest():
        raise ValueError(f"{output_path}: sample index hash is invalid")
    for key in ("y_true", "y_pred", "logit_list"):
        values = predictions.get(key)
        if values is None or np.asarray(values).shape[0] != int(count):
            raise ValueError(f"{output_path}: {key} does not match sample_index_count")


def _validate_channel_drop_rows(evaluations: list[dict[str, Any]], output_path: Path) -> None:
    required = (
        "channel_count",
        "requested_channel_drop_count",
        "dropped_channel_count",
        "dropped_channel_indices",
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
        dropped_count = int(evaluation["dropped_channel_count"])
        dropped_indices = evaluation["dropped_channel_indices"]
        if (
            not isinstance(dropped_indices, list)
            or len(dropped_indices) != dropped_count
            or len(set(dropped_indices)) != dropped_count
            or any(int(index) < 0 or int(index) >= channel_count for index in dropped_indices)
        ):
            raise ValueError(f"{output_path}: dropped channel indices are inconsistent")
        if int(evaluation["all_missing_sequence_count"]) < 0:
            raise ValueError(f"{output_path}: all-missing sequence count is invalid")
        if channel_count == 1:
            if (
                evaluation["channel_drop_applicable"] is not False
                or dropped_count != 0
                or not evaluation["analysis_exclusion_reason"]
            ):
                raise ValueError(f"{output_path}: univariate channel-drop exclusion is invalid")
        elif (
            evaluation["channel_drop_applicable"] is not True
            or evaluation["analysis_exclusion_reason"] is not None
        ):
            raise ValueError(f"{output_path}: channel-drop applicability is invalid")


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

    storages = {
        evaluation.get("predictions", {}).get("storage")
        for evaluation in evaluations
        if isinstance(evaluation, dict)
    }
    storage = next(iter(storages)) if len(storages) == 1 else "mixed"
    if expected_count == 100:
        if payload.get("task", "classification") != "classification":
            raise ValueError(f"{output_path}: public 100-cell output supports classification only")
        for evaluation in evaluations:
            _validate_full_predictions(evaluation.get("predictions"), output_path)
        _validate_channel_drop_rows(evaluations, output_path)
        storage = "full"
    return expected_count, str(storage)


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
    seen_output_artifacts: set[tuple[str, str]] = set()
    logical_output_artifacts: dict[tuple[Any, ...], tuple[str, str]] = {}
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
                        artifact_id = (str(resolved_path.resolve()), _file_sha256(resolved_path))
                        logical_key = tuple(
                            payload.get(key, record.get(key))
                            for key in ("dataset", "model", "seed", "task")
                        )
                        previous = logical_output_artifacts.get(logical_key)
                        if previous is not None and previous != artifact_id:
                            raise ValueError(
                                f"Logical run {logical_key} has conflicting output artifacts: "
                                f"{previous[0]} and {artifact_id[0]}"
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
    parser.add_argument(
        "--export-scope",
        choices=["public", "internal"],
        default="public",
        help="public redacts internal paths/details; internal keeps broader metadata where emitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_cli_args()
    summary = aggregate(
        evidence_root=args.evidence_root,
        out_root=args.out_root,
        output_dir=args.output_dir,
        public_export=args.export_scope == "public",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
