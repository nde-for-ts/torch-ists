#!/usr/bin/env python
"""Aggregate packaged script manifests into public-safe result tables."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any


PUBLIC_RUNTIME_KEYS = {
    "package_version",
    "mirror_commit_sha",
    "mirror_archive_checksum",
    "runtime_baseline_id",
    "wheel_or_source_checkout",
}
INTERNAL_FIELD_FRAGMENTS = (
    "hash",
    "backend",
    "architecture_axes",
    "capacity_config",
    "canonical_profile",
    "profile_alias",
)
FORBIDDEN_PUBLIC_VALUE_FRAGMENTS = (
    "/home/",
    "/media/",
    ".claude",
    "session_",
    "canonical_profile",
    "profile_alias",
    "search_profile_visibility=internal",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


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


def _flatten_runtime_provenance(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        f"runtime_{key}": payload.get(key)
        for key in sorted(PUBLIC_RUNTIME_KEYS)
        if key in payload
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


def _evaluation_rows_from_output(
    *,
    record: dict[str, Any],
    manifest: dict[str, Any],
    output_path: Path,
) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    payload = _read_pickle(output_path)
    if not isinstance(payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        return rows
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    runtime_provenance = meta.get("runtime_provenance") or manifest.get("runtime_provenance")
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
                "missing_seed",
                "output_path",
                "param_path",
                "split_path",
            ],
        )
        _copy_metrics(row, evaluation.get("metrics"))
        row.update(_flatten_runtime_provenance(runtime_provenance))
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
        ["dataset", "model", "seed", "task", "missing_rate", "output_path", "param_path", "split_path", "reason"],
    )
    _copy_metrics(row, record.get("metrics"))
    row.update(_flatten_runtime_provenance(record.get("runtime_provenance") or manifest.get("runtime_provenance")))
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
    row.update(_flatten_runtime_provenance(record.get("runtime_provenance") or manifest.get("runtime_provenance")))
    return row


def _is_public_safe_row(row: dict[str, Any]) -> bool:
    for key, value in row.items():
        key_text = str(key)
        value_text = str(value)
        if any(fragment in key_text for fragment in INTERNAL_FIELD_FRAGMENTS):
            return False
        if any(fragment in value_text for fragment in FORBIDDEN_PUBLIC_VALUE_FRAGMENTS):
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
                    runs.append(_run_record_row(record, manifest, kind=kind))
                    output_path = record.get("output_path")
                    if output_path:
                        robustness_rows.extend(
                            _evaluation_rows_from_output(
                                record=record,
                                manifest=manifest,
                                output_path=_resolve_output_path(output_path, out_root),
                            )
                        )
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
