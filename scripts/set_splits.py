#!/usr/bin/env python
"""Create reproducible split files for packaged script execution."""

import argparse
import json
import sys
from pathlib import Path

from _common import configure_runtime_paths
from dataset_source_contracts import load_dataset_payload, normalize_dataset_source
from task_target_contracts import (
    load_task_targets_npz,
    make_fixed_task_splits,
    resolve_task_split_path,
    validate_task_splits,
)

sys.dont_write_bytecode = True


def _ensure_repo_on_path() -> Path:
    return configure_runtime_paths(__file__)

def _normalize_dataset_source_arg(raw: str) -> str:
    try:
        return normalize_dataset_source(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_dataset(args):
    return load_dataset_payload(
        dataset_name=args.dataset_name,
        dataset_source=args.dataset_source,
        dataset_npz=args.dataset_npz,
        task=args.task,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create split file")
    parser.add_argument("--dataset-name", required=True, help="Dataset id used for split file path")
    parser.add_argument(
        "--dataset-source",
        type=_normalize_dataset_source_arg,
        default="default",
        metavar="{default,npz}",
        help="Dataset source backend. default=packaged built-in dataset loader, npz=local file payload.",
    )
    parser.add_argument(
        "--dataset-npz",
        default=None,
        help="Path to .npz file containing X and y (required when --dataset-source=npz)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation (default 42; eval seeds 0-4 are disjoint).")
    parser.add_argument(
        "--task",
        choices=[
            "classification",
            "regression",
            "forecasting",
            "imputation",
            "outlier_detection",
            "change_detection",
        ],
        default="classification",
        help="Task that owns the split artifact (default: classification).",
    )
    parser.add_argument(
        "--targets-root",
        default="targets",
        help="Root containing <dataset>.npz task target bundles.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=None,
        help="Required target horizon for forecasting splits.",
    )
    parser.add_argument(
        "--if-exists",
        choices=["reuse", "fail"],
        default="reuse",
        help="Behavior when split artifact already exists (default: reuse).",
    )
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        default="split",
        help="Split root directory (default: split). "
        "Alias: --output-dir",
    )
    parser.add_argument(
        "--contract-dir",
        default="evidence",
        help="Directory for JSON run logs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier for related outputs.",
    )
    parser.add_argument(
        "--manifest-existing-policy",
        choices=["fail", "reuse", "overwrite"],
        default="fail",
        help="Behavior when the stage manifest for --run-id already exists (default: fail).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and compute split summary without writing split artifact.",
    )
    return parser


def parse_cli_args():
    parser = build_parser()
    args = parser.parse_args()
    if args.dataset_source == "npz":
        if not args.dataset_npz:
            parser.error("--dataset-npz is required when --dataset-source=npz")
        if "," in str(args.dataset_name):
            parser.error("--dataset-source=npz requires exactly one dataset")
    return args


def main() -> int:
    _ensure_repo_on_path()
    from script_chain_contracts import generate_run_id, portable_artifact_reference, write_manifest

    args = parse_cli_args()
    run_id = args.run_id or generate_run_id("run")

    try:
        from torch_ists.split_utils import (
            get_split_path,
            load_splits,
            make_stratified_splits,
            save_splits,
        )

        split_path = resolve_task_split_path(
            args.out_dir,
            args.dataset_name,
            args.seed,
            task=args.task,
        )
        reuse_existing = False
        payload = _load_dataset(args)
        timestamp_summary = payload.timestamp_summary
        target_spec = None

        if args.task != "classification":
            target, target_mask, target_spec = load_task_targets_npz(
                dataset=args.dataset_name,
                targets_root=args.targets_root,
                task=args.task,
                horizon=args.forecast_horizon,
                num_samples=int(payload.X.shape[0]),
                seq_len=int(payload.X.shape[2]),
                input_dim=int(payload.X.shape[1]),
                input_values=payload.X,
            )

        if split_path.exists():
            if args.if_exists == "fail":
                raise FileExistsError(f"split file already exists: {split_path}")
            train_idx, valid_idx, test_idx = load_splits(split_path)
            reuse_existing = True
        else:
            if args.task == "classification":
                assert payload.y is not None
                train_idx, valid_idx, test_idx = make_stratified_splits(y=payload.y, seed=args.seed)
            else:
                train_idx, valid_idx, test_idx = make_fixed_task_splits(
                    num_samples=int(payload.X.shape[0]),
                    seed=args.seed,
                )
            if not args.dry_run:
                save_splits(split_path, train_idx, valid_idx, test_idx)
        if args.task != "classification":
            assert target_spec is not None
            target_spec = dict(target_spec)
            target_spec["split_supervision"] = validate_task_splits(
                task=args.task,
                target=target,
                target_mask=target_mask,
                splits={"training": train_idx, "validation": valid_idx, "test": test_idx},
            )
        preview_fields = (
            {}
            if args.task == "classification"
            else {
                "task": args.task,
                "split_policy": "deterministic_fixed",
                "target_spec": target_spec,
            }
        )
        manifest_path, _manifest = write_manifest(
            stage="split",
            contract_dir=args.contract_dir,
            run_id=run_id,
            existing_policy=args.manifest_existing_policy,
            payload={
                "status": "pass",
                "dataset": args.dataset_name,
                "seed": int(args.seed),
                "dataset_source": args.dataset_source,
                "dataset_npz": portable_artifact_reference(args.dataset_npz, root=True),
                "timestamp_summary": timestamp_summary,
                "split_path": portable_artifact_reference(split_path),
                "reuse_existing": bool(reuse_existing),
                "dry_run": bool(args.dry_run),
                "counts": {
                    "train": int(len(train_idx)),
                    "valid": int(len(valid_idx)),
                    "test": int(len(test_idx)),
                },
                **preview_fields,
            },
        )

        result = {
            "status": "pass",
            "run_id": run_id,
            "dataset_source": args.dataset_source,
            "dataset_npz": portable_artifact_reference(args.dataset_npz, root=True),
            "timestamp_summary": timestamp_summary,
            "split_path": portable_artifact_reference(split_path),
            "split_manifest_path": portable_artifact_reference(manifest_path),
            "reuse_existing": bool(reuse_existing),
            "dry_run": bool(args.dry_run),
            "counts": {
                "train": int(len(train_idx)),
                "valid": int(len(valid_idx)),
                "test": int(len(test_idx)),
            },
            **preview_fields,
        }
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        fail_manifest_path, _manifest = write_manifest(
            stage="split",
            contract_dir=args.contract_dir,
            run_id=run_id,
            existing_policy=args.manifest_existing_policy,
            payload={
                "status": "fail",
                "dataset": args.dataset_name,
                "seed": int(args.seed),
                "dataset_source": args.dataset_source,
                "dataset_npz": portable_artifact_reference(args.dataset_npz, root=True),
                "split_path": "",
                "dry_run": bool(args.dry_run),
                "counts": {"train": 0, "valid": 0, "test": 0},
                "reason": str(exc),
            },
        )
        print(
            json.dumps(
                {
                    "status": "fail",
                    "run_id": run_id,
                    "reason": str(exc),
                    "split_manifest_path": portable_artifact_reference(fail_manifest_path),
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
