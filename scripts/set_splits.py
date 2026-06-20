#!/usr/bin/env python
"""Create reproducible split files for packaged script execution."""

import argparse
import json
import sys
from pathlib import Path

from _common import configure_runtime_paths
from dataset_source_contracts import load_dataset_payload, normalize_dataset_source

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
    from script_chain_contracts import generate_run_id, write_manifest

    args = parse_cli_args()
    run_id = args.run_id or generate_run_id("run")

    try:
        from torch_ists.split_utils import (
            get_split_path,
            load_splits,
            make_stratified_splits,
            save_splits,
        )

        split_path = get_split_path(Path(args.out_dir), args.dataset_name, args.seed)
        reuse_existing = False
        payload = _load_dataset(args)
        timestamp_summary = payload.timestamp_summary

        if split_path.exists():
            if args.if_exists == "fail":
                raise FileExistsError(f"split file already exists: {split_path}")
            train_idx, valid_idx, test_idx = load_splits(split_path)
            reuse_existing = True
        else:
            train_idx, valid_idx, test_idx = make_stratified_splits(y=payload.y, seed=args.seed)
            if not args.dry_run:
                save_splits(split_path, train_idx, valid_idx, test_idx)

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
                "dataset_npz": None if args.dataset_npz is None else str(args.dataset_npz),
                "timestamp_summary": timestamp_summary,
                "split_path": str(split_path),
                "reuse_existing": bool(reuse_existing),
                "dry_run": bool(args.dry_run),
                "counts": {
                    "train": int(len(train_idx)),
                    "valid": int(len(valid_idx)),
                    "test": int(len(test_idx)),
                },
            },
        )

        result = {
            "status": "pass",
            "run_id": run_id,
            "dataset_source": args.dataset_source,
            "dataset_npz": None if args.dataset_npz is None else str(args.dataset_npz),
            "timestamp_summary": timestamp_summary,
            "split_path": str(split_path),
            "split_manifest_path": str(manifest_path),
            "reuse_existing": bool(reuse_existing),
            "dry_run": bool(args.dry_run),
            "counts": {
                "train": int(len(train_idx)),
                "valid": int(len(valid_idx)),
                "test": int(len(test_idx)),
            },
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
                "dataset_npz": None if args.dataset_npz is None else str(args.dataset_npz),
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
                    "split_manifest_path": str(fail_manifest_path),
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
