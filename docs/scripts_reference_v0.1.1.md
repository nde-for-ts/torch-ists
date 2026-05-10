# torch-ists Scripts Reference v0.1.1

> **Reference Version**: 0.1.1 (matches package version)
> **Package Version**: 0.1.1
> **Date**: 2026-05-10
> **Scope**: Command-line entry points to the bundled benchmark pipeline.

**Related docs**:
[Package overview](../README.md) ·
[Docs index](README.md) ·
[API Reference v0.1.1](api_reference_v0.1.1.md) ·
[Model Reference v0.1.1](model_reference_v0.1.1.md) ·
[Project site](https://nde-for-ts.github.io/)

This document covers the three bundled scripts:
[`scripts/set_splits.py`](../scripts/set_splits.py),
[`scripts/param_search.py`](../scripts/param_search.py), and
[`scripts/model_run.py`](../scripts/model_run.py). Use the API reference
for Python class signatures and the model reference for per-model origin.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [`set_splits.py`](#2-set_splitspy)
3. [`param_search.py`](#3-param_searchpy)
4. [`model_run.py`](#4-model_runpy)
5. [Output Artifacts and Run Manifests](#5-output-artifacts-and-run-manifests)
6. [Internal Helpers](#6-internal-helpers)

---

## 1. Pipeline Overview

The three scripts run in this order:

```
set_splits.py  →  param_search.py  →  model_run.py
   split/            params/             out/
                                     evidence/  (run manifests, all stages)
```

Each step writes to a dedicated root directory and reads from the previous
step's output:

| Stage | Reads from | Writes to |
|---|---|---|
| `set_splits.py` | (raw dataset) | `split/<dataset>/<dataset>_<seed>` |
| `param_search.py` | `split/` | `params/<dataset>/<dataset>_<model>[_cv]` |
| `model_run.py` | `split/`, `params/` | `out/<dataset>/[robustness/]<missing_rate>/<name>` |

All three scripts also append a JSON manifest to `evidence/` for run tracking.

**Environment variable**:
- `TORCH_ISTS_RUNTIME_ROOT` (optional) — override the runtime root when
  running the scripts from outside the package directory.

---

## 2. `set_splits.py`

**Purpose**: Generate stratified train/validation/test splits for a dataset
with a fixed seed.

**Usage**:

```bash
python scripts/set_splits.py --dataset-name ECG200 --seed 0
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset-name` | `str` | (required) | Dataset id used in the split file path. |
| `--dataset-source` | `str` | `default` | Dataset source backend (`{default}`). The legacy alias `refactoring` is mapped to `default`. |
| `--seed` | `int` | `0` | Random seed for the stratified split. |
| `--if-exists` | `{reuse, fail}` | `reuse` | Behavior when the split artifact already exists. `reuse` keeps the existing file; `fail` raises. |
| `--out-dir` (alias `--output-dir`) | `str` | `split` | Root directory for split artifacts. |
| `--contract-dir` | `str` | `evidence` | Directory for run manifests (stage = `split`). |
| `--run-id` | `str` | (auto) | Run identifier for cross-stage manifest correlation. Auto-generated if omitted. |
| `--dry-run` | flag | `False` | Validate inputs and compute the split summary without writing the artifact. |

### Output

- **Split file** (pickle): `<out-dir>/<dataset-name>/<dataset-name>_<seed>`
  containing `(train_idx, valid_idx, test_idx)`.
- **Run manifest**: `<contract-dir>/split_<run-id>.json`.

---

## 3. `param_search.py`

**Purpose**: Run a Ray Tune hyperparameter search for one or more models on
one or more datasets, using the splits produced by `set_splits.py`.

**Usage**:

```bash
python scripts/param_search.py \
    --datasets ECG200,GunPoint \
    --models neuralcde,neural_lsde \
    --seed 0
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--seed` | `int` | `0` | Global seed for split selection and reproducibility. |
| `--datasets` | `str` | (required) | Comma-separated dataset list. |
| `--dataset-source` | `str` | `default` | Dataset source backend (`{default}`). |
| `--models` | `str` | (active list) | Comma-separated model list. Defaults to `get_active_model_list()` (28 active models in v0.1.1). |
| `--missing-rate` | `float` | `0.0` | Single missing rate applied uniformly during tuning. |
| `--num-samples` | `int` | `20` | Ray Tune `num_samples` — hyperparameter trial budget. |
| `--max-num-epochs` | `int` | `10` | Per-trial training budget. |
| `--max-runs` | `int` | `None` | Stop after this many runs across the dataset × model grid. |
| `--max-runs-mode` | `{completed, new}` | `completed` | What counts toward `--max-runs`: `completed` (executed + reused) or `new` (executed only). |
| `--existing-policy` | `{reuse, fail}` | `reuse` | Behavior when a param artifact already exists. |
| `--search-mode` | `{default, cv}` | `default` | `default` runs at `SEED=0` only; `cv` averages over `SEED=0–4` (≈5× compute). |
| `--ray-cpu` | `float` | `16.0` | CPU per Ray trial. |
| `--ray-gpu` | `float` | `0.3` | GPU per Ray trial (`0` for CPU-only). |
| `--device` | `{auto, cpu, cuda}` | `auto` | Per-trial device selection. |
| `--split-root` | `str` | `split` | Input split artifact root. |
| `--param-root` | `str` | `params` | Output parameter artifact root. |
| `--contract-dir` | `str` | `evidence` | Directory for run manifests (stage = `search`). |
| `--run-id` | `str` | (auto) | Run identifier. |
| `--task` | `{classification}` | `classification` | Task head. Only `classification` is stable in v0.1.1. |
| `--targets-root` | `str` | `targets` | Root directory for dataset target bundles (used by experimental task modes). |
| `--model-scope-variant` | `{active}` | `active` | Model scope. Only `active` is supported in v0.1.1. |

### Search Modes

- **`default`** — single-seed search at `SEED=0`. Output filename: `<dataset>_<model>`.
- **`cv`** — cross-validated search averaging over `SEED=0–4`. Output filename: `<dataset>_<model>_cv`. ≈5× the compute.

### Output

- **Param file** (pickle): `<param-root>/<dataset>/<dataset>_<model>[_cv]`
  containing the best hyperparameter dict.
- **Run manifest**: `<contract-dir>/search_<run-id>.json` (records, reused
  records, skipped, summary).

### Workflow Dependencies

Requires split files from `set_splits.py --seed <SEED>` for every dataset
in `--datasets`.

---

## 4. `model_run.py`

**Purpose**: Train models with the searched hyperparameters and evaluate on
the test set, in either standard mode (sweep over missing rates) or
robustness mode (train at `missing=0`, test on multiple rates).

**Usage** (standard mode):

```bash
python scripts/model_run.py \
    --datasets ECG200 \
    --models neural_lsde \
    --seed 0
```

**Usage** (robustness mode):

```bash
python scripts/model_run.py \
    --datasets ECG200 \
    --models neural_lsde \
    --seed 0 \
    --robustness-mode \
    --robustness-test-rates "0.0,0.3,0.5,0.7"
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--seed` | `int` | `0` | Random seed. |
| `--datasets` | `str` | (default list) | Comma-separated dataset list. Defaults to `get_default_selected_datasets()`. |
| `--dataset-source` | `str` | `default` | Dataset source backend. |
| `--models` | `str` | (active list) | Comma-separated model list. Defaults to `get_active_model_list()`. |
| `--missing-rates` | `str` | `0.0,0.3,0.5,0.7` | Comma-separated missing rates for standard mode. Ignored in robustness mode. |
| `--epochs` | `int` | `100` | Epochs per (dataset, missing-rate, model) run. |
| `--max-runs` | `int` | `None` | Stop after this many runs across the run grid. |
| `--max-runs-mode` | `{completed, new}` | `completed` | What counts toward `--max-runs`. |
| `--existing-policy` | `{reuse, fail}` | `reuse` | Behavior when an output artifact already exists. |
| `--check` / `--no-check` | flag | `True` | Pre-flight existence check; skip the run if the output file already exists. |
| `--param-mode` | `{default, cv}` | `default` | Which `param_search.py` output to consume. Must match the search mode used. |
| `--robustness-mode` | flag | `False` | Robustness evaluation: train on `missing=0`, evaluate on multiple test rates. |
| `--robustness-test-rates` | `str` | `0.0,0.3,0.5,0.7` | Test missing rates for robustness mode. Only used when `--robustness-mode` is set. |
| `--save-predictions` | `{full, summary}` | `full` | `full` stores prediction arrays; `summary` stores only shape/dtype metadata. |
| `--split-root` | `str` | `split` | Input split artifact root. |
| `--param-root` | `str` | `params` | Input parameter artifact root. |
| `--out-root` | `str` | `out` | Output prediction artifact root. |
| `--contract-dir` | `str` | `evidence` | Directory for run manifests (stage = `run`). |
| `--run-id` | `str` | (auto) | Run identifier. |
| `--task` | `{classification}` | `classification` | Task head. Only `classification` is stable in v0.1.1. |
| `--targets-root` | `str` | `targets` | Root directory for dataset target bundles. |
| `--model-scope-variant` | `{active}` | `active` | Model scope. |

### Runtime Modes

- **Standard** (default) — triple loop `(dataset × missing-rate × model)`.
  Trains once per missing rate and evaluates on the test set at the same
  rate. Output schema:
  ```json
  {"mode": "standard", "task": "classification",
   "train_missing_rate": <r>, "evaluations": [...], "meta": {...}}
  ```
- **Robustness** (`--robustness-mode`) — double loop `(dataset × model)`.
  Trains once on `missing=0.0`; evaluates on every rate in
  `--robustness-test-rates` using the **train-time normalization
  statistics** (test-time rate changes do not re-fit normalization).
  Output schema:
  ```json
  {"mode": "robustness", "task": "classification",
   "train_missing_rate": 0.0,
   "evaluations": [{"test_missing_rate": <r>, "metrics": {...}}, ...],
   "meta": {...}}
  ```

### Task Support Tiers

- **Stable**: `classification` (default).
- **Experimental** (available through the underlying `ists_model` but not
  yet a stable script entry point in v0.1.1): `regression`, `forecasting`,
  `imputation`, `outlier_detection`, `change_detection`.

### Metrics

| Task | Metrics |
|---|---|
| `classification` | accuracy, macro F1 (`f1`), weighted F1 (`wf1`) |
| `outlier_detection` | BCE, `outlier_acc` |
| `change_detection` | BCE, `change_acc` |
| `regression`, `forecasting`, `imputation` | MSE, MAE |

### Output

- **Prediction file** (pickle):
  `<out-root>/<dataset>[/robustness]/<missing_rate>/<dataset>_[robustness_]<model>_<seed>`.
- **Run manifest**: `<contract-dir>/run_<run-id>.json`.

### Workflow Dependencies

Requires:
- Split files from `set_splits.py --seed <SEED>` for every dataset.
- Param files from `param_search.py --search-mode {default|cv}` matching
  `--param-mode`.

---

## 5. Output Artifacts and Run Manifests

Path patterns produced by the three scripts:

| Artifact | Path |
|---|---|
| Split | `<split-root>/<dataset>/<dataset>_<seed>` |
| Params (default search) | `<param-root>/<dataset>/<dataset>_<model>` |
| Params (cv search) | `<param-root>/<dataset>/<dataset>_<model>_cv` |
| Predictions (standard) | `<out-root>/<dataset>/<missing_rate>/<dataset>_<model>_<seed>` |
| Predictions (robustness) | `<out-root>/<dataset>/robustness/<missing_rate>/<dataset>_robustness_<model>_<seed>` |
| Run manifest | `<contract-dir>/{split,search,run}_<run-id>.json` |

All run manifests share a common envelope (`SCHEMA_VERSION = "run_manifest_v1"`,
defined in `script_chain_contracts.py`) recording the stage, run id,
records executed/reused/skipped, and a stage-specific summary.

---

## 6. Internal Helpers

The following modules under `scripts/` support the three CLI entry points
but are **not part of the public Python API** and may change between
releases. Do not import them from your own code:

- `_common.py` — runtime path setup, model-scope variant resolver,
  `default_method_for_model()` delegation.
- `script_chain_contracts.py` — manifest schema, run-id generation, stable
  hashing.
- `dataset_source_contracts.py` — dataset loading from the default
  registry or NPZ files.
- `task_target_contracts.py` — task target loading for experimental task
  modes.
- `_training_runtime.py` — dataset construction, training and evaluation
  loops, missing-data validation gates.

Some flags are intentionally hidden as internal or experimental
(`argparse.SUPPRESS`) and may change without notice.

---

**End of Scripts Reference v0.1.1**
