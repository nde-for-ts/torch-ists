# torch-ists Scripts Reference

> **Type**: Reference
> **Scope**: Source-checkout command-line entry points to the benchmark pipeline.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Custom Data Guide](custom_data_guide.md) · [Custom Model Guide](custom_model_guide.md) · [Third-Party Notices](third_party_notices.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

This document covers the three source-checkout scripts: [`scripts/set_splits.py`](../scripts/set_splits.py), [`scripts/param_search.py`](../scripts/param_search.py), and [`scripts/model_run.py`](../scripts/model_run.py). Built wheels do not install these files or expose console entry points.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [`set_splits.py`](#2-set_splitspy)
3. [`param_search.py`](#3-param_searchpy)
4. [`model_run.py`](#4-model_runpy)
5. [Output Artifacts](#5-output-artifacts)
---

## 1. Pipeline Overview

The three scripts run in this order:

```
set_splits.py  →  param_search.py  →  model_run.py
   split/            params/             out/
                                     evidence/  (run manifests, all stages)
```

**Install requirement.** The scripts call `preprocess`, whose finalization imports `torchcde` before building interpolation coefficients — this runs for **every** model, including basic baselines such as `cnn`. So the source-checkout script workflow needs `torchcde` even when the selected model is not a continuous-time method. Install `.[all]` (recommended) or at least `.[tune,cde]`; `.[tune]` alone can fail with `No module named 'torchcde'`.

Each step writes to a dedicated root directory and reads from the previous step's output:

| Stage | Reads from | Writes to |
|---|---|---|
| `set_splits.py` | (raw dataset) | `split/<dataset>/<dataset>_<seed>` |
| `param_search.py` | `split/` | `params/<dataset>/<dataset>_<model>[_cv]` |
| `model_run.py` | `split/`, `params/` | `out/<dataset>/[robustness/]<missing_rate>/<name>` |

All three scripts also append a JSON manifest to `evidence/` for run tracking.

**Seed policy** (default: hyperparameter search and evaluation seeds are disjoint):
- `set_splits.py --seed` default = `42` — the hyperparameter search split.
- `param_search.py --seed` default = `42` — selects hyperparameters on the split at seed `42`. **The search seed is held at `42` by policy**: the param artifact filename (`<dataset>_<model>[_cv]`) does not encode the search seed, so overriding `--seed` silently reuses or overwrites the seed-42 artifact. Override only after backing up or deleting `params/` and passing `--existing-policy fail`.
- `model_run.py --seed` is passed explicitly per evaluation point; the standard reported evaluation uses seeds `0`, `1`, `2`, `3`, `4`.
- The hyperparameter search seed (`42`) is disjoint from the evaluation seeds (`0-4`), so every reported number is out-of-sample with respect to hyperparameter selection.
- Reproducing the ICLR 2024 evaluation seed: after generating the fixed search split at seed `42`, pass `--seed 0` to `set_splits.py` and `model_run.py`. Keep `param_search.py --seed` at its default `42` — the search seed is fixed by policy (see above).

**Batch size**:
- `param_search.py` and `model_run.py` do not search `batch_size`; they determine it automatically from the loaded training split and sample footprint.
- The resolved batch-size metadata is recorded in search/run manifests and run payload metadata.

**Search contract**:
- `param_search.py` uses the package default search contract and records search metadata in parameter artifacts and manifests.
- Historical width/depth-only profiles are internal compatibility metadata, not part of the public workflow.

**Environment variable**:
- `TORCH_ISTS_RUNTIME_ROOT` (optional) — override the runtime root when running the scripts from outside the package directory.
- `TORCH_ISTS_DATA_ROOT` (optional) — override the built-in UCR/UEA archive root used by `--dataset-source default`.

- `--help` works from a source checkout with `PYTHONPATH=.`.

---

## 2. `set_splits.py`

**Purpose**: Generate stratified train/validation/test splits for a dataset with a fixed seed.

**Usage**:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/set_splits.py --dataset-name BasicMotions --seed 42
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset-name` | `str` | (required) | Dataset id used in the split file path. |
| `--dataset-source` | `{default,npz}` | `default` | Dataset source backend. `default` loads benchmark archive data by dataset name through `torch_ists.benchmark_loader.get_data`; `npz` loads a local file through `--dataset-npz`. |
| `--dataset-npz` | `str` | `None` | Required when `--dataset-source npz`; NPZ must contain `X [N,D,L]` and `y [N]`. |
| `--seed` | `int` | `42` | Random seed for the stratified split. Default `42` is the hyperparameter search seed (disjoint from evaluation seeds `0-4`); pass `--seed 0..4` to generate the evaluation splits, or `--seed 0` for ICLR 2024 reproducibility. |
| `--if-exists` | `{reuse, fail}` | `reuse` | Behavior when the split artifact already exists. `reuse` keeps the existing file; `fail` raises. |
| `--out-dir` (alias `--output-dir`) | `str` | `split` | Root directory for split artifacts. |
| `--contract-dir` | `str` | `evidence` | Directory for run manifests (stage = `split`). |
| `--run-id` | `str` | (auto) | Run identifier for cross-stage manifest correlation. Auto-generated if omitted. |
| `--dry-run` | flag | `False` | Validate inputs and compute the split summary without writing the artifact. |

### Output

- **Split file** (pickle): `<out-dir>/<dataset-name>/<dataset-name>_<seed>` containing `(train_idx, valid_idx, test_idx)`.
- **Run manifest**: `<contract-dir>/split_<run-id>.json`.

---

## 3. `param_search.py`

**Purpose**: Run a hyperparameter search for one or more models on one or more datasets, using splits from `set_splits.py`. The default is a single `BasicMotions × cnn` configuration. Multi-sample search uses Ray Tune; single-sample search (`--num-samples 1`) runs locally without Ray.

**Usage** (bounded first-run smoke):

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/param_search.py \
    --seed 42 \
    --num-samples 1 --max-num-epochs 1
```

This is a bounded smoke run; the full defaults (`--num-samples 20 --max-num-epochs 10`) are the production settings documented in the flags table below.

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--seed` | `int` | `42` | Global seed for split selection and reproducibility. Default `42` is the fixed search seed by policy (the param artifact filename does not encode the search seed, so overriding silently reuses or overwrites the seed-42 artifact). Disjoint from evaluation seeds `0-4`, so reported numbers are out-of-sample with respect to hyperparameter selection. |
| `--datasets` | `str` | `BasicMotions` | Comma-separated dataset list. Pass a comma-separated list to run additional datasets. |
| `--dataset-source` | `{default,npz}` | `default` | Dataset source backend. `default` loads benchmark archive data by dataset name through `torch_ists.benchmark_loader.get_data`; `npz` loads one local file through `--dataset-npz`. |
| `--dataset-npz` | `str` | `None` | Required when `--dataset-source npz`; `--datasets` must contain exactly one dataset name. |
| `--models` | `str` | `cnn` | Comma-separated model list. Pass a comma-separated list to run additional models. |
| `--missing-rate` | `float` | `0.0` | Single missing rate applied uniformly during tuning. |
| `--num-samples` | `int` | `20` | Ray Tune `num_samples` — hyperparameter trial budget. |
| `--max-num-epochs` | `int` | `10` | Per-trial training budget. |
| `--max-runs` | `int` | `None` | Stop after this many runs across the dataset × model grid. |
| `--max-runs-mode` | `{completed, new}` | `completed` | What counts toward `--max-runs`: `completed` (executed + reused) or `new` (executed only). |
| `--existing-policy` | `{reuse, fail}` | `reuse` | Behavior when a param artifact already exists. **Note**: the artifact filename (`<dataset>_<model>[_cv]`) does not encode the search seed; the search seed is held at `42` by policy. If you override `--seed`, delete or back up the existing `params/` directory and pass `--existing-policy fail` to surface any silently-reused artifact. |
| `--search-mode` | `{default, cv}` | `default` | `default` runs the single-seed policy search (default `--seed 42`, fixed by policy); `cv` averages over `SEED=0–4` (≈5× compute; note that `cv` uses the evaluation seeds for hyperparameter selection). |
| `--ray-cpu` | `float` | `16.0` | CPU per Ray trial. |
| `--ray-gpu` | `float` | `0.5` | GPU per Ray trial (`0` for CPU-only). |
| `--device` | `{auto, cpu, cuda}` | `auto` | Per-trial device selection. |
| `--split-root` | `str` | `split` | Input split artifact root. |
| `--param-root` | `str` | `params` | Output parameter artifact root. |
| `--contract-dir` | `str` | `evidence` | Directory for run manifests (stage = `search`). |
| `--run-id` | `str` | (auto) | Run identifier. |
| `--task` | `{classification}` | `classification` | Task head. Only `classification` is stable. |
| `--targets-root` | `str` | `targets` | Root directory for dataset target bundles when a task uses external targets. |
| `--model-scope-variant` | `{active}` | `active` | Model scope. Only `active` is supported. |

### Search Modes

- **`default`** — single-seed policy search (default `--seed 42`, fixed by policy; disjoint from evaluation seeds `0-4`). Output filename: `<dataset>_<model>`.
- **`cv`** — cross-validated search averaging over `SEED=0–4`. Output filename: `<dataset>_<model>_cv`. ≈5× the compute. Note: `cv` uses the evaluation seeds for hyperparameter selection, so the resulting numbers are not disjoint from the evaluation; prefer `default` when fairness is the priority.

### Output

- **Param file** (pickle): `<param-root>/<dataset>/<dataset>_<model>[_cv]` containing the best hyperparameter dict, resolved effective model parameters, and machine-readable search-contract provenance.
- **Run manifest**: `<contract-dir>/search_<run-id>.json` (records, reused records, skipped, summary). Executed records include resolved batch-size and search-contract metadata.

### Workflow Dependencies

Requires the split file from `set_splits.py` at the configured search seed (default `42`) for every dataset in `--datasets`. For standard evaluation, also generate the evaluation split consumed by `model_run.py` (default seed `0`); generate `set_splits.py --seed 0..4` only when running the full evaluation seed sweep.

---

## 4. `model_run.py`

**Purpose**: Train models with the searched hyperparameters and evaluate on the test set. The default is a single `BasicMotions × cnn × missing_rate=0.0` run. Comma-separated arguments expand this into dataset, model, or missing-rate sweeps.

**Usage** (standard mode, bounded first-run smoke):

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/model_run.py \
    --seed 0 \
    --epochs 1 --max-runs 1
```

This is a bounded smoke run; the full default (`--epochs 100`) is the production setting documented in the flags table below.

**Usage** (robustness mode):

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/model_run.py \
    --seed 0 \
    --robustness-mode
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--seed` | `int` | `0` | Random seed. |
| `--datasets` | `str` | `BasicMotions` | Comma-separated dataset list. Pass a comma-separated list to run additional datasets. |
| `--dataset-source` | `{default,npz}` | `default` | Dataset source backend. `default` loads benchmark archive data by dataset name through `torch_ists.benchmark_loader.get_data`; `npz` loads one local file through `--dataset-npz`. |
| `--dataset-npz` | `str` | `None` | Required when `--dataset-source npz`; `--datasets` must contain exactly one dataset name. |
| `--models` | `str` | `cnn` | Comma-separated model list. Pass a comma-separated list to run additional models. |
| `--missing-rates` | `str` | `0.0` | Comma-separated missing rates for standard mode. Ignored in robustness mode. |
| `--epochs` | `int` | `100` | Epochs per (dataset, missing-rate, model) run. |
| `--max-runs` | `int` | `None` | Stop after this many runs across the run grid. |
| `--max-runs-mode` | `{completed, new}` | `completed` | What counts toward `--max-runs`. |
| `--existing-policy` | `{reuse, fail}` | `reuse` | Behavior when an output artifact already exists. |
| `--check` / `--no-check` | flag | `True` | Pre-flight existence check; skip the run if the output file already exists. |
| `--param-mode` | `{default, cv}` | `default` | Which `param_search.py` output to consume. Must match the search mode used. |
| `--robustness-mode` | flag | `False` | Robustness evaluation: train on `missing=0.0`, evaluate on configured test rates. |
| `--robustness-test-rates` | `str` | `0.0` | Test missing rates for robustness mode. Pass a comma-separated list for a robustness sweep. |
| `--save-predictions` | `{full, summary}` | `full` | `full` stores prediction arrays; `summary` stores only shape/dtype metadata. |
| `--split-root` | `str` | `split` | Input split artifact root. |
| `--param-root` | `str` | `params` | Input parameter artifact root. |
| `--out-root` | `str` | `out` | Output prediction artifact root. |
| `--contract-dir` | `str` | `evidence` | Directory for run manifests (stage = `run`). |
| `--run-id` | `str` | (auto) | Run identifier. |
| `--task` | `{classification}` | `classification` | Task head. Only `classification` is stable. |
| `--targets-root` | `str` | `targets` | Root directory for dataset target bundles. |
| `--model-scope-variant` | `{active}` | `active` | Model scope. |

### Runtime Modes

- **Standard** (default) — runs `(dataset × missing-rate × model)`. With defaults this is one `BasicMotions × 0.0 × cnn` configuration. Trains once per missing rate and evaluates on the test set at the same rate. Output schema:
  ```json
  {"mode": "standard", "task": "classification",
   "train_missing_rate": <r>, "evaluations": [...], "meta": {...}}
  ```
- **Robustness** (`--robustness-mode`) — runs `(dataset × model)`. Trains once on `missing=0.0`; evaluates on every rate in `--robustness-test-rates` using the **train-time normalization statistics** (test-time rate changes do not re-fit normalization). Output schema:
  ```json
  {"mode": "robustness", "task": "classification",
   "train_missing_rate": 0.0,
   "evaluations": [{"test_missing_rate": <r>, "metrics": {...}}, ...],
   "meta": {...}}
  ```

  The robustness runner records missingness mechanism (`MCAR`, `MAR`, `MNAR`, `BLOCK`, `BURST`), `channel_sync`, seed, and `robustness_axes`. This schema is experimental.

### Task Support Tiers

- **Stable**: `classification` (default).
- **Experimental**: `regression`, `forecasting`, `imputation`, `outlier_detection`, `change_detection` (available through `ists_model`).

The task and metric tables in this document describe `ists_model` API outputs. The public source-checkout script workflow (`scripts/param_search.py`, `scripts/model_run.py`) accepts `--task classification` only in this package; the experimental tasks are reached through the `ists_model` API, not the scripts.

### Metrics

| Task | Metrics |
|---|---|
| `classification` | accuracy, macro F1 (`f1`), weighted F1 (`wf1`) |
| `outlier_detection` | BCE, `outlier_acc` |
| `change_detection` | BCE, `change_acc` |
| `regression`, `forecasting`, `imputation` | MSE, MAE |

### Output

- **Prediction file** (pickle): `<out-root>/<dataset>[/robustness]/<missing_rate>/<dataset>_[robustness_]<model>_<seed>`.
- **Run manifest**: `<contract-dir>/run_<run-id>.json`. Executed records include resolved batch-size metadata.

### Workflow Dependencies

Requires:
- Split files from `set_splits.py` for every dataset at every reported evaluation seed (`--seed 0..4` for the standard evaluation; the `param_search.py` step additionally needs the configured search seed, default `--seed 42`).
- Param files from `param_search.py --search-mode {default|cv}` matching `--param-mode`. The run step validates the parameter artifact search contract before reuse.

---

## 5. Output Artifacts

Path patterns produced by the three scripts:

| Artifact | Path |
|---|---|
| Split | `<split-root>/<dataset>/<dataset>_<seed>` |
| Params (default search) | `<param-root>/<dataset>/<dataset>_<model>` |
| Params (cv search) | `<param-root>/<dataset>/<dataset>_<model>_cv` |
| Predictions (standard) | `<out-root>/<dataset>/<missing_rate>/<dataset>_<model>_<seed>` |
| Predictions (robustness) | `<out-root>/<dataset>/robustness/<missing_rate>/<dataset>_robustness_<model>_<seed>` |
| Run manifest | `<contract-dir>/{split,search,run}_<run-id>.json` |

Run manifests record the stage, run id, records executed/reused/skipped, and a stage-specific summary.

---

**End of Scripts Reference**
