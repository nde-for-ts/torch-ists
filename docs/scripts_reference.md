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
| `param_search.py` | `split/` | `params/<dataset>/<dataset>_<model>[_aggregated]` |
| `model_run.py` | `split/`, `params/` | `out/<dataset>/<missing_rate>/<name>` (standard/missingness); `out/<dataset>/robustness/<name>__rates_..__types_..__sync_..` (robustness) |

All three scripts also append a JSON manifest to `evidence/` for run tracking.

**Seed policy** (default: hyperparameter search and evaluation seeds are disjoint):
- `set_splits.py --seed` default = `42` — the hyperparameter search split.
- `param_search.py --seed` default = `42` — selects hyperparameters on the split at seed `42`. **The search seed is held at `42` by policy**: the param artifact filename (`<dataset>_<model>[_aggregated]`) does not encode the search seed, so overriding `--seed` silently reuses or overwrites the seed-42 artifact. Override only after backing up or deleting `params/` and passing `--existing-policy fail`.
- `model_run.py --seed` is passed explicitly per evaluation point; the reported standard evaluation uses seeds `0`, `1`, `2`, `3`, `4`.
- With the default `--search-repeat single`, the hyperparameter search seed (`42`) is disjoint from the evaluation seeds (`0-4`), so reported numbers are out-of-sample with respect to hyperparameter selection. **This does not hold for `--search-repeat aggregated`**, which tunes by averaging over the evaluation seed set `0-4` — treat aggregated search as diagnostic, not as out-of-sample official evaluation over `0-4`.
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
- **Run manifest**: `<contract-dir>/<run-id>/split.manifest.json`.

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
| `--seed` | `int` | `42` | Global seed for split selection and reproducibility. Default `42` is the fixed search seed by policy (the param artifact filename does not encode the search seed, so overriding silently reuses or overwrites the seed-42 artifact). For `--search-repeat single` (default), disjoint from evaluation seeds `0-4` so reported numbers are out-of-sample w.r.t. HPO; `--search-repeat aggregated` tunes over `0-4` and is therefore not out-of-sample (diagnostic). |
| `--datasets` | `str` | `BasicMotions` | Comma-separated dataset list. Pass a comma-separated list to run additional datasets. |
| `--dataset-source` | `{default,npz}` | `default` | Dataset source backend. `default` loads benchmark archive data by dataset name through `torch_ists.benchmark_loader.get_data`; `npz` loads one local file through `--dataset-npz`. |
| `--dataset-npz` | `str` | `None` | Required when `--dataset-source npz`; `--datasets` must contain exactly one dataset name. |
| `--models` | `str` | `cnn` | Comma-separated model list. Pass a comma-separated list to run additional models. |
| `--num-samples` | `int` | `20` | Ray Tune `num_samples` — hyperparameter trial budget. |
| `--max-num-epochs` | `int` | `10` | Per-trial training budget. |
| `--max-runs` | `int` | `None` | Stop after this many runs across the dataset × model grid. |
| `--max-runs-mode` | `{completed, new}` | `completed` | What counts toward `--max-runs`: `completed` (executed + reused) or `new` (executed only). |
| `--existing-policy` | `{reuse, fail}` | `reuse` | Behavior when a param artifact already exists. **Note**: the artifact filename (`<dataset>_<model>[_aggregated]`) does not encode the search seed; the search seed is held at `42` by policy. If you override `--seed`, delete or back up the existing `params/` directory and pass `--existing-policy fail` to surface any silently-reused artifact. |
| `--search-policy` | `{default, constrained}` | `default` | Search policy. `default` is the runnable package search. `constrained` is reserved for the capacity-constrained lane. |
| `--search-repeat` | `{single, aggregated}` | `single` | Repeat policy. `single` uses search seed `42`; `aggregated` averages the fixed seed set `0-4` and uses about 5x compute. |
| `--ray-cpu` | `float` | `16.0` | CPU per Ray trial. |
| `--ray-gpu` | `float` | `0.5` | GPU per Ray trial (`0` for CPU-only). |
| `--device` | `{auto, cpu, cuda}` | `auto` | Per-trial device selection. |
| `--split-root` | `str` | `split` | Input split artifact root. |
| `--param-root` | `str` | `params` | Output parameter artifact root. |
| `--contract-dir` | `str` | `evidence` | Directory for run manifests (stage = `search`). |
| `--run-id` | `str` | (auto) | Run identifier. |
| `--task` | `{classification}` | `classification` | Task head. Only `classification` is stable. |
| `--targets-root` | `str` | `targets` | Root directory for dataset target bundles when a task uses external targets. |

### Search Policy And Repeat

- **`default + single`** — default model-specific search on seed `42`. Output filename: `<dataset>_<model>`.
- **`default + aggregated`** — repeats search over the fixed seed set `0-4` and averages trial scores. Output filename: `<dataset>_<model>_aggregated`.
- **`constrained`** — reserved for capacity-constrained search and not the runnable default in this package checkpoint.

The default search uses model-specific axes where relevant, such as attention `n_heads`, flow `flow_depth`, and fixed Neural RDE depth by model name. Batch size and learning rate are resolved outside architecture search.

Parameter search runs the standard search without synthetic missingness injection (`missing_rate=0.0`). Provided datasets may still contain existing `NaN` values. Missing-rate sweeps are evaluation axes handled by `model_run.py --missing-rates` or `model_run.py --robustness-mode`, not by parameter search.

### Output

- **Param file** (pickle): `<param-root>/<dataset>/<dataset>_<model>[_aggregated]` containing the best hyperparameter dict, resolved effective model parameters, and machine-readable search-contract provenance. Older `_cv` aggregated artifacts are read as a compatibility fallback.
- **Run manifest**: `<contract-dir>/<run-id>/search.manifest.json` (records, reused records, skipped, summary). Executed records include resolved batch-size and search-contract metadata.

### Workflow Dependencies

Requires the split file from `set_splits.py` at the configured search seed (default `42`) for every dataset in `--datasets`. For standard evaluation, also generate the evaluation split consumed by `model_run.py` (default seed `0`); generate `set_splits.py --seed 0..4` only when running the full evaluation seed sweep.

---

## 4. `model_run.py`

**Purpose**: Train models with the searched hyperparameters and evaluate on the test set. The default is a single `BasicMotions × cnn × missing_rate=0.0` run. Comma-separated arguments expand this into dataset, model, or missing-rate sweeps.

**Usage** (standard evaluation, bounded first-run smoke):

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/model_run.py \
    --seed 0 \
    --epochs 1 --max-runs 1
```

This is a bounded smoke run; the full default (`--epochs 100`) is the production setting documented in the flags table below.

**Usage** (missingness evaluation, explicit missingness):

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/model_run.py \
    --seed 0 \
    --missing-rates 0.3 \
    --epochs 1 --max-runs 1
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--seed` | `int` | `0` | Random seed. |
| `--datasets` | `str` | `BasicMotions` | Comma-separated dataset list. Pass a comma-separated list to run additional datasets. |
| `--dataset-source` | `{default,npz}` | `default` | Dataset source backend. `default` loads benchmark archive data by dataset name through `torch_ists.benchmark_loader.get_data`; `npz` loads one local file through `--dataset-npz`. |
| `--dataset-npz` | `str` | `None` | Required when `--dataset-source npz`; `--datasets` must contain exactly one dataset name. |
| `--models` | `str` | `cnn` | Comma-separated model list. Pass a comma-separated list to run additional models. |
| `--missing-rates` | `str` | `0.0` | Comma-separated missing rates for standard/missingness evaluation. `0.0` is standard; positive rates are missingness. Ignored in robustness mode. |
| `--epochs` | `int` | `100` | Epochs per (dataset, missing-rate, model) run. |
| `--max-runs` | `int` | `None` | Stop after this many runs across the run grid. |
| `--max-runs-mode` | `{completed, new}` | `completed` | What counts toward `--max-runs`. |
| `--existing-policy` | `{reuse, fail}` | `reuse` | Behavior when an output artifact already exists. |
| `--check` / `--no-check` | flag | `True` | Pre-flight existence check; skip the run if the output file already exists. |
| `--param-policy` | `{default, constrained}` | `default` | Parameter artifact policy to consume. Must match `param_search.py --search-policy`. |
| `--param-repeat` | `{single, aggregated}` | `single` | Parameter artifact repeat policy to consume. Must match `param_search.py --search-repeat`. |
| `--robustness-mode` | flag | `False` | Robustness evaluation: train on `missing=0.0`, evaluate on configured test rates. |
| `--robustness-test-rates` | `str` | `0.0` | Positive test missing rates for robustness mode; the clean `0.0` `STANDARD` anchor is always included. |
| `--robustness-test-mechanisms` | `str` | `MCAR` | Test missingness mechanisms for robustness mode: `MCAR`, `MAR`, `MNAR`, `BLOCK`, `BURST`. |
| `--robustness-channel-sync-options` | `str` | `false` | Test channel synchronization options for robustness mode. Default `false`; pass `false,true` only for an explicit sync sweep. |
| `--save-predictions` | `{full, summary}` | `full` | `full` stores prediction arrays; `summary` stores only shape/dtype metadata. |
| `--split-root` | `str` | `split` | Input split artifact root. |
| `--param-root` | `str` | `params` | Input parameter artifact root. |
| `--out-root` | `str` | `out` | Output prediction artifact root. |
| `--contract-dir` | `str` | `evidence` | Directory for run manifests (stage = `run`). |
| `--run-id` | `str` | (auto) | Run identifier. |
| `--task` | `{classification}` | `classification` | Task head. Only `classification` is stable. |
| `--targets-root` | `str` | `targets` | Root directory for dataset target bundles. |

### Evaluation Regimes

- **Standard** (default) — runs `(dataset × missing-rate × model)` with `missing_rate=0.0`. With defaults this is one `BasicMotions × 0.0 × cnn` configuration. Standard/missingness evaluation uses `channel_sync=False` for missingness generation. Output schema:
  ```json
  {"mode": "standard", "evaluation_regime": "standard", "task": "classification",
   "train_missing_rate": 0.0, "train_missing_type": "STANDARD",
   "train_channel_sync": false,
   "evaluations": [...], "meta": {...}}
  ```
- **Missingness** — selected by a positive `--missing-rates` value. The default missingness mechanism is `MCAR`, `missing_seed` is the run seed, and `channel_sync=False`.
- **Robustness** (`--robustness-mode`) — runs `(dataset × model)`. With defaults this emits one clean anchor evaluation. Additional positive test rates create an explicit robustness sweep; `0.0` is always included once. Trains once on `missing=0.0`; evaluates on every effective rate using the **train-time normalization statistics** (test-time rate changes do not re-fit normalization). Output schema:
  ```json
  {"mode": "robustness", "task": "classification",
   "train_missing_rate": 0.0, "train_missing_type": "STANDARD",
   "evaluations": [{"test_missing_rate": <r>, "metrics": {...}}, ...],
   "meta": {...}}
  ```

  The robustness runner records the clean anchor with `STANDARD` missingness type, plus missingness mechanism (`MCAR`, `MAR`, `MNAR`, `BLOCK`, `BURST`), `channel_sync`, seed, and `robustness_axes`. This schema is experimental. The robustness `missing_seed` is the run `--seed` (provenance, not an independent axis): a single `--seed` run fixes the model, split, and missingness mask together, so to isolate sensitivity to the missingness mask, run multiple `--seed` jobs and aggregate across them.

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

- **Prediction file** (pickle): standard/missingness `<out-root>/<dataset>/<missing_rate>/<dataset>_<missing_rate>_<model>_<seed>`; robustness `<out-root>/<dataset>/robustness/<dataset>_robustness_<model>_<seed>__rates_<...>__types_<...>__sync_<...>`. The robustness path has **no** `<missing_rate>` segment — the test axes are encoded in the filename suffix.
- **Run manifest**: `<contract-dir>/<run-id>/run.manifest.json`. Executed records include resolved batch-size metadata.

### Workflow Dependencies

Requires:
- Split files from `set_splits.py` for every dataset at every reported evaluation seed (`--seed 0..4` for the standard evaluation; the `param_search.py` step additionally needs the configured search seed, default `--seed 42`).
- Param files from `param_search.py --search-policy` and `--search-repeat` matching `model_run.py --param-policy` and `--param-repeat`. The run step validates the parameter artifact search contract before reuse.

---

## 5. Output Artifacts

Path patterns produced by the three scripts:

| Artifact | Path |
|---|---|
| Split | `<split-root>/<dataset>/<dataset>_<seed>` |
| Params (default single) | `<param-root>/<dataset>/<dataset>_<model>` |
| Params (default aggregated) | `<param-root>/<dataset>/<dataset>_<model>_aggregated` |
| Predictions (standard/missingness) | `<out-root>/<dataset>/<missing_rate>/<dataset>_<missing_rate>_<model>_<seed>` |
| Predictions (robustness) | `<out-root>/<dataset>/robustness/<dataset>_robustness_<model>_<seed>__rates_<...>__types_<...>__sync_<...>` |
| Run manifest | `<contract-dir>/<run-id>/{split,search,run}.manifest.json` |

Run manifests record the stage, run id, records executed/reused/skipped, and a stage-specific summary.

---

**End of Scripts Reference**
