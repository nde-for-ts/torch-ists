# torch-ists Scripts Reference

> **Type**: Reference
> **Scope**: Included command-line scripts for the benchmark pipeline.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Custom Data Guide](custom_data_guide.md) · [Custom Model Guide](custom_model_guide.md) · [Third-Party Notices](third_party_notices.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

This document covers the included scripts: [`scripts/set_splits.py`](../scripts/set_splits.py), [`scripts/param_search.py`](../scripts/param_search.py), [`scripts/model_run.py`](../scripts/model_run.py), and [`scripts/aggregate_results.py`](../scripts/aggregate_results.py). Built wheels do not install these files or expose console entry points.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [`set_splits.py`](#2-set_splitspy)
3. [`param_search.py`](#3-param_searchpy)
4. [`model_run.py`](#4-model_runpy)
5. [`aggregate_results.py`](#5-aggregate_resultspy)
6. [Output Artifacts](#6-output-artifacts)
---

## 1. Pipeline Overview

The main scripts run in this order:

```
set_splits.py  →  param_search.py  →  model_run.py
   split/            params/             out/
                                     evidence/  (JSON run logs, all stages)
                                            ↓
                                    aggregate_results.py
```

**Install requirement.** The scripts call `preprocess`, which imports `torchcde` before building interpolation coefficients. This applies even to basic baselines such as `cnn`. Install `.[all]` (recommended) or at least `.[tune,cde]`; `.[tune]` alone can fail with `No module named 'torchcde'`.

Each step writes to a dedicated root directory and reads from the previous step's output:

| Stage | Reads from | Writes to |
|---|---|---|
| `set_splits.py` | (raw dataset) | `split/<dataset>/<dataset>_<seed>` |
| `param_search.py` | `split/` | `params/<dataset>/<dataset>_<model>[_aggregated]` |
| `model_run.py` | `split/`, `params/` | `out/<dataset>/<missing_rate>/<name>` (standard/missingness); `out/<dataset>/robustness/<name>__rates_..__types_..__sync_..` (robustness) |
| `aggregate_results.py` | `evidence/`, `out/` | `aggregate/{runs,robustness_rows,search_cells,exclusions,fallbacks}.csv` |

The split/search/run scripts also write JSON run logs under `evidence/`.

**Seed usage**:
- `set_splits.py --seed` default = `42` — the hyperparameter search split.
- `param_search.py --seed` default = `42` — selects hyperparameters on the split at seed `42`. The parameter filename (`<dataset>_<model>[_aggregated]`) does not include the search seed. If you override `--seed`, first back up or delete `params/` and use `--existing-policy fail`.
- `model_run.py --seed` is passed explicitly per evaluation point; the reported standard evaluation uses seeds `0`, `1`, `2`, `3`, `4`.
- With the default `--search-repeat single`, the search seed (`42`) is separate from evaluation seeds (`0-4`). `--search-repeat aggregated` averages over `0-4` and is diagnostic.
- Reproducing the ICLR 2024 evaluation seed: after generating the search split at seed `42`, pass `--seed 0` to `set_splits.py` and `model_run.py`. Keep `param_search.py --seed` at its default `42`.

**Batch size**:
- `param_search.py` and `model_run.py` do not search `batch_size`; they determine it automatically from the loaded training split and sample footprint.
- The resolved batch size is recorded in the search and run outputs.

**Search settings**:
- `param_search.py` uses the default search settings and records the selected parameters.
- Older width/depth-only settings are accepted only for compatibility.

**Environment variable**:
- `TORCH_ISTS_RUNTIME_ROOT` (optional) — override the runtime root for custom source-checkout layouts.
- `TORCH_ISTS_DATA_ROOT` (optional) — override the built-in UCR/UEA archive root used by `--dataset-source default`.

- `--help` works from a repository checkout with `PYTHONPATH=.`.
- Local dataset, split, parameter, target, and evidence paths are resolved to absolute paths before Ray workers start.

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
| `--dataset-npz` | `str` | `None` | Required when `--dataset-source npz`; NPZ must contain `X [N,D,L]` and `y [N]`; optional `timestamps [N,L]` are validated and recorded. |
| `--seed` | `int` | `42` | Random seed for the stratified split. Default `42` is the hyperparameter search seed (disjoint from evaluation seeds `0-4`); pass `--seed 0..4` to generate the evaluation splits, or `--seed 0` for ICLR 2024 reproducibility. |
| `--if-exists` | `{reuse, fail}` | `reuse` | Behavior when the split artifact already exists. `reuse` keeps the existing file; `fail` raises. |
| `--out-dir` (alias `--output-dir`) | `str` | `split` | Root directory for split artifacts. |
| `--contract-dir` | `str` | `evidence` | Directory for JSON run logs (stage = `split`). |
| `--run-id` | `str` | (auto) | Run identifier for related outputs. Auto-generated if omitted. |
| `--dry-run` | flag | `False` | Validate inputs and compute the split summary without writing the artifact. |

### Output

- **Split file** (pickle): `<out-dir>/<dataset-name>/<dataset-name>_<seed>` containing `(train_idx, valid_idx, test_idx)`.
- **Run log**: `<contract-dir>/<run-id>/split.manifest.json`.

---

## 3. `param_search.py`

**Purpose**: Run a hyperparameter search for one or more models on one or more datasets, using splits from `set_splits.py`. The default is a single `BasicMotions × cnn` configuration. Multi-sample search uses Ray Tune; single-sample search (`--num-samples 1`) runs locally without Ray.

**Usage** (bounded first-run smoke):

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/param_search.py \
    --seed 42 \
    --num-samples 1 --max-num-epochs 1
```

This is a bounded smoke run; the full defaults (`--num-samples 50 --max-num-epochs 20`) are documented in the flags table below.

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--seed` | `int` | `42` | Seed for split selection and hyperparameter search. The default `42` is disjoint from evaluation seeds `0-4` for `--search-repeat single`; `--search-repeat aggregated` averages over `0-4` and is diagnostic. |
| `--datasets` | `str` | `BasicMotions` | Comma-separated dataset list. Pass a comma-separated list to run additional datasets. |
| `--dataset-source` | `{default,npz}` | `default` | Dataset source backend. `default` loads benchmark archive data by dataset name through `torch_ists.benchmark_loader.get_data`; `npz` loads one local file through `--dataset-npz`. |
| `--dataset-npz` | `str` | `None` | Required when `--dataset-source npz`; `--datasets` must contain exactly one dataset name; optional `timestamps [N,L]` are validated and recorded. |
| `--models` | `str` | `cnn` | Comma-separated model list. Pass a comma-separated list to run additional models. |
| `--num-samples` | `int` | `50` | Ray Tune `num_samples` — hyperparameter trial budget. Saved parameter artifacts record this value and mark `--num-samples 1` artifacts as smoke/local artifacts. |
| `--max-num-epochs` | `int` | `20` | Per-trial training budget. |
| `--search-warmup` | `int` | `5` | Minimum completed epochs before in-trial patience stopping may fire. |
| `--search-patience` | `int` | `5` | Stop a search trial after this many non-improving validation epochs. |
| `--search-scheduler-step-size` | `int` | `10` | StepLR step size for search trials. |
| `--asha-grace-period` | `int` | `5` | Minimum epochs before ASHA may stop a Ray trial. |
| `--allow-incomplete-ray-trial` / `--no-allow-incomplete-ray-trial` | flag | enabled | In single-repeat search, use the best finite terminated Ray trial if no finite trial reaches `--max-num-epochs`; disable to require full-budget completion. |
| `--max-runs` | `int` | `None` | Stop after this many runs across the dataset × model grid. |
| `--max-runs-mode` | `{completed, new}` | `completed` | What counts toward `--max-runs`: `completed` (executed + reused) or `new` (executed only). |
| `--existing-policy` | `{reuse, fail}` | `reuse` | Behavior when a param artifact already exists. The filename does not include the search seed; if you override `--seed`, back up or delete `params/` and use `--existing-policy fail`. |
| `--skip-policy` | `{allow, fail}` | `fail` | Whether skipped dataset/model cells fail the run or keep exit status zero. Use `allow` only for exploratory partial runs. |
| `--search-policy` | `{default, constrained}` | `default` | Search setting. `default` is the package search. `constrained` is an experimental capacity-controlled search. |
| `--search-repeat` | `{single, aggregated}` | `single` | Repeat setting. `single` uses the search seed; `aggregated` averages seeds `0-4` and uses about 5x compute. |
| `--ray-cpu` | `float` | `16.0` | CPU per Ray trial. |
| `--ray-gpu` | `float` | `0.5` | GPU per Ray trial (`0` for CPU-only). |
| `--device` | `{auto, cpu, cuda}` | `auto` | Per-trial device selection. |
| `--split-root` | `str` | `split` | Input split artifact root. |
| `--param-root` | `str` | `params` | Output parameter artifact root. |
| `--contract-dir` | `str` | `evidence` | Directory for JSON run logs (stage = `search`). |
| `--run-id` | `str` | (auto) | Run identifier. |
| `--runtime-baseline-id` | `str` | `None` | Optional caller-supplied flat run label. Leave unset for shared benchmark artifacts. |
| `--task` | `{classification}` | `classification` | Task head. Only `classification` is stable. |
| `--targets-root` | `str` | `targets` | Root directory for dataset target bundles when a task uses external targets. |

### Search Policy And Repeat

- **`default + single`** — default model-specific search on seed `42`. Output filename: `<dataset>_<model>`.
- **`default + aggregated`** — repeats search over the fixed seed set `0-4` and averages trial scores. Output filename: `<dataset>_<model>_aggregated`.
- **`constrained + single`** — experimental capacity-controlled search. Output filename uses the capacity-controlled suffix.
- **`constrained + aggregated`** — same capacity-controlled search with seed aggregation. Output filename uses the capacity-controlled suffix plus `_aggregated`.

The default search uses model-specific axes where relevant, such as attention `n_heads`, flow `flow_depth`, and fixed Neural RDE depth by model name. Batch size and learning rate are resolved outside architecture search.

Parameter search runs the standard search without synthetic missingness injection (`missing_rate=0.0`). Provided datasets may still contain existing `NaN` values. Missing-rate sweeps are evaluation axes handled by `model_run.py --missing-rates` or `model_run.py --robustness-mode`, not by parameter search.

### Output

- **Param file** (pickle): `<param-root>/<dataset>/<dataset>_<model>` with optional capacity-controlled and repeat suffixes, containing the selected hyperparameters, effective model parameters, and run details. Older `_cv` aggregated artifacts are read as a compatibility fallback.
- **Run log**: `<contract-dir>/<run-id>/search.manifest.json` (records, reused records, skipped, summary). Executed records include resolved batch size, search settings, and `runtime_baseline_id` only when explicitly supplied.

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
| `--dataset-npz` | `str` | `None` | Required when `--dataset-source npz`; `--datasets` must contain exactly one dataset name; optional `timestamps [N,L]` are validated and recorded. |
| `--models` | `str` | `cnn` | Comma-separated model list. Pass a comma-separated list to run additional models. |
| `--missing-rates` | `str` | `0.0` | Comma-separated missing rates for standard/missingness evaluation. `0.0` is standard; positive rates are missingness. Ignored in robustness mode. |
| `--epochs` | `int` | `100` | Epochs per (dataset, missing-rate, model) run. |
| `--device` | `{auto, cpu, cuda}` | `auto` | Device selection. `auto` uses CUDA when available; `cpu` forces CPU; `cuda` requires CUDA. |
| `--max-runs` | `int` | `None` | Stop after this many runs across the run grid. |
| `--max-runs-mode` | `{completed, new}` | `completed` | What counts toward `--max-runs`. |
| `--existing-policy` | `{reuse, fail}` | `reuse` | Behavior when `--check` finds an output artifact: reuse checks stored output details; fail raises an error. |
| `--skip-policy` | `{allow, fail}` | `fail` | Whether skipped dataset/model cells fail the run or keep exit status zero. Use `allow` only for exploratory partial runs. |
| `--check` / `--no-check` | flag | `True` | Validate and reuse an existing output file when present; `--existing-policy` controls reuse vs fail behavior. |
| `--param-policy` | `{default, constrained}` | `default` | Parameter search setting to consume. Must match `param_search.py --search-policy`. |
| `--param-repeat` | `{single, aggregated}` | `single` | Parameter repeat setting to consume. Must match `param_search.py --search-repeat`. |
| `--robustness-mode` | flag | `False` | Robustness evaluation: train on `missing=0.0`, evaluate on configured test rates. |
| `--robustness-profile` | `{custom, 100-cell}` | `custom` | `100-cell` selects the fixed classification grid described below and requires full predictions. |
| `--robustness-test-rates` | `str` | `0.0` | Positive test missing rates for robustness mode; the clean `0.0` `STANDARD` anchor is always included. |
| `--robustness-test-mechanisms` | `str` | `MCAR` | Test missingness mechanisms for robustness mode: `MCAR`, `MAR`, `MNAR`, `BLOCK`, `BURST`, `CHANNEL_DROP`. |
| `--robustness-channel-sync-options` | `str` | `false` | Test channel synchronization options. `false` masks channels independently; `true` shares missing time indices across channels. |
| `--save-predictions` | `{full, summary}` | `full` | `full` stores prediction arrays; `summary` stores only shape and dtype. |
| `--split-root` | `str` | `split` | Input split artifact root. |
| `--param-root` | `str` | `params` | Input parameter artifact root. |
| `--out-root` | `str` | `out` | Output prediction artifact root. |
| `--contract-dir` | `str` | `evidence` | Directory for JSON run logs (stage = `run`). |
| `--run-id` | `str` | (auto) | Run identifier. |
| `--runtime-baseline-id` | `str` | `None` | Optional caller-supplied flat run label. Leave unset for shared benchmark artifacts. |
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
- **Robustness** (`--robustness-mode`) — runs `(dataset × model)`. The default `custom` setting emits one clean anchor unless additional axes are supplied. `--robustness-profile 100-cell` materializes one clean `STANDARD` anchor, 90 temporal-missingness evaluations (`9` positive rates × `5` mechanisms × `2` channel-sync settings), and 9 `CHANNEL_DROP` evaluations. It is classification-only and stores full prediction arrays. Training occurs once at `missing=0.0`; every evaluation reuses the **train-time normalization statistics**. Output schema:
  ```json
  {"mode": "robustness", "task": "classification",
   "train_missing_rate": 0.0, "train_missing_type": "STANDARD",
   "evaluations": [{"test_missing_rate": <r>, "metrics": {...}}, ...],
   "meta": {...}}
  ```

  The robustness output records the clean anchor, missingness mechanism, `channel_sync`, and `missing_seed`. `CHANNEL_DROP` rows also record the channel count, requested and realized drop counts, dropped channel indices, applicability, exclusion reason, and the number of samples with no remaining observations. A univariate input emits unchanged placeholders marked as excluded because channel outage is not interpretable for one channel. `missing_seed` is the run `--seed`; run multiple seeds to average over split/model/mask variation.

### Task Support Tiers

- **Stable**: `classification` (default).
- **Experimental**: `regression`, `forecasting`, `imputation`, `outlier_detection`, `change_detection` (available through `ists_model`).

The task and metric tables in this document describe `ists_model` API outputs. The included scripts (`scripts/param_search.py`, `scripts/model_run.py`) accept `--task classification` only; experimental tasks are reached through the `ists_model` API.

### Metrics

| Task | Metrics |
|---|---|
| `classification` | accuracy, balanced accuracy, precision/recall/F1, cross-entropy, Brier score, confidence summaries, confusion matrix |
| `outlier_detection` | BCE, `outlier_acc` |
| `change_detection` | BCE, `change_acc` |
| `regression`, `forecasting`, `imputation` | MSE, MAE |

### Output

- **Prediction file** (pickle): standard/missingness `<out-root>/<dataset>/<missing_rate>/<dataset>_<missing_rate>_<model>_<seed>`; robustness `<out-root>/<dataset>/robustness/<dataset>_robustness_<model>_<seed>__rates_<...>__types_<...>__sync_<...>`. The robustness path has **no** `<missing_rate>` segment — the test axes are encoded in the filename suffix. Output metadata includes the output contract, prediction identity, and classification metrics; `runtime_baseline_id` appears only when explicitly supplied.
- **Run log**: `<contract-dir>/<run-id>/run.manifest.json`. Executed records include resolved batch size, output-contract checks, and `runtime_baseline_id` only when explicitly supplied.

### Workflow Dependencies

Requires:
- Split files from `set_splits.py` for every dataset at every reported evaluation seed (`--seed 0..4` for the standard evaluation; the `param_search.py` step additionally needs the configured search seed, default `--seed 42`).
- Param files from `param_search.py --search-policy` and `--search-repeat` matching `model_run.py --param-policy` and `--param-repeat`. The run step checks that the parameter file matches the requested run before reuse.

### Artifact Reuse Boundaries

- Split artifacts keep the stable `train/valid/test` pickle format and are safe to reuse when the dataset and split path are intentional.
- Param artifacts written with `0.3.2+` integrity metadata are validated by identity, hash, search policy, repeat policy, and search profile. Pre-`0.3.2` param artifacts lack those integrity fields; reuse is accepted only with the explicit legacy audit flag and still requires a saved `lr` plus matching requested search axes. Partial integrity metadata and bad hashes are rejected.
- Output artifacts are reused only when `model_run.py --check` finds the same output-contract hash and validates the stored payload contents. For a 100-condition result this includes exact ordered axes, uniqueness, full prediction identity, and complete channel-outage provenance. Incomplete or older payload shapes are recomputed with `--no-check`; there is no silent cross-contract output reuse.

---

## 5. `aggregate_results.py`

**Purpose**: Build compact CSV tables from script run logs and output payloads. Public multi-condition robustness exports require the exact 100-condition structure and full predictions. Identical output references are counted once using canonical path and file hash; conflicting artifacts for the same dataset/model/seed/task fail instead of being mixed.

**Usage**:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/aggregate_results.py \
    --evidence-root evidence --out-root out --output-dir aggregate
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--evidence-root` | `str` | `evidence` | Root containing `*.manifest.json` files from split/search/run scripts. |
| `--out-root` | `str` | `out` | Root containing `model_run.py` output payloads. |
| `--output-dir` | `str` | `aggregate` | Directory for aggregate CSV/JSON files. |
| `--export-scope` | `{public, internal}` | `public` | `public` writes public-safe table fields and redacts local absolute path prefixes; `internal` keeps broader emitted metadata. |

### Output

- `runs.csv` — run-level executed/reused records.
- `robustness_rows.csv` — one row per evaluation payload entry when output payloads are available.
- `search_cells.csv` — search-level executed/reused records.
- `exclusions.csv` — skipped or failed cells with reasons when present.
- `fallbacks.csv` — search cells that used finite-trial fallback.
- `aggregate_summary.json` — row counts and export metadata.

---

## 6. Output Artifacts

Path patterns produced by the scripts:

| Artifact | Path |
|---|---|
| Split | `<split-root>/<dataset>/<dataset>_<seed>` |
| Params (default single) | `<param-root>/<dataset>/<dataset>_<model>` |
| Params (default aggregated) | `<param-root>/<dataset>/<dataset>_<model>_aggregated` |
| Predictions (standard/missingness) | `<out-root>/<dataset>/<missing_rate>/<dataset>_<missing_rate>_<model>_<seed>` |
| Predictions (robustness) | `<out-root>/<dataset>/robustness/<dataset>_robustness_<model>_<seed>__rates_<...>__types_<...>__sync_<...>` |
| Run log | `<contract-dir>/<run-id>/{split,search,run}.manifest.json` |
| Aggregate tables | `<output-dir>/{runs,robustness_rows,search_cells,exclusions,fallbacks}.csv` |

JSON run logs record the stage, run id, executed/reused/skipped records, and a stage summary.

---

**End of Scripts Reference**
