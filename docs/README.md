# torch-ists Docs

Reference docs for the current package. Built wheels embed the top-level README as package metadata but do not install this `docs/` tree.

**Related docs**: [Package overview](../README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Scripts Reference](scripts_reference.md) · [Data Reference](data_reference.md) · [Custom Model Reference](custom_model_reference.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

## Included Docs

- [Package overview](../README.md) — install, quick start, and current scope
- [API Reference](api_reference.md) — exported runtime API and source-checkout script behavior
- [Model Reference](model_reference.md) — per-model origin paper, canonical repo, and implementation note
- [Scripts Reference](scripts_reference.md) — source-checkout CLI scripts: flags, outputs, and pipeline order
- [Data Reference](data_reference.md) — guidance for adding user-provided datasets and NPZ script inputs
- [Custom Model Reference](custom_model_reference.md) — guidance for using custom models alongside the package

## Installation

- `PYTHONNOUSERSITE=1` is required for the source-checkout scripts and recommended for local runs
- `Python >= 3.10`
- `pip install -e .` from a source checkout installs the runtime package while keeping top-level `scripts/` available in the checkout
- built wheels install the `torch_ists` runtime package only; they do not install top-level `scripts/`, `docs/`, or `PACKAGE_STATUS.json`
- `pip install -e .[tune]` enables the source-checkout `scripts/param_search.py`
- `pip install -e .[ode]` installs the packaged ODE backend requirements (`torchdiffeq`, `torchdyn`)
- `pip install -e .[cde]` installs the packaged NeuralCDE backend requirements (`torchcde`, `torchdiffeq`)
- `pip install -e .[sde]` installs the packaged SDE backend requirements (`torchsde`, `torchcde`)

## Current Scope

- classification is the primary supported workflow
- the source-checkout scripts cover split creation, parameter search, and evaluation
- the docs in this directory describe the packaged models and source-checkout scripts
- the documented model roster contains 47 packaged models
- benchmark loading and CLI behavior are documented in the API, model, and scripts references
- data reference documents the supported user-provided dataset path through `DataLoader`, NPZ inputs, and packaged split artifacts
- custom model reference documents the supported wrapper pattern for user-owned training code
