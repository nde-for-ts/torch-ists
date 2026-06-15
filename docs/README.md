# torch-ists Docs

Reference docs for the `0.1.7` package release.

**Related docs**: [Package overview](../README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Scripts Reference](scripts_reference.md) · [New Data Onboarding](new_data_onboarding.md) · [New Model Onboarding](new_model_onboarding.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

## Included Docs

- [Package overview](../README.md) — install, quick start, and current scope
- [API Reference](api_reference.md) — exported runtime API and packaged script behavior
- [Model Reference](model_reference.md) — per-model origin paper, canonical repo, and implementation note
- [Scripts Reference](scripts_reference.md) — bundled CLI scripts: flags, outputs, and pipeline order
- [New Data Onboarding](new_data_onboarding.md) — guidance for adding user-provided datasets and NPZ script inputs
- [New Model Onboarding](new_model_onboarding.md) — guidance for using external custom models alongside the package

## Installation

- `PYTHONNOUSERSITE=1` is required for the bundled scripts and recommended for local runs
- `Python >= 3.10`
- `pip install -e .` installs the runtime package
- `pip install -e .[tune]` enables `scripts/param_search.py`
- `pip install -e .[ode]` installs the packaged ODE backend requirements (`torchdiffeq`, `torchdyn`)
- `pip install -e .[cde]` installs the packaged NeuralCDE backend requirements (`torchcde`, `torchdiffeq`)
- `pip install -e .[sde]` installs the packaged SDE backend requirements (`torchsde`, `torchcde`)

## Current Scope

- classification is the primary supported workflow
- the bundled scripts cover split creation, parameter search, and evaluation
- the docs in this directory describe the models and scripts shipped in version `0.1.7`
- the documented model roster contains 47 packaged models
- benchmark loading, implementation files, and CLI import behavior are documented in the API, model, and scripts references
- new-data onboarding documents the supported custom dataset path through `DataLoader`, NPZ inputs, and packaged split artifacts
- new-model onboarding is documented as an external-wrapper pattern for user-owned training code
