# torch-ists Docs

Reference docs for the public `0.1.4` package release.

**Related docs**: [Package overview](../README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Scripts Reference](scripts_reference.md) · [New Model Onboarding](new_model_onboarding.md) · [Project site](https://nde-for-ts.github.io/)

## Included Docs

- [Package overview](../README.md) — install, quick start, and current scope
- [API Reference](api_reference.md) — exported runtime API and packaged script contract
- [Model Reference](model_reference.md) — per-model origin paper, canonical repo, and implementation note
- [Scripts Reference](scripts_reference.md) — bundled CLI scripts: flags, outputs, and pipeline order
- [New Model Onboarding](new_model_onboarding.md) — public extension boundary for external custom models

## Installation

- `PYTHONNOUSERSITE=1` is required for the bundled scripts and recommended for local runs
- `Python >= 3.10`
- `pip install -e .` installs the runtime package
- `pip install -e .[tune]` enables `scripts/param_search.py`
- `pip install -e .[ode]` installs the packaged ODE backend requirements (`torchdiffeq`, `torchdyn`)
- `pip install -e .[cde]` installs the packaged NeuralCDE backend requirements (`torchcde`, `torchdiffeq`)
- `pip install -e .[sde]` installs the packaged SDE backend requirements (`torchsde`, `torchcde`)

## Current Scope

- classification is the primary supported packaged workflow
- the bundled scripts cover split creation, parameter search, and evaluation
- the docs in this directory describe the models and scripts shipped in version `0.1.4`
- the documented model roster contains 35 public package models
- UCR/UEA loading, public implementation files, and CLI import behavior are documented in the API, model, and scripts references
- new-model onboarding is documented as an external-wrapper workflow, not a public handler-registration API
