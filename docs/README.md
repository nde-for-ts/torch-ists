# torch-ists Docs

Reference docs for the public `0.1.0` package release.

**Related docs**:
[Package overview](../README.md) ·
[API Reference v0.1.0](api_reference_v0.1.0.md) ·
[Model Reference v0.1.0](model_reference_v0.1.0.md) ·
[Project site](https://nde-for-ts.github.io/)

## Included Docs

- [Package overview](../README.md) — install, quick start, citations
- [API Reference v0.1.0](api_reference_v0.1.0.md) — module / class / function reference
- [Model Reference v0.1.0](model_reference_v0.1.0.md) — origin paper, canonical
  repo, and implementation notes for each model
- [Project site](https://nde-for-ts.github.io/) — overview, papers, and related work

## Installation

- `PYTHONNOUSERSITE=1` is required for the bundled scripts and recommended for local runs
- `Python >= 3.10`
- `pip install -e .` installs the runtime package
- `pip install -e .[tune]` enables `scripts/param_search.py`
- `pip install -e .[ode]` installs the packaged ODE backend requirements (`torchdiffeq`, `torchdyn`)

## Current Scope

- classification is the primary supported packaged workflow
- the bundled scripts cover split creation, parameter search, and evaluation
- additional task modes exposed by `ists_model` are experimental and less tested than classification
- version `0.1.0` includes the ODE models (`gru-dt`, `gru-d`, `gru-ode`, `ode-rnn-spline`, `ode-lstm`, `ode-rnn`, `latentode`, `gru-ode-bayes`)
- flow-family, LatentSDE, and TSDE-family backbones are outside version `0.1.0`
