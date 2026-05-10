# torch-ists Docs

Reference docs for the public `0.1.1` package release.

**Related docs**:
[Package overview](../README.md) ·
[API Reference v0.1.1](api_reference_v0.1.1.md) ·
[Model Reference v0.1.1](model_reference_v0.1.1.md) ·
[Scripts Reference v0.1.1](scripts_reference_v0.1.1.md) ·
[Project site](https://nde-for-ts.github.io/)

## Included Docs

- [Package overview](../README.md) — install, quick start, and current scope
- [API Reference v0.1.1](api_reference_v0.1.1.md) — exported runtime API and packaged script contract
- [Model Reference v0.1.1](model_reference_v0.1.1.md) — per-model origin paper, canonical repo, and implementation note
- [Scripts Reference v0.1.1](scripts_reference_v0.1.1.md) — bundled CLI scripts: flags, outputs, and pipeline order

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
- version `0.1.1` adds the basic/attention bridge: `bilstm`, `gru-simple`, `transformer`
- flow-family, LatentSDE, custom-data, and TSDE-family backbones remain outside version `0.1.1`
