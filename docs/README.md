# torch-ists Docs

Reference docs for the public `0.1.0` package release.

## Included Docs

- package overview: `README.md`
- API reference: `api_reference_v1.0.md`

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
