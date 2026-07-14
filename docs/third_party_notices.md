# torch-ists Third-Party Notices

> **Type**: Notices
> **Scope**: Upstream sources and licenses for third-party code included in the runtime.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Custom Data Guide](custom_data_guide.md) · [Custom Model Guide](custom_model_guide.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

`torch-ists` is MIT-licensed (Copyright (c) 2024-2026 YongKyung Oh; see [LICENSE](../LICENSE)). It also includes third-party code copied or adapted from the upstreams below; each retains its own license. Per-model papers and repositories are in the [Model Reference](model_reference.md). Paths are relative to `torch_ists/module/`.

## Third-party code

| Component | Models | Path | Upstream | License |
|---|---|---|---|---|
| Phased LSTM | `plstm` | `basic/plstm.py` | [dannyneil/pytorch_plstm](https://github.com/dannyneil/pytorch_plstm) | MIT |
| Time-Aware LSTM | `tlstm` | `basic/tlstm.py` | [duskybomb/tlstm](https://github.com/duskybomb/tlstm) | None |
| Time-Gated LSTM | `tglstm` | `basic/tglstm.py` | [FedericOldani/TGLSTM](https://github.com/FedericOldani/TGLSTM) | None |
| SAnD | `sand` | `attention/sand.py` | [khirotaka/SAnD](https://github.com/khirotaka/SAnD) | MIT |
| mTAN | `mtan` | `attention/mtan.py` | [reml-lab/mTAN](https://github.com/reml-lab/mTAN) | MIT |
| Multi-view Integration Attention Model (MIAM) | `miam` | `attention/miam.py` | [YurimALee/MIAM](https://github.com/YurimALee/MIAM) | None |
| ODE-LSTM | `ode-lstm` | `ode/odelstm.py` | [mlech26l/ode-lstms](https://github.com/mlech26l/ode-lstms) | MIT |
| GRU-ODE-Bayes | `gru-ode-bayes` | `ode/source_variants.py` | [edebrouwer/gru_ode_bayes](https://github.com/edebrouwer/gru_ode_bayes) | MIT |
| Latent ODE / ODE-RNN | `latentode`, `ode-rnn` | `ode/source_variants.py` | [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode) | MIT |
| Neural Jump ODE (NJ-ODE) | `njode` | `ode/njode.py` | [HerreraKrachTeichmann/NJODE](https://github.com/HerreraKrachTeichmann/NJODE) | MIT |
| Latent SDE example | `latentsde` | `sde/latentsde.py` | [google-research/torchsde](https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py) | Apache-2.0 |
| Neural CDE | `neuralcde`, `neuralcde-l/r/c/h`, `gru-d`, `gru-dt`, `gru-ode`, `ode-rnn-spline` | `cde/{_controldiffeq/,metamodel.py}` | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | Apache-2.0 |
| Neural RDE | `neuralrde-1/2/3` | `cde/nrde_model.py` | Adapted from the NeuralCDE metamodel lineage; method paper: [Morrill et al. 2021](https://arxiv.org/abs/2009.08295) | Apache-2.0 lineage for copied structure |
| ANCDE | `ancde` | `cde/ancde/` | [sheoyon-jhin/ANCDE](https://github.com/sheoyon-jhin/ANCDE) | Apache-2.0 |
| EXIT | `exit` | `cde/exit/` | [sheoyon-jhin/EXIT](https://github.com/sheoyon-jhin/EXIT) | Apache-2.0 |
| LEAP | `leap` | `cde/leap/` | [alflsowl12/LEAP](https://github.com/alflsowl12/LEAP) | None |
| TorchDiffEqPack | `exit`, `leap` | `cde/{exit,leap}/_controldiffeq/TorchDiffEqPack/` | [juntang-zhuang/TorchDiffEqPack](https://github.com/juntang-zhuang/TorchDiffEqPack) | MIT |

## MIT notices for included adaptations

The complete Apache License 2.0 text for Apache-derived components is included as [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0) and is packaged in source and wheel distributions.

### SAnD and mTAN

Copyright (c) 2019 Hirotaka Kawashima

Copyright (c) 2020 Satya Narayan Shukla

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### GRU-ODE-Bayes

Copyright (c) 2022 Edward De Brouwer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Latent ODE

Copyright (c) 2019 Yulia Rubanova

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Neural Jump ODE (NJ-ODE)

Copyright (c) 2020 HerreraKrachTeichmann

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**None** means the upstream publishes no LICENSE file. Such code is all-rights-reserved by default; confirm redistribution rights with the original authors. Attribution is provided here and, where present, in source headers.

## Runtime dependencies

Installed separately, each under its own license (not bundled):

- **Core**: `torch`, `numpy`, `scipy`, `scikit-learn`
- **Continuous-time backends** (`.[ode]` / `.[cde]` / `.[sde]`): `torchdiffeq`, `torchdyn`, `torchcde`, `torchsde`
- **Neural RDE log-signatures** (source-build optional extra): `signatory`
- **Optional** (`.[tune]` / `.[data]`): `ray[tune]`, `pandas`, `sktime`

## Notes

- The SDE / Stable Neural SDE modules (`sde/`) are the author's own ICLR 2024 work ([yongkyung-oh/Stable-Neural-SDEs](https://github.com/yongkyung-oh/Stable-Neural-SDEs)), same MIT license — not third-party.
- When updating included code, keep its upstream header and source attribution current.
- Re-implemented models that copy no upstream code are credited in the [Model Reference](model_reference.md), not here.

---

**End of Third-Party Notices**
