# torch-ists Attribution and Licenses

> **Type**: Notices
> **Scope**: Origin, source, and license for packaged methods and benchmark datasets.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Model Reference](model_reference.md) · [Scripts Reference](scripts_reference.md) · [Custom Data Guide](custom_data_guide.md) · [Custom Model Guide](custom_model_guide.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

`torch-ists` is MIT-licensed (Copyright (c) 2024-2026 YongKyung Oh; see [LICENSE](../LICENSE)). Methods implemented in this package carry that license; included third-party code retains its upstream license. Standard `nn.*` baselines are omitted. Per-model papers are in the [Model Reference](model_reference.md). Paths are relative to `torch_ists/module/`.

## Package-authored methods

| Method | Models | Path | Origin | License |
|---|---|---|---|---|
| Stable Neural SDEs | `neural_lsde`, `neural_lnsde`, `neural_gsde` | `sde/` | [Oh, Lim, Kim, ICLR 2024](https://arxiv.org/abs/2402.14989) | MIT (this package) |
| DualDynamics | `dualdynamics`, `dualdynamics-r/g/c` | `flow/dualdynamics.py` | [Oh, Lim, Kim, AAAI 2025](https://arxiv.org/abs/2401.04979) | MIT (this package) |
| FlowPath | `flowpath`, `flowpath-r/g/c` | `flow/flowpath.py` | [Oh et al., AAAI 2026](https://arxiv.org/abs/2511.10841) | MIT (this package) |
| Native NeuralFlow operators | `neuralflow`, `neuralflow-r/g/c` | `flow/operators_native.py` | Package-authored adaptations informed by [Biloš et al. 2021](https://arxiv.org/abs/2110.13040) and its [reference repository](https://github.com/mbilos/neural-flows-experiments) | MIT (this package) |
| SDE baselines | `naive_cde`, `neural_ode`, `neural_ode_t`, `neural_sde`, `neural_sde_t` | `sde/` | Package-authored baselines | MIT (this package) |

DualDynamics and FlowPath reuse the native flow operators as a component but remain distinct methods and registry rows.

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
| Neural Jump ODE (NJ-ODE) | `njode` | `ode/njode.py` | Method paper: Herrera, Krach, and Teichmann, *Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering* (ICLR 2021) — [OpenReview](https://openreview.net/forum?id=JFKR3WqwyXR), [arXiv:2006.04727](https://arxiv.org/abs/2006.04727); source: [HerreraKrachTeichmann/NJODE](https://github.com/HerreraKrachTeichmann/NJODE) | MIT |
| Latent SDE example | `latentsde` | `sde/latentsde.py` | [google-research/torchsde](https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py) | Apache-2.0 |
| Neural CDE | `neuralcde`, `neuralcde-l/r/c/h`, `gru-d`, `gru-dt`, `gru-ode`, `ode-rnn-spline` | `cde/{_controldiffeq/,metamodel.py}` | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | Apache-2.0 |
| Neural RDE | `neuralrde-1/2/3` | `cde/nrde_model.py` | Adapted from the NeuralCDE metamodel lineage; method paper: [Morrill et al. 2021](https://arxiv.org/abs/2009.08295) | Apache-2.0 lineage for copied structure |
| ANCDE | `ancde` | `cde/ancde/` | [sheoyon-jhin/ANCDE](https://github.com/sheoyon-jhin/ANCDE) | Apache-2.0 |
| EXIT | `exit` | `cde/exit/` | [sheoyon-jhin/EXIT](https://github.com/sheoyon-jhin/EXIT) | Apache-2.0 |
| LEAP | `leap` | `cde/leap/` | [jeongwhanchoi/LEAP](https://github.com/jeongwhanchoi/LEAP); method paper: Jhin et al., *Learnable Path in Neural Controlled Differential Equations* (AAAI 2023) — [arXiv:2301.04333](https://arxiv.org/abs/2301.04333) | None |
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

Method citation: Herrera, Krach, and Teichmann, *Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering* (ICLR 2021) — [OpenReview](https://openreview.net/forum?id=JFKR3WqwyXR), [arXiv:2006.04727](https://arxiv.org/abs/2006.04727). The following is the upstream MIT copyright and permission notice; its 2020 copyright year is preserved verbatim and is distinct from the paper's 2021 publication year.

Copyright (c) 2020 HerreraKrachTeichmann

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**None** records that no LICENSE file was found in the cited upstream repository at the reviewed revision; the attribution above identifies that provenance without assigning an upstream license.

### NeuralFlow reference implementation

The `neuralflow` family is informed by the residual, GRU-style, and coupling architectures introduced in *Neural Flows: Efficient Alternative to Neural ODEs* (Biloš, Sommer, Rangapuram, Januschowski, Günnemann; NeurIPS 2021). The reference implementation is [`mbilos/neural-flows-experiments`](https://github.com/mbilos/neural-flows-experiments). That repository does not publish a LICENSE file, so this distribution does not bundle its source. The default operators in `flow/operators_native.py` are package-authored adaptations and are not claimed to be source-identical or numerically equivalent to the reference implementation. The optional `flow_backend="nfe"` adapter loads a user-installed checkout pinned to commit `bd19f7c92461e83521e268c1a235ef845a3dd963`; `flow/NFE_PROVENANCE.json` records the validated source files. The adapter and manifest are MIT-licensed package code, not copies of upstream source.

## Benchmark datasets

The built-in benchmark loader downloads UCR/UEA archive datasets at runtime; they are not bundled with this package. Cite the archive when using those datasets:

- Dau et al., *The UCR Time Series Archive* (2019) — [arXiv:1810.07758](https://arxiv.org/abs/1810.07758)
- Bagnall et al., *The UEA multivariate time series classification archive* (2018) — [arXiv:1811.00075](https://arxiv.org/abs/1811.00075)
- Archive site: <http://www.timeseriesclassification.com/>

## Runtime dependencies

Installed separately, each under its own license (not bundled):

- **Core**: `torch`, `numpy`, `scipy`, `scikit-learn`
- **Continuous-time backends** (`.[ode]` / `.[cde]` / `.[sde]`): `torchdiffeq`, `torchdyn`, `torchcde`, `torchsde`
- **Neural RDE log-signatures** (source-build optional extra): `signatory`
- **Optional** (`.[tune]` / `.[data]`): `ray[tune]`, `pandas`, `sktime`

## Notes

- Package-authored methods are listed separately from included third-party code.
- When updating included third-party code, keep its upstream header and source attribution current.
- Per-model papers for all methods are in the [Model Reference](model_reference.md).

---

**End of Attribution and Licenses**
