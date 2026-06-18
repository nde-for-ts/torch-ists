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
| ODE-LSTM | `ode-lstm` | `ode/odelstm.py` | [mlech26l/ode-lstms](https://github.com/mlech26l/ode-lstms) | MIT |
| Neural CDE `controldiffeq` | `neuralcde`, `gru-d`, `gru-dt`, `gru-ode`, `ode-rnn-spline` | `cde/_controldiffeq/` | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | Apache-2.0 |
| Neural RDE | `neuralrde-1/2/3` | `cde/nrde_model.py` | Adapted from the NeuralCDE metamodel lineage; method paper: [Morrill et al. 2021](https://arxiv.org/abs/2009.08295) | Apache-2.0 lineage for copied structure |
| ANCDE | `ancde` | `cde/ancde/` | [sheoyon-jhin/ANCDE](https://github.com/sheoyon-jhin/ANCDE) | Apache-2.0 |
| EXIT | `exit` | `cde/exit/` | [sheoyon-jhin/EXIT](https://github.com/sheoyon-jhin/EXIT) | Apache-2.0 |
| LEAP | `leap` | `cde/leap/` | [alflsowl12/LEAP](https://github.com/alflsowl12/LEAP) | None |
| TorchDiffEqPack | `exit`, `leap` | `cde/{exit,leap}/_controldiffeq/TorchDiffEqPack/` | [juntang-zhuang/TorchDiffEqPack](https://github.com/juntang-zhuang/TorchDiffEqPack) | MIT |
| NeuralFlow NFE backend (optional) | `neuralflow-r/g/c` when `flow_backend='nfe'` | `flow/_neural_flows/` | [mbilos/neural-flows-experiments](https://github.com/mbilos/neural-flows-experiments) | None |

**None** means the upstream publishes no LICENSE file. Such code is all-rights-reserved by default; confirm redistribution rights with the original authors. Included files preserve the upstream URL and author in their headers (the NeuralFlow snapshot instead records its provenance in `flow/_neural_flows/PROVENANCE.json`).

NeuralFlow defaults to the package's own MIT-licensed operators in `flow/operators_native.py`. The NFE snapshot is the optional backend selected only via `flow_backend='nfe'` (requires manual `stribor` install) and is the only path that runs the no-license NFE code; its pinned commit and file hashes are recorded in `flow/_neural_flows/PROVENANCE.json`.

## Runtime dependencies

Installed separately, each under its own license (not bundled):

- **Core**: `torch`, `numpy`, `scipy`, `scikit-learn`
- **Continuous-time backends** (`.[ode]` / `.[cde]` / `.[sde]`): `torchdiffeq`, `torchdyn`, `torchcde`, `torchsde`
- **CDE signatures** (source-build optional extra): `signatory`
- **Optional** (`.[tune]` / `.[data]`): `ray[tune]`, `pandas`, `sktime`

## Notes

- The SDE / Stable Neural SDE modules (`sde/`) are the author's own ICLR 2024 work ([yongkyung-oh/Stable-Neural-SDEs](https://github.com/yongkyung-oh/Stable-Neural-SDEs)), same MIT license — not third-party.
- When updating included code, keep its upstream header and refresh `PROVENANCE.json` for pinned snapshots.
- Re-implemented models that copy no upstream code are credited in the [Model Reference](model_reference.md), not here.

---

**End of Third-Party Notices**
