# torch-ists Model Reference

> **Package Version**: 0.1.5
> **Scope**: Origin paper, canonical repository, and implementation note for every model shipped in `v0.1.5`.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Scripts Reference](scripts_reference.md) · [New Model Onboarding](new_model_onboarding.md) · [Project site](https://nde-for-ts.github.io/)

This document is a companion to [`api_reference.md`](api_reference.md). Use the API reference for class signatures and call patterns; use this document to look up where each model comes from and which upstream implementation our copy is based on.

---

## Table of Contents

1. [Sequence baselines](#1-sequence-baselines)
2. [Attention Family](#2-attention-family)
3. [ODE runtime family](#3-ode-runtime-family)
4. [Additional ODE models](#4-additional-ode-models)
5. [Neural CDE](#5-neural-cde)
6. [SDE baselines](#6-sde-baselines)
7. [Stable Neural SDE](#7-stable-neural-sde)
8. [NeuralFlow](#8-neuralflow)
9. [FlowPath and DualDynamics](#9-flowpath-and-dualdynamics)
10. [How to cite](#10-how-to-cite)

---

## 1. Sequence baselines

| Model | Origin | Implementation in this package |
|---|---|---|
| `cnn` | Standard 1-D convolutional baseline | Plain `nn.Conv1d` stack (this package) |
| `rnn` | Standard RNN baseline | Plain `nn.RNN` (this package) |
| `lstm` | Hochreiter & Schmidhuber, *Long Short-Term Memory* (1997) | Plain `nn.LSTM` (this package) |
| `gru` | Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder* (2014) | Plain `nn.GRU` (this package) |
| `bilstm` | Bidirectional LSTM baseline | Plain bidirectional `nn.LSTM` (this package) |
| `gru-simple` | Simplified GRU baseline | Plain `nn.GRU` baseline variant (this package) |
| `tlstm` | Time-Aware LSTM lineage | Packaged time-aware LSTM row routed through `BasicHandler` |
| `plstm` | Phased LSTM lineage | Packaged phased LSTM row routed through `BasicHandler` |
| `tglstm` | Time-Gated LSTM lineage | Packaged time-gated LSTM row routed through `BasicHandler` |

These baselines do not invoke any ODE / CDE / SDE solver and do not consume interpolation coefficients. They consume the `(values, mask, time-deltas)` channel stack as a regular tensor, and the packaged runtime replaces `NaN` values with `0.0` before the backbone forward pass.

**Blind-spot note.** `cnn`, `rnn`, `lstm`, `gru`, and `bilstm` are intentionally naive baselines in this package. The mask channel is available in the input tensor but is not consumed by these backbones at forward time. After `nan_to_num(..., nan=0.0)`, the model cannot distinguish a true zero or normalized-mean value from a missing position. In the bundled scripts, the sequence values are z-scored with training-only statistics before this step, so the fallback is mathematically equivalent to per-channel mean imputation in raw value space. Outside that normalized workflow it is literal zero-fill. When missingness awareness matters within the public active roster, prefer `gru-simple`, `tlstm`, `plstm`, `tglstm`, `gru-d`, `gru-dt`, and the ODE / CDE / SDE families.

---

## 2. Attention Family

| Model | Origin | Implementation in this package |
|---|---|---|
| `transformer` | Transformer encoder baseline ([*Attention Is All You Need*, 2017](https://arxiv.org/abs/1706.03762)) | Packaged sequence-attention bridge routed through `AttentionHandler` with a fixed classifier head for irregular time-series classification |
| `transformer-mask` | Packaged attention variant built on the same Transformer encoder backbone as `transformer` | Zero-fill sequence-attention variant that additionally applies `src_key_padding_mask` from the packaged observation mask before the classifier head |
| `miam` | Missing Information Attention Model lineage | Packaged time-aware attention row routed through `AttentionHandler`; consumes values, original sequence stack, and normalized sequence timestamps |
| `mtan` | Multi-Time Attention Network lineage | Packaged multi-time attention row routed through `AttentionHandler`; consumes values, original sequence stack, and normalized sequence timestamps |
| `sand` | SAnD-style self-attention lineage | Packaged encoder-style attention row routed through `AttentionHandler`; consumes zero-filled sequence values directly |

Version `0.1.5` includes five attention-backed rows. They use the direct sequence stack and do not consume interpolation coefficients in the packaged runtime.

**Blind-spot note.** The packaged `transformer` follows the same naive missing-data treatment as the sequence baselines: it zero-fills `NaN`, ignores interpolation coefficients, and does not apply `src_key_padding_mask` in the current runtime. Use it as a direct attention baseline rather than a missingness-aware attention model.

**Masked variant note.** `transformer-mask` keeps the same backbone and zero-fill preprocessing, but it applies `src_key_padding_mask` using the packaged observation mask. Treat it as a package-specific mitigation variant, not as a separate external paper-backed method.

**SAnD mask caveat.** `sand` follows the direct zero-filled attention path and does not apply `src_key_padding_mask` in version `0.1.5`. This is an explicit package caveat, not a parity-verified missingness-aware attention claim.

---

## 3. ODE runtime family

This section mixes two ODE-family slices that are both shipped in the public package.

- `gru-d`, `gru-dt`, `gru-ode`, and `ode-rnn-spline` are packaged implementations built on the [NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) baseline code introduced in [*Neural Controlled Differential Equations for Irregular Time Series* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926). Where continuous-time integration is used, their solver behavior follows that code path rather than the original repo-backed implementations from the broader method families.
- `ode-lstm` is the canonical ODE-LSTM lineage from its original paper and reference repository. It appears in the same runtime family because it is part of the packaged ODE surface, not because it shares the NeuralCDE implementation path.

| Model | Package row | Origin paper | Canonical reference repo | Implementation note |
|---|---|---|---|---|
| `gru-d` | re-implemented | [*Recurrent Neural Networks for Multivariate Time Series with Missing Values* (Scientific Reports 2018)](https://arxiv.org/abs/1606.01865) | [Han-JD/GRU-D](https://github.com/Han-JD/GRU-D) | NeuralCDE-baseline GRU-D variant (missing-data-aware). Discrete-step; no ODE solver runs at inference time. |
| `gru-dt` | packaged implementation | — (delta-time variant of GRU-D, maintained alongside `gru-d` in the NeuralCDE baseline code) | — | Discrete-step GRU-D with explicit time-gap features. |
| `gru-ode` | packaged implementation | [*Neural Controlled Differential Equations* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) (NeuralCDE-style ODE wrapper) | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | One packaged GRU-ODE implementation built on the NeuralCDE infrastructure with a GRU-driven ODE field. `gru-ode-bayes` in [§4](#4-additional-ode-models) is a sibling implementation from the same broader method family. |
| `ode-rnn-spline` | packaged implementation | [*Neural Controlled Differential Equations* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) (NeuralCDE-style ODE-RNN with spline control path) | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | One packaged ODE-RNN implementation built on the NeuralCDE infrastructure with a natural-cubic-spline control path. `ode-rnn` in [§4](#4-additional-ode-models) is a sibling implementation from the same broader method family. |
| `ode-lstm` | canonical | [*Learning Long-Term Dependencies in Irregularly-Sampled Time Series* (NeurIPS 2020)](https://arxiv.org/abs/2006.04418) | [mlech26l/ode-lstms](https://github.com/mlech26l/ode-lstms) | LSTM cell with continuous-time hidden state evolved by an ODE between observations. |

### ODE family naming notes

- `gru-ode` / `gru-ode-bayes` and `ode-rnn-spline` / `ode-rnn` should be read as different implementations within the same broader method families.
- The rows in this section keep the packaged NeuralCDE-style implementations.
- The repo-backed reference implementations are listed separately in [§4](#4-additional-ode-models).

---

## 4. Additional ODE models

Canonical implementations from the original papers / repositories.

| Model | Origin paper | Canonical reference repo |
|---|---|---|
| `gru-ode-bayes` | [*GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series* (NeurIPS 2019)](https://arxiv.org/abs/1905.12374) | [edebrouwer/gru_ode_bayes](https://github.com/edebrouwer/gru_ode_bayes) |
| `ode-rnn` | [*Latent ODEs for Irregularly-Sampled Time Series* (NeurIPS 2019)](https://arxiv.org/abs/1907.03907) | [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode) |
| `latentode` | [*Latent ODEs for Irregularly-Sampled Time Series* (NeurIPS 2019)](https://arxiv.org/abs/1907.03907) | [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode) |

`ode-rnn` and `latentode` come from the same NeurIPS 2019 paper but are **different models**: `ode-rnn` is the RNN cell with ODE between observations; `latentode` is the encoder–decoder Latent ODE that uses `ode-rnn` as its encoder.

---

## 5. Neural CDE

| Model | Interpolation | Underlying library | Origin |
|---|---|---|---|
| `neuralcde` | natural cubic spline | [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE) (NeuralCDE reference path) | [*Neural Controlled Differential Equations for Irregular Time Series* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) |
| `neuralcde-l` | linear | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper as above; variant uses the maintained follow-up library |
| `neuralcde-r` | rectilinear | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper |
| `neuralcde-c` | cubic | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper |
| `neuralcde-h` | hermite | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper |

`neuralcde` follows the original `controldiffeq` reference path; the variants `neuralcde-l/-r/-c/-h` are built on `torchcde`, the maintained follow-up library to the original NeuralCDE codebase. The `-l/-r/-c/-h` suffix selects the interpolation scheme used to build the control path.

---

## 6. SDE baselines

| Model | Origin | Implementation in this package |
|---|---|---|
| `naive_cde` | Baseline lineage (this package) | A simple control-path baseline for SDE-handler comparisons. No external paper. |
| `neural_ode` | [*Neural Ordinary Differential Equations* (NeurIPS 2018)](https://arxiv.org/abs/1806.07366) | Routed through the SDE handler with a deterministic-drift configuration. Backed by [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) lineage. |
| `neural_ode_t` | Baseline lineage (this package) | Time-conditioned variant of `neural_ode`. No external paper. |
| `neural_sde` | Baseline lineage (this package) | Generic Neural SDE baseline driven by [`torchsde`](https://github.com/google-research/torchsde). No external paper. |
| `neural_sde_t` | Baseline lineage (this package) | Time-conditioned variant of `neural_sde`. No external paper. |

All five share the SDE handler's forward path (the `Diffusion_model` stack with cubic-spline interpolation built via `torchcde`). The four `neural_*` entries are baseline-lineage models — they exist in this package as reference architectures and are not bound to any single paper claim.

For continuous-time rows in this package, prefer `preprocess(X, model_name=...)` so interpolation mode and channel count match the selected model by default. Sequence baselines, `transformer`, `transformer-mask`, `ode-lstm`, `ode-rnn`, `latentode`, and `gru-ode-bayes` can use the direct sequence stack with `coeffs=None`.

---

## 7. Stable Neural SDE

The Stable Neural SDE family comes from a single paper:

> *Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data* (ICLR 2024) — Oh, Lim, and Kim ([arXiv:2402.14989](https://arxiv.org/abs/2402.14989) · [OpenReview](https://openreview.net/forum?id=4VIgNuQ1pY) · repo: [yongkyung-oh/Stable-Neural-SDEs](https://github.com/yongkyung-oh/Stable-Neural-SDEs))

| Model | Drift–diffusion variant |
|---|---|
| `neural_lsde` | Additive (Langevin-type) noise |
| `neural_lnsde` | Multiplicative noise |
| `neural_gsde` | Geometric multiplicative noise |

`neural_gsde` is a variant within the same paper, not a separate work. All three default to the `srk` solver (matching the upstream reference implementation).

---

## 8. NeuralFlow

Version `0.1.5` adds NeuralFlow as an independent flow-method row on the packaged native backend line:

| Model group | Models | Runtime note |
|---|---|---|
| NeuralFlow comparator | `neuralflow-r`, `neuralflow-g`, `neuralflow-c` | Flow-only row using the direct value/time path. No ODE/CDE solver surface is exposed. |

The suffix selects the native flow operator: `-r` = residual flow, `-g` = GRU-style flow, and `-c` = coupling-style flow. The default public runtime uses the repo-native backend and does not require `stribor`. Guarded NFE/stribor compatibility is not part of the default `0.1.5` public profile.

Compatibility aliases are accepted as input names and normalized to canonical package names:

| Alias family | Canonical package name |
|---|---|
| `neuralflow_x_{r,g,c}` | `neuralflow-{r,g,c}` |

For missing-data handling, `neuralflow-*` keeps the raw missing-value contract on the direct value path.

---

## 9. FlowPath and DualDynamics

FlowPath and DualDynamics are advanced Neural CDE/control-path methodology rows implemented through the packaged `FlowHandler` and `module.flow` surface in version `0.1.5`:

| Model group | Models | Runtime note |
|---|---|---|
| FlowPath | `flowpath-r`, `flowpath-g`, `flowpath-c` | Flow plus CDE control path; solver-backed rows default to `rk4` with fixed-grid step size from the input times. |
| DualDynamics | `dualdynamics-r`, `dualdynamics-g`, `dualdynamics-c` | DualDynamics lineage with coeff-driven control path; solver-backed rows default to `rk4` with fixed-grid step size from the input times. |

The suffix selects the native flow operator: `-r` = residual flow, `-g` = GRU-style flow, and `-c` = coupling-style flow. These rows use the repo-native backend and do not expose `ancde`, `exit`, or `leap` as part of the current public payload.

Compatibility aliases are accepted as input names and normalized to canonical package names:

| Alias family | Canonical package name |
|---|---|
| `neuralflowcde_x_{r,g,c}`, `flowpath_x_{r,g,c}` | `flowpath-{r,g,c}` |
| `dualdynamics_x_{r,g,c}`, `neuralcontrolledflow_x_{r,g,c}` | `dualdynamics-{r,g,c}` |

For missing-data handling, `flowpath-*` keeps the raw missing-value contract on the direct value path. `dualdynamics-*` is coeff-centric and should be used with preprocessing coefficients.

---

## 10. How to cite

If you use the **Stable Neural SDE** family (`neural_lsde`, `neural_lnsde`, `neural_gsde`) in academic work, please cite:

```bibtex
@inproceedings{oh_stable_2024,
    title     = {Stable {Neural} {Stochastic} {Differential} {Equations}
                 in {Analyzing} {Irregular} {Time} {Series} {Data}},
    author    = {Oh, YongKyung and Lim, Dong-Young and Kim, Sungil},
    year      = 2024,
    month     = may,
    booktitle = {The {Twelfth} {International} {Conference} on
                 {Learning} {Representations}, {ICLR} 2024,
                 {Vienna}, {Austria}, {May} 7-11, 2024},
    publisher = {OpenReview.net},
    url       = {https://arxiv.org/abs/2402.14989}
}
```

For models with an explicit origin paper listed above, cite that paper. For package-specific baseline or variant rows without a single external paper, cite this package and name the exact packaged model you used. Two additional references that contextualize this package's overall design are listed in the [main README](../README.md#citation):

- **DualDynamics (AAAI 2025)** — [arXiv:2401.04979](https://arxiv.org/abs/2401.04979)
- **Neural Differential Equations Survey (IJCAI 2025)** — [arXiv:2502.09885](https://arxiv.org/abs/2502.09885)

---

**End of Model Reference**
