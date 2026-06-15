# torch-ists Model Reference

> **Package Version**: 0.1.7
> **Scope**: Origin paper, canonical repository, and implementation note for every model shipped in `0.1.7`.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Scripts Reference](scripts_reference.md) · [New Data Onboarding](new_data_onboarding.md) · [New Model Onboarding](new_model_onboarding.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

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
9. [DualDynamics and FlowPath](#9-dualdynamics-and-flowpath)
10. [Advanced CDE](#10-advanced-cde)

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
| `plstm` | [Neil et al., *Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences* (NeurIPS 2016)](https://arxiv.org/abs/1610.09513) | Packaged phased LSTM row routed through `BasicHandler` |
| `tlstm` | [Baytas et al., *Patient Subtyping via Time-Aware LSTM Networks* (KDD 2017)](https://dl.acm.org/doi/10.1145/3097983.3097997) | Packaged time-aware LSTM row routed through `BasicHandler` |
| `tglstm` | [Sahin and Kozat, *Nonuniformly Sampled Data Processing Using LSTM Networks* (IEEE TNNLS 2019)](https://doi.org/10.1109/TNNLS.2018.2869822) | Packaged time-gated LSTM row routed through `BasicHandler` |

These baselines do not invoke any ODE / CDE / SDE solver and do not consume interpolation coefficients. They consume the `(values, mask, time-deltas)` channel stack as a regular tensor, and the packaged runtime replaces `NaN` values with `0.0` before the backbone forward pass.

**Limitation.** `cnn`, `rnn`, `lstm`, `gru`, and `bilstm` are intentionally naive baselines in this package. The mask channel is available in the input tensor but is not consumed by these backbones at forward time. After `nan_to_num(..., nan=0.0)`, the model cannot distinguish a true zero or normalized-mean value from a missing position. In the source-checkout scripts, the sequence values are z-scored with training-only statistics before this step, so the fallback is mathematically equivalent to per-channel mean imputation in raw value space. Outside that normalized workflow it is literal zero-fill. When missingness awareness matters, prefer `gru-simple`, `plstm`, `tlstm`, `tglstm`, `gru-d`, `gru-dt`, and the ODE / CDE / SDE families.

---

## 2. Attention Family

| Model | Origin | Implementation in this package |
|---|---|---|
| `transformer` | Transformer encoder baseline ([*Attention Is All You Need*, 2017](https://arxiv.org/abs/1706.03762)) | Packaged sequence-attention bridge routed through `AttentionHandler` with a fixed classifier head for irregular time-series classification |
| `transformer-mask` | Packaged attention variant built on the same Transformer encoder backbone as `transformer` | Zero-fill sequence-attention variant that additionally applies `src_key_padding_mask` from the packaged observation mask before the classifier head |
| `sand` | [Song et al., *Attend and Diagnose: Clinical Time Series Analysis using Attention Models* (AAAI 2018)](https://arxiv.org/abs/1711.03905) | Packaged encoder-style attention row routed through `AttentionHandler`; consumes zero-filled sequence values directly |
| `mtan` | [Shukla and Marlin, *Multi-Time Attention Networks for Irregularly Sampled Time Series* (ICLR 2021)](https://arxiv.org/abs/2101.10318) | Packaged multi-time attention row routed through `AttentionHandler`; consumes values, original sequence stack, and normalized sequence timestamps |
| `miam` | [Lee et al., *Multi-View Integrative Attention-Based Deep Representation Learning for Irregular Clinical Time-Series Data* (IEEE J-BHI 2022)](https://arxiv.org/abs/2101.09986) | Packaged time-aware attention row routed through `AttentionHandler`; consumes values, original sequence stack, and normalized sequence timestamps |

Version `0.1.7` includes five attention-backed rows. They use the direct sequence stack and do not consume interpolation coefficients in the packaged runtime.

**Limitation.** `transformer` follows the same naive missing-data treatment as the sequence baselines: it zero-fills `NaN`, ignores interpolation coefficients, and does not apply `src_key_padding_mask` in the current runtime. Use it as a direct attention baseline rather than a missingness-aware attention model.

**Masked variant note.** `transformer-mask` keeps the same backbone and zero-fill preprocessing, but it applies `src_key_padding_mask` using the packaged observation mask. Treat it as a package-specific mitigation variant, not as a separate external paper-backed method.

**SAnD mask note.** `sand` follows the direct zero-filled attention path and does not apply `src_key_padding_mask` in version `0.1.7`. Treat it as a basic attention baseline rather than a missingness-aware attention variant in this version.

---

## 3. ODE runtime family

This section mixes two ODE-family slices that are both shipped in the package.

- `gru-d`, `gru-dt`, `gru-ode`, and `ode-rnn-spline` are packaged implementations built on the [NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) baseline code introduced in [Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926). Where continuous-time integration is used, their solver behavior follows that code path rather than the original repo-backed implementations from the broader method families.
- `ode-lstm` is the canonical ODE-LSTM lineage from its original paper and reference repository. It appears in the same runtime family because it is part of the packaged ODE surface, not because it shares the NeuralCDE implementation path.

| Model | Package row | Origin paper | Canonical reference repo | Implementation note |
|---|---|---|---|---|
| `gru-d` | re-implemented | [Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values* (Scientific Reports 2018)](https://arxiv.org/abs/1606.01865) | [Han-JD/GRU-D](https://github.com/Han-JD/GRU-D) | NeuralCDE-baseline GRU-D variant (missing-data-aware). Discrete-step; no ODE solver runs at inference time. |
| `gru-dt` | packaged implementation | — (delta-time variant of GRU-D, maintained alongside `gru-d` in the NeuralCDE baseline code) | — | Discrete-step GRU-D with explicit time-gap features. |
| `gru-ode` | packaged implementation | [Kidger et al., *Neural Controlled Differential Equations* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) (NeuralCDE-style ODE wrapper) | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | One packaged GRU-ODE implementation built on the NeuralCDE infrastructure with a GRU-driven ODE field. `gru-ode-bayes` in [§4](#4-additional-ode-models) is a sibling implementation from the same broader method family. |
| `ode-rnn-spline` | packaged implementation | [Kidger et al., *Neural Controlled Differential Equations* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) (NeuralCDE-style ODE-RNN with spline control path) | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | One packaged ODE-RNN implementation built on the NeuralCDE infrastructure with a natural-cubic-spline control path. `ode-rnn` in [§4](#4-additional-ode-models) is a sibling implementation from the same broader method family. |
| `ode-lstm` | canonical | [Lechner and Hasani, *Learning Long-Term Dependencies in Irregularly-Sampled Time Series* (NeurIPS 2020)](https://arxiv.org/abs/2006.04418) | [mlech26l/ode-lstms](https://github.com/mlech26l/ode-lstms) | LSTM cell with continuous-time hidden state evolved by an ODE between observations. |

### ODE family naming notes

- `gru-ode` / `gru-ode-bayes` and `ode-rnn-spline` / `ode-rnn` should be read as different implementations within the same broader method families.
- The rows in this section keep the packaged NeuralCDE-style implementations.
- The repo-backed reference implementations are listed separately in [§4](#4-additional-ode-models).

---

## 4. Additional ODE models

Canonical implementations from the original papers / repositories.

| Model | Origin paper | Canonical reference repo |
|---|---|---|
| `gru-ode-bayes` | [De Brouwer et al., *GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series* (NeurIPS 2019)](https://arxiv.org/abs/1905.12374) | [edebrouwer/gru_ode_bayes](https://github.com/edebrouwer/gru_ode_bayes) |
| `ode-rnn` | [Rubanova et al., *Latent ODEs for Irregularly-Sampled Time Series* (NeurIPS 2019)](https://arxiv.org/abs/1907.03907) | [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode) |
| `latentode` | [Rubanova et al., *Latent ODEs for Irregularly-Sampled Time Series* (NeurIPS 2019)](https://arxiv.org/abs/1907.03907) | [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode) |

`ode-rnn` and `latentode` come from the same NeurIPS 2019 paper but are **different models**: `ode-rnn` is the RNN cell with ODE between observations; `latentode` is the encoder–decoder Latent ODE that uses `ode-rnn` as its encoder.

---

## 5. Neural CDE

| Model | Interpolation | Underlying library | Origin |
|---|---|---|---|
| `neuralcde` | natural cubic spline | [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE) (NeuralCDE reference path) | [Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) |
| `neuralcde-l` | linear | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper as above; variant uses the maintained follow-up library |
| `neuralcde-r` | rectilinear | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper |
| `neuralcde-c` | cubic | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper |
| `neuralcde-h` | hermite | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper |

`neuralcde` follows the original `controldiffeq` reference path; the variants `neuralcde-l/-r/-c/-h` are built on `torchcde`, the maintained follow-up library to the original NeuralCDE codebase. The `-l/-r/-c/-h` suffix selects the interpolation scheme used to build the control path.

---

## 6. SDE baselines

| Model | Origin | Implementation in this package |
|---|---|---|
| `naive_cde` | Baseline lineage (this package) | A simple control-path baseline for reference and SDE-family comparisons. No external paper. |
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

Version `0.1.7` retains NeuralFlow as an independent flow-method row on the packaged native backend line:

| Model group | Models | Runtime note |
|---|---|---|
| NeuralFlow comparator | official name `neuralflow`; explicit variants `neuralflow-r`, `neuralflow-g`, `neuralflow-c` | Flow-only row using the direct value/time path. No ODE/CDE solver surface is exposed. |

Origin paper: [*Neural Flows: Efficient Alternative to Neural ODEs* (NeurIPS 2021)](https://arxiv.org/abs/2110.13040).

The official public model name is `neuralflow`, which uses the residual-flow variant by default. The suffix selects the native flow operator when an explicit variant is needed: `-r` = residual flow, `-g` = GRU-style flow, and `-c` = coupling-style flow. The default runtime uses the repo-native backend and does not require `stribor`. Guarded NFE/stribor compatibility is not included in the default `0.1.7` runtime profile.

Accepted model names for this family:

| Public name | Runtime selection |
|---|---|
| `neuralflow`, `neuralflow-r` | residual-flow variant |
| `neuralflow-g` | GRU-style flow variant |
| `neuralflow-c` | coupling-style flow variant |

For missing-data handling, `neuralflow` and its explicit variants keep raw missing values on the direct value path.

---

## 9. DualDynamics and FlowPath

DualDynamics and FlowPath are advanced Neural CDE/control-path methodology rows implemented through the packaged `FlowHandler` in version `0.1.7`, listed by publication year (DualDynamics: AAAI 2025; FlowPath: AAAI 2026). They share the packaged native flow-operator runtime with NeuralFlow, but they are not flow-only comparators: they consume interpolation coefficients and use CDE-style control paths.

| Model group | Models | Runtime note |
|---|---|---|
| DualDynamics | official name `dualdynamics`; explicit variants `dualdynamics-r`, `dualdynamics-g`, `dualdynamics-c` | DualDynamics lineage with coeff-driven control path; solver-backed rows default to `rk4` with fixed-grid step size from the input times. |
| FlowPath | official name `flowpath`; explicit variants `flowpath-r`, `flowpath-g`, `flowpath-c` | Flow plus CDE control path; solver-backed rows default to `rk4` with fixed-grid step size from the input times. |

Origin papers:
- DualDynamics: [*DualDynamics: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis* (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34173) · [arXiv:2401.04979](https://arxiv.org/abs/2401.04979)
- FlowPath: [*FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification* (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39643) · [arXiv:2511.10841](https://arxiv.org/abs/2511.10841)

The official model names `dualdynamics` and `flowpath` use residual-flow variants by default. The suffix selects the native flow operator when an explicit variant is needed: `-r` = residual flow, `-g` = GRU-style flow, and `-c` = coupling-style flow. Their method taxonomy remains CDE/control-path; the flow suffix only selects the packaged native flow operator. These rows use the repo-native backend; `ancde`, `exit`, and `leap` are included in version `0.1.7` as advanced CDE control-path rows (see the Advanced CDE section).

Accepted model names for these families:

| Public name | Runtime selection |
|---|---|
| `dualdynamics`, `dualdynamics-r` | residual-flow dual-dynamics variant |
| `dualdynamics-g` | GRU-style flow dual-dynamics variant |
| `dualdynamics-c` | coupling-style flow dual-dynamics variant |
| `flowpath`, `flowpath-r` | residual-flow control-path variant |
| `flowpath-g` | GRU-style flow control-path variant |
| `flowpath-c` | coupling-style flow control-path variant |

For missing-data handling, `dualdynamics`, `flowpath`, and their explicit variants are coeff-centric CDE/control-path models and should be used with `preprocess(X, model_name=...)` to build matching interpolation coefficients.

---

## 10. Advanced CDE

`ancde`, `exit`, and `leap` are advanced CDE control-path models added in version `0.1.6` and retained in version `0.1.7`. Like the Neural CDE family and the DualDynamics/FlowPath rows, they consume interpolation coefficients and route through the CDE handler, using the control-path control-differential-equation formulation. Each uses its own modified vendored control-path solver and is independent of the pip `torchcde` package and the `neuralcde` base backend. The default solver is `rk4` with a fixed-grid step size derived from the input times.

| Model | Paper | Behavior |
|---|---|---|
| `ancde` | Jhin et al., *Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting* (ICDM 2021) | Attention-gated control path (ANCDE). Returns `(out, hn)`. No auxiliary loss. |
| `exit` | Jhin et al., *EXIT: Extrapolation and Interpolation-based Neural Controlled Differential Equations for Time-series Classification and Forecasting* | CDE with kinetic-energy regularization. Returns `(out, hn, reg)`, where `reg` is the auxiliary kinetic-energy regularization term. |
| `leap` | *LEAP: Learnable Path CDE* | Learnable-path CDE. Returns `(out, hn, loss)`, where `loss` is an auxiliary loss term. |

**LEAP normalized-time caveat.** The packaged `leap` implementation uses a normalized `[0, 1]` control-path time (forward-time), differing from the upstream integer-grid assumption. Results may not directly match experiments that relied on integer-grid control-path time. Use this implementation with awareness of that difference.

**EXIT/LEAP auxiliary-output note.** `exit` and `leap` return auxiliary outputs in addition to `(out, hn)`. When using `ists_classifier` or `ists_model` with `return_aux_loss=True`, the auxiliary term is returned alongside the prediction. `ancde` does not return an auxiliary output.

**Solver note.** All three models use their own modified vendored control-path solver. This solver is independent of the pip `torchcde` package and of the `neuralcde` base backend. The default is `rk4` with a fixed-grid step size from the input times. Install `.[cde]` or `.[all]` for the full dependency set.

**Preprocessing.** Use `preprocess(X, model_name=...)` with `ancde`, `exit`, or `leap` to obtain matching natural-cubic-spline coefficients (`interpolation='natural'`).

---

**End of Model Reference**
