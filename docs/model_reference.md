# torch-ists Model Reference

> **Scope**: Origin papers, reference repositories, and implementation notes for packaged models.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Scripts Reference](scripts_reference.md) · [Data Reference](data_reference.md) · [Custom Model Reference](custom_model_reference.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

Use the API reference for class signatures and call patterns; use this document for model origins and reference implementations.

---

## Table of Contents

1. [Sequence baselines](#1-sequence-baselines)
2. [Attention-based methods](#2-attention-based-methods)
3. [Neural ODE methods](#3-neural-ode-methods)
4. [ODE-RNN, Latent ODE, and GRU-ODE-Bayes](#4-ode-rnn-latent-ode-and-gru-ode-bayes)
5. [Neural CDE](#5-neural-cde)
6. [ANCDE, EXIT, and LEAP](#6-ancde-exit-and-leap)
7. [NeuralFlow](#7-neuralflow)
8. [DualDynamics and FlowPath](#8-dualdynamics-and-flowpath)
9. [SDE baselines](#9-sde-baselines)
10. [Stable Neural SDEs](#10-stable-neural-sdes)

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
| `plstm` | [Neil et al., *Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences* (NeurIPS 2016)](https://arxiv.org/abs/1610.09513) | Packaged phased LSTM row |
| `tlstm` | [Baytas et al., *Patient Subtyping via Time-Aware LSTM Networks* (KDD 2017)](https://dl.acm.org/doi/10.1145/3097983.3097997) | Packaged time-aware LSTM row |
| `tglstm` | [Sahin and Kozat, *Nonuniformly Sampled Data Processing Using LSTM Networks* (IEEE TNNLS 2019)](https://doi.org/10.1109/TNNLS.2018.2869822) | Packaged time-gated LSTM row |

These baselines do not invoke ODE / CDE / SDE solvers or consume interpolation coefficients. They use the sequence stack directly and replace `NaN` values with `0.0` before the backbone forward pass (the default `imputation_strategy="zero"`; `last_observed` and `channel_mean` are experimental opt-in alternatives — see the API reference).

**Limitation.** `cnn`, `rnn`, `lstm`, `gru`, and `bilstm` do not use the mask channel. After zero-fill, they cannot distinguish a true zero from a missing value. When missingness awareness matters, prefer rows that use mask, delta, or interpolation coefficients.

---

## 2. Attention-based methods

| Model | Origin | Implementation in this package |
|---|---|---|
| `transformer` | Transformer encoder baseline ([*Attention Is All You Need*, 2017](https://arxiv.org/abs/1706.03762)) | Packaged sequence-attention row with a fixed classifier head |
| `transformer-mask` | Packaged attention variant built on the same Transformer encoder backbone as `transformer` | Zero-fill sequence-attention variant that additionally applies `src_key_padding_mask` from the packaged observation mask before the classifier head |
| `sand` | [Song et al., *Attend and Diagnose: Clinical Time Series Analysis using Attention Models* (AAAI 2018)](https://arxiv.org/abs/1711.03905) | Packaged encoder-style attention row; consumes zero-filled sequence values directly |
| `mtan` | [Shukla and Marlin, *Multi-Time Attention Networks for Irregularly Sampled Time Series* (ICLR 2021)](https://arxiv.org/abs/2101.10318) | Packaged multi-time attention row; consumes values, sequence stack, and normalized timestamps |
| `miam` | [Lee et al., *Multi-View Integrative Attention-Based Deep Representation Learning for Irregular Clinical Time-Series Data* (IEEE J-BHI 2022)](https://arxiv.org/abs/2101.09986) | Packaged time-aware attention row; consumes values, sequence stack, and normalized timestamps |

These rows use the direct sequence stack and do not consume interpolation coefficients.

**Mask notes.** `transformer` and `sand` zero-fill missing values and do not apply a padding mask. `transformer-mask` keeps the same Transformer backbone but applies `src_key_padding_mask`.


---

## 3. Neural ODE methods

This section lists ODE-style recurrent methods. The GRU-D/GRU-ODE/ODE-RNN-spline rows follow the NeuralCDE baseline code path; `ode-lstm` follows the ODE-LSTM lineage.

| Model | Package row | Origin paper | Canonical reference repo | Implementation note |
|---|---|---|---|---|
| `gru-d` | re-implemented | [Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values* (Scientific Reports 2018)](https://arxiv.org/abs/1606.01865) | [Han-JD/GRU-D](https://github.com/Han-JD/GRU-D) | NeuralCDE-baseline GRU-D variant (missing-data-aware). Discrete-step; no ODE solver runs at inference time. |
| `gru-dt` | packaged implementation | — (delta-time variant of GRU-D, maintained alongside `gru-d` in the NeuralCDE baseline code) | — | Discrete-step GRU-D with explicit time-gap features. |
| `gru-ode` | packaged implementation | [Kidger et al., *Neural Controlled Differential Equations* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) (NeuralCDE-style ODE wrapper) | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | One packaged GRU-ODE implementation built on the NeuralCDE infrastructure with a GRU-driven ODE field. `gru-ode-bayes` in [§4](#4-ode-rnn-latent-ode-and-gru-ode-bayes) is a sibling implementation from the same broader method. |
| `ode-rnn-spline` | packaged implementation | [Kidger et al., *Neural Controlled Differential Equations* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) (NeuralCDE-style ODE-RNN with spline control path) | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | One packaged ODE-RNN implementation built on the NeuralCDE infrastructure with a natural-cubic-spline control path. `ode-rnn` in [§4](#4-ode-rnn-latent-ode-and-gru-ode-bayes) is a sibling implementation from the same broader method. |
| `ode-lstm` | canonical | [Lechner and Hasani, *Learning Long-Term Dependencies in Irregularly-Sampled Time Series* (NeurIPS 2020)](https://arxiv.org/abs/2006.04418) | [mlech26l/ode-lstms](https://github.com/mlech26l/ode-lstms) | LSTM cell with continuous-time hidden state evolved by an ODE between observations. |

### ODE naming notes

`gru-ode` / `gru-ode-bayes` and `ode-rnn-spline` / `ode-rnn` are different packaged implementations of related method lines. The canonical ODE-RNN, Latent ODE, and GRU-ODE-Bayes rows are listed in [§4](#4-ode-rnn-latent-ode-and-gru-ode-bayes).

---

## 4. ODE-RNN, Latent ODE, and GRU-ODE-Bayes

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

ANCDE, EXIT, LEAP, DualDynamics, and FlowPath also use CDE-style control paths and consume interpolation coefficients.

---

## 6. ANCDE, EXIT, and LEAP

`ancde`, `exit`, and `leap` are Neural CDE methods from the controlled differential equation literature. Like Section 5 Neural CDE, they consume interpolation coefficients and use CDE control paths. The default solver is `rk4` with a fixed-grid step size derived from the input times.

| Model | Paper | Behavior |
|---|---|---|
| `ancde` | Jhin et al., *Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting* (ICDM 2021) | Attention-gated control path (ANCDE). Returns `(out, hn)`. No auxiliary loss. |
| `exit` | Jhin et al., *EXIT: Extrapolation and Interpolation-based Neural Controlled Differential Equations for Time-series Classification and Forecasting* | CDE with kinetic-energy regularization. Returns `(out, hn, reg)`, where `reg` is the auxiliary kinetic-energy regularization term. |
| `leap` | *LEAP: Learnable Path CDE* | Learnable-path CDE. Returns `(out, hn, loss)`, where `loss` is an auxiliary loss term. |

**LEAP time scale.** `leap` uses normalized `[0, 1]` control-path time, so it may differ from experiments that used an integer-grid path.

**EXIT/LEAP auxiliary-output note.** `exit` and `leap` return auxiliary outputs in addition to `(out, hn)`. When using `ists_classifier` or `ists_model` with `return_aux_loss=True`, the auxiliary term is returned alongside the prediction. `ancde` does not return an auxiliary output.

**Solver and dependencies.** All three use `rk4` with a fixed-grid step size from input times. Install `.[cde]` or `.[all]`.

**Preprocessing.** Use `preprocess(X, model_name=...)` with `ancde`, `exit`, or `leap` to obtain matching natural-cubic-spline coefficients (`interpolation='natural'`).

---

## 7. NeuralFlow

NeuralFlow is an independent flow method:

| Model group | Models | Runtime note |
|---|---|---|
| NeuralFlow comparator | default name `neuralflow`; explicit variants `neuralflow-r`, `neuralflow-g`, `neuralflow-c` | Flow-only row using the direct value/time path. |

Origin paper: [*Neural Flows: Efficient Alternative to Neural ODEs* (NeurIPS 2021)](https://arxiv.org/abs/2110.13040).

Use `neuralflow` for the residual-flow default. The suffix selects an explicit flow variant: `-r` = residual flow, `-g` = GRU-style flow, and `-c` = coupling-style flow. The package implementation does not require `stribor`.

Accepted model names:

| Public name | Runtime selection |
|---|---|
| `neuralflow`, `neuralflow-r` | residual-flow variant |
| `neuralflow-g` | GRU-style flow variant |
| `neuralflow-c` | coupling-style flow variant |

For missing-data handling, `neuralflow` and its explicit variants keep raw missing values on the direct value path.

---

## 8. DualDynamics and FlowPath

DualDynamics and FlowPath are flow-based Neural CDE / control-path methods. They consume interpolation coefficients, use CDE-style control paths, and include flow components; they are not flow-only comparators.

| Model group | Models | Runtime note |
|---|---|---|
| DualDynamics | default name `dualdynamics`; explicit variants `dualdynamics-r`, `dualdynamics-g`, `dualdynamics-c` | DualDynamics lineage with coefficient-driven control paths; defaults to `rk4` with fixed-grid step size from the input times. |
| FlowPath | default name `flowpath`; explicit variants `flowpath-r`, `flowpath-g`, `flowpath-c` | Flow plus CDE control path; defaults to `rk4` with fixed-grid step size from the input times. |

Origin papers:
- DualDynamics: [*DualDynamics: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis* (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34173) · [arXiv:2401.04979](https://arxiv.org/abs/2401.04979)
- FlowPath: [*FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification* (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39643) · [arXiv:2511.10841](https://arxiv.org/abs/2511.10841)

Use `dualdynamics` and `flowpath` for residual-flow defaults. The suffix selects an explicit flow variant: `-r` = residual flow, `-g` = GRU-style flow, and `-c` = coupling-style flow. These remain CDE/control-path methods.

Accepted model names:

| Public name | Runtime selection |
|---|---|
| `dualdynamics`, `dualdynamics-r` | residual-flow dual-dynamics variant |
| `dualdynamics-g` | GRU-style flow dual-dynamics variant |
| `dualdynamics-c` | coupling-style flow dual-dynamics variant |
| `flowpath`, `flowpath-r` | residual-flow control-path variant |
| `flowpath-g` | GRU-style flow control-path variant |
| `flowpath-c` | coupling-style flow control-path variant |

For missing-data handling, use `preprocess(X, model_name=...)` to build matching interpolation coefficients.

---

## 9. SDE baselines

| Model | Origin | Implementation in this package |
|---|---|---|
| `naive_cde` | Baseline lineage (this package) | A simple control-path baseline for reference and SDE comparisons. No external paper. |
| `neural_ode` | [*Neural Ordinary Differential Equations* (NeurIPS 2018)](https://arxiv.org/abs/1806.07366) | Deterministic-drift baseline backed by [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) lineage. |
| `neural_ode_t` | Baseline lineage (this package) | Time-conditioned variant of `neural_ode`. No external paper. |
| `neural_sde` | Baseline lineage (this package) | Generic Neural SDE baseline driven by [`torchsde`](https://github.com/google-research/torchsde). No external paper. |
| `neural_sde_t` | Baseline lineage (this package) | Time-conditioned variant of `neural_sde`. No external paper. |

All five use cubic-spline interpolation via `torchcde`. The four `neural_*` entries are baseline-lineage reference architectures, not single-paper claims.

For continuous-time methods in this package, prefer `preprocess(X, model_name=...)` so interpolation mode and channel count match the selected model by default. Sequence baselines, `transformer`, `transformer-mask`, `ode-lstm`, `ode-rnn`, `latentode`, and `gru-ode-bayes` can use the direct sequence stack with `coeffs=None`.

---

## 10. Stable Neural SDEs

The section title uses the plural form from the ICLR 2024 paper title. The packaged names map to the three model classes proposed in the paper:

> *Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data* (ICLR 2024) — Oh, Lim, and Kim ([arXiv:2402.14989](https://arxiv.org/abs/2402.14989) · [OpenReview](https://openreview.net/forum?id=4VIgNuQ1pY) · repo: [yongkyung-oh/Stable-Neural-SDEs](https://github.com/yongkyung-oh/Stable-Neural-SDEs))

| Package model | Full model name in the paper | Drift–diffusion variant |
|---|---|---|
| `neural_lsde` | Langevin-type SDE (LSDE) | Additive / Langevin-type noise |
| `neural_lnsde` | Linear Noise SDE (LNSDE) | Linear multiplicative noise |
| `neural_gsde` | Geometric SDE (GSDE) | Geometric multiplicative noise |

All three default to the `srk` solver.

---

**End of Model Reference**
