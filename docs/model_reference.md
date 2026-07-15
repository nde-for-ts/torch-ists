# torch-ists Model Reference

> **Type**: Reference
> **Scope**: Origin papers, reference repositories, and implementation notes for packaged models.

**Related docs**: [Package overview](../README.md) · [Docs index](README.md) · [API Reference](api_reference.md) · [Scripts Reference](scripts_reference.md) · [Custom Data Guide](custom_data_guide.md) · [Custom Model Guide](custom_model_guide.md) · [Third-Party Notices](third_party_notices.md) · [CIKM 2025 Tutorial site](https://nde-for-ts.github.io/)

Use the API reference for class signatures and call patterns; use this document for model origins and reference implementations.

---

## Table of Contents

1. [Sequence baselines](#1-sequence-baselines)
2. [Attention-based methods](#2-attention-based-methods)
3. [Neural ODE methods](#3-neural-ode-methods)
4. [ODE-RNN, Latent ODE, GRU-ODE-Bayes, and Neural Jump ODE (NJ-ODE)](#4-ode-rnn-latent-ode-gru-ode-bayes-and-neural-jump-ode-nj-ode)
5. [Neural CDE and Neural RDE](#5-neural-cde-and-neural-rde)
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

These models use the sequence stack directly with `coeffs=None`; `NaN` is replaced by `0.0` at the backbone input.

**PLSTM numerical note.** The packaged `plstm` clamps the phased time-gate denominator factors at `1e-3`. This intentionally differs from the unclamped reference implementation to avoid float32 sub-normal denominator underflow observed in long final-training runs.

**Limitation.** `cnn`, `rnn`, `lstm`, `gru`, and `bilstm` do not use the mask channel. After zero-fill, they cannot distinguish a true zero from a missing value. When missingness awareness matters, prefer rows that use mask, delta, or interpolation coefficients.

---

## 2. Attention-based methods

| Model | Origin | Implementation in this package |
|---|---|---|
| `transformer` | Transformer encoder baseline ([*Attention Is All You Need*, 2017](https://arxiv.org/abs/1706.03762)) | Packaged sequence-attention row |
| `transformer-mask` | Same backbone as `transformer` ([2017](https://arxiv.org/abs/1706.03762)) | Same as `transformer` but applies `src_key_padding_mask` from the observation mask. |
| `sand` | [Song et al., *Attend and Diagnose: Clinical Time Series Analysis using Attention Models* (AAAI 2018)](https://arxiv.org/abs/1711.03905) | Encoder-style attention; consumes zero-filled sequence values directly. |
| `mtan` | [Shukla and Marlin, *Multi-Time Attention Networks for Irregularly Sampled Time Series* (ICLR 2021)](https://arxiv.org/abs/2101.10318) | Variational multi-time attention; each forward samples the latent representation. |
| `miam` | [Lee et al., *Multi-View Integrative Attention-Based Deep Representation Learning for Irregular Clinical Time-Series Data* (IEEE J-BHI 2022)](https://doi.org/10.1109/JBHI.2022.3172549) | Multi-view Integration Attention Model; consumes values, masks, deltas, and normalized timestamps. |

**Mask notes.** `transformer` and `sand` zero-fill missing values and do not apply a padding mask. `transformer-mask` keeps the same Transformer backbone but applies `src_key_padding_mask`; fully masked rows are handled as zero-filled finite fallback rows.

**Inference notes.** mTAN remains stochastic under `eval()` because variational latent sampling is part of the method. Set the random seed for repeatable single-sample inference or aggregate multiple samples. MIAM requires `hidden_dim >= 2` for its feature normalization.

---

## 3. Neural ODE methods

ODE-style recurrent methods. The GRU-D, GRU-ODE, and ODE-RNN-spline rows are built on the NeuralCDE infrastructure; `ode-lstm` is adapted from the ODE-LSTM implementation. The package adaptations of ODE-RNN, Latent ODE, GRU-ODE-Bayes, and Neural Jump ODE (NJ-ODE) are listed in [§4](#4-ode-rnn-latent-ode-gru-ode-bayes-and-neural-jump-ode-nj-ode).

| Model | Package row | Origin paper | Canonical reference repo | Implementation note |
|---|---|---|---|---|
| `gru-d` | re-implemented | [Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values* (Scientific Reports 2018)](https://arxiv.org/abs/1606.01865) | [Han-JD/GRU-D](https://github.com/Han-JD/GRU-D) | NeuralCDE-baseline GRU-D variant (missing-data-aware). Discrete-step; no ODE solver runs at inference time. |
| `gru-dt` | packaged | — (delta-time variant of GRU-D, maintained alongside `gru-d` in the NeuralCDE baseline code) | — | Discrete-step GRU-D with explicit time-gap features. |
| `gru-ode` | packaged | [Kidger et al., *Neural Controlled Differential Equations* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) (NeuralCDE-style ODE wrapper) | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | GRU-driven ODE field on the NeuralCDE infrastructure. `gru-ode-bayes` in [§4](#4-ode-rnn-latent-ode-gru-ode-bayes-and-neural-jump-ode-nj-ode) is a separate package adaptation. |
| `ode-rnn-spline` | packaged | [Kidger et al., *Neural Controlled Differential Equations* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) (NeuralCDE-style ODE-RNN with spline control path) | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | ODE-RNN with a natural-cubic-spline control path on the NeuralCDE infrastructure. `ode-rnn` in [§4](#4-ode-rnn-latent-ode-gru-ode-bayes-and-neural-jump-ode-nj-ode) uses a direct value/time path. |
| `ode-lstm` | adapted | [Lechner and Hasani, *Learning Long-Term Dependencies in Irregularly-Sampled Time Series* (NeurIPS 2020)](https://arxiv.org/abs/2006.04418) | [mlech26l/ode-lstms](https://github.com/mlech26l/ode-lstms) | LSTM cell with continuous-time hidden state evolved by an ODE between observed events. |

**Runtime note.** `ode-lstm` updates only samples with an observed event and holds the other hidden and cell states unchanged. It handles singleton batches with a zero-based adaptive grid and passes elapsed durations between observations; fixed/adaptive solver selection remains model-specific.

`gru-d` and `gru-dt` are discrete updates and use no numerical solver. The packaged defaults are Euler for `gru-ode`, RK4 for `ode-rnn-spline`, and adaptive Dopri5 for `ode-lstm`.

---

## 4. ODE-RNN, Latent ODE, GRU-ODE-Bayes, and Neural Jump ODE (NJ-ODE)

Package implementations informed by the original papers and repositories. They adapt event-based reference code to the package's batched sequence interface while retaining the method-specific training terms described below.

| Model | Origin paper | Reference repository | Package implementation |
|---|---|---|---|
| `gru-ode-bayes` | [De Brouwer et al., *GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series* (NeurIPS 2019)](https://arxiv.org/abs/1905.12374) | [edebrouwer/gru_ode_bayes](https://github.com/edebrouwer/gru_ode_bayes) | Learned Gaussian prediction distribution, GRU-ODE drift, mask-aware Bayesian observation preparation, pre-jump negative log-likelihood, and post-jump Gaussian KL. Observation intervals use bounded Euler sub-steps. |
| `ode-rnn` | [Rubanova et al., *Latent ODEs for Irregularly-Sampled Time Series* (NeurIPS 2019)](https://arxiv.org/abs/1907.03907) | [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode) | ODE-RNN adaptation for direct sequence representations. |
| `latentode` | [Rubanova et al., *Latent ODEs for Irregularly-Sampled Time Series* (NeurIPS 2019)](https://arxiv.org/abs/1907.03907) | [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode) | Reverse ODE-RNN recognition, learned posterior mean and scale, reparameterized initial latent state, batched generative solve, masked Gaussian reconstruction term, and initial-state KL. Recognition, reconstruction, and decoded trajectories respect each sample's last valid step. |
| `njode` | [Herrera et al., *Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering* (ICLR 2021)](https://arxiv.org/abs/2006.04727) | [HerreraKrachTeichmann/NJODE](https://github.com/HerreraKrachTeichmann/NJODE) | Dense-grid adaptation with the paper's hidden-state ODE field, mask-aware self-imputation, observation-time jump, event-normalized native loss, and bounded Euler sub-steps. |

`ode-rnn` and `latentode` come from the same NeurIPS 2019 paper but expose different package architectures: `ode-rnn` evolves a recurrent state directly between observations, while `latentode` learns a variational initial state and decodes a generative latent trajectory. Sample-level task heads use the inferred initial latent representation; sequence-level heads use the decoded trajectory. GRU-ODE-Bayes, Latent ODE, and NJ-ODE expose their method-specific objectives through `return_aux_loss=True`.

The packaged defaults are Euler for `ode-rnn`, GRU-ODE-Bayes, and NJ-ODE, and adaptive Dopri5 for Latent ODE.

For variable-length batches, ODE-RNN evolves fixed-step intervals in batch, Latent ODE ignores padding-only timestamps during recognition and decoding, and both methods use each sample's final valid index for their returned representation. LatentSDE likewise evaluates path KL at each sample's own terminal step and holds the returned latent path constant across padding.

The package's six task heads are shared adapters around these backbones. In particular, the NJ-ODE forecasting head projects the context representation to a fixed output horizon; it is not the paper's native future-time filtering path because the current forecasting target does not include a future timestamp grid.

---

## 5. Neural CDE and Neural RDE

Neural CDE methods:

| Model | Interpolation | Underlying library | Origin |
|---|---|---|---|
| `neuralcde` | natural cubic spline | [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE) (NeuralCDE reference path) | [Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series* (NeurIPS 2020)](https://arxiv.org/abs/2005.08926) |
| `neuralcde-l` | linear | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper as above |
| `neuralcde-r` | rectilinear | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper |
| `neuralcde-c` | cubic | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper |
| `neuralcde-h` | hermite | [torchcde](https://github.com/patrick-kidger/torchcde) | Same paper |

Neural RDE methods:

| Model | Base path | Log-signature depth | Underlying library | Origin |
|---|---|---|---|---|
| `neuralrde-1` | natural cubic spline | 1 | [torchcde](https://github.com/patrick-kidger/torchcde) + [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE) | [Morrill et al., *Neural Rough Differential Equations for Long Time Series* (ICML 2021)](https://arxiv.org/abs/2009.08295) |
| `neuralrde-2` | natural cubic spline | 2 | [torchcde](https://github.com/patrick-kidger/torchcde) + [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE) | Same Neural RDE paper |
| `neuralrde-3` | natural cubic spline | 3 | [torchcde](https://github.com/patrick-kidger/torchcde) + [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE) | Same Neural RDE paper |

`neuralcde` uses the original `controldiffeq` reference; the `-l/-r/-c/-h` variants use the `torchcde` library and select the interpolation scheme for the control path. `neuralrde-1/2/3` first builds a natural-spline path, then applies log-signature windows at the requested depth. Keep native missing values as `NaN` before `preprocess(...)`; the interpolation path is built from that missing-aware input.

ANCDE, EXIT, LEAP, DualDynamics, and FlowPath also use CDE-style control paths and consume interpolation coefficients.

---

## 6. ANCDE, EXIT, and LEAP

`ancde`, `exit`, and `leap` are Neural CDE methods that consume interpolation coefficients and use CDE control paths. The default solver is `rk4` with a fixed-grid step size derived from the input times.

| Model | Paper | Behavior |
|---|---|---|
| `ancde` | Jhin et al., *Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting* (ICDM 2021). Reference repo: [sheoyon-jhin/ANCDE](https://github.com/sheoyon-jhin/ANCDE) | Attention-gated control path. No auxiliary loss. |
| `exit` | Jhin et al., *EXIT: Extrapolation and Interpolation-based Neural Controlled Differential Equations for Time-series Classification and Forecasting* (ACM Web Conference 2022). Reference repo: [sheoyon-jhin/EXIT](https://github.com/sheoyon-jhin/EXIT) | Control path with kinetic-energy regularization. Returns an auxiliary term when called with `return_aux_loss=True`. |
| `leap` | [Jhin et al., *Learnable Path in Neural Controlled Differential Equations* (AAAI 2023)](https://arxiv.org/abs/2301.04333) | Learnable-path CDE. Returns an auxiliary loss when called with `return_aux_loss=True`. |

- **Solver and dependencies**: all three use `rk4` with a fixed-grid step size from input times; install `.[cde]` or `.[all]`.
- **Preprocessing**: use `preprocess(X, model_name=...)` to obtain matching natural-cubic-spline coefficients.
- **LEAP time scale**: `leap` uses normalized `[0, 1]` control-path time, which may differ from runs that used an integer-grid path.

---

## 7. NeuralFlow

| Model group | Models | Runtime note |
|---|---|---|
| NeuralFlow comparator | default name `neuralflow`; explicit variants `neuralflow-r`, `neuralflow-g`, `neuralflow-c` | Flow-only; uses the direct value/time path and no numerical solver. |

Origin paper: [*Neural Flows: Efficient Alternative to Neural ODEs* (NeurIPS 2021)](https://arxiv.org/abs/2110.13040).

Use `neuralflow` for the residual-flow default. The suffix selects an explicit flow variant: `-r` = residual flow, `-g` = GRU-style flow, and `-c` = coupling-style flow.

Active NeuralFlow solver settings are rejected because there is no integration step to configure; the neutral shared-call setting `adjoint=False` remains accepted. Its strict raw-missing path uses mask-aware zero filling and supports only `imputation_strategy="zero"`.

For missing-data handling, `neuralflow` and its explicit variants keep raw missing values on the direct value path. Missing positions should remain `NaN` with mask value `0`.

---

## 8. DualDynamics and FlowPath

DualDynamics and FlowPath are flow-based Neural CDE / control-path methods. They consume interpolation coefficients, use CDE-style control paths, and include flow components; they are not flow-only comparators. Both default to `rk4` with a fixed-grid step size from the input times.

| Model group | Models |
|---|---|
| DualDynamics | default name `dualdynamics`; explicit variants `dualdynamics-r`, `dualdynamics-g`, `dualdynamics-c` |
| FlowPath | default name `flowpath`; explicit variants `flowpath-r`, `flowpath-g`, `flowpath-c` |

Origin papers:
- DualDynamics: [*DualDynamics: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis* (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34173) · [arXiv:2401.04979](https://arxiv.org/abs/2401.04979)
- FlowPath: [*FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification* (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39643) · [arXiv:2511.10841](https://arxiv.org/abs/2511.10841)

Use `dualdynamics` and `flowpath` for residual-flow defaults. The suffix selects an explicit flow variant: `-r` = residual flow, `-g` = GRU-style flow, and `-c` = coupling-style flow. These remain CDE/control-path methods.

For missing-data handling, use `preprocess(X, model_name=...)` to build matching interpolation coefficients. `flowpath-*` also keeps the raw missing-value path; `dualdynamics-*` follows the coefficient-driven control path.

---

## 9. SDE baselines

| Model | Origin | Implementation in this package |
|---|---|---|
| `naive_cde` | Baseline (this package) | Control-path baseline for SDE comparisons. No external paper. |
| `neural_ode` | [*Neural Ordinary Differential Equations* (NeurIPS 2018)](https://arxiv.org/abs/1806.07366) | Zero-diffusion deterministic baseline integrated through the shared [`torchsde`](https://github.com/google-research/torchsde) wrapper. |
| `neural_ode_t` | Baseline (this package) | Time-conditioned variant of `neural_ode`. No external paper. |
| `neural_sde` | Baseline (this package) | Generic Neural SDE baseline; uses [`torchsde`](https://github.com/google-research/torchsde). No external paper. |
| `neural_sde_t` | Baseline (this package) | Time-conditioned variant of `neural_sde`. No external paper. |
| `latentsde` | [Li et al., *Scalable Gradients for Stochastic Differential Equations* (AISTATS 2020)](https://arxiv.org/abs/2001.01328) | Full-path mask-aware posterior for the initial latent state, OU prior, initial and path KL, and a physical latent trajectory separated from the augmented KL accumulator. The SDE path follows the [`torchsde` Latent SDE example](https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py). |

The first five rows use cubic-spline interpolation via `torchcde` and accept the Ito-compatible Euler, Milstein, and SRK methods. `latentsde` consumes values, per-channel masks, deltas, and normalized time directly; its default solver is Euler with direct differentiation, and `kl=False` disables both initial and path KL. In variable-length batches, path KL is evaluated at each sample's last valid step and the returned latent state is held constant across its padded suffix.

Use `preprocess(X, model_name=...)` to build the sequence stack and model-specific preprocessing outputs. LatentSDE does not consume the interpolation coefficients. See [API Reference §7.1](api_reference.md#71-get_data_config) for the full per-model settings.

---

## 10. Stable Neural SDEs

> *Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data* (ICLR 2024) — Oh, Lim, and Kim ([arXiv:2402.14989](https://arxiv.org/abs/2402.14989) · [OpenReview](https://openreview.net/forum?id=4VIgNuQ1pY) · repo: [yongkyung-oh/Stable-Neural-SDEs](https://github.com/yongkyung-oh/Stable-Neural-SDEs))

| Package model | Full model name in the paper | Drift–diffusion variant |
|---|---|---|
| `neural_lsde` | Langevin-type SDE (LSDE) | Additive / Langevin-type noise |
| `neural_lnsde` | Linear Noise SDE (LNSDE) | Linear multiplicative noise |
| `neural_gsde` | Geometric SDE (GSDE) | Geometric multiplicative noise |

All three default to the `srk` solver.

---

**End of Model Reference**
