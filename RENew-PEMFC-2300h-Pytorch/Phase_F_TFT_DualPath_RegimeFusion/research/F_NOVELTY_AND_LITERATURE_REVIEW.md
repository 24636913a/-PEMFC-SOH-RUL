# Novelty and Literature Positioning (Design-Stage Only)

## Verified foundations

1. Lim, Arik, Loeff and Pfister, **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**, *International Journal of Forecasting* (2021), DOI [10.1016/j.ijforecast.2021.03.012](https://doi.org/10.1016/j.ijforecast.2021.03.012), defines the native ingredients relevant here: variable selection, recurrent local processing, gated residual components, static context, and interpretable attention for multi-horizon forecasting.
2. Bai, Kolter and Koltun, **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling** (2018), [arXiv:1803.01271](https://arxiv.org/abs/1803.01271), provides the generic causal/dilated convolution rationale. It does not establish that a particular dilation is physical for PEMFC data.
3. Yang and Wang, **Performance degradation prediction of PEMFC under dynamic operating conditions based on WOA-BiGRU** (2024), DOI [10.3969/j.issn.1674-8425(z).2024.01.035](https://opaj.napstic.cn/periodicalArticle/0120240305876145), is an example of dynamic-condition PEMFC degradation modelling with an optimized bidirectional recurrent method. Its bidirectional setup is not a causal template for this project.
4. Recent PEMFC hybrid/multiscale work exists, including **Forecasting the output performance of PEMFCs via a novel deep learning framework considering varying operating conditions and time scales** (Applied Energy, 2025), [record](https://ideas.repec.org/a/eee/appene/v389y2025ics0306261925004933.html), and **A physically-informed hybrid deep learning framework for multi-timescale voltage prediction of proton exchange membrane fuel cells under dynamic loading** (Journal of Power Sources, 2026), DOI [10.1016/j.jpowsour.2026.240948](https://doi.org/10.1016/j.jpowsour.2026.240948). These establish that multi-timescale/fusion ideas are not novel in isolation.

## What native TFT already does

The audited TFT already performs time-varying variable selection, local recurrent processing, temporal self-attention, and multiple gates/GRNs. Hence a manuscript must not call “using TFT gates”, “using attention”, “using feature selection”, or merely “combining local and global time scales” a new contribution.

Nor is a BiLSTM-TFT-style replacement appropriate here: bidirectional processing is incompatible with the strict-history forecasting information set when it makes a representation for a historical time depend on later history in a way not available at that time. The proposed F1 retains a causal recurrent path and a causal convolution branch; it is a within-TFT encoder augmentation, not a second forecaster or a generic LSTM substitution.

## Defensible, conditional contribution framing

If and only if the frozen F0–F2 experiment supports it, the paper may frame two **tested hypotheses**:

1. **Dual-path temporal encoding hypothesis:** an additional causal multi-scale convolution over TFT's VSN-selected history can improve 48→24 DeltaV prediction beyond the same-stage native strict-history TFT.
2. **Regime-conditioned fusion hypothesis:** an observed-past operating-regime summary can control the fixed F1 mixture and add value without giving the decoder future physical covariates.

These are method hypotheses, not established contributions at Phase F design time. The method is not a PINN, does not identify PEMFC degradation mechanisms from gates, and must not be marketed as novel solely because it combines familiar blocks. The potential novelty lies in the constrained integration, strict information firewall, and controlled F0→F1→F2 evaluation on dynamic PEMFC voltage forecasting.

## Literature limitation

This is a targeted source check, not a systematic review. It found no verified source establishing an identical `pytorch-forecasting` strict-history TFT modification with this exact causal multi-scale encoder insertion and scalar regime mixture for this RealHyFC task. That absence is not proof of novelty; a manuscript-grade claim still requires a formal search before submission.
