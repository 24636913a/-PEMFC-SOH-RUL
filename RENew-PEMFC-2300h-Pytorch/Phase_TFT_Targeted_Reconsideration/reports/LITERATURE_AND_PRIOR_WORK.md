# Literature and prior-work boundaries

## Temporal Fusion Transformer

The original TFT is Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* (International Journal of Forecasting, 2021). Its design combines local recurrent processing, variable selection, gated residual networks and interpretable attention for multi-horizon forecasting. This phase is explicitly a **strict-history TFT adaptation**, rather than a claim that it exactly reproduces the original known-future-covariate setting. [Publisher record](https://doi.org/10.1016/j.ijforecast.2021.03.012) · [Author preprint](https://arxiv.org/abs/1912.09363).

TFT has already been applied directly to PEMFC work. A 2025 *Renewable Energy* PEMFC study uses a TFT-based approach and modifies the local LSTM encoder-decoder with BiLSTM. Consequently, neither “first use of TFT in PEMFC” nor the generic idea of replacing TFT's local LSTM route can be claimed here. [Article record](https://www.sciencedirect.com/science/article/abs/pii/S0960148125006810).

The frozen implementation-reference commits are recorded in `SOURCE_MANIFEST.yaml`: Google Research TFT commit `e49bbfe381c9c0e564b937f1c4e163a2273c65cc` and PyTorch Forecasting TFT commit `5a8c20f7001f442439644616257218fdc85be2d3`. They were used only for architecture review, not as runtime code.

## xLSTM and mLSTM

xLSTM is described by Beck et al., *xLSTM: Extended Long Short-Term Memory* (NeurIPS 2024). This phase froze the official NX-AI/xlstm tag `v2.0.4`, commit `add5001330eb62d66867a5eca211050240ca74f5`, for engineering tests of native state continuity. [Paper](https://arxiv.org/abs/2405.04517) · [official source](https://github.com/NX-AI/xlstm).

Matrix LSTM is also not new to PEMFC: *Remaining useful life prediction of PEMFC based on matrix long short-term memory* was published in the *International Journal of Hydrogen Energy* (2025), DOI [`10.1016/j.ijhydene.2025.02.302`](https://doi.org/10.1016/j.ijhydene.2025.02.302). Therefore, even if an mLSTM-TFT route had passed, its allowable claim would only have been: embedding a matrix-memory mechanism into TFT's local recurrent path and testing it under this frozen RealHyFC protocol. It could not claim the first PEMFC use of mLSTM.

## Implication for this phase

The preregistered Vanilla TFT gate failed before any sLSTM/mLSTM performance experiment was authorized. Thus this phase establishes neither an xLSTM gain nor an mLSTM-TFT contribution. Novelty and model-selection claims remain outside its evidence.
