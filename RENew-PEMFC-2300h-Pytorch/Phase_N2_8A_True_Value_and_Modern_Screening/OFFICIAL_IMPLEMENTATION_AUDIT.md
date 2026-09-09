# Official Implementation Audit

## TimeXer

- Repository: `https://github.com/thuml/TimeXer`; fixed vendor commit `76011909357972bd55a27adba2e1be994d81b327`.
- Vendored source: `vendor/TimeXer`, including `models/TimeXer.py`, `layers/SelfAttention_Family.py`, `layers/Embed.py`, `layers/Transformer_EncDec.py`, and normalization components.
- Retained forward path: official `EnEmbedding` patches endogenous voltage, official `DataEmbedding_inverted` embeds the thirteen exogenous F2 variables, official global endogenous token and self/cross attention `EncoderLayer` blocks are retained, followed by official `FlattenHead`.
- Adapter: F2 columns are reordered only at the wrapper boundary so `cell_voltage_mean` is the official MS-mode final/endogenous channel; all 13 exogenous variables remain historical. `use_norm=False` is an official configuration setting because the train-only external scaler and standardized DeltaV target are the frozen comparable training protocol. Output is the official 24-step head, trained against standardized DeltaV and reconstructed only for secondary voltage metrics.

## TimeMixer++

- Checked named official source: `https://anonymous.4open.science/r/TimeMixerPP`; retrieval on this run returned HTTP 410 and cannot be vendored/pinned.
- Checked authors' public `https://github.com/kwuking/TimeMixer` at fixed commit `e24610583b36fdd8c76cc17a8df4e65759a5f460`. Its sole model path is old `models/TimeMixer.py`; it has no TimeMixer++ MRTI, TID, MCM, or MRM implementation.
- Therefore **no old TimeMixer model was renamed, adapted, or reported as TimeMixer++**. The requested high-fidelity TimeMixer++ experiment is blocked until an accessible author-confirmed versioned implementation is supplied or released.
