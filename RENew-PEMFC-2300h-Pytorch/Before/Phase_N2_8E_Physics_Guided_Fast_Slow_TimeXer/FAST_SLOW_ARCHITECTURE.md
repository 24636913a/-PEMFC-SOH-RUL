# Fast-slow architecture

The official TimeXer historical backbone is preserved at commit `76011909357972bd55a27adba2e1be994d81b327`. It produces the historical global latent and official base forecast. Each future condition row `z_h` goes only through `MLP(z_h) -> gamma_h,beta_h`; the corresponding horizon state is modulated as `(1+gamma_h)H_h+beta_h`. No CrossAttention, pooling over future steps, or future-time concatenation is present.

The fast prediction is the frozen-form official head plus a zero-initialized step-aligned correction. The slow prediction is `LayerNorm(Hhist) -> Linear(64,32) -> GELU -> Linear(32,24)` with a zero-initialized final layer, and is supervised only by the DeltaV data loss plus optional regularizers.

| component | parameters | extra/original |
|---|---:|---:|
| official TimeXer backbone | 115096 | 0.000000% |
| SA-TimeXer total | 125657 | 9.175818% |
| FS/PGFS-TimeXer total | 128657 | 11.782338% |

The model graph enforces strict h-to-h conditioning; a derivative audit is emitted for every fast-slow fold.
