# Frozen architecture diagram

```text
Raw F2 [B,48,14]
  → transpose [B,14,48]
  → shared variable MLP [B,14,D]
  → per-sample rFFT(D), top-2 distinct periods
  → for each period branch: outer segmentation → masked Conv1D
  → concat/valid-fill to D → Q,K,V → E=[B,14,14]
  → for each outer segment: inner patches → Conv1D → bridge S=E(patch·Wv)
  → GPRU → dynamic graph A_jl → two-hop A/Aᵀ mix-hop → masked intra mean
  → period-level shared GPRU/graph/mix-hop → masked inter-period mean
  → shared branch MLP
  → two-scalar branch fusion
  → shared variable forecast MLP [B,14,24]
  → transpose [B,24,14] → voltage channel 0
  → subtract standardized origin voltage → DeltaV [B,24]
```

All arrows labelled “bridge” in `PAPER_TO_INSPIRED_MAPPING.md` are
exploratory choices, not claims about the unavailable official PDFNet source.
