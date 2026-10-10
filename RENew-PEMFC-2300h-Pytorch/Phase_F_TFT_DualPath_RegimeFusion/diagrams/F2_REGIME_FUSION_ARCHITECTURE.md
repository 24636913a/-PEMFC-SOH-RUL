# F2: Regime-Aware Fusion Delta from F1

```mermaid
flowchart LR
  X[Scaled 48-step observed history] --> J[Five fixed regime channels]
  J --> R[Temporal mean\nr: B×5]
  R --> G[MLP 5→8→1 + sigmoid\ng: B×1]
  L[E_LSTM: B×48×64] --> F[Regime mixture\nE_F2=g·E_LSTM+(1-g)·E_conv]
  C[E_conv: B×48×64] --> F
  G --> F
  F --> N[Same F1 native downstream TFT]
  N --> Y[24-step DeltaV]
```

F2 changes only the F1 fusion equation. `g` is computed from observed history only; it has no decoder or future physical input.
