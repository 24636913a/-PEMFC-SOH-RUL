# Relation to Phase N2.5

Phase N2.5 contained project-local, auditable sLSTM/mLSTM/Hybrid xLSTM-style
implementations and found them weaker than its contemporary LSTM. That is
project-specific negative evidence for the broad xLSTM category only.

This phase instead evaluates the author forecasting implementation from
`muslehal/xLSTMTime`. Its native path includes a moving-average
trend/seasonal split, 48→24 per-variable projections, a 256-dimensional mLSTM
stack operating after projection, and a native 24-step head. It is therefore
not identical to the N2.5 time-axis local blocks. No statement in this phase
claims that xLSTMTime had previously failed on RealHyFC.
