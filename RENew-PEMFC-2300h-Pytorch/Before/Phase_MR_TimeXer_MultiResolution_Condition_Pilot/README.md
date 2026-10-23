# Phase MR — TimeXer Multi-Resolution Condition Pilot

This is a strictly Development-only, two-gate pilot.  The frozen 30-minute
SA-TimeXer remains unmodified.  The only candidate change is a direct 10-minute
historical condition forecaster that predicts the same future 24 30-minute
condition targets as P2.

The run first reproduces frozen P0/P1/P2.  It then compares P2 and MR on a
common-origin set, using causal target-completion-purged OOF for MR training
predictions.  Frozen-TimeXer M0/M1 inference is permitted only after the
preregistered condition gate passes.
