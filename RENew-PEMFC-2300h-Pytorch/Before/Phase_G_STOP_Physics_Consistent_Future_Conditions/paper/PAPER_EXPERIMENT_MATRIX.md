# Experiment Matrix

| Stage | Variant | Learned component | Formal status | Seeds | Outcome |
|---|---|---|---|---|---|
| G1 | raw physical audit | none | formal audit | n/a | PASS |
| G1 | true q only | none; true future diagnostic | diagnostic only | 42 | physically inconsistent input combination |
| G1 | true physical current group | none; true future diagnostic | diagnostic only | 42 | confirms future-current information gap |
| G2/G3 | P0 | repaired Ridge futures; frozen SA-TimeXer | formal | 42 | baseline reproduced |
| G2/G3 | P1 | minimal physical projection; frozen SA-TimeXer | formal | 42 | physical pass; Gate B fail |
| G2/G3 | P2 | q-aware light forecast; frozen SA-TimeXer | formal | 42 | Gate B pass |
| G4 | P0 retrained | unchanged SA-TimeXer | formal control | 42 | 24.538 mV |
| G4 | P2 retrained | unchanged SA-TimeXer | conditional formal | 42 | 21.553 mV; multiseed trigger fail |
| G5 | L0 | original causal-SG LSTM | formal control | 42 | 7.926 mV |
| G5 | L1 | P0 residual head | formal | 42 | 21.043 mV; fail |
| G5 | L2 | P2 residual head | formal | 42 | 21.467 mV; fail |
| Test | all | none | sealed | none | NOT_RUN |
