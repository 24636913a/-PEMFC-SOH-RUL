# Phase AHC — Sparse-Anchor Affine Health Calibration Characterization

**Status:** `POSTHOC_HYPOTHESIS_CHARACTERIZATION_PROTOCOL_FROZEN`

AHC characterizes the already prespecified and executed CHM-D B2 model,
`Y = aR + b, a >= 0`, using only frozen Development artifacts. It is not a new
model search, a confirmatory validation, validated SOH, Test evaluation, or
RUL study. The analysis is restricted to checkpoints 300/600/850/1150/1500/
1750 h and the frozen P1S three-interval median functional.

The executable protocol is `configs/AHC_POSTHOC_PROTOCOL.yaml`. The runner
uses only the three listed frozen CSV inputs, writes Development-only outputs,
and refuses raw, polarization, EIS, 2000 h, 2300 h, or Test sources.
