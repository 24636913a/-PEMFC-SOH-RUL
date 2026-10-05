# Phase AHC-TV — One-Time Late-Life Test Validation Protocol

**Status:** `AHC_TV_EXECUTION_CONTRACT_FROZEN_AWAITING_MANUAL_TEST_OPENING_DECISION`

This is a protocol-freeze-only phase. It does not read Test raw data, polarization, EIS, 2000 h, 2300 h, or after-Air-Air data; it does not run Test metrics. The sole preregistered question is whether the Development-frozen P1S affine calibration is more accurate than a Development-frozen nominal-age affine baseline at the two late-life checkpoints 2000 h and 2300 h.

The protocol provides no authorization to unblind Test. A separate manual decision is required before any one-shot execution.

The original protocol remains intact. `configs/AHC_TV_EXECUTION_CONTRACT_V1.yaml`
is its frozen pre-unblinding supplement: it fixes the exact PAA inheritance,
the seven-dimensional Development support, metadata-only/target-reader
isolation, dual-checkpoint prediction freezing, and the block on historical
N1 Test-derived outputs. This phase has not read real 2000 h or 2300 h data.
