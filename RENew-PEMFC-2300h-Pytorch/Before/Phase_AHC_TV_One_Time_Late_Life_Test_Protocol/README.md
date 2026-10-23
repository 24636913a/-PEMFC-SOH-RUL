# Phase AHC-TV — One-Time Late-Life Test Validation Protocol

**Status:** `READY_FOR_EXPLICIT_OPEN_AHC_TV_TEST`

This is a protocol-freeze-only phase. It does not read Test raw data, polarization, EIS, 2000 h, 2300 h, or after-Air-Air data; it does not run Test metrics. The sole preregistered question is whether the Development-frozen P1S affine calibration is more accurate than a Development-frozen nominal-age affine baseline at the two late-life checkpoints 2000 h and 2300 h.

The protocol provides no authorization to unblind Test. A separate manual decision is required before any one-shot execution.

The final pre-open verification passed without opening real 2000 h or 2300 h
Test sources.  The phase is ready only to wait for an explicit
`OPEN_AHC_TV_TEST` decision; it must not initiate one-shot execution itself.

The original protocol remains intact. `configs/AHC_TV_EXECUTION_CONTRACT_V1.yaml`
is its frozen pre-unblinding supplement: it fixes the exact PAA inheritance,
the seven-dimensional Development support, metadata-only/target-reader
isolation, dual-checkpoint prediction freezing, and the block on historical
N1 Test-derived outputs. This phase has not read real 2000 h or 2300 h data.

`scripts/run_ahc_tv_one_shot.py` is now the sole future formal runtime entry.
It requires an explicit external authorization argument, first runs target-blind
Stage P for both checkpoints, freezes both the prediction table and
target-free provenance, and permits paired Stage T target reading only after
the hardened verification gate accepts those artifacts. The present freeze ran
only temporary synthetic integration data; it did not create any real runtime
artifact or read actual Test sources.

## Final direct-runtime verification (V2)

The final Development-only audit passes only because the actual formal Stage-P
runtime was replayed at all six frozen PAA checkpoints, and the audit runner
executed its required unittest suites itself:

- Formal Stage P Development replay: **6/6 PASS**.
- Synthetic runtime E2E: **ACTUALLY EXECUTED, PASS** (7 tests).
- One-shot state machine: **ACTUALLY EXECUTED, PASS** (7 tests).
- Real Test: **NOT OPENED**.

The formal runtime and retained independent reference replay reproduce all 18
selected interval identities, timestamps, P1S retention values, condition
vectors, and support flags. `scripts/ahc_tv_raw_alignment_core.py` is the
authoritative raw-to-interval implementation. This is an implementation
correction only, not a scientific-protocol change. The phase is therefore
`READY_FOR_EXPLICIT_OPEN_AHC_TV_TEST`, and remains prohibited from initiating
that one-shot action itself.
