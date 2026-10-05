# One-shot unblinding procedure

An approved execution first verifies the original protocol, execution contract,
frozen-input authority, and frozen alignment-builder hashes. Reader M then
reads only `DateTime` from each polarization CSV to resolve the two actual
characterization starts. It constructs both Test P1S inputs from strictly
pre-characterization raw operating data, verifies the seven-dimensional
support gate, and generates both P1S-affine, nominal-age, and raw predictions.

It then writes and SHA256-freezes
`inputs/FROZEN_TEST_PREDICTIONS_BEFORE_TARGET_UNBLINDING.csv`, containing both
2000 h and 2300 h rows. A matching SHA256 only proves that the artifact did
not change after freezing; it does not prove that the file is a complete
prediction. Before Reader T opens a target column,
`authorize_and_verify_target_unblinding()` requires the literal external
authorization marker `OPEN_AHC_TV_TEST`, verifies all frozen authority hashes,
enforces the exact 21-column schema, recomputes the P1S/raw/age predictions,
and rechecks support and temporal evidence. The 2000 h target cannot affect
the 2300 h prediction. No refitting, tuning, Test-alignment change, or Test
follow-up is allowed after target unblinding.
