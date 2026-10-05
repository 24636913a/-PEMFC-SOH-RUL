# One-shot unblinding procedure

An approved execution may use only `scripts/run_ahc_tv_one_shot.py` with the
literal external authorization marker. It first verifies the original
protocol, execution contract, runtime-integration contract, frozen-input
authority, and frozen alignment-builder hashes. Reader M then reads only
`DateTime` from each filename-bound polarization CSV to resolve both actual
characterization starts before Stage P opens raw operating data. Stage P slices
each checkpoint with `DateTime < characterization_start`, derives the P1S
retentions and support evidence from raw rows, and generates both P1S-affine,
nominal-age, and raw predictions.

Stage P writes and SHA256-freezes the two-row prediction CSV, target-free
provenance JSON, and a completion marker in the explicit runtime directory.
Before Stage T opens either target column,
`authorize_and_verify_target_unblinding()` requires the literal external
authorization marker `OPEN_AHC_TV_TEST`, verifies all frozen authority hashes,
enforces the exact 21-column schema, recomputes the P1S/raw/age predictions,
and hash-verifies both Stage P provenance and completion marker. Stage T binds
both 2000 h and 2300 h sources before it opens either target file, then reads
both targets as one atomic paired operation. The 2000 h target cannot affect
the 2300 h prediction. No refitting, tuning, Test-alignment change, or Test
follow-up is allowed after target unblinding.
