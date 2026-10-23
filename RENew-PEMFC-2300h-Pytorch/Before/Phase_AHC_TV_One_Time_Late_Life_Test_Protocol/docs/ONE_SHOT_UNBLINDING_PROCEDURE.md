# One-shot unblinding procedure

An approved execution may use only `scripts/run_ahc_tv_one_shot.py` with the
literal external authorization marker. The formal CLI has no runtime-directory,
overwrite, retry, force, or reset option. It first verifies the original
protocol, execution contract, runtime-integration contract, frozen-input
authority, and frozen alignment-builder hashes. Only then does it exclusively
create the phase-global `runtime/AHC_TV_ONE_SHOT_EXECUTION_LEDGER.json`, with
the initial status `EXECUTION_STARTED_BEFORE_FIRST_TEST_IO`; an existing ledger
blocks all Test I/O. Reader M then reads only
`DateTime` from each filename-bound polarization CSV to resolve both actual
characterization starts before Stage P opens raw operating data. Stage P slices
each checkpoint with `DateTime < characterization_start`, derives the P1S
retentions and support evidence from raw rows, and generates both P1S-affine,
nominal-age, and raw predictions.

Stage P writes and SHA256-freezes the two-row prediction CSV, target-free
provenance JSON, and a completion marker in the canonical
`runtime/ONE_SHOT_TEST_RUN` directory. Every formal artifact is an
exclusive-create operation. The ledger advances only forward through Stage P,
paired target unblinding, target/error freeze, decision freeze, and completion;
any post-start exception is permanently recorded as
`ABORTED_AFTER_TEST_EXECUTION_STARTED` and cannot be retried.
Before Stage T opens either target column,
`authorize_and_verify_target_unblinding()` requires the literal external
authorization marker `OPEN_AHC_TV_TEST`, verifies all frozen authority hashes,
enforces the exact 21-column schema, recomputes the P1S/raw/age predictions,
and hash-verifies both Stage P provenance and completion marker. Stage T binds
both 2000 h and 2300 h sources before it opens either target file, then reads
both targets as one atomic paired operation. The 2000 h target cannot affect
the 2300 h prediction. No refitting, tuning, Test-alignment change, or Test
follow-up is allowed after target unblinding. Immediately after both target
reads, `FROZEN_TEST_TARGETS_AND_ERRORS.csv` and
`TEST_UNBLINDING_DECISION.json` are frozen with SHA256 sidecars. Any offline
post-processing must consume the former artifact and may not reopen a
polarization CSV.
