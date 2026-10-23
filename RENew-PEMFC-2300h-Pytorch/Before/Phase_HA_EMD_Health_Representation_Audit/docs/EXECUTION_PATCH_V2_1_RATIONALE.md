# Phase HA execution patch V2.1 rationale

`PRE_ENDPOINT_ANALYSIS_AMENDMENT_V2.yaml` and its SHA256 remain immutable.
During the first V2 run, all endpoint calculations completed in memory but final
gate assembly raised `AttributeError` when there was no matched-support stratum
and `ASR_AUDIT` was therefore an empty table without a `stratum` column.  No
normal endpoint output was persisted and no health state was issued.  The
historical record is retained verbatim in `TECHNICAL_BLOCKER_V2.md`.

V2.1 makes one execution-only correction: an empty ASR table is interpreted as
a failed stability gate.  It is not a value, an imputation, or an ASR result.
This does not alter the registered N1 firewall, cache checks, representations,
matching design, calipers, bootstrap block lengths, 2,000 iterations, seed,
RDR/recovery rules, or final decision rules.  The patch is frozen with exact
source-code hashes and must be verified before a retry.

The effective protocol for the retry is therefore `v1 + amendment_v2 +
execution_patch_v2_1`.  The original V2 execution hashes are kept as the
authenticated predecessor; V2.1 supplies the authenticated current hashes.
