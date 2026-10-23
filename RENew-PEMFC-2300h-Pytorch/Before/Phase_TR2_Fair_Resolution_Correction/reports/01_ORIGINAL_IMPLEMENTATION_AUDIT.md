# Original implementation audit

Status: `POST_HOC_METHOD_CORRECTION`. This document audits the committed Phase TR
source, not a recollection of its reported results.

## Confirmed issue 1 — future-observation weighting in predictions

In `Phase_TR_Time_Resolution_Audit/scripts/run_time_resolution_audit.py`, function
`windows` (line 159 in the audited version) stores each future target's
`valid_current_count` as `weights`.  `predict_compare` line 215 then computes
`np.average(q[i:i+3], weights=r["weights"][i:i+3])` and applies the same weighting
to the predicted mixed probability.  Those weights come from the future target
measurements and are unavailable when a forecast is issued.  This is a causal
evaluation defect even though using the same raw counts to construct the true
30-minute q label is legitimate.

## Confirmed issue 2 — non-identical mixed-state event label

The same line 215 constructs the aggregated truth class as
`int(r["mixed"][i:i+3].max())`.  That marks a 30-minute bin mixed only if at least
one 10-minute child is internally mixed.  It misses a valid parent containing one
all-on child (`q=0`) and one all-zero child (`q=1`), although the parent raw counts
make its q strictly between zero and one.  The evaluated probability and truth
therefore did not necessarily denote the same 30-minute event.

## Confirmed issue 3 — non-common training-origin sets and capacity confounding

`predict_compare` independently creates `w30t` and `w10t` at line 205, fits models
at line 208, and intersects only validation origins afterward (lines 214–217).
Thus the training rows and their target counts may differ.  In addition, the
30-minute flattened history has 144 inputs while the 10-minute flattened history
has 432 inputs, with no aggregation-control model.  The former difference can
confound sample support; the latter prevents attributing every difference solely
to resolution.

Phase TR's `NO_GO_KEEP_30MIN` remains unchanged. TR2 does not overwrite it and
does not present its post-hoc corrected result as a blind confirmation.
