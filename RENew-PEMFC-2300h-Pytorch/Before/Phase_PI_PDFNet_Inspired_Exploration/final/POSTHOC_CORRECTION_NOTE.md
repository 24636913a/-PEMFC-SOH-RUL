# Phase PI posthoc correction note

Correction baseline commit: `cd006de9f25ca7c7b58bd5e56a0d74db980366ce`
Correction type: `POSTHOC_REPORT_AND_AUDIT_CORRECTION`

## Historical inconsistency corrected

At the correction baseline, `outputs/FAIRNESS_AUDIT.json` reported `status = PASS` while reporting `timesnet_cache_identity_pass = false`. The false cache-identity flag was a reporting-logic error: `test_accessed=False` is a correct safety state, but it was included in a generic all-True boolean aggregation.

The corrected audit explicitly checks the six cache identity fields separately from window conditions and negative safety assertions. All six identity fields are true for every saved fold × seed record; `test_accessed`, future covariates, future 13-variable supervision, and external decomposition are correctly required to be false. Thus the corrected `FAIRNESS_AUDIT.status` remains `PASS`, now for a valid derived reason.

The prior final report also incorrectly said the exploratory architecture “showed sufficient potential” because it treated `multiseed=NOT_RUN` as a positive state. The corrected conclusion is `PDFNET_INSPIRED_NO_PROMISE`: the preregistered PDFNet-inspired exploratory architecture did not show sufficient potential under the frozen RealHyFC protocol.

## What did not change

- No model was trained, no seed-42 run was repeated, and seeds 2024/3407 were not run.
- No formal prediction metric, fold result, gate result, or `FINAL_EXPLORATORY_DECISION.yaml` research state was changed.
- No Test data, Test prediction, or Test metric was accessed.
- `TIMESNET_FREEZE` and `BACKBONE_SEARCH_CLOSED` remain in force.

The correction regenerated only the derived fairness display and final Markdown report. Local binary-cache status: `LOCAL_BINARY_CACHE_PRESENT_NOT_RECOMPUTED`. No cache was recomputed or used to regenerate formal performance data.
