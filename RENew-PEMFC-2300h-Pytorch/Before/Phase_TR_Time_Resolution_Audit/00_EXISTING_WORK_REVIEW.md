# Existing-work review and scope boundary

## Read-only sources reviewed

This phase was created after reading the plans, result/decision documents, and
relevant baseline code in `Phase_N1_Data_Foundation`,
`Phase_G_STOP_Physics_Consistent_Future_Conditions`, `Phase_OPR_TimeXer`,
`Phase_TTD_TimeXer_Failure_Diagnosis`, `Phase_R0_Rare_State_Identifiability`, and
`Phase_R1_Calibrated_Two_Scenario`.  These directories remain read-only.

`Phase_N1` established `OpHrs2 [h]` as the ageing coordinate, causal
left-closed/right-open aggregation, 30-minute as the primary representation, and
the frozen development endpoint at 1819.997738 h.  A 48-step 30-minute history is
**24 h**, not 48 h.  The inherited expanding folds are: 72.788690–946.393214 h
(Fold 1 train), validation to 1164.794345 h (Fold 1); then training expands to
1164.794345 h and 1383.195476 h, with Fold 3 validation ending at 1819.997738 h.

Earlier future-condition studies define `q` as the fraction of raw samples with
`abs(I.S.act) <= 0.10 A`.  It is an operational zero-current fraction, not a
validated shutdown label.  Their P0/P1/P2 scenarios distinguish unavailable,
protocol-known, and causally forecast future conditions.  This audit implements
only the strict unavailable-future setting: no actual future current, voltage, or
q enters a forecast.

`Phase_OPR` and `Phase_TTD` identified the important hard regimes rather than a
generic architecture failure: Fold 1 12-hour errors can be driven by state-change
outliers and Fold 3 can remain difficult during sustained nonzero-current
operation.  `Phase_R0` attributed much of this to limited identifiability of
future q from the past.  `Phase_R1` tested causal probability calibration and
found it did not supply a reliable corrective path.  Those results are not
repeated, recalibrated, or re-labelled here.

## New contribution of Phase TR

Phase TR asks a narrower, preregistered question: whether 10-minute causal bins
reveal credible within-30-minute zero-current transitions **and**, on exactly
matched 30-minute targets and forecast origins, improve a fixed simple causal
forecast of q.  It reconstructs both resolutions directly from the development
raw stream, audits timestamp/data quality, uses fixed Logistic Regression and
Ridge models only, and does not train or modify TimeXer.  A positive result would
only justify a future 10-minute TimeXer experiment; it cannot itself establish a
TimeXer accuracy gain.
