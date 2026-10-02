# Phase PAV — PAA Anchor Validity Qualification Audit

`POSTHOC_QUALIFICATION_AUDIT`

This phase qualifies, but does not replace, the frozen Phase PAA result
`P1S_PREANCHOR_COMPATIBILITY_EXPLORATORY_PASS`.  It has two deliberately
narrow checks:

1. Whether each allowed Development nominal-polarization CSV internally
   records a time that places the three frozen PAA P1S measurements before the
   polarization characterization.
2. Whether the frozen median-producing interval of each PAA anchor is itself
   in PAA's complete-cycle joint common-support region.

The PAA state token, 48 h interval rule, three-interval rule, 2-of-3 support
rule, PAA calculations, and historical PAA outputs are read-only.  No model,
continuous SOH, or RUL target is produced.  Test polarization and EIS files
are rejected by the production path.

Run from the repository root while `HEAD` is
`02b0319fac3380b0c5b76cf999f4ae8ae71df1ea`:

```powershell
python RENew-PEMFC-2300h-Pytorch/Phase_PAV_PAA_Anchor_Validity_Qualification/scripts/run_paa_anchor_validity_qualification.py
python -m unittest discover -s RENew-PEMFC-2300h-Pytorch/Phase_PAV_PAA_Anchor_Validity_Qualification/tests -v
```

The production script accesses only the six named Development nominal
polarization files, the BoL header for structural reference, and frozen Phase
PAA files.  It never infers a polarization time from a filename or filesystem
metadata.
