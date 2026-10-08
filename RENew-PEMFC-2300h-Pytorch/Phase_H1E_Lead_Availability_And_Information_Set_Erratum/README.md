# Phase H1E — Lead availability and information-set erratum

This is a read-only Development-side erratum for Phase CPRF-H1. It does not train, refit, rescore, or select any model. It preserves H1 numerical results and clarifies two interpretation limits:

1. whether actual target P1S measurement-end lead was known at prediction origin; and
2. why Persistence, actual-lead baselines, and Raw-F2-only TimesNet do not have equal information sets.

Run:

```powershell
python scripts/run_h1e_erratum.py
```

The runner accepts only the committed H1 summary and frozen H1 sample authority plus existing Development-side RCR/PAA protocol artifacts. It never reads Test data, predictions, or targets and does not execute model code.
