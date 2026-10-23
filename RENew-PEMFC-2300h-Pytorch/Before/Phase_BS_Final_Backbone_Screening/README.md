# Phase BS — Final Backbone Screening

This is the terminal, Development-only RealHyFC backbone screen frozen before
any new Development result. It compares only TimesNet, SimpleTM, DUET, and
ModernTCN under the identical Raw F2 `[B,48,14]`, 48-step history / 24-step
DeltaV task. The recorded decision is **`TIMESNET_FREEZE`**.

## Reproducibility boundary

`configs/PREREGISTERED_PROTOCOL.yaml` is protected by
`configs/PREREGISTERED_PROTOCOL.sha256`. The original execution began at
repository commit `2083d1e46e830d998abc537fd2d671a3319d1fa5`; the runner
checks that base before accessing the registered Development prefix.

The phase reads only 3,007 rows from the N1 causal table and retains exactly
3,005 Development rows after the frozen end-boundary filter. Test rows,
targets, predictions and metrics are prohibited; see `RUN_AUDIT.json`.

The official source checkouts live locally under `vendor/` and are ignored by
Git because they are third-party repositories. Their URLs, immutable commits,
licenses and canonical source hashes are in the frozen protocol. To recreate a
fresh local execution environment, clone each repository into the named
`vendor_root` at the specified commit before running the preflight.

## Commands used

```powershell
python tests/test_backbone_protocol.py
python scripts/run_preflight.py
python scripts/run_backbone_screening.py
```

The final command uses existing complete caches when they already pass the
artifact validation; it does not change model settings or access Test.

## Main artifacts

- `FINAL_DECISION.yaml` — terminal outcome and scope flags.
- `FINAL_BACKBONE_REPORT.md` — formal and target-time sensitivity results.
- `outputs/per_fold.csv` — every completed model/fold/seed result.
- `outputs/artifact_integrity.json` — key, truth, anchor and boundary audit.
- `outputs/preflight/` — official source and GPU implementation audit.
