# Phase DF — Leakage-Safe Decomposition Features

This phase tests only whether 48-step, origin-conditioned causal-block VMD or
EMD voltage representations add value beyond the frozen RealHyFC Raw F2 and
SimpleTrend controls. The formal backbone remains `TIMESNET_FREEZE`; this is
not a backbone search, SOH study, RUL study, Test evaluation, or a context
length study.

All decomposition inputs are exactly `V[t-47:t]`. A decomposition value is
therefore an origin-window-level feature, never a global timestamp feature.

## Reproduction order

From this directory, use the frozen Python environment and run the stages in
order:

```powershell
D:/ruanjian/python3.11.9/python.exe scripts/run_stage0.py
D:/ruanjian/python3.11.9/python.exe scripts/run_stage1.py
D:/ruanjian/python3.11.9/python.exe scripts/build_local_artifact_manifest.py
D:/ruanjian/python3.11.9/python.exe scripts/generate_final_artifacts.py
D:/ruanjian/python3.11.9/python.exe -m unittest tests.test_phase_df_protocol -v
```

Stage 1 can run extra seeds only when the pre-registered seed-42 gates allow
them. Ignored caches, checkpoints, predictions and epoch logs are recorded by
`outputs/LOCAL_ARTIFACT_MANIFEST.csv`; no Test input is accepted by the
production reader.
