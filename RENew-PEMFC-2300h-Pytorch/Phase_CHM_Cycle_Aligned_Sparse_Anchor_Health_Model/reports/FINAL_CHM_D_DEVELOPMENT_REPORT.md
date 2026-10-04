# Phase CHM-D — Development Execution

## 1. Execution integrity

- Base HEAD: `a3e56dd5bc55a6f7ab4b4aa373adbb190714be73`
- Authority: `FINAL_V2A`
- Frozen SHA checks: PASS
- Test/raw/polarization/EIS accessed: NO
- Neural training was Development-only.

## 2. Frozen baseline LOAO

- B0 MAE/RMSE: 0.01238958 / 0.01241887
- B1 MAE/RMSE: 0.00129687 / 0.00212422
- B2 MAE/RMSE: 0.00047687 / 0.00062858
- Each B1/B2 fold excludes its held-out target; details are frozen in `outputs/CHM_D_BASELINE_LOAO.csv`.

## 3. Nested LOAO profile selection

- Inner LOAO used only the five outer-training anchors, three frozen profiles, and seeds 42/2024/3407.
- Outer held-out targets did not select either their outer profile or the final all-six profile.
- Per-fold selections are in `outputs/CHM_D_OUTER_PROFILE_SELECTION.csv`.

## 4. Aggregate deterministic metrics

- CHM MAE/RMSE: 0.00151714 / 0.00175833
- Bias: 0.00041068; descriptive Spearman (n=6): 0.82857143
- The six official CHM estimates are three-seed medians; metrics are Development-only.

## 5. Gates 1–4

- Gate_1: PASS
- Gate_2: PASS
- Gate_3: FAIL
- Gate_4: PASS

## 6. Frozen bootstrap execution

- Triggered after Gates 1–4: NO.
- Bootstrap outcomes and any structured failures are retained in the corresponding `outputs/CHM_D_*BOOTSTRAP*.csv` files.

## 7. Gate 5

- Gate_5: FAIL.
- This gate uses the preregistered held-out coverage rule; no Test information was used.

## 8. Primary Development qualification

- Status: `DEVELOPMENT_HEALTH_MODEL_NOT_QUALIFIED`.
- The status follows Gates 1–5 only. S1 cannot overturn primary qualification.

## 9. Final all-six latent trajectory

- This is a Development-only Cycle-Aligned Latent Health State, not validated SOH.
- Final profile: `B_ANCHOR_EMPHASIS`; outer target information was not used for this choice.

## 10. S1 secondary robustness

- Status: `S1_ROBUSTNESS_SUPPORTED`.
- S1 uses the frozen reduced-anchor sensitivity and is not an independent validation set or a primary gate.

## 11. Causal-online diagnostic

- Only 1150/1500/1750 h have legal prefix fits; aggregate n=3 is secondary only.
- MAE/RMSE: 0.00278916 / 0.00288106.

## 12. Limitations

- Only six sparse Development anchors; PAA/PAV ancestry remains exploratory.
- Three of six median sources are inside full joint support.
- `LOCAL_WINDOW_SCATTER_PROXY` is not independent replicate variance.
- S1 is not an independent validation set; Test remains unopened.
- Latent health is not validated SOH and no RUL conclusion is made.
- No automatic Test phase is started by this result.
