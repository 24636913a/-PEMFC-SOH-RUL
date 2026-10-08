# Phase POT — Model Implementation Audit

- Task: 48 half-hour causal steps → 24 future `cell_voltage_mean` ΔV steps.
- Formal population: 3005 Development rows obtained only by `pd.read_csv(..., nrows=3007)` followed by `bin_end_ophrs2_h < 1819.997738`.
- Test rows/targets accessed in this phase: **No**.
- M1/M2: read-only historical controls, revalidated at exact origin, target-row, and truth level.
- M3/M4: same frozen TimesNet architecture, train-only scaling, chronological inner stopping, seed reinitialisation for final outer-train refit.
- Parameters: Raw = 149,046; EMD = 149,304. EMD adds two causal origin-conditioned slow-varying features and is not interpreted as true SOH.
- M5/M6: not run by the Stage 0 gate, rather than replaced by an alternative physics penalty.

Current phase Test status is `CURRENT_PHASE_TEST_NOT_ACCESSED`. This does not erase the separate historical fact `HISTORICAL_TEST_CONTACT_EXISTS`.
