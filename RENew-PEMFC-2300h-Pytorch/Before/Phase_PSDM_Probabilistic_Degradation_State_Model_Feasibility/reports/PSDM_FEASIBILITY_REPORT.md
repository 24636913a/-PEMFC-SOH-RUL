# PSDM feasibility — Development-only probabilistic degradation state model

## Decision

**PSDM_FEASIBILITY_NOT_ESTABLISHED**
Physical-rate constraint: **PHYSICAL_RATE_CONSTRAINT_NOT_YET_SUPPORTED**.

The model state is named a **latent degradation health state**, not true SOH.  PSDM constrains expected transition degradation rate through `softplus(z)>0`; filtering updates may nevertheless yield local posterior corrections.  Its predictive distribution is a Gaussian EKF approximation, not an exact Bayesian posterior.

## Data boundary and prior incident

The historical pre-training PSDM attempt accessed a mixed N1 table; it produced no model results and no Git changes.  This current run retained `historical_mixed_n1_access=true`, but used the committed Stage-A 12-row anchor snapshot and did not read raw polarization, mixed N1, Test, or late-life target data.  Raw F2 was not used.  Causal P1S count: 131.

A post-fit reporting aggregation defect was corrected from the fixed on-disk result files; this recovery did not call an optimizer, filter fitting routine, or model-evaluation routine.

## Primary H6 evaluation

| fold | model | n_validation | MAE_mV_equivalent | RMSE_mV_equivalent | Gaussian_CRPS | Gaussian_NLL | coverage_80 | coverage_95 | mean_width_95 |
|---|---|---|---|---|---|---|---|---|---|
| fold_1 | PERSISTENCE | 18 | 1.05101 | 1.22144 |  |  |  |  |  |
| fold_1 | ULLT | 18 | 1.64842 | 2.61686 |  |  |  |  |  |
| fold_1 | PSDM | 18 | 0.998118 | 1.08625 |  |  |  |  |  |
| fold_1 | GAUSSIAN_PERSISTENCE | 18 |  |  | 0.00100037 | -4.93481 | 0.666667 | 0.944444 | 0.00592246 |
| fold_1 | ULLT | 18 |  |  | 0.00195189 | -4.14161 | 0.944444 | 0.944444 | 0.0182086 |
| fold_1 | PSDM | 18 |  |  | 0.000971248 | -4.8078 | 0.5 | 0.666667 | 0.00388949 |
| fold_2 | PERSISTENCE | 9 | 0.382692 | 0.413242 |  |  |  |  |  |
| fold_2 | ULLT | 9 | 1.09896 | 1.89177 |  |  |  |  |  |
| fold_2 | PSDM | 9 | 0.693926 | 0.802146 |  |  |  |  |  |
| fold_2 | GAUSSIAN_PERSISTENCE | 9 |  |  | 0.000430248 | -5.53353 | 1 | 1 | 0.00571111 |
| fold_2 | ULLT | 9 |  |  | 0.00146241 | -4.45063 | 0.888889 | 1 | 0.0139723 |
| fold_2 | PSDM | 9 |  |  | 0.000638337 | -5.41507 | 0.777778 | 1 | 0.00407792 |
| fold_3 | PERSISTENCE | 18 | 0.378606 | 0.529658 |  |  |  |  |  |
| fold_3 | ULLT | 18 | 0.796334 | 1.2304 |  |  |  |  |  |
| fold_3 | PSDM | 18 | 0.430633 | 0.450611 |  |  |  |  |  |
| fold_3 | GAUSSIAN_PERSISTENCE | 18 |  |  | 0.000465103 | -5.53817 | 0.944444 | 1 | 0.00528812 |
| fold_3 | ULLT | 18 |  |  | 0.00116078 | -4.56696 | 0.944444 | 1 | 0.0146152 |
| fold_3 | PSDM | 18 |  |  | 0.00038561 | -5.7946 | 1 | 1 | 0.00388737 |

Equal-fold averaging is primary; pooled values are descriptive only.  H3 is secondary.

| fold | model | n_validation | MAE_mV_equivalent | Gaussian_CRPS | coverage_95 |
|---|---|---|---|---|---|
| fold_1 | PERSISTENCE | 18 | 0.620547 |  |  |
| fold_1 | ULLT | 18 | 0.710421 |  |  |
| fold_1 | PSDM | 18 | 0.829595 |  |  |
| fold_1 | GAUSSIAN_PERSISTENCE | 18 |  | 0.000585519 | 1 |
| fold_1 | ULLT | 18 |  | 0.000828049 | 0.944444 |
| fold_1 | PSDM | 18 |  | 0.000781815 | 1 |
| fold_2 | PERSISTENCE | 12 | 0.208058 |  |  |
| fold_2 | ULLT | 12 | 0.512045 |  |  |
| fold_2 | PSDM | 12 | 0.556371 |  |  |
| fold_2 | GAUSSIAN_PERSISTENCE | 12 |  | 0.00028802 | 1 |
| fold_2 | ULLT | 12 |  | 0.000662881 | 0.916667 |
| fold_2 | PSDM | 12 |  | 0.000526487 | 0.916667 |
| fold_3 | PERSISTENCE | 24 | 0.270291 |  |  |
| fold_3 | ULLT | 24 | 0.389062 |  |  |
| fold_3 | PSDM | 24 | 0.360803 |  |  |
| fold_3 | GAUSSIAN_PERSISTENCE | 24 |  | 0.000318354 | 1 |
| fold_3 | ULLT | 24 |  | 0.000526345 | 1 |
| fold_3 | PSDM | 24 |  | 0.000351652 | 1 |

## Parameters

| horizon | fold | model | q | r | initial_rate_physical_per_raw_cycle | training_NLL | optimizer_success | optimizer_iterations | selected_deterministic_start |
|---|---|---|---|---|---|---|---|---|---|
| H3 | fold_1 | ULLT | 2.23264e-07 | 1.92368e-08 | -0.000563963 | -390.06 | True | 24 | 2 |
| H3 | fold_1 | PSDM | 0.00673795 | 8.12231e-07 | 0.000229108 | -355.001 | True | 22 | 2 |
| H3 | fold_2 | ULLT | 1.10661e-07 | 7.12944e-08 | -0.00018626 | -500.403 | True | 18 | 0 |
| H3 | fold_2 | PSDM | 0.00673795 | 8.42508e-07 | 0.000230097 | -457.363 | True | 25 | 2 |
| H3 | fold_3 | ULLT | 1.27242e-07 | 5.63666e-08 | -0.000287103 | -591.96 | True | 23 | 0 |
| H3 | fold_3 | PSDM | 0.00673795 | 8.81232e-07 | 0.000229899 | -536.714 | True | 45 | 0 |
| H6 | fold_1 | ULLT | 2.23264e-07 | 1.92368e-08 | -0.000563963 | -390.06 | True | 24 | 2 |
| H6 | fold_1 | PSDM | 0.00673795 | 8.12231e-07 | 0.000229108 | -355.001 | True | 22 | 2 |
| H6 | fold_2 | ULLT | 1.10661e-07 | 7.12944e-08 | -0.00018626 | -500.403 | True | 18 | 0 |
| H6 | fold_2 | PSDM | 0.00673795 | 8.42508e-07 | 0.000230097 | -457.363 | True | 25 | 2 |
| H6 | fold_3 | ULLT | 1.27242e-07 | 5.63666e-08 | -0.000287103 | -591.96 | True | 23 | 0 |
| H6 | fold_3 | PSDM | 0.00673795 | 8.81232e-07 | 0.000229899 | -536.714 | True | 45 | 0 |

## Gates

* A point prediction: **FAIL**
* B probabilistic CRPS: **FAIL**
* C 95% coverage: **PASS**
* D 80% coverage: **PASS**
* E Fold 3 CRPS: **PASS**

## Post-hoc Development anchor description

| jstar_A_cm2 | comparison | Spearman_rho | n | POSTHOC_DESCRIPTIVE_ONLY |
|---|---|---|---|---|
| 1 | Raw_P1S_anchor_vs_polarization | 0.942857 | 6 | True |
| 1 | PSDM_filtered_H_vs_polarization | 0.942857 | 6 | True |
| 1.6 | Raw_P1S_anchor_vs_polarization | 0.828571 | 6 | True |
| 1.6 | PSDM_filtered_H_vs_polarization | 0.828571 | 6 | True |

These six-anchor Spearman values are descriptive only: they were not used for parameter fitting, model choice, or gates.  No EOL threshold or RUL result is defined.  The phase stops here.
