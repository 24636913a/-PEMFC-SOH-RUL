# Frozen Model, Inputs, Loss, and Optimization

## Inputs and scaler audit

The network receives only the retention deviation `R_j-1` and standardized
deviations of the seven frozen P1S conditions. For each condition `x`, the
reference median and IQR are calculated from the earliest ten legal complete
cycles:

\[
z=(x-\operatorname{median}_{ref})/s_{ref}.
\]

`s_ref` is IQR; if IQR is zero it is the reference standard deviation. If both
are zero, the feature is `CONSTANT_REFERENCE_FEATURE`, is fixed at zero, and is
excluded from the network. Future code must write `REFERENCE_SCALER_AUDIT.json`
with included and constant features. This is expected to audit, rather than
silently remove, possibly constant P1S current density.

## Network and observation loss

The only primary network is

```text
input → Linear(d,16) → SiLU → Linear(16,1).
```

There is one hidden layer of width 16 and no condition residual head. The dense
observation calibration is

\[
\phi(R)=\operatorname{softplus}(a_{raw})R+b,
\qquad
L_{obs}=\sum_j\operatorname{Huber}_{0.001}(H_j-\phi(R_j)).
\]

The positive slope prevents higher retention from mapping to lower health.
`Huber delta=0.001` is frozen before every CHM fit and was chosen as several
times the prior P1/P1S cycle-local variation scale, not from CHM performance.

## Smoothness and total objective

`L_smooth` is calculated only for triples of complete-cycle HOU whose
`raw_cycle_id` are consecutive. It never crosses a raw-cycle gap and never
forces a second-difference relation through a partial HOU.

\[
L=\lambda_{anchor}L_{anchor}+\lambda_{obs}L_{obs}
+\lambda_{smooth}L_{smooth}.
\]

The frozen grid has 27 combinations: `λ_anchor ∈ {1,3,10}`,
`λ_obs ∈ {0.3,1,3}`, and `λ_smooth ∈ {0.01,0.1,1}`. Optuna, continuous search,
and Test-driven changes are prohibited.

## Optimization and seeds

Each future fit uses AdamW (`lr=1e-3`, weight decay `1e-4`), full-batch HOU
sequence optimization, gradient-clip norm 1.0, maximum 5000 epochs, and at
least 500 epochs. The only seeds are 42, 2024, and 3407.

Every 200 epochs after the minimum epoch, early stopping compares only the best
training composite loss. If relative improvement is below `1e-5`, it stops.
Held-out anchors and Test data are never monitored. Inner selection averages
MAE over the three seeds; an outer point prediction is their median. No seed
may be deleted after results are observed.
