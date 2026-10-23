# Model Formulation

## Estimand and cycle unit

The proposed estimand is a **Cycle-Aligned Latent Health State**, denoted
`H_j`, on the timestamp-sorted Health Observation Unit (HOU) sequence. The
primary reported trajectory selects the 146 RCR-frozen complete-cycle HOU; six
frozen pre-anchor partial HOU are auxiliary states only. It is not called SOH
in Development and is not an RUL target.

The complete-cycle identity, cycle boundaries, and P1S observation are inherited
without modification. In particular, the sole P1S state is `operation_point=221`,
`I.S.set=221.200 A`, token `221||221.200`; P1S uses the middle-50% stable
dwell, the active ten cells, and the earliest ten legal complete cycles as its
fixed reference. Let the resulting cycle observation be `R_c` (P1S
operating-voltage retention). `R_c` is a dense degradation-related observation;
`R_c ≠ SOH_c`.

## Primary network

The one proposed primary model is the small **Monotone Cycle Latent-State
Network**. BoL is a target-normalization reference and is not a HOU. The first
observable state and the HOU transition are:

\[
H_1=\operatorname{sigmoid}(\eta), \qquad
H_j=H_{j-1}\exp\left[-10^{-3}\operatorname{softplus}(g_\theta(X_j))\right].
\]

`sigmoid(eta)` initializes to 0.995 solely for numerical optimization, never as
a BoL health label. The `10^-3` transition scale is a frozen numerical
parameterization constant, not a physical threshold. The exponential factor,
rather than clipping or an extra penalty, guarantees positive non-increasing
states. `g_θ` is exactly `Linear(d,16) → SiLU → Linear(16,1)`; a condition
residual head and every large backbone are excluded.

## Inputs

For cycle `c`, `X_c` contains only information available by the end of that
cycle:

1. `R_j-1`, the P1S retention deviation from its frozen reference;
2. the seven P1S conditions standardized by the earliest-ten-complete-cycle
   reference median and IQR (reference standard deviation only when IQR is
   zero); and
3. the previous latent state through the recurrence.

When both reference IQR and standard deviation are zero, the condition becomes
`CONSTANT_REFERENCE_FEATURE`, is audited, set to zero, and excluded. Raw
conditions and their duplicate deviations are not both input.

`OpHrs2`, cycle index, absolute `DateTime`, all future-cycle values, future
polarization anchors, and every Test artifact are prohibited inputs. Age may
appear only as a plot axis, split coordinate, or retrospective diagnostic.

## Retrospective and causal-online modes

These modes must remain distinct.

- **Offline retrospective mode** fits model parameters on its permitted
  Development anchors and reconstructs the historical sequence with only
  prefix `X_1, …, X_c` inside the state recurrence. Its Leave-One-Anchor-Out
  (LOAO) analysis is a sparse-label generalization diagnostic, not a
  deployment-causality claim, because fitting a held-out early anchor can
  otherwise use later Development anchors.
- **Causal online mode** estimates `H_j` using only HOU inputs and anchors
  available no later than the target. 300 h is cold start; 600/850 h have too
  few prior non-BoL anchors for prefix model selection. Only 1150/1500/1750 h
  enter its `n=3` secondary aggregate, and no later anchor is ever used.

Both modes forbid future operating data in the recurrence. Reports must label
their mode and may not merge their scores.

## Sparse anchor target

At each PAV-qualified Development checkpoint `k`, the sparse target is

\[
Y_k = V_{\mathrm{pol},k}(1.0\;\mathrm{A/cm^2}) /
      V_{\mathrm{pol},\mathrm{BoL}}(1.0\;\mathrm{A/cm^2}).
\]

The six non-BoL anchors (300, 600, 850, 1150, 1500, 1750 h) use frozen PAA/PAV
pre-characterization alignment. For each checkpoint, prediction is the median
of the three frozen selected HOU states, including frozen partial HOU where
selected. They are sparse supervision, never a dense ground-truth trajectory.
PAV verified temporal order 6/6 and found only 3/6 final median sources inside
complete-cycle joint support. That limitation is represented through
uncertainty-aware supervision, not retrospective interval removal.
