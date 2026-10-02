# Model Formulation

## Estimand and cycle unit

The proposed estimand is a **Cycle-Aligned Latent Health State**, denoted
`H_c`, for each of the 146 RCR-frozen complete cycles. It is not called SOH in
Development and is not an RUL target.

The cycle identity, cycle boundaries, and P1S observation are inherited without
modification. In particular, the sole P1S state is `operation_point=221`,
`I.S.set=221.200 A`, token `221||221.200`; P1S uses the middle-50% stable
dwell, the active ten cells, and the earliest ten legal complete cycles as its
fixed reference. Let the resulting cycle observation be `R_c` (P1S
operating-voltage retention). `R_c` is a dense degradation-related observation;
`R_c ≠ SOH_c`.

## Primary network

The one proposed primary model is the small **Monotone Cycle Latent-State
Network**:

\[
H_0 = 1, \qquad
H_c = H_{c-1} - s\,\operatorname{softplus}(g_\theta(X_c)).
\]

`s` is a fixed positive step scale to be frozen only after technical review.
The structural decrement, rather than clipping or an extra penalty, guarantees
that a transition cannot increase `H_c`. `g_θ` has at most two MLP layers,
hidden width at most 32, and a target of fewer than 10,000 parameters. No
Transformer, xLSTM, Mamba, KAN, large attention module, or graph model is in
scope.

## Inputs

For cycle `c`, `X_c` contains only information available by the end of that
cycle:

1. `R_c` and the P1S voltage difference from its frozen reference;
2. the fixed seven P1S stable-dwell conditions: current density, anode inlet
   temperature, anode inlet pressure, anode inlet RH, hydrogen flow, anode
   stoichiometry, and cathode stoichiometry;
3. each of those seven conditions relative to the frozen reference-condition
   median; and
4. the previous latent state through the recurrence. Intra-cycle P1S
   variability remains optional until explicitly frozen before execution.

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
- **Causal online mode** estimates `H_c` using only `X_1, …, X_c` and anchors
  available no later than cycle `c`. Its evaluation uses chronological prefix
  anchor holdouts; BoL is an allowed initial reference, but a held-out or later
  non-BoL anchor is never used. This is the primary claim hierarchy.

Both modes forbid future operating data in the recurrence. Reports must label
their mode and may not merge their scores.

## Sparse anchor target

At each PAV-qualified Development checkpoint `k`, the sparse target is

\[
Y_k = V_{\mathrm{pol},k}(1.0\;\mathrm{A/cm^2}) /
      V_{\mathrm{pol},\mathrm{BoL}}(1.0\;\mathrm{A/cm^2}).
\]

The six non-BoL anchors (300, 600, 850, 1150, 1500, 1750 h) use frozen PAA/PAV
pre-characterization alignment. They are sparse supervision, never a dense
ground-truth trajectory. PAV verified their temporal order 6/6; it also found
that only 3/6 final median-producing intervals are directly inside complete
cycle joint common support. That limitation is represented through
uncertainty-aware supervision, not retrospective interval removal.
