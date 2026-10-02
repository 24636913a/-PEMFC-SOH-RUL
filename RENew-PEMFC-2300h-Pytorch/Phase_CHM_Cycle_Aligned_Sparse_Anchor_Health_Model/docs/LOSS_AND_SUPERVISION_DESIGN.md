# Loss and Sparse-Supervision Design

## Anchor supervision

All six PAV-qualified Development anchors are retained in the primary fit. For
an anchor-aligned cycle `a_k`, define

\[
L_{\mathrm{anchor}} = \sum_k \widetilde w_k (H_{a_k}-Y_k)^2.
\]

Weights are not hand tuned by checkpoint and cannot use model residuals. The
pre-execution estimator is

\[
\sigma_k^2 = \sigma_{\mathrm{pol},k}^2 + \sigma_{\mathrm{cycle},k}^2,
\quad
w_k=(\sigma_k^2+\epsilon)^{-1},
\quad
\widetilde w_k=w_k/\operatorname{mean}(w).
\]

`σ_pol,k` must be derived only from the checkpoint's frozen 1.0 A/cm²
polarization repeat-voltage dispersion. `σ_cycle,k` must be the pre-frozen
robust spread of the three already selected PAA P1S retention intervals. If one
component cannot be computed from the allowed Development artifact, the
execution protocol must fail closed rather than invent a precision value. The
exact robust estimator, epsilon, and any unit conversion are open technical
decisions and must be frozen before the first fit.

## Dense-observation consistency

The model must not learn from only six points. It uses a deliberately low
freedom affine calibration map,

\[
\phi(R)=aR+b,
\qquad
L_{\mathrm{obs}}=\sum_c \rho\{H_c-\phi(R_c)\},
\]

where `ρ` is a pre-frozen robust loss. No deep calibration MLP is permitted.
The model formulation should constrain the direction of `a` consistently with
the representation before execution; it must not learn an unrestricted lookup
table from `R_c` to `H_c`.

## Smoothness and total loss

\[
L_{\mathrm{smooth}}=\sum_{c\ge2}(H_c-2H_{c-1}+H_{c-2})^2,
\]

\[
L=\lambda_{\mathrm{anchor}}L_{\mathrm{anchor}}
+\lambda_{\mathrm{obs}}L_{\mathrm{obs}}
+\lambda_{\mathrm{smooth}}L_{\mathrm{smooth}}.
\]

Monotonicity comes only from the softplus decrement in the transition. A
separate monotonicity penalty and hard clipping are forbidden.

The non-executed candidate grid is:

| Component | Candidate values |
| --- | --- |
| `λ_anchor` | 1, 3, 10 |
| `λ_obs` | 0.3, 1, 3 |
| `λ_smooth` | 0.01, 0.1, 1 |

This is not an authorization for a grid search. The final finite grid and its
Development-only selection rule remain open until technical review.

## Registered sensitivity S1

Sensitivity S1 uses only the three anchors whose final frozen median source is
inside joint common support: 300, 850, and 1750 h. It cannot replace the
primary all-six-anchor result and cannot be used to select a model. Its sole
purpose is to test whether the conclusion depends entirely on the three
boundary-qualified median sources at 600, 1150, and 1500 h.
