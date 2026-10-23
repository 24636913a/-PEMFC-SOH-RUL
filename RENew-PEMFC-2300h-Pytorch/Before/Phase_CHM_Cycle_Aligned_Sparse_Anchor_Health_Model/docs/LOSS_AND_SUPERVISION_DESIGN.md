# Loss and Sparse-Supervision Design

> Candidate V1 supersedes the original draft notation below: every anchor
> prediction is the median of three frozen HOU states, and all numeric settings
> are frozen in `PREREGISTERED_CHM_PROTOCOL_CANDIDATE_V1.yaml`.

## Anchor supervision

All six PAV-qualified Development anchors are retained in the primary fit. For
an anchor-aligned cycle `a_k`, define

\[
\widehat{Y}_k=\operatorname{median}(H_{k,1},H_{k,2},H_{k,3}),
\qquad L_{\mathrm{anchor}} = \sum_k \widetilde w_k (\widehat{Y}_k-Y_k)^2.
\]

Weights are not hand tuned by checkpoint and cannot use model residuals. The
pre-execution estimator is

\[
\sigma_k^2 = \sigma_{\mathrm{pol},k}^2 + \sigma_{\mathrm{int},k}^2 + \sigma_{\mathrm{support},k}^2,
\quad
w_k=(\sigma_k^2+\epsilon)^{-1},
\quad
\widetilde w_k=w_k/\operatorname{mean}(w).
\]

`σ_pol,k` is the sample SD of frozen 1.0 A/cm² polarization repeats divided by
BoL voltage; `σ_int,k=1.4826 MAD` of the three frozen PAA retentions; and
`σ_support,k` is the all-three versus joint-support median difference. A missing
component or fewer than two polarization repeats fails closed. Epsilon is
`1e-8`; weights normalize to mean one across training anchors only.

## Dense-observation consistency

The model must not learn from only six points. It uses a deliberately low
freedom affine calibration map,

\[
\phi(R)=\operatorname{softplus}(a_{raw})R+b,
\qquad
L_{\mathrm{obs}}=\sum_c \rho\{H_c-\phi(R_c)\},
\]

where `ρ` is Huber loss with frozen delta `0.001`. No deep calibration MLP is permitted.
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

The frozen grid is:

| Component | Candidate values |
| --- | --- |
| `λ_anchor` | 1, 3, 10 |
| `λ_obs` | 0.3, 1, 3 |
| `λ_smooth` | 0.01, 0.1, 1 |

This is not authorization for an unconstrained search. Candidate V1 uses nested
inner LOAO over outer-training anchors and deterministic tie breaking; it still
requires final protocol review before any fit.

## Registered sensitivity S1

Sensitivity S1 uses only the three anchors whose final frozen median source is
inside joint common support: 300, 850, and 1750 h. It cannot replace the
primary all-six-anchor result and cannot be used to select a model. Its sole
purpose is to test whether the conclusion depends entirely on the three
boundary-qualified median sources at 600, 1150, and 1500 h.
