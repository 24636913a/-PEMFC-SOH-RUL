# Initial State and Monotone Transition

## BoL is not a latent state

BoL polarization is used only as the denominator in the sparse target

\[
Y_k=V_{\mathrm{pol},k}(1.0\;\mathrm{A/cm^2}) /
V_{\mathrm{pol},\mathrm{BoL}}(1.0\;\mathrm{A/cm^2}).
\]

It has no corresponding P1S HOU. Therefore BoL is neither `H_0` nor the first
observable state.

## First observable state

The first actual HOU state is learned:

\[
H_1=\operatorname{sigmoid}(\eta),
\qquad \operatorname{sigmoid}(\eta_{\mathrm{init}})=0.995.
\]

Equivalently, `eta_init = logit(0.995)`. The value 0.995 is an optimization
initialization only; it is not a BoL SOH, a health label, or an extra sparse
anchor.

## Frozen transition

For later HOU in their immutable timestamp order,

\[
H_j=H_{j-1}\exp\left[-10^{-3}\operatorname{softplus}(g_\theta(X_j))\right].
\]

The fixed `10^-3` is a numerical parameterization constant, not a physical
degradation-rate threshold. Since the exponential factor lies in `(0,1]`, the
trajectory remains positive and non-increasing without hard clipping or a
separate monotonicity penalty. The transition is applied to complete and
partial HOU; the main trajectory report later selects only complete HOU.

Age, cycle index, DateTime, nominal checkpoint age, future operating data, and
future anchors are prohibited inputs. Timestamps are sorting/audit fields only.
