# Frozen Baselines

All baselines use the same frozen anchor functional as the primary model. For
checkpoint `k`, each baseline receives the three PAA-selected HOU observations
and returns an anchor-level prediction. No baseline substitutes a nearest
complete cycle for a partial HOU.

## B0 — raw P1S anchor functional

\[
\widehat{Y}^{B0}_k=\operatorname{median}(R_{k,1},R_{k,2},R_{k,3}).
\]

There is no fitted age trend, no cycle index, and no hidden age covariate.

## B1 — isotonic calibration

Use `sklearn.isotonic.IsotonicRegression(increasing=True,
out_of_bounds='clip')`. Fit only training anchors' P1S three-interval median to
their polarization target `Y`. The held-out anchor target is unavailable.

## B2 — nonnegative affine calibration

Fit the deterministic constrained least-squares relation

\[
Y=aR+b, \qquad a\ge0,
\]

on training anchor-level P1S medians. The nonnegative slope preserves the
retention-to-health direction. A spline, polynomial, or another baseline cannot
be substituted after results are seen.

All B0/B1/B2 point comparisons use the same outer LOAO checkpoints and same
anchor functional as CHM. Baselines do not receive age, a held-out anchor, or
Test data.
