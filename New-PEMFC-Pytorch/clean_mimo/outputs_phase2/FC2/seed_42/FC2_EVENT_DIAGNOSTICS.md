# FC2 dynamic-event diagnostics

Scope: the held-out Phase-2 test origins only. `delta_U1_outcome` is the observed h=1 outcome used to describe event severity; it is never used as a feature, removed as an outlier, or used to select a model.

## Distribution of `Utot(t+1) - Utot(t)` (V)

| mean | std | min | max | P1 | P5 | P50 | P95 | P99 | median | MAD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -0.00016332 | 0.04079228 | -0.45686102 | 0.48623037 | -0.00648475 | -0.00325191 | -0.00022697 | 0.00171304 | 0.00417829 | -0.00022697 | 0.00068069 |

Top-1% absolute-change threshold: `0.03577893 V`; flagged origins: `4/391`.
Three-sigma reference threshold: `0.12237683 V` from the mean.

## Top-1% held-out events and origin-hour load context

| origin (h) | target (h) | ΔU1 outcome (V) | I mean (A) | I std (A) | I p2p (A) | ΔI mean (A) | ΔI std (A) | ΔI p2p (A) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 666.000000 | 667.000000 | 0.486230 | 70.061882 | 0.003346 | 0.015000 | 0.002017 | -0.000111 | 0.000000 |
| 667.000000 | 668.000000 | -0.456861 | 48.222500 | 30.730316 | 70.373000 | -21.839382 | 30.726970 | 70.358000 |
| 830.000000 | 831.000000 | 0.346462 | 70.084429 | 0.008903 | 0.049000 | -0.003321 | -0.001189 | 0.000000 |
| 831.000000 | 832.000000 | -0.290781 | 54.201020 | 27.041962 | 94.470000 | -15.883408 | 27.033059 | 94.421000 |

## Factual diagnostic reading

The top positive-change origins retain low within-hour current variability, so the ensuing recovery begins in the target interval rather than being fully announced by the origin-hour load descriptors.

The top negative-change origins have high within-hour current variability and large one-hour increases in current spread/range, consistent with load-transition coupling.

These observations describe the held-out outcomes; they do not establish causality and do not authorize removal of any event. Together they support treating FC2 raw-voltage forecasting as a mixture of slow degradation and reversible condition-linked variation, some of which is not fully predictable without future operating conditions.

The CSV preserves every held-out origin.  It includes observed-at-origin Utot, current descriptors, temperature, pressure, flow, humidity, and each variable's one-hour historical change so the large events can be inspected for condition-transition or recovery coupling without deleting them.
