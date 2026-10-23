# P1S-B2 Health Indicator Consolidation

## Health-indicator definition

`HI_P1S = 1.1988114938324186 × R_P1S -0.2090362463118213`, where
`R_P1S = V_P1S / 0.71383023635`.  This is the frozen Raw P1S-B2 calibrated
health indicator: a densely observable, cycle-aligned degradation-related
health proxy, not independently validated true SOH or a validated RUL state.
The `middle_50` P1S stable window was used because it exactly reproduces the
complete-cycle values used in the frozen Development anchor alignment.

## Coverage and trajectory

The source has 141 unique legal complete P1S cycles from
86.11 h to 1712.79 h.  All lie below the
1819.997738 h Development boundary; therefore the cycle-level
late-life event count is zero.  The 2000 h and 2300 h points in figures and
validation are immutable, already-unblinded ELTE checkpoint evidence, not new
P1S observations, not a reopened target, and not an independent Test.

## Trend construction

Raw cycle observations are preserved as `HI_RAW`.  Three predeclared trend
representations were generated: light Savitzky–Golay smoothing, decreasing
isotonic regression, and a fixed-bin monotonic PCHIP spline.  The primary
trend is decreasing isotonic regression because it is the simplest method with
zero upward violations, uses no late-life target criterion, and leaves the raw
indicator unchanged.  It should not be interpreted as a replacement for raw
cycle measurements.

## Uncertainty description

`sigma_measurement` is the sample standard deviation across the two frozen
per-cycle stable-window estimates (`middle_50` and `last_25`), i.e. a
window-sensitivity / measurement-variability proxy.  `sigma_calibration` is
the standard deviation of the six pre-existing leave-one-Development-anchor-out
B2 functions at each retention.  The displayed interval is raw HI ± 1.96 times
their quadrature total.  It is a descriptive calibration-plus-measurement
interval, not a probability-coverage guarantee.

## Polarization consistency

The six Development anchor MAE is 0.2160 mV.
The two already-unblinded exploratory late-life checkpoints have MAE
0.1398 mV.  The late-life result remains exploratory and
same-stack only: 2000 h is out of frozen joint support and 2300 h has 2/3
support-compatible intervals.  Neither was used to fit B2, choose a trend, or
change uncertainty settings.

## Answer and limitation

**YES.** Within its available data coverage, the frozen P1S-B2 mapping is
sufficiently stable and dense to serve as the primary degradation health
indicator for subsequent degradation-trajectory forecasting and
threshold-conditional RUL research.  This decision does not create true SOH
ground truth, does not define an EOL threshold, and does not solve the absence
of post-1819.997738 h cycle-level P1S observations.  Any later RUL work must
remain explicit about right censoring and threshold conditionality.

## No post-target retuning

No 2000 h/2300 h raw target file was read by PHI.  Existing ELTE output was
used only for descriptive validation.  B2, the reference voltage, support
ranges, trend choice, and uncertainty method were not retuned after ELTE.
