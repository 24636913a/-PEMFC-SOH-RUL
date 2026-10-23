# Method and evidence labels

## Frozen project evidence

Phase HA uses the first-two-IMF-removed EMD remainder exactly as frozen in
Phase DF: a 48-point, origin-conditioned voltage history is standardized by its
own mean and `max(std, 0.001 V)`, then the first at most two IMFs are removed.
The cached endpoint is feature position `[47, 0]`; it is not a global
timestamp-level reconstruction.  Phase DF source hashes and cache provenance
are checked before the audit starts.

The EMD endpoint field requested by this phase is named `EMD_endpoint_V` for
traceability with the inherited feature dictionary.  Its numerical quantity is
the standardized EMD remainder, hence it is dimensionless rather than an
absolute voltage in volts.  Absolute-voltage comparisons are only made within
Raw and SimpleTrend representations; cross-representation stability comparisons
use unitless ratios or within-representation standardized residual quantities.

## Literature-supported methods

Empirical mode decomposition is an adaptive data analysis method that separates
a signal into intrinsic mode functions.  The originating description is Huang
et al., *The empirical mode decomposition and the Hilbert spectrum for nonlinear
and non-stationary time series analysis*, Proc. R. Soc. A (1998),
[DOI:10.1098/rspa.1998.0193](https://doi.org/10.1098/rspa.1998.0193).

Pearson, Spearman, Kendall, ordinary least squares, and Theil--Sen slopes are
reported as descriptive trend summaries.  Theil--Sen is the robust median of
pairwise slopes (Sen, 1968, *JASA*, 63, 1379--1389).  None of these methods by
themselves establishes a physical degradation mechanism or an SOH scale.

## Project analysis design

The 24 h/48 h non-overlapping blocks, current-density P33/P67 strata,
common-support threshold, block bootstrap, ASR, RDR, and final gates are
**PROJECT_ANALYSIS_DESIGN**.  They are transparent decision aids for this
single Development trajectory, not universal PHM standards.

