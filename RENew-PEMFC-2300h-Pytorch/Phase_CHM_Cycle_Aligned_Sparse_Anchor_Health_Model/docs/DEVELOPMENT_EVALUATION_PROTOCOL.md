# Development Evaluation Protocol

## Baselines

Three low-complexity baselines are mandatory; none may be replaced by a modern
backbone.

- **B0 — raw P1S anchor functional:** the median of the same three frozen PAA
  interval retentions at every anchor.
- **B1 — isotonic calibration:**
  `sklearn.isotonic.IsotonicRegression(increasing=True, out_of_bounds='clip')`
  on training anchor-level P1S medians only.
- **B2 — nonnegative affine calibration:** deterministic constrained least
  squares `Y=aR+b`, `a≥0`, on the same training anchor-level representation.

No baseline receives age, cycle index, or a held-out anchor value.

## Two non-random chronological evaluations

1. **LOAO retrospective diagnostic.** Hold out each non-BoL anchor at 300,
   600, 850, 1150, 1500, and 1750 h in turn. Train only on the other five
   non-BoL Development anchors plus the fixed BoL reference. The held-out
   polarization value cannot enter training, lambda selection, early stopping,
   anchor-noise tuning, or any calibration decision. For each outer fold, use
   nested LOAO across the five permitted anchors to select among the frozen 27
   lambda combinations by three-seed mean inner MAE; ties within `1e-8` follow
   protocol lambda-list order.
2. **Causal-online chronological-prefix diagnostic.** For anchor `k`, use
   only the fixed BoL reference and non-BoL anchors strictly earlier than `k`.
   The held-out anchor and every later anchor are unavailable. 300 h is cold
   start; 600/850 h are insufficient for prefix selection. Only 1150/1500/1750
   form the secondary `n=3` online aggregate.

The state recurrence is causal in both tracks. The retrospective LOAO score is
not presented as online deployment performance.

## Metrics and reporting

Each held-out checkpoint must be reported individually. The minimum report is:

1. held-out prediction, target, signed/absolute/squared error, and 95%
   interval plus inclusion flag for every checkpoint;
2. aggregate MAE, RMSE, bias, descriptive Spearman rho, calibration slope, and
   calibration intercept across exactly six held-out predictions;
3. latent monotonic-violation count and complete-cycle roughness; and
4. small-sample 5/6 uncertainty coverage gate.

Per-anchor Spearman is prohibited. Mean-only tables are prohibited, and no
success claim may rely on aggregate Spearman alone.

## Development qualification gate

The primary model may receive `DEVELOPMENT_HEALTH_MODEL_QUALIFIED` only when
all of the following are met using frozen Development protocol choices:

1. all six LOAO anchors have legal predictions;
2. aggregate LOAO MAE is better than B0;
3. it is no worse than the best interpretable monotone baseline;
4. there is no structural rise in latent health;
5. at least 5/6 held-out anchors are inside their 95% interval; and
6. the three quantitative S1 no-reversal thresholds all pass.

A developmental PASS does not open Test and does not authorize RUL.
