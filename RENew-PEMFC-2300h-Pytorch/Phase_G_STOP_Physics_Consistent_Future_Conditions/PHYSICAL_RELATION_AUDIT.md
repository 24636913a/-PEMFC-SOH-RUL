# Physical Relation Audit

Status: **PASS** (Development only; Test unopened).

The raw definition is `q = count(|I.S.act| <= 0.10 A) / raw sample count` in each causal 0.5 h bin. Population standard deviation (`ddof=0`) is used, and the audited Development raw rows contain no missing current values. All current, current-density, and q summaries were recomputed from the same bins.

Fold-1 Train estimates a fixed active area of **280.001183 cm²** from the median current/current-density ratio. The zero-intercept slopes for mean, standard deviation, and peak-to-peak are 280.000126, 279.999311, and 279.999869 cm². Their Pearson correlations are 0.999999905, 0.999999633, and 0.999999765; median relative identity errors are 3.889e-05, 1.008e-04, and 4.058e-06.

The verified relations are `J=I/A`, nonnegative current/statistical ranges, `0<=q<=1`, `mean_I=(1-q)mu_on`, and the population-mixture variance identity. The mixture mean and standard-deviation median absolute reconstruction errors are both numerically zero. Peak-to-peak is **not** derivable from mean and variance and remains independently predicted. No rule linking q to zero voltage is valid. There are no q=1 bins, so q=1 classification is **NOT_ESTIMABLE**.

The originally drafted absolute identity threshold was below source export quantization. Before any downstream inference, it was replaced by the preregistered slope/correlation/relative-error criterion documented in the plan. Raw Test rows parsed: **0**.
