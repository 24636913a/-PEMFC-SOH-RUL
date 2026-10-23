# Fold 3 continuous-nonzero audit

Object B follows the historical OPR pointwise definition: target rows are `regime_transition=continuous_nonzero`, not an entire trajectory label. It contains 6502 prediction points, 328 unique target times and four temporal segments. P1/P2 RMSE is 26.684831/32.812165 mV; D has mean +364.557952 mV², median -0.235172 mV², and positive fraction 47.17%.

For independent target times, the pre-specified de-duplication is the shortest available horizon per target. It yields 328 targets and P1/P2 RMSE 29.776766/49.117095 mV. This is a diagnostic de-duplication only; the original pointwise result remains official.

The four segment-level records are in `outputs/fold3_nonzero/fold3_independent_segments.csv`. Segment 15 contributes +1.313 million net D mV², segments 16 and 13 also contribute positive net D, while segment 14 is slightly net favorable to P2. This is a distributed segment pattern, unlike Fold 1's one-target concentration.
