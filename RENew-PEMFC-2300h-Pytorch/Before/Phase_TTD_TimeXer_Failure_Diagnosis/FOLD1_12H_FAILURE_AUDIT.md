# Fold 1, 12 h failure audit

Object A contains 233 unique 12 h target times. P1 RMSE is 31.682133 mV and P2 RMSE is 41.296982 mV. Mean D is +701.683178 mV², median D is +0.016941 mV², and 50.64% of points have positive D.

Positive D totals 294553.388 mV² and negative D totals -131061.207 mV², giving net +163492.180 mV². The worst unique target at 1130.0 h alone contributes 274021.435 mV², or 93.03% of all positive D; it is a `mixed_to_nonzero` target. P2 predicts mixed probability 0.6906 and q=0.7311 while actual q=0, with P2 current 41.11 A versus actual 150.18 A. The top 5 targets capture 98.36% of positive D.

Therefore this object is dominated by two state-mismatch outliers, not by a distributed 12 h degradation failure. The remaining 231 S0-L points favor P2: P1/P2 RMSE 28.369/16.222 mV and net D -125122.3 mV².
