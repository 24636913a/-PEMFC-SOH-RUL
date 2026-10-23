# Next-phase decision

The preregistered GO rule requires at least 5 percent q RMSE reduction in two or more folds, no more than 5 percent Fold 3 worsening, non-worse estimable Brier in two or more folds, and at least ten independent validation mixed-state events. The rule was frozen in configs/frozen_protocol.yaml before metrics were generated.

## Decision

**NO_GO_KEEP_30MIN**

Comparable evaluation was valid but frozen GO rule failed: q RMSE improvement >=5% in 3/3 folds; Fold 3 change 31.9%; non-worse Brier in 0 folds; independent mixed-state events on the common 30-minute target scale 252.

This is a resolution-audit decision only. It does not train or claim a better TimeXer.

## Required answers

**Q1.** Yes for expression: 21.0% of valid 30-minute bins contain more than one valid 10-minute q state, so 10 minutes reveals genuine within-bin transitions. This is not automatically a predictive gain.

**Q2.** Partly. A fixed past-only Ridge reduces unified q error, but the fixed mixed-state Logistic probabilities deteriorate. The finer state is therefore not reliably forecastable under every required metric.

**Q3.** Unified q RMSE improves by Fold 1 44.3%, Fold 2 55.3%, Fold 3 31.9%; however mixed-state Brier worsens by 197.0%, 126.3%, and 55.2% respectively. Hence the fair composite answer is no.

**Q4.** q-RMSE gains are consistent, including Fold 3, but Brier deterioration is also consistent (0/3 non-worse folds). Common-scale independent mixed-event support is 252, so lack of support is not the reason for the negative decision.

**Q5.** No. Keep the frozen 30-minute representation for the next authorized model phase; do not start a 10-minute TimeXer experiment from this audit.
