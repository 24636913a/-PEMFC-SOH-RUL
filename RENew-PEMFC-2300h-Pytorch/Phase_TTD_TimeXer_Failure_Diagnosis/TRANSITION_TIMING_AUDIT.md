# Transition timing audit

State is actual mixed if q>0 and predicted mixed if frozen P2 probability>=0.5. Same-origin, same-type transitions were matched one-to-one within ±2 steps, with ties resolved to the earlier predicted transition.

Across folds, matching is near exact: F1 mixed-to-nonmixed has 1171 matches, 0 false positives and 1 false negative; F1 nonmixed-to-mixed has 1172 matches, 1 false positive and 0 false negatives. F2 and F3 are similarly near exact; F3 has 19/2342 early-by-one mixed-to-nonmixed matches and 52/2340 early-by-one nonmixed-to-mixed matches. No matched event is early/late by two steps in the retained summary.

Shift diagnostics give their minimum current MAE and state mismatch at zero shift in each fold (F1 1.865 A/0.447%, F2 1.884 A/0.051%, F3 2.377 A/0.706%). Shifting by ±1 raises current MAE to roughly 96–99 A and state mismatch to about 41–42%. Thus a global index shift or transition-timing error is not supported.
