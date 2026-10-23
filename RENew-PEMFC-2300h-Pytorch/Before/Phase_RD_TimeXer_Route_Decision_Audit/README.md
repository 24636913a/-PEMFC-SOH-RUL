# Phase RD — TimeXer Route Decision Audit

Development-only, cache-first attribution audit of the saved RealHyFC TimeXer evidence. This phase performs no model training, architecture edit, hyperparameter search, checkpoint update, or Test access.

The audit reuses the frozen OPR decomposition and P1/P2 pointwise attribution caches, plus the predeclared TTD Fold 1 and Fold 3 case objects. Its purpose is to distinguish verified numerical facts from unidentifiable mechanism claims before any future research authorization.

Run from this directory:

`python scripts/run_route_audit.py`

Then validate the contracts:

`python -m unittest tests/test_route_audit.py`
