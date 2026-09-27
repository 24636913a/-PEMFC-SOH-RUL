# Phase H0 — Existing Artifact Comparability Audit

This is a cache-only, Development-only alignment and protocol-comparability audit of already saved RealHyFC TimeXer predictions. It never trains, fine-tunes, instantiates, or forwards a model; never loads raw data or a data builder; and never reads Test rows, Test predictions, or Test performance.

Run from this directory:

`python scripts/run_existing_h0_audit.py`

Validate the fail-closed contracts:

`python -m unittest tests/test_existing_h0_audit.py`
