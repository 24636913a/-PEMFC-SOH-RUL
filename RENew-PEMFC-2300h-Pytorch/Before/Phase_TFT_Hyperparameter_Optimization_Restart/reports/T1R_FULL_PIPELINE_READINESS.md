# T1-R full pipeline readiness

`T1R_REAL_SMOKE_TEST_PASS` means code and isolated synthetic acceptance checks are ready for a separately authorized Development-only execution. It does not report training, Optuna, B/C evaluation, model performance, or Test use.

Future command (not executed): `python scripts/run_t1r.py --mode execute --authorization EXECUTE_T1R_SEPARATE_TRAINING` after a separately audited authorization file is created.
