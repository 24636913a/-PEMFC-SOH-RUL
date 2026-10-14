# 96h P1S Health-Aware TFT Dynamic Dual-Path Forecasting

Independent stage on main 4c2b8ab8b489465023b5d02047a9660cb53b73ba.

CPU alignment reproduced 3005 Development rows and 1380/205 original windows. Conservative P1S availability and fixed 96h qualification retained 870 Train / 178 Validation origins and 224 distinct validation target rows. All three models use the same samples. No Assessment/Test windows, raw rebuild, Optuna, M2, SOH head, or physics loss.

The original F2 is loaded read-only. B/C only expand the historical gate from 5-8-1 to 10-8-1 (+40 trainable parameters). Native TFT, convolution encoder, decoder, attention and head remain structurally unchanged and remain trainable. B uses five fixed zero inputs; C uses five past P1S features. Initialization and batch orders are paired by Seed42/52/62.

## Execution

Use Python 3.11.9 with the frozen Phase H runtime. Formal training entry is one-shot and refuses repeated execution.

- CPU alignment: `python -B scripts/data_alignment.py` (completed; do not overwrite while a run exists).
- Pretraining tests: `python -B -m unittest discover -s tests -v` (15 passed).
- Freeze: `python -B scripts/run_training.py freeze` (completed, original synthetic-preflight revision retained).
- GPU preflight: `python -B scripts/run_training.py smoke` (completed).
- Nine formal fits: `python -B scripts/run_training.py run` (one-shot registered runs only).
- After completion: `python -B scripts/verify_results.py`, `python -B scripts/render_results.py`, `python -B scripts/write_reports.py`.

Read `outputs/EXECUTION_LEDGER.json` and per-run `outputs/training_logs/*/run_state.json` before any execution. Failed/interrupted formal runs must be retained and reviewed; do not reinitialize or select a favorable retry. The original Windows fsync error occurred in synthetic preflight before any formal run; its protocol/code/logs are preserved under `configs/PREFLIGHT_REVISION_1`.

Main numerical outputs are `ALL_RUN_METRICS.csv`, `THREE_SEED_MEANS.csv`, `PAIRED_SEED_COMPARISONS.csv`, predictions, horizon metrics, gate weights and best checkpoints. The 5% mean improvement and two paired-seed requirement are fixed. Endpoint deterioration is assessed transparently using the original qualitative criterion; no additional zero-worsening cutoff is imposed. Research is exploratory reused Development Validation.

## Completed outcome

All nine formal runs completed with zero formal retries. Query-equal mean RMSE: F2_MATCHED 8.734973 mV; zero-input control 8.260648 mV; health-aware 10.702212 mV. The health model worsened by 22.52% and 29.56% against the two controls, with zero paired-seed wins against both. Decision: HEALTH_CONTEXT_NO_CLEAR_GAIN. Endpoint deterioration and all tails remain in the reports. No M2/physics/SOH/RUL extension or gate redesign follows.

Independent checks verified 38,448 prediction queries, all checkpoint reload identities and hashes, shared sampling/RNG, 1,585 event selections and 1,048 health vectors. Read 00_最终实验总结.md first.
