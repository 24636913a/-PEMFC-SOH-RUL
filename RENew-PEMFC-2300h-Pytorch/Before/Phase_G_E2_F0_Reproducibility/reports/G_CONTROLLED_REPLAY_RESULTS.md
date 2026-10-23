# Controlled replay — technical stop, not a training result

Protocol SHA: `21e27f8fb99d93f77d1ff1e098476a4d0050ccad442bb077a8380ca12961febb`.

G-E-A reached the first Forward and was stopped by strict deterministic CUDA
Backward at epoch1/batch0. No optimizer update or completed epoch. G-E-B,
G-F-A and G-F-B were NOT_RUN under the registered fail-closed rule. No replay
RMSE, first divergent batch, or repeat consistency result can be reported.

See `outputs/replay_logs/G-E-A/TECHNICAL_FAILURE.json`, `console.log`,
`initial.json`, `environment.json`, and `input_audit.json`; the execution ledger
preserves the started trajectory and blocks automatic rerun.

The recorder emits a batch record after a completed optimizer step. Therefore
no completed-batch JSONL exists for this failure. The partial first-batch
forward/loss summaries were not durably flushed before the backward exception;
they are unavailable and have not been reconstructed or invented.

Strict determinism was not disabled. No CPU fallback, warn-only, model change,
extra condition or automatic second attempt was executed. Historical code,
results and checkpoints remain untouched. This is PHASE_G_EXECUTION_BLOCKED,
not evidence that either execution path is unstable or consistently repeatable.
