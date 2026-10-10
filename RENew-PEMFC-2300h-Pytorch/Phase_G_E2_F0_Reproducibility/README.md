# Phase G — E2/F0 training trajectory reproducibility

Diagnostic only: four independent Seed42 paths, exactly 12 epochs each; no
Assessment inference, Test, tuning, or formal model replacement. Historical E/F
artifacts are read-only. E and F retain separate training loops and their best
state handling. Symmetric tensor/RNG/optimizer summaries synchronize CUDA and
may alter timing; the controlled runtime is not the historical runtime.

Actual status: **PHASE_G_EXECUTION_BLOCKED**. G-E-A failed at its first
Backward because CUDA `upsample_linear1d_backward_out_cuda` has no strict
deterministic implementation in this environment. Completed epochs=0,
optimizer updates=0; the other three trajectories were not started. The planned
four×12 budget was not completed, and no repeat-consistency claim is possible.
Do not execute `--run` again or silently change determinism/device/model.

Run from this phase using Python 3.11.9:

```powershell
python -B scripts/audit_historical_e2_f0.py
python -B -m unittest discover -s tests -v
python -B scripts/run_controlled_replay.py --freeze
python -B scripts/run_controlled_replay.py --run
python -B scripts/analyze_replay.py
```

The runner refuses to repeat an entered trajectory or silently disable
deterministic algorithms. Results are NOT new E2/F0 scientific model scores.
