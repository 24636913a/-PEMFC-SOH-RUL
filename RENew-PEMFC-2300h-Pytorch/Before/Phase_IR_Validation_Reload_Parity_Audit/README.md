# Phase I-R — Validation Reload Parity Audit

Base main: `27d3001f0838d7fa8ee311516c5eaf9bdb2c63ec`.

Completed audit: `PHASE_IR_H_AUTHORITY_PARITY_VERIFIED` and `PHASE_IR_I_EXECUTION_SEMANTICS_DIFFERENCE_OBSERVED`. All 45 controlled paths and 17 post-freeze regression tests completed; saved differences passed independent recalculation. A/B/D exactly reproduce the saved reference for all nine checkpoints; C/E differ from it but match each other exactly. See `outputs/IR_SUMMARY.json` and `reports/IR_FINAL_RECOMMENDATION.md`. The original Phase I remains BLOCKED and is not restarted.

Scope: nine frozen Phase H primary checkpoints × five controlled read-only paths, **one forward per path on the first 32 frozen Validation origins**. Each path uses a fresh Python child and the checkpoint's saved fixed prediction as reference. No training, Assessment construction/inference/cache access, Test access, model changes, scaler fitting or historical-file writes.

## Execution

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/run_ir_validation_parity.py --freeze
python -B scripts/run_ir_validation_parity.py --run
```

The finite execution ledger refuses another matrix run. Workers refuse replacement of existing path evidence; no retries or device fallback occur. The production H/I inference functions are not edited. Phase I's BLOCKED state remains intact.

## Paths

| Path | Construction / loading | Context | Parameter flags | Interface |
| --- | --- | --- | --- | --- |
| A | CUDA | no_grad | original | original H e.predict -> c.predict |
| B | CUDA | inference_mode | original | same one-batch model.predict_delta |
| C | CUDA | no_grad | disabled | original e.predict |
| D | CPU then CUDA | no_grad | original | original e.predict |
| E | CPU then CUDA | inference_mode | disabled | unmodified I.predict_frozen |

Only inherited Validation windows `[2103,2404)` are constructed. Bounded N1 access is `nrows=3007` then Development filtering to 3005 rows. Full Validation input/label/origin/target hashes are verified against H before retaining only the fixed32 arrays. Persisted E scaler and E EWMA Top10 transform are reused without fit.

Every path saves actual/saved/difference arrays before judging exact parity; numerical mismatch does not skip the other registered paths. Shape, input and checkpoint identity failures stop. Parameter and buffer entries, original/configured/final gradient flags, grad=None and forbidden-operation guards prove read-only execution. Requires_grad=True is not training.

Independent audit (no model invocation):

```powershell
python -B scripts/audit_ir_differences.py
```

All differences are diagnostic **standardized DeltaV** or mV after multiplying by the original target scale. Optional fixed32 RMSE is neither full 205-window Validation nor Assessment performance. Exact parity remains `np.array_equal`, not allclose. Reports and `outputs/IR_SUMMARY.json` are generated from saved evidence. Any future Assessment repair needs a separately authorized revised protocol and new code identity; this audit cannot reopen Phase I.
