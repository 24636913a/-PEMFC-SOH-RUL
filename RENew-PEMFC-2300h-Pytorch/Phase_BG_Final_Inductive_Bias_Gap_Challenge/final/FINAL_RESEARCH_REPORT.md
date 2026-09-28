# Phase BG — final inductive-bias gap challenge

## Decision

**Final backbone: `TIMESNET_FREEZE`.** `BACKBONE_SEARCH_CLOSED` is now
terminal. This is a one-time structured gap challenge following the historical
Phase BS screening, not a new model search. Phase BS's immutable reference is
commit `68f880908f9fd56d35c3788d7a19a0b2d69b4ecd`; its registered protocol
SHA-256 is `fd710c4f…afd6788`, and the protected-file drift audit is empty.

Phase BS remains the only scored incumbent here. Its Development-only formal
forecast-query RMSEs for TimesNet were 14.207 mV (seed 42), 13.623 mV (2024),
and 13.750 mV (3407). These values are historical context; no challenger was
allowed to generate a Development score after its earlier mandatory gate
failed.

## Frozen contract and fairness controls

The contract was recovered exactly: `pandas.read_csv(..., nrows=3007)`, then
the ageing boundary filter `< 1819.997738 h`, retaining 3,005 Development
rows. The 48-step Raw F2 history, 24-step forecast, three outer folds,
chronological isolated stopper, train-only scaling, standardized absolute-V
output contract, and DeltaV MSE mapping matched the verified Phase BS
TimesNet caches for every fold and seed. See
`00_protocol_and_audit/WINDOW_IDENTITY_AUDIT.csv`.

No active candidate accessed Test. Test feature rows, target rows, candidate
predictions, and metrics are all zero. Raw source data and external
decomposition were also not accessed. The only Development access was the
registered prefix needed to revalidate the inherited window identity; it did
not produce challenger performance.

## Candidate outcomes

### PDFNet — `SOURCE_GATE_FAIL`

PDFNet was the fixed primary challenger. Its paper identity could be found,
but no auditable association from the paper to a named author and executable
official source was established. No repository was accepted on topical
similarity alone. Therefore it was never cloned, adapted, resource-tested, or
scored. This is a provenance stop, not a negative RealHyFC performance result.

### xLSTMTime — `RESOURCE_INFEASIBLE`

The author-linked xLSTMTime forecasting repository passed source and
architecture review at frozen commit `db43526e…e76622`. Its native model first
performs 48→24 trend/seasonal projection, then sends a variable-axis
256-dimensional representation through three mLSTM blocks; it is not a new
48-position recurrent time-axis baseline. This is distinct from the earlier
project-local N2.5 xLSTM/sLSTM experiment, so the historical category-level
negative evidence was not mislabelled as an author-model result.

Before any Development scoring, the immutable author implementation failed
construction at the synthetic resource gate. Importing its sLSTM CUDA path
called `torch.utils.cpp_extension.include_paths(cuda=True)`, unsupported by
the installed PyTorch API. No source edit, dependency workaround, substitute
implementation, configuration change, model training, or Development score
was used. The full evidence is `02_xLSTMTime/RESOURCE_AUDIT.json`.

### CFPT — `CFPT_PROTOCOL_INCOMPATIBLE`

CFPT's official ICML/PMLR-linked source passed provenance and architecture
review. Its Cross-Frequency Interaction branch separates low/high spectra and
uses interaction blocks; its Periodic-Aware Timestamp Modeling branch encodes
future decoder timestamp marks into a 2-D periodic tensor and blends that
forecast with the frequency prediction.

Route A is not author-natively history-only: verified official source
unconditionally uses `x_mark_dec`. Route B is not fair under the frozen
control: official TimesNet functionally consumes `x_mark_enc` but ignores
`x_mark_dec`, whereas CFPT consumes `x_mark_dec`. Supplying an ignored
argument does not satisfy equal information. Making TimesNet consume it would
modify the author body, add a custom adapter, or change Raw F2—each prohibited.
CFPT was therefore stopped before resource preflight or any score. This is not
evidence of poor CFPT performance; it is a protocol incompatibility.

## Final research interpretation

All three frozen candidates exited before an eligible score-based promotion:
one at source provenance, one at a reproducible author-source resource
failure, and one at an explicit fair-protocol failure. Consequently no result
supports replacing the Phase BS incumbent. `TIMESNET_FREEZE` remains the
final backbone, and no fourth backbone may be introduced in this research
route.

## Handoff: decomposition and degradation-feature study

The permitted next activity is **design only** for a leakage-safe causal or
rolling decomposition protocol. A later approved phase may compare Raw F2,
EMD-derived, VMD-derived, and selected degradation-related modes only after
the preprocessing fit/apply boundaries, chronological split, and evaluation
protocol are separately frozen. Phase BG runs no VMD, EMD, feature selection,
SOH, RUL, model search, or Test evaluation.
