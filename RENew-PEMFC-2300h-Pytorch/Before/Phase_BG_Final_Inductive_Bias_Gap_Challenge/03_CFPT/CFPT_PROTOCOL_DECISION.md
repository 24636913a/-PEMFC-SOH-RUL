# CFPT protocol decision — `INCOMPATIBLE`

## Decision before any Development score

**CFPT is protocol-incompatible with this one-time challenge.** The decision
was made from SHA-verified official source only. No CFPT model was constructed,
resource-preflighted, trained, or scored; no Development CSV or Test data was
read by this gate.

## Route A: Raw-F2 history-only

**Rejected.** In the verified `Model.forecast`, `self.time_enc(x_mark_dec)` is
called unconditionally and its result is blended with `freq_pred` as
`beta * freq_pred + (1 - beta) * time_pred`. Passing no decoder mark does not
constitute an author-supported disablement of the periodic timestamp body.
Removing or bypassing that branch would change the official CFPT body.

## Route B: equal known-time covariates

The required marks could, in principle, be deterministic: origin timestamp
plus the fixed 0.5-hour cadence and horizon. They would not require a future
sensor value. But a fair Route B requires *both* official bodies to
functionally consume identical encoder and decoder marks.

The frozen official TimesNet `Model.forecast` uses `x_mark_enc` in
`enc_embedding(x_enc, x_mark_enc)` and never uses `x_mark_dec`; the latter is
only carried through its public signature. CFPT, by contrast, functionally
uses `x_mark_dec`. Passing an ignored argument to TimesNet would not give both
models the same information. Making TimesNet consume it would require a custom
head/adapter, an author-body edit, or an addition to frozen Raw F2—all
prohibited here.

## Consequence

`PROTOCOL_GATE_AUDIT.json` and the production test establish
`route=INCOMPATIBLE`. This is not evidence that CFPT forecasts RealHyFC poorly;
it is a correct stop before an unfair comparison. The next and final state is
`BACKBONE_SEARCH_CLOSED`.
