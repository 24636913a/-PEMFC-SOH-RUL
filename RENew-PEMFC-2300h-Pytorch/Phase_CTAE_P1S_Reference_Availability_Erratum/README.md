# Phase CTAE — P1S Reference Availability Erratum

This Development-only erratum narrows CTA route T2. Fixed-reference P1S retention cannot be available until all ten frozen reference-cycle voltages have been observed. The first ten complete P1S cycles remain PAA reference warm-up records and are excluded from formal causal target evaluation even though `V_ref` can be calculated at the end of cycle 10.

The corrected formal target is `POST_REFERENCE_RAW_P1S_RETENTION_AS_CAUSAL_HEALTH_TARGET`.

`R_P1S(t) = V_P1S(t) / V_ref` is formal only for non-reference evaluation cycles after fixed-reference availability. Global six-anchor B2 and all full-sequence PHI trends remain retrospective-only.

No model training, Test access, target retuning, support redefinition or trend replay is performed here.
