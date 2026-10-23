# Test firewall

This pre-unblinding contract phase accesses no Test raw data, Test polarization,
Test EIS, 2000 h data, 2300 h data, or after-Air-Air data.
`2300h_after-Air-Air` is permanently excluded from the primary Test. Any Test
execution requires a new `OPEN_AHC_TV_TEST` manual instruction.

The project historical exposure is **YES**: Phase N1 parsed and persisted
late-life Test polarization values for labelled future-validation outputs.
AHC model-selection use is **NO**; AHC-TV protocol-freeze Test access is
**NO**; future AHC-TV reuse of those N1 derived outputs is **PROHIBITED**; and
future target unblinding is **ONE_SHOT_ONLY_AFTER_PREDICTION_FREEZE**. The
historical outputs must not be described as a permanently sealed Test set.
