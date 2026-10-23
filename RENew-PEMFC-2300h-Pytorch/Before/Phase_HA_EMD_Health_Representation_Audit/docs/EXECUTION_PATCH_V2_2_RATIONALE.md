# Phase HA execution patch V2.2 rationale

V2.1 correctly changed the final ASR gate to treat an empty table as a failed
stability gate.  Its retry exposed the same schema assumption earlier in the
RDR reporting function: when no stratum has matched support, the empty
contrast dataframe has no `stratum` field.  V2.2 makes that production route
return `RDR_NOT_IDENTIFIABLE_IN_STABLE_SUBSET` for every stratum.

This is an execution-totality correction only.  It changes no data access,
cache validation, EMD parameter, unit, condition-matching variable, scale,
caliper, block definition, bootstrap iteration count, seed, RDR threshold,
recovery rule, endpoint formula, or final scientific gate.  In particular, it
does not manufacture a contrast or a revision-to-drift result.  V2 and V2.1
files and blocker records remain immutable historical records.
