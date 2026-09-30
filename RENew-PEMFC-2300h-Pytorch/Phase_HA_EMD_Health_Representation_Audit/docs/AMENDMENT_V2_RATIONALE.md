# Pre-endpoint-analysis technical amendment V2

**Status:** `FROZEN_AFTER_TECHNICAL_BLOCKER_BEFORE_FIRST_ENDPOINT_STATISTICS`

This amendment preserves the v1 protocol and `TECHNICAL_BLOCKER.md` unchanged.
It was written after knowledge of Phase DF's `EMD_SPECIFIC_ADVANCE`, the
historical approximate 105.9 mV retrospective-revision warning, and the v1
technical blocker, but before the first Phase HA endpoint-health statistic.
It must therefore not be represented as blind Phase DF preregistration.

The v1 blocker compared two different hash semantics.  In Phase DF,
`CACHE_PROVENANCE_MANIFEST.csv.cache_sha256` is the SHA256 of
`features.tobytes()`, while `LOCAL_ARTIFACT_MANIFEST.csv.sha256` is the SHA256
of the complete NPZ file bytes.  Consequently, `c379... != 5681...` neither
establishes cache corruption nor establishes cache validity.  It is a
`HASH_SEMANTICS_IMPLEMENTATION_ERROR`.  V2 independently validates the two
hashes without exchanging their meanings.

V2 also replaces a full N1-file checksum with an allowed logical-prefix
fingerprint, corrects the inherited EMD endpoint unit to V and V/h, implements
joint support plus outcome-free 1:1 condition matching, and replaces IID block
resampling with registered time-dependent moving-block procedures.  These are
technical/statistical implementation repairs only: they do not change Phase
DF's frozen forecast results, EMD parameters, or research status.
