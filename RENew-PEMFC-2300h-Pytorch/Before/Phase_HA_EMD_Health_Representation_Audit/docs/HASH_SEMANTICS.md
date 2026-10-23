# Cache hash semantics

Phase HA V2 validates two independent objects for every local Phase DF cache:

| Object | Expected source | Observed calculation |
|---|---|---|
| Feature payload | `CACHE_PROVENANCE_MANIFEST.csv.cache_sha256` | `SHA256(np.ascontiguousarray(features).tobytes())` |
| NPZ artifact file | `LOCAL_ARTIFACT_MANIFEST.csv.sha256` | SHA256 of the `.npz` file bytes |

The values may legitimately differ because the NPZ archive contains metadata,
array serialization, and ZIP structure in addition to the feature payload.
V2 rejects a cache unless both independently available references pass, archive
metadata passes, input-history/parameter hashes pass, and the `[48,2]`
float32 finite feature contract passes.  It never compares the file hash to the
payload hash.
