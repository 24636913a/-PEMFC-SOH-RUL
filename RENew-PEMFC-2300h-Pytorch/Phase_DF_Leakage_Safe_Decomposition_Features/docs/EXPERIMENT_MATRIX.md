# Frozen experiment matrix

| Route | Input | Stage 0 | Seed 42 | Extra seeds |
| --- | --- | --- | --- | --- |
| RawF2_TimesNet | 48×14 | cached-control identity audit | reused cache | reused cache only |
| SimpleTrend_TimesNet | 48×16 | required | if Stage 0 passes | after its Raw gate, or as the required paired control for an authorized decomposition candidate |
| VMDFeature_TimesNet | 48×16 | mode, endpoint and resource gates | only if Stage 0 passes | only after Raw and decomposition-specific gates |
| EMDFeature_TimesNet | 48×16 | IMF, endpoint and resource gates | only if Stage 0 passes | only after Raw and decomposition-specific gates |

The frozen TimesNet architecture is shared by all routes. Changing `enc_in` and
`c_out` from 14 to 16 is permitted only because the two appended features must
be predicted under the official multivariate TimesNet contract; no 14-feature
checkpoint is loaded into a 16-feature route.
