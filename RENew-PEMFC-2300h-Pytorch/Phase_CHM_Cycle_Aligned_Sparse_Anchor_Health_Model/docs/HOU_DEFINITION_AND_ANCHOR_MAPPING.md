# Health Observation Unit Definition and Anchor Mapping

## HOU is the recurrent observation unit

The CHM recurrent time axis consists of **Health Observation Units** (HOU),
strictly ordered by `measurement_end_datetime`. This ordering is immutable in
training, LOAO, bootstrap, and reporting.

| HOU kind | Count | Frozen source | Role |
| --- | ---: | --- | --- |
| `COMPLETE_CYCLE_HOU` | 146 | RCR frozen complete cycles, with PAA's frozen middle-50% P1S calculation | Main health trajectory and bootstrap population |
| `PREANCHOR_PARTIAL_HOU` | 6 | `PARTIAL_PRE_300`, `600`, `850`, `1150`, `1500`, `1750` | Anchor-local auxiliary observation only |

Every HOU must carry `hou_id`, `hou_kind`, `source_interval_id`,
`measurement_end_datetime`, `measurement_end_ophrs2_h`, P1S retention, the
seven-condition vector, and a complete-cycle ID when applicable. The future
CHM data builder must fail closed if these values are missing, non-unique, or
not strictly ordered by `measurement_end_datetime`.

The main paper trajectory remains exactly the 146 `COMPLETE_CYCLE_HOU`. The six
partial HOU do not change that count, do not become complete cycles, do not
enter the complete-cycle bootstrap population, and do not receive
complete-cycle second-difference smoothness.

## Frozen anchor functional

For checkpoint `k`, the three interval IDs in frozen
`P1S_ANCHOR_ALIGNMENT.csv` define three HOU states:

\[
H_{k,1}, H_{k,2}, H_{k,3}.
\]

The **only** formal prediction for the sparse polarization anchor is

\[
\widehat{Y}_k=\operatorname{median}(H_{k,1},H_{k,2},H_{k,3}),
\qquad
L_{\mathrm{anchor}}=\sum_k w_k(\widehat{Y}_k-Y_k)^2.
\]

No anchor may be replaced by a nearest complete cycle, nearest cycle ID, or a
new partial interval. In particular, the 600 h anchor includes
`PARTIAL_PRE_600`; its mapping is therefore not reducible to one complete-cycle
ID.

The baselines use the identical functional. B0 reports
`median(R_{k,1}, R_{k,2}, R_{k,3})`; B1 and B2 train and predict at this same
anchor-level representation. This keeps primary and baseline estimands
identical.

## Support qualification carried forward

All six anchors retain PAV's `TEMPORAL_ORDER_VERIFIED` qualification. The PAA
registered 2-of-3 joint-support rule remains unchanged. Three of six final
median sources are directly in joint support; all six anchors remain in the
primary uncertainty-aware supervision set. Sensitivity S1 contains only 300,
850, and 1750 h and cannot select a model.
