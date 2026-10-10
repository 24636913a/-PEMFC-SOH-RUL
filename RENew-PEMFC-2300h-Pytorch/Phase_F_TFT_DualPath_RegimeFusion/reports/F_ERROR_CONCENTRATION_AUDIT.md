# Phase F Error Concentration

Protocol SHA256: `c8ee4be440b9828cad04b615ddd19c15781033fdb22bbadeb1df18a59cf74cbf`. Base: `d7190bc86a1507015207c99702de6153b1006ee6`. Seed42 only. Development exploratory, not independent Test.

PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS; C1_D0_E0_REPRODUCIBILITY_UNRESOLVED; PHASE_F_TFT_ARCHITECTURE_EXPLORATION_ONLY; TIMESNET_FREEZE; BACKBONE_SEARCH_CLOSED.

|Model|12h RMSE mV|P95 mV|P99 mV|Max mV|Target-equal RMSE mV|Max-SSE row|Multiplicity|Max-row SSE share|Top5 SSE share|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|F0|18.309446|17.763852|31.043233|317.225657|21.178687|2952|19|29.299%|84.255%|
|F1|30.598323|14.982781|25.090261|521.948291|25.723957|2504|24|41.300%|87.445%|
|F2|12.667647|14.223970|24.165475|310.136396|22.213520|2952|19|34.239%|90.744%|

Target-time-equal RMSE first averages squared errors within each target, then weights target times equally; it does not average predictions into an ensemble. Formal primary RMSE remains forecast-query-weighted. All outliers remain in primary scores. Target SSE contributions and horizon error curves are saved for both splits. Assessment has 9072 records from 378 overlapping origins, not 9072 independent observations; no significance tests or anomaly-driven retuning were performed.
