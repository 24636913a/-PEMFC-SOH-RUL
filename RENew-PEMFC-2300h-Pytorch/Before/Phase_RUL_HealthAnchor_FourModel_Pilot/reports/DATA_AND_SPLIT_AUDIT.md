# Data and split audit

Source commit: `2950160fdb762511138c1bf4a159c6c898a9d193`. The analysis loaded exactly the first 3,005 rows of the Phase N1 30-min causal table (maximum bin end 1819.997738 h) and verified no `split=test` rows. Health labels were restricted to BoL–1750 h.

The polarization files' `OpHrs` values are not used as the ageing coordinate. The `DateTime` of each permitted characterization was matched to its nearest development causal bin (maximum accepted difference: 2 h). BoL precedes Loadcycle availability and therefore has no dynamic input history.

| stage | nominal_stage_h | aligned_ophrs2_h | alignment_delta_h | polarization_datetime | reference_voltage_V | SOH | alignment_method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BoL | 0.000000 |  |  | 2024-12-09 07:35:40.005000 | 0.698953 | 1.000000 | NO_LOADCYCLE_COVERAGE |
| 300 | 300.000000 | 312.0 | 0.41305555555555556 | 2025-02-09 00:33:22.005000 | 0.688217 | 0.984639 | DATETIME_NEAREST_BIN |
| 600 | 600.000000 | 566.0 | 0.615 | 2025-02-24 01:36:43.005000 | 0.684750 | 0.979679 | DATETIME_NEAREST_BIN |
| 850 | 850.000000 | 854.0 | 0.7569444444444444 | 2025-03-20 02:02:04.005000 | 0.683395 | 0.977740 | DATETIME_NEAREST_BIN |
| 1150 | 1150.000000 | 1130.5 | 1.1119444444444444 | 2025-04-18 05:10:57.005000 | 0.679818 | 0.972623 | DATETIME_NEAREST_BIN |
| 1500 | 1500.000000 | 1463.0 | 1.1202777777777777 | 2025-05-26 22:30:13.005000 | 0.680685 | 0.973863 | DATETIME_NEAREST_BIN |
| 1750 | 1750.000000 | 1719.0 | 0.8986111111111111 | 2025-06-10 16:29:51.005000 | 0.679232 | 0.971785 | DATETIME_NEAREST_BIN |

The 2000 h and 2300 h files were not opened by this phase. The historical RUL90 report records that test targets had been accessed descriptively before this phase; that seal is **not restored** by this run.
