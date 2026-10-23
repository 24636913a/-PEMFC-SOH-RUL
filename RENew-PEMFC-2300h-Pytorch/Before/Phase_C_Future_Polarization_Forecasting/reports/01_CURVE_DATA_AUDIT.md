# Curve data audit

Seven normal nominal Development files were streamed directly: BoL, 300, 600, 850, 1150, 1500 and 1750 h. AfterAirAirStart files were not opened. Active voltage is the frozen median over U1–U5 and U7–U11; U6 is excluded. All files have median cell count 10 and active area 280 cm².

The sample-order current trace has many sign changes rather than a reliably labelled up/down sweep. No consistent scan-direction branch can therefore be identified without inventing a rule. Before modelling, the frozen rule was set to pooled direct target-window medians, `|j-j*|≤0.01 A cm⁻²`; it preserves the prior RUL90 j=1.0 definition. All five common grid points have at least three direct samples in every curve, so no extrapolation or interpolation was required.

| stage | raw_sample_count | diagnostic_start_datetime | diagnostic_end_datetime | j_min_A_cm2 | j_max_A_cm2 | cell_count_median | active_area_cm2_median | u6_median_V | u6_negative_ratio | scan_sign_change_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BoL | 14363 | 2024-12-09 07:35:40.005 | 2024-12-09 11:35:02.005 | 0.00000 | 1.80364 | 10.00000 | 280.00000 | -0.01636 | 1.00000 | 4914 |
| 300 | 10700 | 2025-02-09 00:33:22.005 | 2025-02-09 03:31:41.005 | 0.00000 | 1.80364 | 10.00000 | 280.00000 | -0.01495 | 1.00000 | 3423 |
| 600 | 10090 | 2025-02-24 01:36:43.005 | 2025-02-24 04:24:52.005 | 0.00000 | 1.80364 | 10.00000 | 280.00000 | -0.01331 | 1.00000 | 3255 |
| 850 | 9785 | 2025-03-20 02:02:04.005 | 2025-03-20 04:45:08.005 | 0.00000 | 1.80364 | 10.00000 | 280.00000 | -0.01288 | 1.00000 | 3159 |
| 1150 | 9000 | 2025-04-18 05:10:57.005 | 2025-04-18 07:40:56.005 | 0.00000 | 1.80364 | 10.00000 | 280.00000 | -0.01288 | 1.00000 | 2926 |
| 1500 | 8880 | 2025-05-26 22:30:13.005 | 2025-05-27 00:58:12.005 | 0.00000 | 1.80364 | 10.00000 | 280.00000 | -0.01312 | 1.00000 | 2858 |
| 1750 | 9660 | 2025-06-10 16:29:51.005 | 2025-06-10 19:10:50.005 | 0.00000 | 1.80364 | 10.00000 | 280.00000 | -0.01300 | 1.00000 | 3109 |
