# Same Seed42 repeatability

|run_id|best_epoch|stop_epoch|RMSE_mV|MAE_mV|R2|
|---|---|---|---|---|---|
|H-F0-Repeat-A|187|227|8.76148906172451|5.6090517908255775|0.997982791148852|
|H-F0-Repeat-B|286|326|8.094854266394512|4.22162396480643|0.9982780795343963|
|H-F0-Seed42|277|317|8.307160723546348|6.413513220669751|0.9981865723007312|

{
  "2": {
    "RMSE_values_mV": [
      8.76148906172451,
      8.094854266394512
    ],
    "absolute_spread_mV": 0.6666347953299976,
    "relative_spread": 0.07909601535203027,
    "threshold_mV": 0.8428171664059511,
    "pass": true
  },
  "3": {
    "RMSE_values_mV": [
      8.76148906172451,
      8.094854266394512,
      8.307160723546348
    ],
    "absolute_spread_mV": 0.6666347953299976,
    "relative_spread": 0.0794763869882277,
    "threshold_mV": 0.8387834683888458,
    "pass": true
  }
}

Two-run D=|RA−RB|; T=max(0.5mV,0.10×mean). Three-run D=max−min uses the same mean-based threshold. All runs are independent initializations in fresh processes. PASS means no material discrepancy above the registered threshold, not bitwise determinism. Repeat-A/B are excluded from multi-seed n=3.
