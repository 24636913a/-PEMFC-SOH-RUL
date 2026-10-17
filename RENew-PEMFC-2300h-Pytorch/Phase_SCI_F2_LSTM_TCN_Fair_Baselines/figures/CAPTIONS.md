# Candidate figures

- figures/F01_five_model_RMSE: Bars: equal-seed mean; whiskers: sample SD; points: seeds 42/52/62. No selected F2 seeds. Data: `tables/FIVE_MODEL_PER_SEED.csv`.

- figures/F02_MAE_and_endpoint: All 4920 queries retained; endpoint: 205 queries per seed. SD is not a confidence interval. Data: `tables/FIVE_MODEL_PER_SEED.csv`.

- figures/F03_24_step_RMSE: Lines: mean of three seeds; shading: observed seed range. One fixed 24-step task. Data: `outputs/metrics/F03_plot_data.csv`.

- figures/F04_history_conditions: Groups use frozen Train terciles over observed 48-step histories; future conditions are not known. Data: `outputs/metrics/F04_plot_data.csv`.

- figures/F05_fixed_origin_voltage: All five models: seed 62, single fixed origin row 2205; no window stitching. Raw origin voltage plus predicted DeltaV. Data: `outputs/predictions/FIXED_CASES.csv`.

- figures/F06_difficult_origin_error: All five models: seed 62, single fixed origin row 2224; no window stitching. Targets 2247/2248 highlighted. Data: `outputs/predictions/FIXED_CASES.csv`.
