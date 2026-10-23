# T0 data and leakage audit

The formal Seed-42 run used only `pd.read_csv(..., nrows=3007)` followed by `bin_end_ophrs2_h < 1819.997738`, yielding exactly 3005 Development rows. The recorded Development-prefix fingerprint is `e7470dec603689595b03c3b06a1a3509a2a74b4cc4f68c985992f914ebc29f2e`. Full-file reads and hashes are prohibited by the frozen protocol.

The three frozen expanding folds retained every legal validation origin: Fold 1 = 233, Fold 2 = 245, Fold 3 = 466. Every target lies below the Development boundary. Model inputs are the 48 observed Raw F2 history steps only. Decoder data contain the deterministic `relative_forecast_step` coordinate and no future measured current, voltage, temperature, pressure, RH, flow, stoichiometry, or any Test observation.

`test_accessed=false` for Stage 0, all formal fold results, and the run audit.
