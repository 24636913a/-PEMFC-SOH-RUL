# Protocol recovery

Phase BG reads the immutable Phase BS YAML and verifies its SHA-256 before calling the unchanged Phase BS helper functions directly. The legacy `load_config()` HEAD guard is not removed, altered, or monkey-patched; it is not called because it correctly guards the historical execution HEAD.

The exact data path is `pandas.read_csv(..., nrows=3007)`, followed by `bin_end_ophrs2_h < 1819.997738`, retaining 3,005 Development rows. All recovered window identities and frozen TimesNet control caches matched. No Test row or raw source record was accessed.

## Restored task

- Raw F2: `[B,48,14]`
- Forecast: 24 steps at 0.5 h
- Target: future `cell_voltage_mean - origin cell_voltage_mean`
- Loss: train-scale standardized DeltaV MSE
- Folds: Phase BS expanding folds with chronological 15% isolated inner stopper
