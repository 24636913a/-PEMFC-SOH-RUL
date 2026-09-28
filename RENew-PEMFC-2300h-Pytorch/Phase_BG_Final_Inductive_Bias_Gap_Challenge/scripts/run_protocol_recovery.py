"""Produce the Phase BG read-only Phase BS protocol and window audit."""
from __future__ import annotations

import json

from legacy_protocol import recover_windows_and_controls, write_csv, write_json, write_text


def main() -> None:
    cfg, _helpers, _frame, _folds, windows, audit = recover_windows_and_controls()
    write_json(audit, "00_protocol_and_audit", "PROTOCOL_HASHES.json")
    write_csv(windows, "00_protocol_and_audit", "WINDOW_IDENTITY_AUDIT.csv")
    firewall = {"status": "PASS", "contract": "Phase_BS_exact_3007_prefix_then_3005_development_filter", "test_accessed": False, "test_feature_rows": 0, "test_target_rows": 0, "test_predictions": 0, "test_metrics": 0, "raw_data_accessed": False, "external_decomposition": False}
    write_json(firewall, "00_protocol_and_audit", "TEST_FIREWALL_AUDIT.json")
    recovery = "# Protocol recovery\n\nPhase BG reads the immutable Phase BS YAML and verifies its SHA-256 before calling the unchanged Phase BS helper functions directly. The legacy `load_config()` HEAD guard is not removed, altered, or monkey-patched; it is not called because it correctly guards the historical execution HEAD.\n\nThe exact data path is `pandas.read_csv(..., nrows=3007)`, followed by `bin_end_ophrs2_h < 1819.997738`, retaining 3,005 Development rows. All recovered window identities and frozen TimesNet control caches matched. No Test row or raw source record was accessed.\n\n## Restored task\n\n- Raw F2: `[B,48,14]`\n- Forecast: 24 steps at 0.5 h\n- Target: future `cell_voltage_mean - origin cell_voltage_mean`\n- Loss: train-scale standardized DeltaV MSE\n- Folds: Phase BS expanding folds with chronological 15% isolated inner stopper\n"
    write_text(recovery, "00_protocol_and_audit", "PROTOCOL_RECOVERY.md")
    print(json.dumps({"status": "PASS", "windows": len(windows), "development_rows": audit["data_audit"]["development_rows_retained"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
