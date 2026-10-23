from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
import run_formal_ablation as rfa  # noqa: E402
from pot_common import build_frozen_folds, load_inherited_protocols, load_protocol, safe_read_development  # noqa: E402


class HistoricalControlCompatibilityTests(unittest.TestCase):
    def test_m1_m2_have_exact_legal_development_origin_and_target_keys(self) -> None:
        cfg, _ = load_protocol()
        bs_cfg, _df_cfg, common = load_inherited_protocols(cfg)
        frame, _ = safe_read_development(cfg, bs_cfg)
        folds = build_frozen_folds(frame, cfg, bs_cfg, common)
        metrics, _predictions, audit = rfa._historical_records(folds, common)
        self.assertEqual(len(metrics), 18)
        self.assertEqual(len(audit), 18)
        self.assertTrue(all(row["origin_and_target_keys_match"] for row in audit))
        self.assertTrue(all(row["truth_match"] for row in audit))
        self.assertTrue(all(row["target_age_max_h"] < 1819.997738 for row in audit))

    def test_resolved_optuna_configs_are_outer_validation_free(self) -> None:
        cfg, protocol_sha = load_protocol()
        resolved = rfa.load_resolved(protocol_sha)
        self.assertFalse(resolved["outer_validation_used_for_selection"])
        self.assertFalse(resolved["test_accessed"])
        self.assertEqual(set(resolved["search_results"]), {"Optuna_RawF2_TimesNet", "Optuna_EMDFeature_TimesNet"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
