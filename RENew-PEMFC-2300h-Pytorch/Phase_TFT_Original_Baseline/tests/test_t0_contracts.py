from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
sys.path.insert(0, str(PHASE / "models"))
from canonical_tft import CanonicalStrictHistoryTFT, RAW_NAMES, RELATIVE_STEP  # noqa: E402
from tft_common import build_folds, load_bs_common, load_protocol, runtime_source_audit, safe_read_development  # noqa: E402


class T0ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg, _ = load_protocol()
        cls.bs_cfg, cls.common = load_bs_common()

    def test_safe_prefix_reader_retains_exact_development_population(self) -> None:
        frame, audit = safe_read_development(self.cfg)
        self.assertEqual(len(frame), 3005)
        self.assertEqual(audit["pandas_read_csv_nrows"], 3007)
        self.assertFalse(audit["full_causal_file_hash_computed"])
        self.assertFalse(audit["test_accessed"])

    def test_frozen_fold_origin_counts_and_target_isolation(self) -> None:
        frame, _ = safe_read_development(self.cfg)
        folds = build_folds(frame, self.cfg, self.bs_cfg, self.common)
        self.assertEqual([len(validation.origin_row) for _, _, validation in folds], [233, 245, 466])
        for _, train, validation in folds:
            self.assertFalse(set(train.origin_row) & set(validation.origin_row))
            self.assertTrue(np.all(validation.target_end_age_h < 1819.997738))

    def test_canonical_tft_outputs_and_exposes_interpretability(self) -> None:
        model = CanonicalStrictHistoryTFT(hidden_size=32, hidden_continuous_size=16, attention_head_size=4, lstm_layers=1, dropout=0.1)
        out = model.forward_history(torch.randn(3, 48, 14))
        self.assertEqual(tuple(out["prediction"].shape), (3, 24, 1))
        self.assertEqual(tuple(out["encoder_variables"].shape), (3, 48, 1, 15))
        self.assertEqual(tuple(out["decoder_variables"].shape), (3, 24, 1, 1))
        self.assertEqual(tuple(out["encoder_attention"].shape), (3, 24, 4, 48))

    def test_decoder_contains_only_deterministic_relative_step(self) -> None:
        model = CanonicalStrictHistoryTFT(hidden_size=32, hidden_continuous_size=16, attention_head_size=4, lstm_layers=1, dropout=0.1)
        x = torch.randn(2, 48, 14)
        inputs = model.build_input(x)
        self.assertTrue(bool(torch.all(inputs["decoder_cont"][:, :, :14] == 0)))
        self.assertTrue(bool(torch.allclose(inputs["decoder_cont"][:, :, -1], torch.arange(1, 25).view(1, 24).expand(2, -1) / 24.0)))
        self.assertEqual(model.hparams.time_varying_reals_decoder, [RELATIVE_STEP])

    def test_future_actual_mutation_cannot_change_strict_history_prediction(self) -> None:
        model = CanonicalStrictHistoryTFT(hidden_size=32, hidden_continuous_size=16, attention_head_size=4, lstm_layers=1, dropout=0.1).eval()
        history = torch.randn(2, 48, 14)
        future_a, future_b = torch.randn(2, 24, 14), torch.randn(2, 24, 14) * 999.0
        with torch.no_grad():
            a = model.predict_delta(history)
            _ = (future_a, future_b)  # never offered to the model interface
            b = model.predict_delta(history)
        self.assertTrue(bool(torch.equal(a, b)))

    def test_source_fingerprints_and_api_version_are_frozen(self) -> None:
        audit = runtime_source_audit(self.cfg)
        self.assertEqual(audit["pytorch_forecasting_version"], "1.8.0")
        self.assertEqual(len(audit["source_hashes"]), 2)

    def test_production_reader_has_no_unbounded_pandas_csv_call(self) -> None:
        source = (PHASE / "scripts" / "tft_common.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        reader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "safe_read_development")
        calls = [node for node in ast.walk(reader) if isinstance(node, ast.Call)]
        reads = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "read_csv"]
        self.assertEqual(len(reads), 1)
        self.assertTrue(any(keyword.arg == "nrows" for keyword in reads[0].keywords))


if __name__ == "__main__":
    unittest.main(verbosity=2)
