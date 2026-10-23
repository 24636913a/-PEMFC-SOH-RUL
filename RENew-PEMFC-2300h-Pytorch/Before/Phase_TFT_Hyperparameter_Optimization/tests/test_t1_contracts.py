from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from t1_common import (  # noqa: E402
    audit_t0_authority, build_folds, load_bs_common, load_canonical_tft,
    load_protocol, safe_read_development, valid_width_combinations,
    verify_t0_validation_authority,
)


class T1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg, _ = load_protocol()
        cls.bs_cfg, cls.common = load_bs_common()

    def test_t0_identity_and_protocol_are_unchanged(self) -> None:
        self.assertTrue(audit_t0_authority(self.cfg)["pass"])

    def test_safe_prefix_reader_and_window_authority(self) -> None:
        frame, audit = safe_read_development(self.cfg)
        self.assertEqual(len(frame), 3005)
        self.assertEqual(audit["pandas_read_csv_nrows"], 3007)
        self.assertFalse(audit["full_causal_file_hash_computed"])
        folds = build_folds(frame, self.cfg, self.bs_cfg, self.common)
        self.assertEqual([len(v.origin_row) for _, _, v in folds], [233, 245, 466])
        self.assertTrue(verify_t0_validation_authority(folds)["pass"])

    def test_all_frozen_widths_are_legal_and_affect_parameter_count(self) -> None:
        model_class, count_parameters = load_canonical_tft()
        values = valid_width_combinations(self.cfg)
        self.assertEqual(len(values), 8)
        low = model_class(hidden_size=16, hidden_continuous_size=8, attention_head_size=4, lstm_layers=1, dropout=0.05)
        high = model_class(hidden_size=64, hidden_continuous_size=32, attention_head_size=4, lstm_layers=1, dropout=0.2)
        self.assertGreater(count_parameters(high), count_parameters(low))
        self.assertEqual(tuple(low.predict_delta(torch.randn(2, 48, 14)).shape), (2, 24))

    def test_production_reader_has_bounded_csv_call(self) -> None:
        source = (PHASE / "scripts" / "t1_common.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        reader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "safe_read_development")
        reads = [node for node in ast.walk(reader) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "read_csv"]
        self.assertEqual(len(reads), 1)
        self.assertTrue(any(keyword.arg == "nrows" for keyword in reads[0].keywords))


if __name__ == "__main__":
    unittest.main(verbosity=2)
