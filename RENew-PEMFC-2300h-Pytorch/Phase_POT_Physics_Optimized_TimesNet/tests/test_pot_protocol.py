from __future__ import annotations

import ast
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
import pot_common as pc  # noqa: E402


class POTProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg, _ = pc.load_protocol()
        cls.bs_cfg, _df_cfg, cls.common = pc.load_inherited_protocols(cls.cfg)

    def test_safe_reader_has_fixed_prefix_and_development_population(self) -> None:
        frame, audit = pc.safe_read_development(self.cfg, self.bs_cfg)
        self.assertEqual(audit["pandas_read_csv_nrows"], 3007)
        self.assertEqual(audit["development_rows_retained"], 3005)
        self.assertFalse(audit["full_causal_file_hash_computed"])
        self.assertFalse(audit["test_accessed"])
        self.assertEqual(len(frame), 3005)

    def test_folds_preserve_target_row_isolation_and_counts(self) -> None:
        frame, _ = pc.safe_read_development(self.cfg, self.bs_cfg)
        folds = self.common.build_outer_folds(frame, self.bs_cfg)
        expected = self.cfg["expected_windows"]
        for name, train, validation in folds:
            self.assertEqual(len(train.origin_row), expected[name]["outer_train_origins"])
            self.assertEqual(len(validation.origin_row), expected[name]["outer_validation_origins"])
            self.assertFalse(set(train.origin_row) & set(validation.origin_row))
            self.assertTrue(np.all(validation.target_end_age_h < 1819.997738))
            fit, stopper = self.common.split_inner_fit_stopper(train, 0.15)
            self.assertFalse(set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)))

    def test_timesnet_output_and_delta_contract(self) -> None:
        model = pc.TimesNetVoltageAdapter(self.bs_cfg, 14, 0.1)
        x = torch.randn(4, 48, 14)
        delta = pc.model_delta_prediction(model, x)
        self.assertEqual(tuple(delta.shape), (4, 24))
        self.assertTrue(bool(torch.isfinite(delta).all()))

    def test_physics_candidate_is_differentiable_and_zero_without_support(self) -> None:
        pred = torch.tensor([0.4, 0.8], dtype=torch.float64, requires_grad=True)
        loss = pc.candidate_polarization_order_loss(pred, torch.tensor([1]), torch.tensor([0]))
        grad = pc.finite_gradient_norm(loss, [pred])
        self.assertGreater(grad, 0.0)
        self.assertEqual(float(pc.candidate_polarization_order_loss(pred, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))), 0.0)

    def test_emd_cache_depends_only_on_origin_history_not_target_values(self) -> None:
        frame, _ = pc.safe_read_development(self.cfg, self.bs_cfg)
        _name, train, _validation = self.common.build_outer_folds(frame, self.bs_cfg)[0]
        mini = train.take(np.arange(2, dtype=int))
        first = pc.load_emd_cache_additions(mini)
        mutated = replace(mini, y_delta_V=mini.y_delta_V + 999.0)
        second = pc.load_emd_cache_additions(mutated)
        for origin in first:
            self.assertTrue(np.array_equal(first[origin], second[origin]))

    def test_no_unbounded_csv_reader_or_n1_full_hash_in_production_reader(self) -> None:
        source = Path(pc.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        reader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "safe_read_development")
        calls = [node for node in ast.walk(reader) if isinstance(node, ast.Call)]
        pd_reads = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "read_csv"]
        self.assertTrue(pd_reads)
        for node in pd_reads:
            self.assertTrue(any(k.arg == "nrows" and isinstance(k.value, ast.Constant) and k.value.value == 3007 for k in node.keywords))
        self.assertIn("full causal CSV hashing is forbidden", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
