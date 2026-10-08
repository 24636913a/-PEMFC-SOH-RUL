from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
import pot_common as pc  # noqa: E402


class FinalArtifactContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg, _ = pc.load_protocol()
        cls.bs_cfg, _df_cfg, cls.common = pc.load_inherited_protocols(cls.cfg)

    def test_raw_f2_feature_contract_is_exact_and_future_covariates_are_prohibited(self) -> None:
        features = self.cfg["task"]["raw_f2_feature_order"]
        self.assertEqual(len(features), 14)
        self.assertEqual(features[0], "cell_voltage_mean")
        self.assertEqual(features[-1], "zero_current_fraction")
        self.assertEqual(self.cfg["task"]["future_covariates"], "prohibited")

    def test_metrics_are_inverse_scaled_millivolt_error(self) -> None:
        bs_cfg, _df_cfg, common = pc.load_inherited_protocols(self.cfg)
        truth = np.zeros((1, 24), dtype=np.float64)
        prediction = np.tile(np.array([0.001, -0.002], dtype=np.float64), 12).reshape(1, 24)
        metric = common.delta_metrics(truth, prediction, np.arange(1, 25, dtype=np.int64).reshape(1, 24))
        self.assertAlmostEqual(metric["mae_mV"], 1.5, places=9)
        self.assertAlmostEqual(metric["rmse_mV"], np.sqrt(2.5), places=9)

    def test_production_training_has_no_physics_branch_or_future_actual_inputs(self) -> None:
        source = (PHASE / "scripts" / "run_formal_ablation.py").read_text(encoding="utf-8")
        self.assertIn("PHYSICS_LOSS_NOT_ESTABLISHED", source)
        self.assertIn("M5/M6", source)
        self.assertNotIn("future_actual", source.lower())

    def test_model_dimension_contracts_raw_and_emd(self) -> None:
        self.assertEqual(self.cfg["task"]["input_shape_raw"], ["batch", 48, 14])
        self.assertEqual(self.cfg["task"]["input_shape_emd"], ["batch", 48, 16])
        self.assertEqual(self.cfg["emd_feature"]["information_set"], "same_48_step_origin_conditioned_causal_block")

    def test_stage0_is_the_only_authority_for_physics_branch_execution(self) -> None:
        tree = ast.parse((PHASE / "scripts" / "run_formal_ablation.py").read_text(encoding="utf-8"))
        text = ast.unparse(tree)
        self.assertIn("PHYSICS_BRANCH_STATUS_UNEXPECTED", text)
        self.assertNotIn("candidate_polarization_order_loss(", text)

    def test_resolved_config_file_is_frozen_before_formal_refit(self) -> None:
        resolved = PHASE / "configs" / "RESOLVED_OPTUNA_CONFIGS.yaml"
        resolved_hash = PHASE / "configs" / "RESOLVED_OPTUNA_CONFIGS.sha256"
        self.assertTrue(resolved.is_file())
        self.assertTrue(resolved_hash.is_file())
        self.assertIn("FROZEN_AFTER_FOLD1_INNER_ONLY_OPTUNA_BEFORE_OUTER_EVALUATION", resolved.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
