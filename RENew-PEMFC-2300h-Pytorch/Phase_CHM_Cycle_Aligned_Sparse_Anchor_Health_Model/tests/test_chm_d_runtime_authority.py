from __future__ import annotations

import sys
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))
sys.path.insert(0, str(PHASE))

from chm_data import FORBIDDEN_MODEL_INPUTS, load_frozen_chm_data, verify_runtime_authority
from chm_evaluation import aggregate_metrics, outer_training_anchors
from chm_model import CHMModel


class CHMDRuntimeAuthorityTests(unittest.TestCase):
    def test_01_runtime_authority_hashes_pass_before_optimizer_creation(self) -> None:
        runtime = verify_runtime_authority()
        self.assertEqual(runtime["authority"]["authority_version"], "FINAL_V2A")
        self.assertEqual(runtime["authority"]["current_status"], "FINAL_V2A_FROZEN_READY_FOR_CHM_D")

    def test_02_only_three_frozen_tables_are_runtime_variable_inputs(self) -> None:
        runtime = verify_runtime_authority()
        self.assertEqual(runtime["authority"]["CHM_D_runtime_variable_data_inputs_only"][:3], ["inputs/CHM_HOU_TABLE.csv", "inputs/CHM_SPARSE_ANCHOR_TARGETS.csv", "inputs/CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv"])
        self.assertTrue(all("test" not in item.lower() and "loadcycle" not in item.lower() for item in runtime["authority"]["CHM_D_runtime_variable_data_inputs_only"]))

    def test_03_hou_chronology_and_counts_are_exact(self) -> None:
        data = load_frozen_chm_data()
        self.assertEqual(len(data.hou), 152)
        self.assertEqual(int(data.hou["hou_kind"].eq("COMPLETE_CYCLE_HOU").sum()), 146)
        self.assertEqual(int(data.hou["hou_kind"].eq("PREANCHOR_PARTIAL_HOU").sum()), 6)
        self.assertTrue(data.hou["measurement_end_datetime"].is_monotonic_increasing)

    def test_04_age_time_and_cycle_are_not_model_inputs(self) -> None:
        data = load_frozen_chm_data()
        self.assertFalse(set(data.scaler.input_feature_names).intersection(FORBIDDEN_MODEL_INPUTS))
        self.assertEqual(data.model_inputs.shape[0], 152)

    def test_05_outer_training_anchors_exclude_heldout_target(self) -> None:
        for checkpoint in (300, 600, 850, 1150, 1500, 1750):
            training = outer_training_anchors(checkpoint)
            self.assertEqual(len(training), 5)
            self.assertNotIn(checkpoint, training)

    def test_06_recurrence_is_structurally_nonincreasing_without_penalty(self) -> None:
        model = CHMModel(3)
        states = model.latent_trajectory(torch.zeros((8, 3), dtype=torch.float64)).detach().numpy()
        self.assertTrue(np.all(np.diff(states) <= 0.0))

    def test_07_vectorized_recurrence_matches_direct_frozen_product(self) -> None:
        model = CHMModel(2)
        inputs = torch.tensor([[0.0, 0.0], [0.2, -0.1], [0.4, 0.3], [-0.2, 0.1]], dtype=torch.float64)
        states = model.latent_trajectory(inputs)
        decrement = torch.exp(-1e-3 * torch.nn.functional.softplus(model.network(inputs).squeeze(-1)))
        direct = [torch.sigmoid(model.eta)]
        for index in range(1, len(inputs)):
            direct.append(direct[-1] * decrement[index])
        self.assertTrue(torch.allclose(states, torch.stack(direct), rtol=0.0, atol=1e-14))

    def test_08_final_report_has_all_preregistered_sections(self) -> None:
        spec = importlib.util.spec_from_file_location("chm_d_runner", PHASE / "scripts" / "run_chm_d_development.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        metric = {"MAE": 0.1, "RMSE": 0.2, "bias": 0.0, "Spearman_n6_descriptive_only": 0.5, "calibration_slope_target_on_prediction": 1.0, "calibration_intercept_target_on_prediction": 0.0}
        result = {"base_head": "base", "metrics": {"CHM": metric, "B0": metric, "B1": metric, "B2": metric}, "gates": {f"Gate_{index}": "PASS" for index in range(1, 6)}, "uncertainty": {"bootstrap_triggered": True}, "primary_status": "DEVELOPMENT_HEALTH_MODEL_QUALIFIED", "final_profile": {"selected": "A_BALANCED"}, "S1": {"status": "S1_ROBUSTNESS_SUPPORTED"}, "causal_online": {"MAE": 0.1, "RMSE": 0.2}}
        with tempfile.TemporaryDirectory() as temporary:
            previous_reports = runner.REPORTS
            runner.REPORTS = Path(temporary)
            try:
                runner.build_report(result)
                report = (Path(temporary) / "FINAL_CHM_D_DEVELOPMENT_REPORT.md").read_text(encoding="utf-8")
            finally:
                runner.REPORTS = previous_reports
        for index in range(1, 13):
            self.assertIn(f"## {index}.", report)

    def test_09_aggregate_rmse_uses_numpy_square_on_prediction_arrays(self) -> None:
        predictions = pd.DataFrame({"target": [0.2, 0.5], "official_prediction_three_seed_median": [0.1, 0.8]})
        metrics = aggregate_metrics(predictions)
        self.assertAlmostEqual(metrics["MAE"], 0.2)
        self.assertAlmostEqual(metrics["RMSE"], float(np.sqrt((0.1**2 + 0.3**2) / 2)))
