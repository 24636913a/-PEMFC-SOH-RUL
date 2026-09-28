"""Protocol-level tests for the frozen, Development-only backbone screen.

These tests exercise production validation and metric functions.  They never
open the causal CSV beyond a mocked registered-prefix call and never load a
model checkpoint, prediction cache, raw record, or Test artifact.
"""
from __future__ import annotations

import copy
import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))

import backbone_common as common  # noqa: E402
import run_backbone_screening as runner  # noqa: E402


MODELS = ["TimesNet", "SimpleTM", "DUET", "ModernTCN"]


def synthetic_frame(cfg: dict, n_rows: int = 180, *, start_h: float = 72.788690) -> pd.DataFrame:
    """Create a legal 0.5 h development-like frame without reading any CSV."""
    starts = start_h + .5 * np.arange(n_rows)
    data: dict[str, object] = {
        "bin_start_ophrs2_h": starts,
        "bin_end_ophrs2_h": starts + .5,
        "last_datetime": pd.date_range("2020-01-01", periods=n_rows, freq="30min").astype(str),
        "temporal_segment_id": np.ones(n_rows, dtype=int),
        "bin_valid_for_windows": np.ones(n_rows, dtype=bool),
    }
    for i, name in enumerate(cfg["data_firewall"]["f2_raw_feature_order"]):
        data[name] = (i + 1) + np.arange(n_rows, dtype=float) * .001
    data["cell_voltage_mean"] = 0.7 - np.arange(n_rows, dtype=float) * .0001
    return pd.DataFrame(data)


class BackboneProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = common.load_config()

    def test_frozen_contract_and_unique_candidate_set(self) -> None:
        self.assertEqual(MODELS, list(self.cfg["preflight"]["required_models"]))
        self.assertEqual(self.cfg["task"]["input_shape"], ["batch", 48, 14])
        self.assertEqual(len(self.cfg["data_firewall"]["f2_raw_feature_order"]), 14)
        self.assertEqual(self.cfg["data_firewall"]["f2_raw_feature_order"][0], "cell_voltage_mean")
        self.assertEqual(self.cfg["scope"]["external_feature_decomposition"], "prohibited")
        self.assertTrue(self.cfg["scope"]["test_access"] == "prohibited")

    def test_registered_prefix_loader_requests_exact_nrows_and_filters_without_test_read(self) -> None:
        fake = synthetic_frame(self.cfg, n_rows=3007)
        fake.loc[3005:, "bin_end_ophrs2_h"] = float(self.cfg["development_end_h_exclusive"])
        calls: list[dict] = []

        def fake_read_csv(*_args, **kwargs):
            calls.append(kwargs)
            return fake.copy()

        with patch.object(common.pd, "read_csv", side_effect=fake_read_csv):
            development, audit = common.read_development_prefix(self.cfg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["nrows"], 3007)
        self.assertEqual(len(development), 3005)
        self.assertEqual(audit["test_rows_read"], 0)
        self.assertEqual(audit["rows_after_registered_prefix_read"], 0)

    def test_windows_use_raw_f2_in_order_and_delta_anchor_target(self) -> None:
        frame = synthetic_frame(self.cfg)
        windows = common.build_windows(frame, self.cfg, float(frame.bin_start_ophrs2_h.iloc[0]), float(frame.bin_end_ophrs2_h.iloc[-1]))
        origin = int(windows.origin_row[0])
        expected_x = frame.loc[origin - 47:origin, self.cfg["data_firewall"]["f2_raw_feature_order"]].to_numpy(np.float32)
        expected_delta = frame.loc[origin + 1:origin + 24, "cell_voltage_mean"].to_numpy(np.float32) - frame.loc[origin, "cell_voltage_mean"]
        self.assertEqual(windows.x.shape[1:], (48, 14))
        np.testing.assert_allclose(windows.x[0], expected_x)
        np.testing.assert_allclose(windows.y_delta_V[0], expected_delta, rtol=1e-5, atol=5e-8)
        self.assertAlmostEqual(float(windows.anchor_V[0]), float(frame.loc[origin, "cell_voltage_mean"]))
        self.assertFalse(any("future" in name.lower() for name in self.cfg["data_firewall"]["f2_raw_feature_order"]))

    def test_all_models_share_one_raw_input_contract_without_future_tensor(self) -> None:
        adapter_source = inspect.getsource(common.model_delta_prediction)
        training_source = inspect.getsource(runner.inner_select_epoch) + inspect.getsource(runner.full_outer_refit)
        self.assertIn("x_scaled[:, -1, 0:1]", inspect.getsource(common.delta_from_standardized_absolute))
        self.assertNotIn("future_covariate", adapter_source)
        self.assertNotIn("future_covariate", training_source)
        self.assertIn("_loader(x_fit, y_fit", training_source)
        self.assertIn("_loader(x_train, y_train", training_source)
        self.assertEqual(set(runner.MODELS), set(MODELS))

    def test_swt_is_not_an_external_loader_feature_transform(self) -> None:
        phase_wrapper_source = Path(common.__file__).read_text(encoding="utf-8") + Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("pywt", phase_wrapper_source)
        self.assertNotIn("pywt", inspect.getsource(common.FoldPreprocessor))
        self.assertIn("model.SimpleTM", inspect.getsource(common.SimpleTMVoltageAdapter))
        self.assertNotIn("SWT", inspect.getsource(common.FoldPreprocessor))

    def test_window_rejects_target_at_development_boundary(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["development_end_h_exclusive"] = 130.0
        frame = synthetic_frame(cfg, n_rows=180, start_h=72.788690)
        with self.assertRaisesRegex(RuntimeError, "Development boundary"):
            common.build_windows(frame, cfg, 72.788690, float(frame.bin_end_ophrs2_h.iloc[-1]))

    def test_inner_stopper_removes_target_row_overlap(self) -> None:
        frame = synthetic_frame(self.cfg, n_rows=260)
        windows = common.build_windows(frame, self.cfg, float(frame.bin_start_ophrs2_h.iloc[0]), float(frame.bin_end_ophrs2_h.iloc[-1]))
        fit, stopper = common.split_inner_fit_stopper(windows, .15)
        self.assertFalse(set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)))
        self.assertLess(int(fit.origin_row.max()), int(stopper.origin_row.min()))

    def test_standardized_absolute_inverse_is_anchor_delta_in_volts(self) -> None:
        x_scaled = np.zeros((2, 48, 14), dtype=np.float32)
        x_scaled[:, -1, 0] = [2.0, -1.0]
        future_absolute_scaled = np.asarray([[2.25, 1.75], [-.5, -1.5]], dtype=np.float32)
        delta_scaled = common.delta_from_standardized_absolute(future_absolute_scaled, x_scaled)
        scale = .02
        np.testing.assert_allclose(delta_scaled * scale, np.asarray([[.005, -.005], [.01, -.01]], dtype=np.float32))

    def test_query_and_target_time_weighting_are_distinct_mse_definitions(self) -> None:
        truth = np.zeros((2, 24), dtype=float)
        pred = truth.copy()
        pred[0, 1] = .004  # target_row=1 has two forecast queries with unequal squared error.
        target_row = np.vstack([np.arange(24), np.arange(1, 25)])
        metrics = common.delta_metrics(truth, pred, target_row)
        expected_query_rmse = np.sqrt((.004 ** 2) / 48) * 1000
        se_by_target = []
        for target in np.unique(target_row):
            se_by_target.append(np.mean(((pred - truth)[target_row == target] * 1000) ** 2))
        expected_target_rmse = np.sqrt(np.mean(se_by_target))
        self.assertAlmostEqual(metrics["rmse_mV"], expected_query_rmse)
        self.assertAlmostEqual(metrics["target_time_equal_rmse_mV"], expected_target_rmse)
        self.assertNotEqual(metrics["rmse_mV"], metrics["target_time_equal_rmse_mV"])

    def test_p99_and_max_keep_tail_errors(self) -> None:
        truth = np.zeros((50, 24), dtype=float)
        pred = truth.copy()
        pred.reshape(-1)[:12] = .1
        metrics = common.delta_metrics(truth, pred, np.arange(1200).reshape(50, 24))
        self.assertGreater(metrics["p99_abs_error_mV"], 0.0)
        self.assertEqual(metrics["max_abs_error_mV"], 100.0)

    def test_seed42_gate_boundaries_and_failed_challenger_extra_seed_block(self) -> None:
        rows = []
        values = {
            "TimesNet": (100.0, 100.0, 100.0, 100.0, 100.0),
            # (fold1/2 RMSE, fold3 RMSE, P99, fold1/2 target-equal, fold3 target-equal)
            # SimpleTM lands exactly on every permitted gate boundary after three-fold means.
            "SimpleTM": (91.0, 103.0, 105.0, 102.5, 110.0),
            "DUET": (90.0, 103.1, 104.0, 101.0, 109.0),
            "ModernTCN": (90.0, 102.0, 104.0, 101.0, 109.0),
        }
        for model, (other_fold_rmse, fold3, p99, other_fold_target, fold3_target) in values.items():
            for fold in ("fold_1", "fold_2", "fold_3"):
                rows.append({"model": model, "seed": 42, "fold": fold, "rmse_mV": fold3 if fold == "fold_3" else other_fold_rmse, "mae_mV": 1.0, "r2": .5, "endpoint_rmse_mV": 1.0, "p95_abs_error_mV": 1.0, "p99_abs_error_mV": p99, "max_abs_error_mV": p99, "target_time_equal_rmse_mV": fold3_target if fold == "fold_3" else other_fold_target, "parameter_count": 10, "peak_gpu_memory_MiB": 1.0, "total_inner_plus_refit_time_s": 1.0})
        summary = common.summarize_seed(pd.DataFrame(rows))
        preflight = pd.DataFrame({"model": MODELS, "status": ["PASS"] * 4})
        gate = runner.seed42_gate(summary, preflight, self.cfg).set_index("model")
        self.assertTrue(bool(gate.loc["SimpleTM", "seed42_gate_pass"]))
        self.assertFalse(bool(gate.loc["DUET", "seed42_gate_pass"]))
        self.assertFalse(bool(gate.loc["DUET", "additional_seeds_authorized"]))

    def test_official_source_fingerprints_and_single_configuration_are_frozen(self) -> None:
        fingerprints = common.source_fingerprint_audit(self.cfg)
        self.assertTrue(bool(fingerprints["pass"].all()))
        self.assertEqual(set(fingerprints["model"]), set(MODELS))
        self.assertFalse(self.cfg["official_sources"]["ModernTCN"]["config"]["use_multi_scale"])
        self.assertEqual(self.cfg["official_sources"]["SimpleTM"]["config"]["m"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
