"""Synthetic-only regression tests for the frozen Phase DF causality contract."""
from __future__ import annotations

import inspect
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from decomposition_features import CONTEXT, causal_slope8, emd_feature, method_feature, simple_trend_features, vmd_feature
from df_protocol_adapter import augment_windows, load_bs_common_readonly, load_bs_protocol, load_df_protocol, model_delta_prediction, read_development_prefix
from run_stage0 import _endpoint_pairs, _resolve_alpha
from run_stage1 import _decomposition_gate, _raw_gate


def history() -> np.ndarray:
    t = np.arange(CONTEXT, dtype=np.float64)
    return .72 + .0008 * t + .005 * np.sin(2 * np.pi * t / 12) + .002 * np.sin(2 * np.pi * t / 5)


@dataclass(frozen=True)
class SyntheticWindow:
    x: np.ndarray
    origin_row: np.ndarray


class PhaseDFProtocolTests(unittest.TestCase):
    def test_simpletrend_exact_hand_calculation_and_slope_units(self) -> None:
        values = np.arange(CONTEXT, dtype=np.float64) * .1
        result = simple_trend_features(values).features
        self.assertEqual(result.shape, (48, 2))
        self.assertEqual(float(result[0, 1]), 0.0)
        self.assertAlmostEqual(float(result[7, 0]), .35, places=7)
        self.assertAlmostEqual(float(result[7, 1]), .2, places=7)  # 0.1 V / 0.5 h
        self.assertEqual(float(causal_slope8(values)[0]), 0.0)

    def test_vmd_fixed_k_dc_ordering_and_shape(self) -> None:
        result = vmd_feature(history(), alpha=1000)
        self.assertEqual(result.features.shape, (48, 2))
        self.assertTrue(np.isfinite(result.features).all())
        frequencies = result.audit["final_center_frequencies_cycles_per_sample"]
        self.assertEqual(frequencies, sorted(frequencies))
        with self.assertRaises(ValueError):
            vmd_feature(history(), alpha=1000, DC=1)
        with self.assertRaises(ValueError):
            vmd_feature(history(), alpha=1000, K=3)

    def test_emd_imf_residue_reconstruction_and_insufficient_imfs(self) -> None:
        result = emd_feature(history())
        self.assertEqual(result.features.shape, (48, 2))
        self.assertLess(result.audit["normalized_reconstruction_max_abs"], 1e-8)
        constant = emd_feature(np.ones(CONTEXT, dtype=np.float64))
        self.assertEqual(constant.features.shape, (48, 2))
        self.assertEqual(constant.audit["imf_count"], 0)
        self.assertTrue(np.isfinite(constant.features).all())

    def test_production_feature_cannot_receive_target_or_full_series(self) -> None:
        self.assertNotIn("target", inspect.signature(method_feature).parameters)
        with self.assertRaises(TypeError):
            method_feature("EMD", history(), target=np.zeros(24))
        with self.assertRaises(ValueError):
            method_feature("SimpleTrend", np.ones(49))

    def test_future_perturbation_and_prefix_invariance(self) -> None:
        origin = 60
        series = np.concatenate([np.linspace(.70, .74, 61), np.linspace(2.0, 10.0, 19)])
        changed = series.copy(); changed[origin + 1:] = np.linspace(-100.0, 100.0, len(changed) - origin - 1)
        before_history = series[origin - 47:origin + 1]
        after_history = changed[origin - 47:origin + 1]
        prefix_history = series[:origin + 1][origin - 47:origin + 1]
        for method, alpha in (("SimpleTrend", None), ("VMD", 1000), ("EMD", None)):
            before = method_feature(method, before_history, vmd_alpha=alpha).features
            after = method_feature(method, after_history, vmd_alpha=alpha).features
            prefix = method_feature(method, prefix_history, vmd_alpha=alpha).features
            self.assertTrue(np.array_equal(before, after), method)
            self.assertTrue(np.array_equal(before, prefix), method)

    def test_feature_cache_is_origin_window_level_and_appends_after_raw_f2(self) -> None:
        raw = np.zeros((2, 48, 14), dtype=np.float32)
        origins = np.asarray([100, 101], dtype=np.int64)
        window = SyntheticWindow(x=raw, origin_row=origins)
        features = {100: np.ones((48, 2), np.float32), 101: np.full((48, 2), 2, np.float32)}
        augmented = augment_windows(window, features)
        self.assertEqual(augmented.x.shape, (2, 48, 16))
        self.assertTrue(np.array_equal(augmented.x[:, :, :14], raw))
        self.assertEqual(float(augmented.x[1, 0, 14]), 2.0)
        with self.assertRaisesRegex(RuntimeError, "CACHE_AUDIT_FAIL"):
            augment_windows(window, {100: features[100]})

    def test_endpoint_pairs_require_exact_half_hour_and_same_segment(self) -> None:
        origins = np.asarray([10, 11, 12, 14])
        ages = np.asarray([100.0, 100.5, 101.0, 102.0])
        self.assertEqual(_endpoint_pairs(origins, ages, {10: 1, 11: 1, 12: 2, 14: 2}), [(0, 1)])
        self.assertEqual(_endpoint_pairs(origins, np.asarray([100.0, 100.4, 100.8, 101.6]), {10: 1, 11: 1, 12: 1, 14: 1}), [])

    def test_alpha_resolution_is_unsupervised_and_has_registered_tie_break(self) -> None:
        rows = pd.DataFrame([
            {"alpha": 500, "finite_feature_rate": 1.0, "mode_separation_pass_rate": .91, "endpoint_unstable": False, "endpoint_revision_p95_V": .002},
            {"alpha": 1000, "finite_feature_rate": 1.0, "mode_separation_pass_rate": .91, "endpoint_unstable": False, "endpoint_revision_p95_V": .002},
            {"alpha": 2000, "finite_feature_rate": 1.0, "mode_separation_pass_rate": .90, "endpoint_unstable": False, "endpoint_revision_p95_V": .001},
        ])
        self.assertEqual(_resolve_alpha(rows), (500, "PASS"))
        self.assertEqual(_resolve_alpha(rows.assign(finite_feature_rate=.99)), (None, "MODE_OR_ENDPOINT_NOT_IDENTIFIABLE"))

    def test_firewall_reads_3007_then_filters_to_3005_without_test_access(self) -> None:
        df_cfg, _ = load_df_protocol(); bs_cfg, _ = load_bs_protocol(df_cfg)
        columns = list(dict.fromkeys(bs_cfg["data_firewall"]["required_columns"]))
        table = pd.DataFrame({column: np.zeros(3007) for column in columns})
        table["bin_end_ophrs2_h"] = np.r_[np.full(3005, 1819.0), np.full(2, 1820.0)]
        with patch("df_protocol_adapter.pd.read_csv", return_value=table) as mocked:
            development, audit = read_development_prefix(df_cfg, bs_cfg)
        self.assertEqual(mocked.call_args.kwargs["nrows"], 3007)
        self.assertEqual(len(development), 3005)
        self.assertEqual(audit["boundary_rows_filtered"], 2)
        self.assertFalse(audit["test_accessed"])

    def test_production_firewall_rejects_test_labelled_or_unregistered_source_path(self) -> None:
        df_cfg, _ = load_df_protocol(); bs_cfg, _ = load_bs_protocol(df_cfg)
        forbidden = {**df_cfg, "data_firewall": {**df_cfg["data_firewall"], "source_relative_path": "Phase_N1_Data_Foundation/outputs/Test/causal_features_30min.csv"}}
        with patch("df_protocol_adapter.pd.read_csv") as mocked:
            with self.assertRaisesRegex(RuntimeError, "DATA_FIREWALL_FAIL"):
                read_development_prefix(forbidden, bs_cfg)
        mocked.assert_not_called()

    def test_frozen_fold_expectations_and_identity_are_registered(self) -> None:
        df_cfg, _ = load_df_protocol()
        self.assertEqual(list(df_cfg["expected_windows"]), ["fold_1", "fold_2", "fold_3"])
        self.assertEqual(df_cfg["expected_windows"]["fold_1"]["outer_train_origins"], 1048)
        self.assertEqual(df_cfg["expected_windows"]["fold_3"]["outer_validation_origins"], 466)

    def test_train_only_scaler_does_not_fit_validation(self) -> None:
        common = load_bs_common_readonly()
        train = np.zeros((2, 48, 16), dtype=np.float32)
        train[:, :, 0] = 1.0
        validation = np.full((1, 48, 16), 999.0, dtype=np.float32)
        scaler = common.FoldPreprocessor().fit(train)
        fitted_mean = scaler.mean.copy()
        scaler.transform(validation)
        self.assertTrue(np.array_equal(scaler.mean, fitted_mean))

    def test_delta_adapter_contract(self) -> None:
        class AbsoluteEcho(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x[:, -24:, 0]
        x = torch.zeros((3, 48, 16), dtype=torch.float32)
        x[:, -24:, 0] = 2.0
        self.assertEqual(tuple(model_delta_prediction(AbsoluteEcho(), x).shape), (3, 24))

    def test_seed_gates_require_raw_and_simpletrend_comparisons(self) -> None:
        raw = pd.Series({"formal_RMSE_mV": 10.0, "Fold3_RMSE_mV": 10.0, "P99_mV": 10.0, "target_time_equal_RMSE_mV": 10.0, "Fold3_target_time_equal_RMSE_mV": 10.0, "endpoint_RMSE_mV": 10.0})
        candidate = pd.Series({"formal_RMSE_mV": 9.0, "Fold3_RMSE_mV": 10.0, "P99_mV": 10.0, "target_time_equal_RMSE_mV": 10.0, "Fold3_target_time_equal_RMSE_mV": 10.0, "endpoint_RMSE_mV": 10.0})
        trend = pd.Series({"formal_RMSE_mV": 9.5, "Fold3_RMSE_mV": 10.0, "P99_mV": 10.0, "endpoint_RMSE_mV": 10.0})
        cfg, _ = load_df_protocol()
        self.assertTrue(all(_raw_gate(candidate, raw, cfg).values()))
        self.assertTrue(all(_decomposition_gate(candidate, trend, cfg).values()))


if __name__ == "__main__":
    unittest.main()
