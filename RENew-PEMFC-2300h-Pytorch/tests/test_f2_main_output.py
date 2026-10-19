"""Output-only regression tests; no training or target selection.

Run: python -B -m unittest discover -s tests -p test_f2_main_output.py
"""
import ast
from contextlib import redirect_stderr
import io
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('f2_main_output_tested', ROOT / 'RealHyFC_F2_Main.py')
MAIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MAIN
SPEC.loader.exec_module(MAIN)
BASE = 'b660d8af5e2e9a7734a5b0fa6292c5244a967864'


def synthetic_predictions():
    rows = []
    for origin in [100, 101, 104]:
        for step in range(1, 25):
            truth = step * .001
            rows.append(dict(partition='VALIDATION', seed=42, origin_row=origin,
                             target_row=origin + step, horizon_step=step,
                             target_age_h=(origin + step) * .5,
                             y_true_delta_V=truth, y_pred_delta_V=truth + .002,
                             true_raw_target_voltage_V=.7 + truth,
                             predicted_voltage_V=.702 + truth))
    return pd.DataFrame(rows)


class F2OutputTests(unittest.TestCase):
    def test_scientific_core_unchanged(self):
        old = subprocess.check_output(['git', 'show', BASE + ':RENew-PEMFC-2300h-Pytorch/RealHyFC_F2_Main.py'],
                                      cwd=ROOT.parent).decode('utf-8')
        new = (ROOT / 'RealHyFC_F2_Main.py').read_text(encoding='utf-8')
        def definitions(text):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        previous, current = definitions(old), definitions(new)
        # Every original definition except presentation/CLI and its import whitelist stays exact.
        excluded = {'initialize_f2', 'historical_checkpoint', 'load_checkpoint', 'predict_partitions', 'main'}
        for name in previous.keys() - excluded:
            with self.subTest(name=name):
                self.assertEqual(previous[name], current[name])

    def test_fixed_lead_has_one_target_per_origin(self):
        frame = synthetic_predictions()
        fixed = MAIN.fixed_horizon_plot_data(frame.sample(frac=1, random_state=4))
        self.assertEqual(fixed.horizon_step.tolist(), [24, 24, 24])
        self.assertEqual(fixed.target_row.tolist(), [124, 125, 128])
        self.assertEqual(len(frame), 72)  # plot filtering never mutates the metric population

    def test_breaks_gaps_without_interpolation(self):
        fixed = MAIN.fixed_horizon_plot_data(synthetic_predictions())
        lines = MAIN.voltage_plot_segments(fixed)
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines.iloc[2].isna().all())
        np.testing.assert_array_equal(lines.dropna().target_age_h, fixed.target_age_h)

    def test_plot_rule_cannot_remove_bad_metric_targets(self):
        frame = synthetic_predictions()
        frame.loc[frame.horizon_step == 1, 'y_pred_delta_V'] += .1
        before = frame.copy(deep=True)
        metrics = MAIN.calculate_metrics(frame)
        MAIN.fixed_horizon_plot_data(frame)
        errors = frame.y_pred_delta_V.to_numpy() - frame.y_true_delta_V.to_numpy()
        self.assertEqual(metrics['queries'], 72)
        self.assertAlmostEqual(metrics['RMSE_mV'], np.sqrt(np.mean(errors ** 2)) * 1000, places=12)
        self.assertGreater(metrics['RMSE_mV'], MAIN.calculate_metrics(
            frame.loc[frame.horizon_step == 24])['RMSE_mV'])
        pd.testing.assert_frame_equal(before, frame)

    def test_summary_does_not_mix_partitions(self):
        rows = [dict(seed=seed, partition=partition, RMSE_mV=value, MAE_mV=value / 2,
                     R2=.9, queries=4920, scientific_status='reused_validation')
                for seed, value in zip(MAIN.SUPPORTED_SEEDS, [5., 6., 7., 8., 9., 10.])
                for partition in MAIN.PARTITIONS]
        for row in rows:
            if row['partition'] != 'VALIDATION': row['RMSE_mV'] = 999.
        summary = MAIN.evaluation_summary(rows, 'VALIDATION', list(MAIN.SUPPORTED_SEEDS))
        self.assertEqual(summary['RMSE_V'], .0075)
        self.assertEqual(summary['queries_per_seed'], [4920] * 6)
        self.assertIn('no ensemble', summary['aggregation'])

    def test_mixed_partition_plot_rejected(self):
        frame = synthetic_predictions()
        frame.loc[frame.origin_row == 104, 'partition'] = 'TRAIN'
        with self.assertRaisesRegex(RuntimeError, 'PARTITIONS_MIXED'):
            MAIN.fixed_horizon_plot_data(frame)

    def test_missing_summary_seed_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'SEED_OR_PARTITION_MISMATCH'):
            MAIN.evaluation_summary([], 'VALIDATION', [42])

    def test_supported_cli_and_default(self):
        self.assertEqual(MAIN.SUPPORTED_SEEDS, (42, 103, 193, 202, 210, 212))
        self.assertEqual(MAIN.parse_arguments([]).seed, '42')
        for seed in [*map(str, MAIN.SUPPORTED_SEEDS), 'all']:
            self.assertEqual(MAIN.parse_arguments(['--seed', seed]).seed, seed)

    def test_legacy_seeds_rejected_during_parsing(self):
        for seed in ['52', '62']:
            error = io.StringIO()
            with self.subTest(seed=seed), redirect_stderr(error), self.assertRaises(SystemExit) as failure:
                MAIN.parse_arguments(['--seed', seed])
            self.assertEqual(failure.exception.code, 2)
            self.assertIn('invalid choice', error.getvalue())

    def test_six_asset_identities(self):
        for seed in MAIN.SUPPORTED_SEEDS:
            row = MAIN.historical_checkpoint(seed)
            self.assertEqual(row.seed, seed)
            self.assertGreater(row.best_epoch, 0)
            self.assertEqual(len(row.checkpoint_sha256), 64)
            reference = MAIN.read_csv(MAIN.ROOT.parent / row.reference_prediction)
            self.assertEqual(len(reference), 4920)
            self.assertEqual(reference.origin_row.nunique(), 205)


if __name__ == '__main__':
    unittest.main()
