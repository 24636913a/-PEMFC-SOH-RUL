"""Synthetic-only tests of new execution/cache plumbing; no RealHyFC I/O."""
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / 'scripts'))
import run_f_training as runner
import audit_f_predictions as audit


class TestFormalExecution(unittest.TestCase):
    def setUp(self):
        root = PHASE / 'outputs/synthetic_execution_tests'
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_three_real_model_training_and_checkpoint_plumbing(self):
        cfg = yaml.safe_load(runner.PROTOCOL.read_text(encoding='utf-8'))
        cfg = copy.deepcopy(cfg)
        # Test-only budget injection; never amend the frozen execution protocol.
        cfg['hyperparameters']['max_epochs'] = 1
        rng = np.random.default_rng(111)
        train = runner.e.c.Windows(rng.normal(size=(4, 48, 14)).astype('float32'),
                  rng.normal(size=(4, 24)).astype('float32'), np.arange(47, 51),
                  np.arange(48, 52)[:, None] + np.arange(24)[None, :], np.arange(4))
        val = runner.e.c.Windows(rng.normal(size=(2, 48, 14)).astype('float32'),
                  rng.normal(size=(2, 24)).astype('float32'), np.arange(100, 102),
                  np.arange(101, 103)[:, None] + np.arange(24)[None, :], np.arange(2))
        scaler = runner.e.CommonScaler().fit(train.x)
        tx, vx = scaler.transform(train.x), scaler.transform(val.x)
        state = {'run_id': 'SYNTHETIC_ONLY', 'git_commit': 'SYNTHETIC', 'groups': {}}
        for folder in ('audits', 'training_logs', 'checkpoints'):
            (self.directory / folder).mkdir()
        torch.set_num_threads(1)
        with patch.object(runner, 'OUT', self.directory):
            init = runner.initialization_audit(cfg)
            for group in runner.GROUPS:
                info = runner.fit_one(group, cfg, 'synthetic_protocol', 'synthetic_code', {}, init,
                    train, tx, vx, val, scaler, torch.device('cpu'), state)
                self.assertEqual(info['stop_epoch'], 1)
                self.assertEqual(state['groups'][group]['status'], 'COMPLETED_CHECKPOINT_FROZEN')
                self.assertTrue(state['groups'][group]['checkpoint_reload']['pass'])
                saved = torch.load(self.directory / f'checkpoints/{group}_best_seed42.pt', weights_only=True)
                runner.e.set_seed(42)
                original = runner.make_model(group, cfg, torch.device('cpu')).state_dict()
                self.assertTrue(any(not torch.equal(v, saved['state_dict'][k]) for k, v in original.items()))
                self.assertEqual(saved['model_group'], group)
                self.assertEqual(saved['protocol_sha256'], 'synthetic_protocol')
            with patch.object(torch.optim, 'AdamW', side_effect=AssertionError('must not retrain')):
                runner.fit_one('F0', cfg, 'synthetic_protocol', 'synthetic_code', {}, init,
                    train, tx, vx, val, scaler, torch.device('cpu'), state)

    def test_prefix_suffix_invariance_and_no_full_hash(self):
        ages = np.concatenate([np.linspace(73, 1819, 3005), [1820, 1821]])
        data = pd.DataFrame({'bin_end_ophrs2_h': ages, 'cell_voltage_mean': .7})
        a, b = self.directory / 'a.csv', self.directory / 'b.csv'
        data.to_csv(a, index=False)
        pd.concat([data, pd.DataFrame({'bin_end_ophrs2_h': [2300]*9, 'cell_voltage_mean': [999]*9})]).to_csv(b, index=False)
        with patch.object(runner.e.c, 'N1', a):
            left = runner.e.c.load_development({})
        with patch.object(runner.e.c, 'N1', b):
            right = runner.e.c.load_development({})
        pd.testing.assert_frame_equal(left, right)
        self.assertEqual(len(left), 3005)
        with self.assertRaisesRegex(RuntimeError, 'PROHIBITED'):
            runner.file_hash(self.directory / 'causal_features_30min.csv')

    def test_independent_metrics_and_fail_closed_keys_truth(self):
        origins = np.repeat(np.arange(2103, 2308), 24)
        horizons = np.tile(np.arange(1, 25), 205)
        rng = np.random.default_rng(321)
        y = rng.normal(0, .01, len(origins))
        f = pd.DataFrame({'experiment_group': 'F0', 'split': 'validation', 'seed': 42,
             'protocol_sha256': 'synthetic', 'origin_row': origins, 'horizon_step': horizons,
             'target_row': origins+horizons, 'target_age_h': 1400.,
             'y_true_delta_V': y, 'y_pred_delta_V': y+rng.normal(0, .003, len(y))})
        ref = f[['origin_row', 'horizon_step', 'target_row', 'y_true_delta_V']]
        audit.validate_frame(f, 'F0', 'validation', 'synthetic', ref)
        for k, v in audit.metrics_from_records(f).items():
            self.assertAlmostEqual(v, runner.e.numpy_metrics(f)[k], places=10)
        path = self.directory / 'predictions.csv'
        f.to_csv(path, index=False, float_format='%.17g')
        reloaded = pd.read_csv(path, float_precision='round_trip')
        self.assertEqual(audit.metrics_from_records(f), audit.metrics_from_records(reloaded))
        for kind in ('missing', 'duplicate', 'truth', 'boundary'):
            changed = f.copy()
            if kind == 'missing': changed = changed.iloc[:-1]
            if kind == 'duplicate': changed.iloc[1] = changed.iloc[0]
            if kind == 'truth': changed.loc[0, 'y_true_delta_V'] += .1
            if kind == 'boundary': changed.loc[0, 'target_age_h'] = 2000
            with self.assertRaises(RuntimeError, msg=kind):
                audit.validate_frame(changed, 'F0', 'validation', 'synthetic', ref)


if __name__ == '__main__':
    unittest.main(verbosity=2)
