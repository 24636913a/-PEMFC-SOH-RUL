"""Synthetic-only tests calling the production Phase L guards/contracts."""
import importlib.util
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PHASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('l_audit', PHASE / 'scripts/audit_l_readonly.py')
l = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l)


class FirewallTests(unittest.TestCase):
    def test_test_numeric_read_before_IO(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('I/O must not occur')):
            with self.assertRaises(l.FirewallError):
                l.read_document(l.REPO / '数据/Polcurve_Nominal_RHFC_2000h.csv')

    def test_unbounded_mixed_source(self):
        with self.assertRaises(l.FirewallError):
            l.guard_operation('mixed_source_read')

    def test_mixed_file_hash(self):
        p = l.REPO / l.ROOT / 'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
        with self.assertRaises(l.FirewallError):
            l.verify_digest(p, '0' * 64)

    def test_mixed_git_blob(self):
        with self.assertRaises(l.FirewallError):
            l.guard_operation('mixed_source_git_blob')

    def test_model_inference_and_load(self):
        for op in ['model_forward', 'model_load', 'train', 'new_test_metric']:
            with self.subTest(op=op), self.assertRaises(l.FirewallError):
                l.guard_operation(op)

    def test_checkpoint_write(self):
        with self.assertRaises(l.FirewallError):
            l.guard_path(l.REPO / l.ROOT / 'Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F2-Seed62_best.pt', 'write')

    def test_formal_unblind_even_simulated(self):
        with self.assertRaises(l.FirewallError):
            l.transition('READY_FOR_EXPLICIT_AUTHORIZATION', 'AUTHORIZED_ONE_SHOT', True, True, True)

    def test_missing_evidence(self):
        with self.assertRaises(l.FirewallError):
            l.require_evidence({'AHC_LEDGER': {}}, ['AHC_LEDGER', 'RUL_LEDGER'])

    def test_checkpoint_byte_identity_tamper(self):
        folder = PHASE / 'outputs/synthetic_fixtures'
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as td:
            p = Path(td) / 'synthetic_checkpoint.bin'
            p.write_bytes(b'synthetic identity, not a model')
            expected = l.sha(p.read_bytes())
            paths = {str(p.resolve())}
            self.assertEqual(l.verify_digest(p, expected, 'checkpoint_hash', paths), expected)
            p.write_bytes(b'tampered synthetic identity')
            with self.assertRaises(l.FirewallError):
                l.verify_digest(p, expected, 'checkpoint_hash', paths)

    def test_qualification_missing_cannot_become_READY(self):
        with self.assertRaises(l.FirewallError):
            l.transition('RUNTIME_SYNTHETIC_VERIFIED', 'READY_FOR_EXPLICIT_AUTHORIZATION', False, True, True)

    def test_static_production_scope(self):
        self.assertTrue(l.static_audit()['pass'])


def rows_fixture():
    return [{'origin': o, 'step': h, 'target': o + h, 'truth': (o + h) * .01,
             'prediction': (o + h) * .01 + ((h % 3) - 1) * (o + 1) * .001}
            for o in [0, 1, 2] for h in range(1, 25)]


class MetricWindowTests(unittest.TestCase):
    def test_query_weighted_metric_and_horizon_all24(self):
        rows = rows_fixture()
        m = l.synthetic_metrics(rows, synthetic=True)
        expected = math.sqrt(sum((r['prediction'] - r['truth']) ** 2 for r in rows) / 72) * 1000
        self.assertAlmostEqual(m['forecast_query_weighted_RMSE_mV'], expected, places=12)
        self.assertEqual(m['queries'], 72)
        self.assertEqual(len(m['horizon_RMSE_mV']), 24)

    def test_target_time_squared_errors_NOT_prediction_ensemble(self):
        rows = rows_fixture()
        targets = sorted({r['target'] for r in rows})
        expected = math.sqrt(statistics.mean(statistics.mean((r['prediction'] - r['truth']) ** 2
                    for r in rows if r['target'] == t) for t in targets)) * 1000
        wrong = math.sqrt(statistics.mean((statistics.mean(r['prediction'] for r in rows if r['target'] == t)
                  - t * .01) ** 2 for t in targets)) * 1000
        got = l.synthetic_metrics(rows, synthetic=True)['target_time_equal_weight_RMSE_mV']
        self.assertAlmostEqual(got, expected, places=12)
        self.assertGreater(abs(got - wrong), .01)

    def test_seed_macro_is_not_pooled_or_best(self):
        r = l.synthetic_seed_summary({42: 1., 52: 2., 62: 6.})
        self.assertEqual(r['arithmetic_mean_RMSE'], 3.)
        self.assertAlmostEqual(r['sample_SD'], statistics.stdev([1., 2., 6.]))
        self.assertNotEqual(r['arithmetic_mean_RMSE'], r['pooled_RMSE_equal_query_counts_only'])

    def test_missing_seed_and_duplicate_query_fail(self):
        with self.assertRaises(l.FirewallError):
            l.synthetic_seed_summary({42: 1., 62: 2.})
        r = rows_fixture()
        with self.assertRaises(l.FirewallError):
            l.synthetic_metrics(r + [r[0]], synthetic=True)

    def test_real_metric_path_denied(self):
        with self.assertRaises(l.FirewallError):
            l.synthetic_metrics(rows_fixture())

    def test_window_continuity_and_preorigin_context(self):
        rows = [{'start_age': i * .5, 'end_age': (i + 1) * .5, 'valid': True,
                 'finite_schema': True, 'segment': 1} for i in range(72)]
        self.assertTrue(l.validate_window(rows, 47, 24., 36.))
        rows[10]['segment'] = 2
        with self.assertRaises(l.FirewallError):
            l.validate_window(rows, 47, 24., 36.)

    def test_duplicate_age_gap_and_target_boundary(self):
        rows = [{'start_age': i * .5, 'end_age': (i + 1) * .5, 'valid': True,
                 'finite_schema': True, 'segment': 1} for i in range(72)]
        with self.assertRaises(l.FirewallError):
            l.validate_window(rows, 47, 24.1, 36.)
        rows[10]['end_age'] = rows[9]['end_age']
        with self.assertRaises(l.FirewallError):
            l.validate_window(rows, 47, 24., 36.)

    def test_whole_bin_width_and_origin_region(self):
        rows = [{'start_age': i * .5, 'end_age': (i + 1) * .5, 'valid': True,
                 'finite_schema': True, 'segment': 1} for i in range(72)]
        rows[3]['start_age'] -= .1
        with self.assertRaises(l.FirewallError):
            l.validate_window(rows, 47, 24., 36.)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    out = PHASE / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'L_SYNTHETIC_TEST_LOG.txt').open('w', encoding='utf-8', newline='\n') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    l.write_json('outputs/L_SYNTHETIC_TEST_RESULTS.json', {'passed': result.wasSuccessful(),
         'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
         'scope': 'synthetic data and production L guard functions only; no model forward or Test I/O',
         'command': 'python -B tests/test_l_synthetic_firewall.py'})
    print('L_SYNTHETIC_TESTS', result.testsRun, 'PASS' if result.wasSuccessful() else 'FAIL')
    raise SystemExit(0 if result.wasSuccessful() else 1)
