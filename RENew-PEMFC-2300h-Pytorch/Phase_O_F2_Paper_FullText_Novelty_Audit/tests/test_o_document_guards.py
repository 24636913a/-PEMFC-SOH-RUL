"""Document-only access tests. No data, weights, predictions or ML imports."""
import sys
import unittest
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_o_document_audit as production


class DocumentGuards(unittest.TestCase):
    def test_static_source_allowed(self):
        relative = production.ROOT+'Phase_TFT_Original_Baseline/models/canonical_tft.py'
        self.assertEqual(production.allowed_path(relative).suffix, '.py')

    def test_raw_source_denied_before_io(self):
        with self.assertRaisesRegex(ValueError, 'NOT_ALLOWLISTED'):
            production.source_bytes('数据/Loadcycle_0-14.csv')

    def test_n1_source_denied_before_io(self):
        with self.assertRaisesRegex(ValueError, 'NOT_ALLOWLISTED'):
            production.source_bytes(production.ROOT+'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv')

    def test_weight_source_denied_before_io(self):
        with self.assertRaisesRegex(ValueError, 'NOT_ALLOWLISTED'):
            production.source_bytes(production.ROOT+'Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F2-Seed42_best.pt')

    def test_prediction_source_denied_before_io(self):
        with self.assertRaisesRegex(ValueError, 'NOT_ALLOWLISTED'):
            production.source_bytes(production.ROOT+'Phase_I2_Frozen_Assessment_Evaluation/outputs/predictions/I2_F2_Seed42_predictions.csv')

    def test_test_source_denied_before_io(self):
        with self.assertRaisesRegex(ValueError, 'NOT_ALLOWLISTED'):
            production.source_bytes('Test/Polcurve_Nominal_RHFC_2300h.csv')

    def test_digest_change_fails(self):
        expected = production.digest(b'synthetic approved document')
        production.verify_digest(b'synthetic approved document', expected)
        with self.assertRaisesRegex(ValueError, 'HASH_MISMATCH'):
            production.verify_digest(b'tampered synthetic document', expected)

    def test_historical_identity_not_weight_read(self):
        result = production.identities()
        self.assertEqual(len(result['identities']), 9)
        self.assertFalse(result['current_weight_bytes_verified'])
        self.assertEqual(result['weight_files_opened'], 0)


if __name__ == '__main__':
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DocumentGuards)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    text = stream.getvalue()
    print(text, end='')
    out = production.PHASE/'outputs/O_DOCUMENT_GUARD_TEST_LOG.txt'
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding='utf-8')
    production.write_json(production.PHASE/'outputs/O_DOCUMENT_GUARD_TEST_RESULTS.json', {
        'tests_run':result.testsRun, 'failures':len(result.failures),
        'errors':len(result.errors), 'pass':result.wasSuccessful(),
        'scope':'production document allowlist/hash/manifest functions; no numerical performance recomputation'})
    raise SystemExit(0 if result.wasSuccessful() else 1)
