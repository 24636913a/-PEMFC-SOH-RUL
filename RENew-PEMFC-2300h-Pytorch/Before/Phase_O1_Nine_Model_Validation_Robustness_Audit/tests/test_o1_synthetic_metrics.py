import io
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_o1_cache_audit as p


def synthetic():
    return pd.DataFrame([{'run_id':'H-F0-Seed42','model':'F0','seed':42,
        'origin_row':origin,'target_row':origin+h,'horizon_step':h,
        'y_true_delta_V':0.,'y_pred_delta_V':.001*h,'protocol_sha256':'synthetic'}
        for origin in (2103,2104) for h in (1,2)])


class SyntheticMetrics(unittest.TestCase):
    def validate(self, f):
        p.validate_frame(f,'H-F0-Seed42','F0',42,'synthetic',n_origins=2,horizon=2)

    def test_legal(self):
        self.validate(synthetic())

    def test_duplicate_blocked(self):
        f=synthetic();f.iloc[3]=f.iloc[0]
        with self.assertRaisesRegex(p.AuditError,'DUPLICATE_KEY'):
            self.validate(f)

    def test_missing_horizon_blocked(self):
        with self.assertRaisesRegex(p.AuditError,'QUERY_COUNT'):
            self.validate(synthetic().iloc[:-1])

    def test_truth_mismatch_blocked(self):
        f=synthetic();f.loc[0,'y_true_delta_V']=.01
        with self.assertRaisesRegex(p.AuditError,'TRUTH_MISMATCH'):
            p.align_truth(f,synthetic())

    def test_protocol_mismatch_blocked(self):
        f=synthetic();f.loc[0,'protocol_sha256']='other'
        with self.assertRaisesRegex(p.AuditError,'PROTOCOL_MISMATCH'):
            self.validate(f)

    def test_key_mismatch_blocked(self):
        f=synthetic();f.loc[0,'target_row']=2120
        with self.assertRaisesRegex(p.AuditError,'COMMON_KEY'):
            p.align_truth(f,synthetic())

    def test_explicit_key_order_alignment(self):
        f=synthetic().iloc[::-1]
        aligned=p.align_truth(f,synthetic())
        self.assertEqual(len(aligned),4)
        self.assertEqual(aligned.origin_row.iloc[0],2103)

    def test_target_relation_and_boundary(self):
        f=synthetic();f.loc[0,'target_row']=2404
        with self.assertRaises(p.AuditError):
            self.validate(f)

    def test_repeat_excluded(self):
        f=synthetic();f['run_id']='H-F0-Repeat-A'
        with self.assertRaisesRegex(p.AuditError,'REPEAT'):
            p.validate_frame(f,'H-F0-Repeat-A','F0',42,'synthetic',2,2)

    def test_source_firewall(self):
        for source in ('Test/x.csv',p.H+'outputs/validation_predictions/H-F0-Repeat-A.csv',
                       p.ROOT+'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'):
            with self.assertRaisesRegex(p.AuditError,'SOURCE_NOT_ALLOWED'):
                p.source(source)

    def test_formal_and_target_equal_different(self):
        f=pd.DataFrame({'target_row':[2104,2105,2105],'horizon_step':[1,2,1],
                        'y_true_delta_V':[0.,0.,0.],'y_pred_delta_V':[.001,.003,.004]})
        tm,t=p.target_metrics(f)
        self.assertAlmostEqual(p.error_metrics([1,3,4])['RMSE_mV'],math.sqrt(26/3))
        self.assertAlmostEqual(tm['target_time_equal_RMSE_mV'],math.sqrt(6.75))
        self.assertNotEqual(tm['target_time_equal_RMSE_mV'],p.error_metrics([1,3,4])['RMSE_mV'])
        self.assertEqual(t.multiplicity.tolist(),[1,2])

    def test_no_prediction_ensemble(self):
        f=pd.DataFrame({'target_row':[2104,2104],'horizon_step':[1,2],
                        'y_true_delta_V':[0.,0.],'y_pred_delta_V':[.004,-.004]})
        self.assertEqual(p.target_metrics(f)[0]['target_time_equal_RMSE_mV'],4.)

    def test_sse_concentration(self):
        f=pd.DataFrame({'target_row':[2104,2105,2105],'horizon_step':[1,2,1],
                        'y_true_delta_V':[0.,0.,0.],'y_pred_delta_V':[.001,.003,.004]})
        m,t=p.target_metrics(f)
        self.assertAlmostEqual(m['top1_target_SSE_share'],25/26)
        self.assertEqual(m['max_SSE_target_row'],2105)
        self.assertEqual(m['max_SSE_target_multiplicity'],2)
        self.assertAlmostEqual(m['top5_target_SSE_share'],1.)
        self.assertAlmostEqual(math.fsum(t.SSE_mV2),26.)

    def test_unique_not_independent(self):
        with self.assertRaisesRegex(p.AuditError,'NOT_INDEPENDENT'):
            p.target_semantics(independence_claim=True)

    def test_seed_statistics_not_pooled(self):
        # Production metrics applied to each seed; arithmetic mean is not pooling.
        a=p.error_metrics([1.,1.])['RMSE_mV'];b=p.error_metrics([3.])['RMSE_mV']
        self.assertEqual(np.mean([a,b]),2.)
        self.assertNotEqual(np.mean([a,b]),p.error_metrics([1.,1.,3.])['RMSE_mV'])


if __name__=='__main__':
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticMetrics)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),end='')
    out=p.PHASE/'outputs';out.mkdir(exist_ok=True)
    (out/'O1_SYNTHETIC_TEST_LOG.txt').write_text(stream.getvalue(),encoding='utf-8')
    p.write_json(out/'O1_SYNTHETIC_TEST_RESULTS.json',{'pass':result.wasSuccessful(),
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'scope':'production cache-validation/metric functions with synthetic data only'})
    raise SystemExit(0 if result.wasSuccessful() else 1)
