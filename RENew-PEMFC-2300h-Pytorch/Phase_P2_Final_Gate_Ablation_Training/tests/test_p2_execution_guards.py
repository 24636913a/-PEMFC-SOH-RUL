import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
import p2_common as c
from audit_p2_predictions import metric_bundle,validate_frame
from analyze_p2_results import decide

def synthetic_frame():
    origins=np.arange(2103,2308);h=np.tile(np.arange(1,25),205);o=np.repeat(origins,24)
    y=np.linspace(-.03,.01,4920)
    return pd.DataFrame({'run_id':'synthetic','model':'G0','seed':42,'protocol_sha256':'synthetic',
        'origin_row':o,'target_row':o+h,'horizon_step':h,'y_true_delta_V':y,'y_pred_delta_V':y+np.linspace(-.002,.002,4920)})

class ExecutionGuards(unittest.TestCase):
    def test_git_newline_representation_does_not_replace_H_byte_hash_check(self):
        observed=b'{"value":1}\r\n';blob=b'{"value":1}\n'
        c.verify_static_source(observed,blob,'synthetic',c.sha_bytes(observed))
        with self.assertRaisesRegex(RuntimeError,'BYTE_HASH_MISMATCH'):
            c.verify_static_source(observed,blob,'synthetic',c.sha_bytes(blob))
        with self.assertRaisesRegex(RuntimeError,'SOURCE_CHANGED'):
            c.verify_static_source(b'{"value":2}\r\n',blob,'synthetic')

    def test_n1_unbounded_and_3005_reads_and_full_hash_rejected(self):
        for kwargs in ({},{'nrows':3005},{'nrows':3007,'iterator':True}):
            with self.assertRaises(PermissionError):c.validate_read_csv_request(c.N1,kwargs)
        c.validate_read_csv_request(c.N1,{'nrows':3007})
        with self.assertRaises(PermissionError):c.file_hash(c.N1)
        with self.assertRaises(PermissionError):c.N1.read_bytes()

    def test_Assessment_Test_raw_and_historical_checkpoint_actual_open_denied(self):
        for path in (c.REPO/'Assessment_predictions.csv',c.REPO/'Test.csv',c.H/'outputs/checkpoints/H-F2-Seed42_best.pt'):
            with self.assertRaises(PermissionError):path.read_bytes()

    def test_checkpoint_wrong_seed_or_protocol_production_rejected(self):
        valid={'run_id':'r','model':'G0','seed':42,'protocol_sha256':'p','code_identity_sha256':'c','input_fingerprints':{}}
        c.require_checkpoint_identity(valid,'r','G0',42,'p','c',{})
        for key,value in (('seed',52),('protocol_sha256','tampered'),('model','G1')):
            ck=copy.deepcopy(valid);ck[key]=value
            with self.assertRaises(RuntimeError):c.require_checkpoint_identity(ck,'r','G0',42,'p','c',{})

    def test_cache_key_truth_seed_and_missing_row_production_rejected(self):
        frame=synthetic_frame();truth=frame[['origin_row','target_row','horizon_step','y_true_delta_V']].copy()
        validate_frame(frame,truth,'synthetic','G0',42,'synthetic')
        variants=[frame.iloc[:-1].copy(),pd.concat([frame.iloc[:-1],frame.iloc[[0]]],ignore_index=True)]
        for column,value in (('seed',52),('model','G2'),('y_true_delta_V',9.)):
            changed=frame.copy();changed.loc[0,column]=value;variants.append(changed)
        for changed in variants:
            with self.assertRaises(RuntimeError):validate_frame(changed,truth,'synthetic','G0',42,'synthetic')

    def test_target_equal_is_mean_squared_errors_not_prediction_ensemble(self):
        frame=synthetic_frame();metric,_,_=metric_bundle(frame)
        err=(frame.y_pred_delta_V-frame.y_true_delta_V)*1000
        expected=np.sqrt(pd.DataFrame({'target':frame.target_row,'sq':err**2}).groupby('target').sq.mean().mean())
        self.assertAlmostEqual(metric['target_time_equal_RMSE_mV'],expected,places=12)
        self.assertAlmostEqual(metric['RMSE_mV'],np.sqrt(np.mean(err**2)),places=12)

    def test_nonfinite_predictions_production_rejected(self):
        frame=synthetic_frame();frame.loc[0,'y_pred_delta_V']=np.inf
        with self.assertRaisesRegex(RuntimeError,'NONFINITE'):metric_bundle(frame)

    def test_sampler_order_mismatch_production_rejected(self):
        with patch.object(c,'OUT',c.PHASE/'outputs/synthetic_guard_tests'):
            c.validate_common_sampler(42,1,[3,1,2])
            c.validate_common_sampler(42,1,[3,1,2])
            with self.assertRaisesRegex(RuntimeError,'SAMPLER_ORDER'):c.validate_common_sampler(42,1,[1,3,2])

    def test_code_identity_changed_invalidates_protocol(self):
        with patch.object(c,'execution_identity',return_value={'files':{},'sha256':'tampered'}):
            with self.assertRaisesRegex(RuntimeError,'CODE_IDENTITY_CHANGED'):c.load_protocol()

    def test_registered_decision_threshold_not_lowered(self):
        rows=[]
        for group in ('G0','G1','G2'):
            for seed in (42,52,62):
                value=10. if group!='G2' else 9.8
                rows.append({'model':group,'seed':seed,'RMSE_mV':value,'MAE_mV':5.,'P99_abs_error_mV':20.,
                             'endpoint_12h_RMSE_mV':10.,'target_time_equal_RMSE_mV':10.})
        self.assertFalse(decide(rows)['all_predefined_criteria_PASS'])
        for r in rows:
            if r['model']=='G2':r['RMSE_mV']=9.
        self.assertTrue(decide(rows)['all_predefined_criteria_PASS'])

    def test_incomplete_nine_run_identity_not_scientific_failure(self):
        with self.assertRaisesRegex(RuntimeError,'IDENTITY_INCOMPLETE'):decide([])
