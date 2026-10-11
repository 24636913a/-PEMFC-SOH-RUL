"""Tests call the actual cache-only reference validator; no inference."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import complete_saved_cache_audit as completion
j=completion.j

class CacheAuditTest(unittest.TestCase):
    def frame(self):
        origin=np.repeat(np.arange(2103,2308),24);step=np.tile(np.arange(1,25),205)
        return pd.DataFrame({'origin_row':origin,'horizon_step':step,'target_row':origin+step,
            'y_true_delta_V':np.linspace(-.5,.5,4920),'y_pred_delta_V':np.linspace(-.5,.5,4920)+.003})
    def test_primary_three_columns_and_secondary_cache_recalculation(self):
        frame=self.frame();metrics=j.project_metrics(frame);old={k:metrics[k] for k in ['RMSE_mV','MAE_mV','R2']}
        result=completion.check_h_reference(frame,frame.copy(),old)
        self.assertEqual(result,metrics);self.assertIn('endpoint_RMSE_mV',result)
    def test_original_prediction_mutation_fails(self):
        a=self.frame();b=a.copy();b.loc[0,'y_pred_delta_V']+=1
        with self.assertRaisesRegex(RuntimeError,'POINT_MISMATCH'):completion.check_h_reference(a,b,{})
    def test_original_target_key_mutation_fails(self):
        a=self.frame();b=a.copy();b.loc[0,'target_row']+=1
        with self.assertRaisesRegex(RuntimeError,'KEYS_MISMATCH'):completion.check_h_reference(a,b,{})
    def test_incorrect_primary_value_fails(self):
        a=self.frame();old=j.project_metrics(a);old['RMSE_mV']+=.1
        with self.assertRaisesRegex(RuntimeError,'PRIMARY_METRIC_MISMATCH'):completion.check_h_reference(a,a,old)
    def test_Test_source_rejected_before_io(self):
        with patch.object(completion.Path,'read_bytes',side_effect=AssertionError('I/O happened')):
            with self.assertRaisesRegex(RuntimeError,'NOT_ALLOWLISTED'):
                completion.read_h_validation(j.PROJECT/'Test/2000.csv',{'additional_Validation_sources':{}})

if __name__=='__main__':unittest.main()
