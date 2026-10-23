import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from models import train_validation,inner_split,error_metrics
from run_experiment import decide

class ProtocolTests(unittest.TestCase):
 def frame(self):
  return pd.DataFrame({'fold':['fold_2']*4,'role':['train','train','validation','validation'],
   'L96_eligible':[True]*4,'target_p1s_measurement_end_ophrs2_h':[5.,11.,15.,16.],
   'origin_p1s_measurement_end_ophrs2_h':[4.,6.,10.,12.]})
 def test_96_alone_keeps_fold_two(self):
  a,b,p,c=train_validation(self.frame(),'fold_2');self.assertEqual(len(b),2);self.assertEqual(len(a),1);self.assertEqual(len(p),1);self.assertEqual(c,10.)
 def test_outer_label_at_cutoff_allowed(self):
  d=self.frame();d.loc[1,'target_p1s_measurement_end_ophrs2_h']=10.;self.assertEqual(len(train_validation(d,'fold_2')[0]),2)
 def test_unqualified_validation_does_not_set_cutoff(self):
  d=self.frame();d.loc[2,'L96_eligible']=False;self.assertEqual(train_validation(d,'fold_2')[3],12.)
 def test_inner_purge(self):
  d=pd.DataFrame({'target_p1s_measurement_end_ophrs2_h':np.arange(20.),'origin_p1s_measurement_end_ophrs2_h':np.arange(20.)-3})
  fit,select,purge,c=inner_split(d);self.assertEqual(len(select),3);self.assertEqual(len(purge),2);self.assertLessEqual(fit.target_p1s_measurement_end_ophrs2_h.max(),c)
 def test_metrics_units_and_tail(self):
  d=error_metrics([0.,0.],[.001,-.003],.7);self.assertAlmostEqual(d['MAE_mV_equivalent'],1.4);self.assertAlmostEqual(d['max_abs_error_mV_equivalent'],2.1)
 def metric_frame(self):
  return pd.DataFrame([dict(horizon='H1',fold=f,model=m,n=2,MAE_mV_equivalent=v,RMSE_mV_equivalent=v,P95_abs_error_mV_equivalent=v,max_abs_error_mV_equivalent=v) for f in ['fold_1','fold_2','fold_3'] for m,v in [('PERSISTENCE',1.),('RIDGE_96H',.9)]])
 def test_supported_gate(self):self.assertEqual(decide(self.metric_frame(),'H1')['status'],'96H_SHORT_HORIZON_SIGNAL_SUPPORTED')
 def test_missing_fold_not_evaluable(self):
  d=self.metric_frame();d.loc[d.fold=='fold_2','n']=0;self.assertEqual(decide(d,'H1')['status'],'THREE_FOLD_NOT_EVALUABLE')
 def test_tail_failure_mixed(self):
  d=self.metric_frame();d.loc[(d.fold=='fold_2')&(d.model=='RIDGE_96H'),'max_abs_error_mV_equivalent']=1.01;self.assertEqual(decide(d,'H1')['status'],'96H_SHORT_HORIZON_MIXED')
 def test_equal_fold_not_pooled(self):
  d=self.metric_frame();d.loc[d.fold=='fold_3','n']=100;self.assertAlmostEqual(decide(d,'H1')['Ridge_equal_fold_MAE_mV'],.9)
if __name__=='__main__':unittest.main()
