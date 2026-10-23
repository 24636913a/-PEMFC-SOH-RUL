"""No training loops: identity, causality, probes, metrics and selection guards."""
import copy
import unittest
import numpy as np
import pandas as pd
import torch
import frozen_f2 as f
import diagnostics as d
from io_utils import metrics
from selection import qualify


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.frame,cls.windows,cls.scaler,cls.prepared,cls.excluded=f.load_selection_data()

    def test_original_model_and_probe_identity(self):
        f.model_identity();torch.manual_seed(202)
        model=f.RegimeAwareDualPathTFT(hidden_size=64,hidden_continuous_size=32,attention_head_size=4,lstm_layers=1,dropout=.1).eval()
        observed=d.attach(copy.deepcopy(model)).eval();x=torch.from_numpy(self.prepared['TRAIN'][:2])
        with torch.no_grad():a=model.predict_delta(x);b=observed.predict_delta(x)
        self.assertTrue(torch.equal(a,b));self.assertEqual(f.state_sha(model),f.state_sha(observed))
        self.assertEqual(sum(p.numel() for p in observed.parameters()),332436)
        observed.zero_grad();torch.nn.functional.mse_loss(observed.predict_delta(x),torch.zeros(2,24)).backward()
        grad=d.gradient_row(observed)
        self.assertGreater(grad['cnn_parameter_grad_norm'],0);self.assertGreater(grad['lstm_encoder_parameter_grad_norm'],0)

    def test_inputs_labels_and_no_holdout(self):
        self.assertEqual(set(self.windows),{'TRAIN','VALIDATION'})
        self.assertEqual([len(self.windows[k].origins) for k in ['TRAIN','VALIDATION']],[1750,218])
        for part,w in self.windows.items():
            raw=self.frame.cell_voltage_mean.to_numpy(float)
            self.assertTrue(np.array_equal(w.y,(raw[w.targets]-raw[w.origins,None]).astype('float32')))
            unused=[i for i in range(14) if i not in f.MASKS['top10']]
            self.assertEqual(np.count_nonzero(self.prepared[part][:,:,unused]),0)
        self.assertLess(self.windows['TRAIN'].targets.max(),self.windows['VALIDATION'].origins.min())

    def test_no_future_decoder_conditions(self):
        x=torch.from_numpy(self.prepared['TRAIN'][:2]);batch=f.CanonicalStrictHistoryTFT.build_input(x)
        self.assertEqual(torch.count_nonzero(batch['decoder_cont'][:,:,:14]),0)
        self.assertEqual(tuple(batch['decoder_cont'].shape),(2,24,15))

    def test_metric_units(self):
        p=pd.DataFrame(dict(y_true_delta_V=[-.001,.001],y_pred_delta_V=[0.,.002],origin_row=[1,1],target_row=[2,3]))
        result=metrics(p);self.assertAlmostEqual(result['RMSE_mV'],1.);self.assertAlmostEqual(result['MAE_mV'],1.)
        self.assertAlmostEqual(result['R2'],0.)

    def test_fixed_qualification_and_holdout_rejection(self):
        control=pd.DataFrame(dict(seed=[42,202,212],partition='VALIDATION',RMSE_mV=10.,MAE_mV=5.,P99_mV=20.))
        candidate=control.copy();candidate['RMSE_mV']=9.5
        self.assertTrue(qualify(candidate,control)['qualified'])
        candidate.loc[0,'RMSE_mV']=10.
        self.assertFalse(qualify(candidate,control)['qualified'])
        candidate['partition']='TEST'
        with self.assertRaises(RuntimeError):qualify(candidate,control)


if __name__=='__main__':unittest.main()
