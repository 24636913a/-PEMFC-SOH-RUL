import ast
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'scripts'))
from audit_data import CONFIG,continuity,windows,disjoint_count
from audit_targets import trailing
import native_models as nm
from train_longterm import prepare,new_model,metrics

class QualificationTests(unittest.TestCase):
    def test_partition_freeze(self):
        c=json.loads(CONFIG.read_text(encoding='utf-8'))
        self.assertEqual(c['partition_rows'],dict(TRAIN=[0,2690],VALIDATION=[2690,3074],HOLDOUT=[3074,3843]))
    def test_smoothing_causal_and_break_reset(self):
        x=np.arange(150,dtype=float);valid=np.ones(150,dtype=bool);edge=valid.copy();edge[0]=False;edge[80]=False
        y=trailing(x,valid,edge);z=x.copy();z[70:]+=999
        np.testing.assert_array_equal(y[:70],trailing(z,valid,edge)[:70])
        self.assertTrue(np.isnan(y[80:127]).all());self.assertTrue(np.isfinite(y[127]))
    def test_disjoint_does_not_count_overlap_as_replication(self):
        w=pd.DataFrame(dict(target_first_row=[1,2,3,4,5],target_last_row=[4,5,6,7,8]))
        self.assertEqual(disjoint_count(w),2)
    def test_query_and_target_equal_metrics_differ(self):
        points=pd.DataFrame(dict(origin_row=[0,1,1],target_row=[2,2,3],horizon_step=[1,2,2],
            true_delta_proxy_V=[0.,1.,2.],predicted_delta_proxy_V=[1.,4.,2.],true_proxy_V=[2.,2.,3.]))
        m=metrics(points)
        self.assertAlmostEqual(m['RMSE_mV'],np.sqrt(10/3)*1000)
        self.assertAlmostEqual(m['target_time_equal_RMSE_mV'],np.sqrt(2.5)*1000)
        self.assertAlmostEqual(m['terminal_RMSE_mV'],np.sqrt(4.5)*1000)
    def test_native_learned_architecture_unchanged(self):
        old=ast.parse((HERE.parent/'F2_Performance_Improvement/frozen_f2.py').read_text(encoding='utf-8'))
        new=ast.parse((HERE/'scripts/native_models.py').read_text(encoding='utf-8'))
        a={n.name:n for n in old.body if isinstance(n,ast.ClassDef)};b={n.name:n for n in new.body if isinstance(n,ast.ClassDef)}
        for name in ['CanonicalStrictHistoryTFT','_LeftPaddedDepthwiseConv1d','CausalMultiScaleHistoryBranch','DualPathStrictHistoryTFT','RegimeFusionGate','RegimeAwareDualPathTFT']:
            aa=a[name];bb=b[name]
            if name=='CanonicalStrictHistoryTFT':
                aa.body=[n for n in aa.body if not isinstance(n,ast.FunctionDef) or n.name not in ['build_input','forward_history']]
                bb.body=[n for n in bb.body if not isinstance(n,ast.FunctionDef) or n.name not in ['build_input','forward_history']]
            self.assertEqual(ast.dump(aa),ast.dump(bb),name)
    def test_same_inputs_and_labels_and_scaler(self):
        for horizon in [48,96]:
            d,i=prepare(horizon);train=d['TRAIN'];val=d['VALIDATION']
            self.assertLess(train['targets'].max(),val['origins'].min())
            np.testing.assert_allclose(i['feature_mean'],train['xraw'].astype(float).reshape(-1,14).mean(0),rtol=0,atol=0)
            self.assertTrue((val['targets']<3074).all());self.assertTrue((val['targets']>=2690).all())
            self.assertTrue((train['x'][:,:,[1,2,5,12]]==0).all())
            self.assertEqual(train['x'].shape[1:],(48,14))
    def test_gpu_lengths_and_parameters(self):
        torch.set_num_threads(1)
        for horizon in [48,96]:
            model=new_model('F2',horizon,202).cuda().eval();x=torch.randn(2,48,14,device='cuda')
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),332436)
            pack=model.build_input(x)
            self.assertEqual(pack['decoder_cont'].shape,(2,horizon,15))
            self.assertTrue(torch.equal(pack['decoder_cont'][:,:,:14],torch.zeros(2,horizon,14,device='cuda')))
            with torch.no_grad():self.assertEqual(model.predict_delta(x).shape,(2,horizon))
            del model

if __name__=='__main__':unittest.main(verbosity=2)
