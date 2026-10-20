import ast
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'scripts'))
from common import CONFIG,REFERENCE,read,read_csv,metrics
from data import prepare,FEATURES,window_indices
from models import new_model,configure
from train import state_sha,gradient_groups

class FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=Path(read(HERE/'outputs/ACTIVE_RUN.json')['path']);configure()
    def test_protocol_and_budget(self):
        c=read(CONFIG)
        self.assertEqual(c['seed'],202);self.assertEqual(len(c['training_jobs']),6)
        self.assertEqual(c['partitions'],dict(TRAIN=[0,2690],VALIDATION=[2690,3074],HOLDOUT=[3074,3843]))
    def test_native_classes_preserved_except_length_interface(self):
        old=ast.parse((HERE.parent/'F2_Performance_Improvement/frozen_f2.py').read_text(encoding='utf-8'))
        new=ast.parse((HERE/'scripts/native_models.py').read_text(encoding='utf-8'))
        a={n.name:n for n in old.body if isinstance(n,ast.ClassDef)};b={n.name:n for n in new.body if isinstance(n,ast.ClassDef)}
        for name in ['CanonicalStrictHistoryTFT','_LeftPaddedDepthwiseConv1d','CausalMultiScaleHistoryBranch','DualPathStrictHistoryTFT','RegimeFusionGate','RegimeAwareDualPathTFT']:
            for node in [a[name],b[name]]:
                if name=='CanonicalStrictHistoryTFT':node.body=[x for x in node.body if not isinstance(x,ast.FunctionDef) or x.name not in ['build_input','forward_history']]
            self.assertEqual(ast.dump(a[name]),ast.dump(b[name]),name)
    def test_new_horizon_shapes_and_no_future_inputs(self):
        for h in [6,12,24]:
            m=new_model('F2',h).cuda().eval();x=torch.randn(2,48,14,device='cuda');pack=m.build_input(x)
            self.assertEqual(tuple(pack['decoder_cont'].shape),(2,h,15))
            self.assertTrue((pack['decoder_cont'][:,:,:14]==0).all())
            with torch.no_grad():self.assertEqual(tuple(m.predict_delta(x).shape),(2,h))
            self.assertEqual(sum(p.numel() for p in m.parameters()),332436)
    def test_C0_C1_initial_backbone_and_predictions(self):
        a=new_model('C0',24).cuda().eval();b=new_model('C1',24).cuda().eval()
        self.assertEqual(state_sha(a),state_sha(b,True))
        self.assertEqual(state_sha(a),read(REFERENCE/'TRAINING_LEDGER.json')['initialization_sha256'])
        self.assertEqual(sum(p.numel() for p in b.parameters()),333510)
        self.assertAlmostEqual(float(b.residual_alpha),.025,places=7)
        x=torch.randn(2,48,14,device='cuda')
        with torch.no_grad():self.assertTrue(torch.equal(a.predict_delta(x),b.predict_delta(x)))
    def test_C1_synthetic_backward_no_optimizer_or_formal_training(self):
        b=new_model('C1',24).cuda().train();x=torch.randn(2,48,14,device='cuda');target=torch.randn(2,24,device='cuda')
        loss=torch.nn.functional.mse_loss(b.predict_delta(x),target);loss.backward();g=gradient_groups(b)
        for key in ['CNN','LSTM_encoder','LSTM_decoder','gate','residual_head']:self.assertTrue(np.isfinite(g[key]) and g[key]>0,key)
        self.assertEqual(g['alpha'],0.) # zero last-layer initialization deliberately preserves native output
    def test_horizon_specific_scaler_and_labels(self):
        for h in [6,12,24]:
            _,d,p=prepare(self.out,h);flat=d['TRAIN']['xraw'].astype(float).reshape(-1,14)
            np.testing.assert_array_equal(p['feature_mean'],flat.mean(0))
            self.assertTrue((d['TRAIN']['x'][:,:,[1,2,5,12]]==0).all())
            self.assertLess(d['TRAIN']['targets'].max(),d['VALIDATION']['origins'].min())
            self.assertEqual(d['VALIDATION']['x'].shape[1:],(48,14))
            if h==24:
                p_old=read_csv(REFERENCE/'predictions_VALIDATION.csv')
                np.testing.assert_array_equal(d['VALIDATION']['targets'].ravel(),p_old.target_row)
                np.testing.assert_array_equal(d['VALIDATION']['y'].ravel(),p_old.y_true_delta_V)
    def test_common_h24_origin_subset(self):
        for p in ['TRAIN','VALIDATION','HOLDOUT']:
            common=set(read_csv(self.out/f'windows/H24_{p}.csv').origin_row)
            for h in [6,12]:self.assertTrue(common.issubset(set(read_csv(self.out/f'windows/H{h}_{p}.csv').origin_row)))
    def test_target_equal_is_not_query_weighted(self):
        p=pd.DataFrame(dict(origin_row=[0,1,1],target_row=[2,2,3],y_true_delta_V=[0.,1.,2.],y_pred_delta_V=[1.,4.,2.]))
        m=metrics(p);self.assertAlmostEqual(m['RMSE_mV'],np.sqrt(10/3)*1000)
        self.assertAlmostEqual(m['target_time_equal_RMSE_mV'],np.sqrt(2.5)*1000)
    def test_shared_AddNorm_diagnostic_hook_preserves_predictions_CPU(self):
        from evaluate import infer_diagnostics
        for kind,h in [('F2',6),('F2',12),('C1',24)]:
            model=new_model(kind,h).cpu().eval();x=np.random.default_rng(1).standard_normal((3,48,14)).astype(np.float32)
            self.assertIs(model.post_lstm_add_norm_encoder,model.post_lstm_add_norm_decoder)
            with torch.no_grad():expected=model.predict_delta(torch.from_numpy(x)).numpy()
            before=state_sha(model);actual,stats,base,correction=infer_diagnostics(model,x,kind)
            np.testing.assert_array_equal(actual,expected);self.assertEqual(state_sha(model),before)
            self.assertEqual(len(stats),3);self.assertTrue((stats.LSTM_latent_RMS>0).all())

if __name__=='__main__':unittest.main(verbosity=2)
