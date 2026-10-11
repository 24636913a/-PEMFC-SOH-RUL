"""Production-function regression tests; synthetic inputs only; no training."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import j_common as j
from canonical_tft import CanonicalStrictHistoryTFT
from models.causal_multiscale_branch import CausalMultiScaleHistoryBranch

class MeanSigmoid(torch.nn.Module):
    def forward(self,x):return torch.sigmoid(x.mean(1,keepdim=True))

class TinyGateModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.regime_gate=MeanSigmoid();self.weight=torch.nn.Parameter(torch.ones(1))
    def predict_delta(self,x):
        g=self.regime_gate(x[:,:,j.INDICES].mean(1))
        return (g*self.weight).expand(-1,24)

class TestJ(unittest.TestCase):
    def legal(self):
        origins=np.arange(2103,2308);targets=origins[:,None]+np.arange(1,25)
        return origins,targets,np.full(4920,1500.)
    def test_legal_validation_keys(self): j.validate_keys(*self.legal())
    def test_duplicate_origin_rejected(self):
        o,t,a=self.legal();o[1]=o[0]
        with self.assertRaisesRegex(RuntimeError,'DUPLICATE'):j.validate_keys(o,t,a)
    def test_Assessment_target_rejected(self):
        o,t,a=self.legal();o[-1]=2390;t[-1]=o[-1]+np.arange(1,25)
        with self.assertRaisesRegex(RuntimeError,'BOUNDARY'):j.validate_keys(o,t,a)
    def test_inconsistent_target_key_rejected(self):
        o,t,a=self.legal();t[0,0]+=1
        with self.assertRaisesRegex(RuntimeError,'KEY_IDENTITY'):j.validate_keys(o,t,a)
    def test_Test_age_rejected(self):
        o,t,a=self.legal();a[-1]=1820
        with self.assertRaisesRegex(RuntimeError,'BOUNDARY'):j.validate_keys(o,t,a)
    def test_full_N1_hash_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'FULL_N1'):j.digest(j.c.N1)
    def test_unbounded_read_rejected(self):
        with j.prefix_guard([]):
            with self.assertRaisesRegex(RuntimeError,'UNBOUNDED'):pd.read_csv(j.c.N1)
    def test_wrong_prefix_count_rejected(self):
        with j.prefix_guard([]):
            with self.assertRaisesRegex(RuntimeError,'UNBOUNDED'):pd.read_csv(j.c.N1,nrows=3005)
    def test_full_read_bytes_rejected(self):
        with j.prefix_guard([]):
            with self.assertRaisesRegex(RuntimeError,'FULL_N1_BYTES'):j.c.N1.read_bytes()
    def test_prefix_suffix_invariance_and_prefix_sensitivity(self):
        runtime=j.PHASE/'runtime_synthetic_tests';runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='prefix_firewall_',dir=runtime) as folder:
            root=Path(folder);base=pd.DataFrame({'bin_end_ophrs2_h':np.r_[np.linspace(72,1800,3005),1820,1821], 'feature':np.arange(3007,dtype=float)})
            a=pd.concat([base,pd.DataFrame({'bin_end_ophrs2_h':[2300.], 'feature':[42.]})],ignore_index=True)
            b=pd.concat([base,pd.DataFrame({'bin_end_ophrs2_h':[2400.,2500.], 'feature':[-900.,1e9]})],ignore_index=True)
            changed=a.copy();changed.loc[10,'feature']=999
            hashes=[];frames=[]
            for name,frame in [('A',a),('B',b),('C',changed)]:
                p=root/(name+'.csv');frame.to_csv(p,index=False)
                with patch.object(j.c,'N1',p):dev,_=j.read_dev()
                frames.append(dev);hashes.append(j.hashlib.sha256(dev.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()).hexdigest())
            self.assertEqual(hashes[0],hashes[1]);self.assertNotEqual(hashes[0],hashes[2]);pd.testing.assert_frame_equal(frames[0],frames[1])
    def test_Test_raw_path_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'ALLOWLISTED'):j.allowed(j.PROJECT/'数据/Polcurve_Nominal_RHFC_2000h.csv')
    def test_Assessment_cache_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'ALLOWLISTED'):j.allowed(j.i.F_PRED[0])
    def test_historical_write_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'HISTORICAL_WRITE'):j.write_target(j.H/'outputs/forbidden.json')
    def test_gate_count_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'COUNT'):j.validate_gates(np.ones(2),np.zeros((3,5)),3)
    def test_gate_nonfinite_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'NONFINITE'):j.validate_gates(np.array([np.nan]),np.zeros((1,5)),1)
    def test_gate_range_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'OUT_OF_RANGE'):j.validate_gates(np.array([1.1]),np.zeros((1,5)),1)
    def test_passive_hook_and_fixed_hook_restoration(self):
        m=TinyGateModel().eval();x=np.arange(3*48*14,dtype=np.float32).reshape(3,48,14)/1000
        before=j.ir.snapshot(m);flags={k:p.requires_grad for k,p in m.named_parameters()}
        normal=j.e.predict(m,x,32,torch.device('cpu'))
        hooked,g,z=j.passive_predict(m,x,torch.device('cpu'))
        self.assertTrue(np.array_equal(normal,hooked));self.assertTrue(j.hooks_empty(m))
        fixed=j.fixed_predict(m,x,torch.device('cpu'));self.assertTrue(np.array_equal(fixed,np.full((3,24),.5,np.float32)))
        self.assertTrue(j.hooks_empty(m));restored=j.e.predict(m,x,32,torch.device('cpu'));self.assertTrue(np.array_equal(restored,normal))
        j.verify_state(j.ir.state_audit(m,before,flags))
    def test_hook_removed_on_exception(self):
        m=TinyGateModel();x=np.zeros((2,48,14),np.float32)
        with patch.object(j.e,'predict',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):j.fixed_predict(m,x,torch.device('cpu'))
        self.assertTrue(j.hooks_empty(m))
    def test_state_change_rejected(self):
        m=TinyGateModel();before=j.ir.snapshot(m);flags={k:p.requires_grad for k,p in m.named_parameters()}
        with torch.no_grad():m.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError,'STATE_OR_GRADIENT'):j.verify_state(j.ir.state_audit(m,before,flags))
    def test_backward_and_optimizer_prohibited(self):
        with j.no_training_guard():
            with self.assertRaisesRegex(RuntimeError,'PROHIBITED'):torch.tensor(1.,requires_grad=True).backward()
            with self.assertRaisesRegex(RuntimeError,'PROHIBITED'):torch.optim.Adam([torch.nn.Parameter(torch.ones(1))])
    def test_decoder_no_future_physical_covariates(self):
        d=CanonicalStrictHistoryTFT.build_input(torch.ones(2,48,14))
        self.assertTrue(torch.equal(d['decoder_cont'][:,:,:14],torch.zeros(2,24,14)))
        self.assertTrue(torch.equal(d['decoder_cont'][0,:,-1],torch.arange(1,25)/24))
    def test_convolution_future_perturbation_invariance(self):
        branch=CausalMultiScaleHistoryBranch(64,.1).eval()
        x=torch.arange(48*64,dtype=torch.float32).reshape(1,48,64)/100
        changed=x.clone();changed[:,30:]+=5
        with torch.no_grad():a=branch(x);b=branch(changed)
        self.assertTrue(torch.equal(a[:,:30],b[:,:30]));self.assertEqual(a.shape,x.shape)
    def test_EWMA_future_perturbation_invariance(self):
        x=np.arange(48*14,dtype=np.float32).reshape(1,48,14);b=x.copy();b[:,24:,0]+=100
        self.assertTrue(np.array_equal(j.e.ewma(x)[:,:24],j.e.ewma(b)[:,:24]))
    def test_hash_tamper_fails_production(self):
        p=j.PHASE/'runtime_synthetic_tests/identity.txt';p.parent.mkdir(exist_ok=True);p.write_text('original',encoding='utf-8')
        records={p.relative_to(j.REPO).as_posix():j.digest(p)};p.write_text('tampered',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'IDENTITY_CHANGED'):j.verify_records(records)
    def test_gate_descriptive_thresholds(self):
        s=j.gate_statistics(np.array([.01,.5,.99]))
        self.assertEqual(s['low_lt_0p1_fraction'],1/3);self.assertEqual(s['high_gt_0p9_fraction'],1/3);self.assertEqual(s['middle_0p4_to_0p6_fraction'],1/3)
    def test_gate_decision_no_benefit_is_unresolved(self):
        s=pd.DataFrame({'SD':[.1]*3,'temporal_range':[.3]*3})
        c=pd.DataFrame({'prediction_difference_RMSE_mV':[1.]*3,'learned_minus_fixed_RMSE_mV':[.1]*3})
        self.assertEqual(j.gate_decision(s,c),'PHASE_J_GATE_DYNAMIC_EFFECT_UNRESOLVED')
        c.learned_minus_fixed_RMSE_mV=-1.
        self.assertEqual(j.gate_decision(s,c),'PHASE_J_GATE_DYNAMIC_AND_PREDICTIVE_EFFECT_OBSERVED')

if __name__=='__main__':unittest.main()
