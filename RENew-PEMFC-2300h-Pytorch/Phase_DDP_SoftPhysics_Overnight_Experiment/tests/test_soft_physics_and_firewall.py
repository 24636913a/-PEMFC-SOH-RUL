"""Real production loss/model calls; synthetic data only."""
import inspect,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import ddp_common as io
from soft_physics import *
torch.set_num_threads(1)

class SoftPhysicsTests(unittest.TestCase):
    def test_restore_uses_unfiltered_anchor_and_scale_not_mean(self):
        delta=torch.full((2,24),0.5,dtype=torch.float64)
        answer=restore_absolute_voltage(delta,torch.tensor([0.6,0.7]),0.2)
        self.assertTrue(torch.allclose(answer[:,0],torch.tensor([0.7,0.8],dtype=torch.float64),atol=1e-7))
    def test_synthetic_inside_interval_loss_and_gradient_zero(self):
        v=torch.tensor([0.2,0.8],requires_grad=True)
        loss=voltage_domain_penalty(v,0.1,0.0,1.0)
        loss.backward()
        self.assertEqual(loss.item(),0.0);self.assertTrue(torch.equal(v.grad,torch.zeros(2)))
    def test_synthetic_lower_violation_gradient_nonzero_correct_direction(self):
        v=torch.tensor([-0.2],requires_grad=True)
        loss=voltage_domain_penalty(v,0.1,0.0,None);loss.backward()
        self.assertTrue(torch.isfinite(v.grad).all());self.assertLess(v.grad.item(),0)
        self.assertGreater(loss.item(),0)
    def test_synthetic_upper_violation_gradient_nonzero(self):
        # 1.0 is an artificial API fixture, never the RealHyFC bound.
        v=torch.tensor([1.2],requires_grad=True)
        loss=voltage_domain_penalty(v,0.1,None,1.0);loss.backward()
        self.assertGreater(v.grad.item(),0);self.assertTrue(torch.isfinite(v.grad).all())
    def test_zero_weight_exact_MSE_and_gradient(self):
        p=torch.full((2,24),-10.0,requires_grad=True);y=torch.zeros_like(p)
        total,data,physics,_=combined_objective(p,y,torch.full((2,),0.6),0.1,0.0)
        expected=F.mse_loss(p,y)
        self.assertTrue(torch.equal(total,expected));self.assertGreater(physics.item(),0)
        g1=torch.autograd.grad(total,p,retain_graph=True)[0];g2=torch.autograd.grad(expected,p)[0]
        self.assertTrue(torch.equal(g1,g2))
    def test_invalid_scales_and_bounds_fail(self):
        for s in (0,-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):voltage_domain_penalty(torch.ones(1),s)
        with self.assertRaises(ValueError):voltage_domain_penalty(torch.ones(1),1,1,0)
        with self.assertRaises(ValueError):voltage_domain_penalty(torch.ones(1),1,None,None)
    def test_finite_and_shape_guards(self):
        with self.assertRaises(ValueError):restore_absolute_voltage(torch.zeros(2,23),torch.ones(2),0.1)
        with self.assertRaises(ValueError):voltage_domain_penalty(torch.tensor([float("nan")]),0.1)
    def test_admission_rejects_inactive_or_missing_gate(self):
        with self.assertRaises(PermissionError):require_training_admission({"formal_training_admitted":False})
        with self.assertRaises(PermissionError):require_training_admission({"formal_training_admitted":True})
        yes={"formal_training_admitted":True,**{k:True for k in
            ("source_identity_pass","physical_domain_pass","train_activity_pass","weight_balance_pass","model_parity_pass")}}
        require_training_admission(yes)

class FrozenG2Tests(unittest.TestCase):
    def test_real_frozen_G2_forward_backward_lambda0_parity_and_decoder(self):
        a,_=io.load_legacy()
        config=io.read_json(io.P3/"configs/P3_FROZEN_FULL450_PROTOCOL.json")["H"]
        a.set_seed(42);cohort=a.build_cohort(42,config,torch.device("cpu"));m=cohort["G2"]
        before=a.state_digest(m.state_dict());m.eval()
        x=torch.from_numpy(a.synthetic_input(2))
        prediction=m.predict_delta(x)
        self.assertEqual(tuple(prediction.shape),(2,24))
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),332436)
        total,data,phys,_=combined_objective(prediction,torch.zeros_like(prediction),torch.full((2,),0.7),0.1,0.0)
        self.assertTrue(torch.equal(total,F.mse_loss(prediction,torch.zeros_like(prediction))))
        params=tuple(p for p in m.parameters() if p.requires_grad)
        g1=torch.autograd.grad(total,params,allow_unused=True,retain_graph=True)
        g2=torch.autograd.grad(data,params,allow_unused=True)
        for left,right in zip(g1,g2):
            if left is None:self.assertIsNone(right)
            else:self.assertTrue(torch.equal(left,right))
        self.assertEqual(before,a.state_digest(m.state_dict()))
        with torch.no_grad():self.assertTrue(torch.equal(prediction,m.predict_delta(x)))
        decoder=m.build_input(x)["decoder_cont"]
        self.assertTrue(torch.equal(decoder[:,:,:14],torch.zeros_like(decoder[:,:,:14])))
        self.assertNotIn("future",str(inspect.signature(m.predict_delta)))

class FirewallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (io.PHASE/"runtime").mkdir(parents=True,exist_ok=True)
        io.install_firewall()
    def test_N1_unbounded_or_3005_reader_fail_closed(self):
        for kw in ({},{"nrows":3005},{"nrows":3007,"chunksize":20}):
            with self.assertRaises(PermissionError):pd.read_csv(io.N1,**kw)
    def test_N1_read_bytes_and_full_hash_fail_closed(self):
        with self.assertRaises(PermissionError):io.N1.read_bytes()
        with self.assertRaises(PermissionError):io.file_hash(io.N1)
    def test_Test_raw_polarization_and_historical_write_fail_closed(self):
        forbidden=[io.PROJECT/"Test/targets.csv",io.PROJECT/"Polcurve_Nominal_RHFC_2000h.csv",
                   Path("E:/数据/Loadcycle_151-162.csv")]
        for path in forbidden:
            with self.assertRaises(PermissionError):io.permitted_open(path)
        with self.assertRaises(PermissionError):io.permitted_open(io.P3/"scripts/p3_common.py",True)
    def test_prefix_suffix_invariant_and_Dev_mutation_sensitive(self):
        with tempfile.TemporaryDirectory(dir=io.PHASE/"runtime") as tmp:
            tmp=Path(tmp)
            age=np.r_[np.linspace(80,1700,3005),1900.,2000.]
            base=pd.DataFrame({"bin_end_ophrs2_h":age,"cell_voltage_mean":np.full(3007,0.7)})
            a=tmp/"A.csv";b=tmp/"B.csv";c=tmp/"C.csv"
            base.to_csv(a,index=False)
            with b.open("w",encoding="utf-8") as stream:
                stream.write(base.to_csv(index=False));stream.write("arbitrary,suffix\n"*50)
            changed=base.copy();changed.loc[15,"cell_voltage_mean"]=0.71;changed.to_csv(c,index=False)
            da,ha=io.read_development_prefix(a);db,hb=io.read_development_prefix(b);dc,hc=io.read_development_prefix(c)
            self.assertEqual(ha,hb);pd.testing.assert_frame_equal(da,db);self.assertNotEqual(ha,hc)
    def test_complete_manifest_one_tampered_hash_rejected(self):
        with tempfile.TemporaryDirectory(dir=io.PHASE/"runtime") as tmp:
            p=Path(tmp)/"source.json";q=Path(tmp)/"other.json"
            p.write_text("{}");q.write_text("{}")
            records={str(p):io.file_hash(p),str(q):io.file_hash(q)}
            io.verify_hash_manifest(records)
            records[str(q)]="0"*64
            with self.assertRaises(RuntimeError):io.verify_hash_manifest(records)

if __name__=="__main__":unittest.main()
