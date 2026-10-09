from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np
PHASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PHASE/'scripts'))
from t1r_decision import *  # noqa: F403
from run_t1r import make_loader,validate_isolation
from t1r_common import load_protocol

def metrics(rmse,fold3=30,p99=10,target=10): return {'rmse':rmse,'fold3_rmse':fold3,'p99':p99,'target_equal_rmse':target}
class DecisionTests(unittest.TestCase):
    def test_c_beats_a_but_not_b_is_not_optuna_advance(self): self.assertNotEqual(seed42_decision(metrics(26),metrics(18),metrics(18.5),audits_pass=True)['status'],ADVANCE)
    def test_budget_only_requires_b_substantial_and_c_not_better(self): self.assertEqual(seed42_decision(metrics(26.158),metrics(18),metrics(18.5),audits_pass=True)['status'],EXTENDED)
    def test_c_beats_b_but_fold3_fails(self): self.assertEqual(seed42_decision(metrics(26.158),metrics(18.5),metrics(18,40),audits_pass=True)['status'],INSUFFICIENT)
    def test_c_beats_b_but_p99_fails(self): self.assertEqual(seed42_decision(metrics(26.158),metrics(18.5,p99=10),metrics(18,p99=12),audits_pass=True)['status'],INSUFFICIENT)
    def test_seed42_pass_requires_multiseed(self):
        first=seed42_decision(metrics(26.158),metrics(20),metrics(18),audits_pass=True); self.assertEqual(first['status'],PENDING); self.assertEqual(multiseed_decision(first,None,audits_pass=True)['status'],PENDING)
    def test_multiseed_pass_and_fail(self):
        first={'status':PENDING}; good=[{'b_rmse':20,'c_rmse':18,'c_fold3_rmse':30}]*3; bad=[{'b_rmse':18,'c_rmse':19,'c_fold3_rmse':30}]*3
        self.assertEqual(multiseed_decision(first,good,audits_pass=True)['status'],ADVANCE); self.assertEqual(multiseed_decision(first,bad,audits_pass=True)['status'],INSUFFICIENT)
    def test_trial_budget_and_audit_fail_closed(self):
        self.assertFalse(check_trial_budget(17,12)['pass']); self.assertFalse(check_trial_budget(16,11)['pass']); self.assertEqual(seed42_decision(metrics(1),metrics(1),metrics(0.5),audits_pass=False)['status'],TECHNICAL)
    def test_reused_loader_matches_t0_sequence_contract(self):
        x=np.arange(40,dtype=np.float32).reshape(20,2); y=np.arange(20,dtype=np.float32).reshape(20,1)
        a=make_loader(x,y,4,42); b=make_loader(x,y,4,42)
        order_a=[[int(v) for _,batch in [(xb,yb)] for v in batch[:,0]] for xb,yb in a]
        order_b=[[int(v) for _,batch in [(xb,yb)] for v in batch[:,0]] for xb,yb in b]
        self.assertEqual(order_a,order_b)
    def test_protocol_and_isolation(self):
        cfg,_=load_protocol(); validate_isolation(cfg); self.assertEqual(cfg['search']['study_name_prefix'],'T1R')
if __name__=='__main__': unittest.main(verbosity=2)
