from __future__ import annotations
import sys, unittest
from dataclasses import dataclass, field
from pathlib import Path
PHASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PHASE/'scripts'))
from t1r_common import load_protocol
from t1r_optuna import OptunaContractError, audit_study

@dataclass
class Trial:
    number:int; state:str='COMPLETE'; value:float=1.; params:dict=field(default_factory=dict); user_attrs:dict=field(default_factory=lambda:{'best_epoch':4,'run_namespace':'T1R'})
class Study:
    def __init__(self,trials,name='T1R_fold_1'): self.trials=trials; self.study_name=name
    @property
    def best_trial(self):
        done=[t for t in self.trials if t.state=='COMPLETE']; return min(done,key=lambda t:t.value) if done else None
def good(cfg):
    resolved=dict(cfg['search']['fixed_trial_0'])
    ts=[Trial(i,value=float(16-i),params=(dict(resolved) if i==0 else {'learning_rate':.001}),user_attrs={'best_epoch':4,'run_namespace':'T1R','resolved_params':dict(resolved)}) for i in range(16)]
    return Study(ts)

class OptunaBudgetTests(unittest.TestCase):
    def setUp(self): self.cfg,_=load_protocol()
    def test_study_object_closure_passes(self): self.assertTrue(audit_study(good(self.cfg),self.cfg)['pass'])
    def test_bad_attempt_count_low_completed_running_trial0_and_missing_best_fail(self):
        cases=[]; cases.append(Study(good(self.cfg).trials[:12]))
        low=good(self.cfg); [setattr(t,'state','FAILED') for t in low.trials[11:]]; [t.user_attrs.update({'failure_reason':'x'}) for t in low.trials[11:]]; cases.append(low)
        run=good(self.cfg); run.trials[-1].state='RUNNING'; cases.append(run)
        zero=good(self.cfg); zero.trials[0].user_attrs['resolved_params']['batch_size']=32; cases.append(zero)
        old=good(self.cfg); old.study_name='T1_fold_1'; cases.append(old)
        for study in cases:
            with self.subTest(study=study.study_name):
                with self.assertRaises(OptunaContractError): audit_study(study,self.cfg)
    def test_failed_trial_requires_reason(self):
        s=good(self.cfg); s.trials[-1].state='FAILED'
        with self.assertRaises(OptunaContractError): audit_study(s,self.cfg)

if __name__=='__main__': unittest.main()
