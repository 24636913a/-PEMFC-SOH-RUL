from __future__ import annotations
import sys, unittest
from dataclasses import dataclass, field
from pathlib import Path
PHASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PHASE/'scripts'))
from run_t1r import PipelineFailure, execute_pipeline
from t1r_common import load_protocol

@dataclass
class Trial:
    number:int; state:str='COMPLETE'; value:float=1.; params:dict=field(default_factory=dict); user_attrs:dict=field(default_factory=lambda:{'best_epoch':8,'run_namespace':'T1R'})
class Study:
    def __init__(self,cfg,fold,broken=None):
        self.study_name=f'T1R_{fold}'; self.trials=[Trial(i,value=float(16-i),params=(dict(cfg['search']['fixed_trial_0']) if i==0 else {'learning_rate':.001})) for i in range(16)]
        if broken=='attempts': self.trials=self.trials[:-1]
        if broken=='running': self.trials[-1].state='RUNNING'
    @property
    def best_trial(self): return min([t for t in self.trials if t.state=='COMPLETE'],key=lambda x:x.value)
class FakeAdapter:
    def __init__(self,cfg,sha,fail_seed42=False,broken=None): self.cfg,self.sha,self.fail_seed42,self.broken=cfg,sha,fail_seed42,broken; self.freeze_calls=0; self.evaluated=[]
    def study(self,fold): return Study(self.cfg,fold,self.broken)
    def freeze_selected(self,audits): self.freeze_calls+=1
    def t0_queries(self):
        return [{'fold':f,'origin_row':i,'target_row':i+1,'horizon_step':1} for f,n in {'fold_1':233,'fold_2':245,'fold_3':466}.items() for i in range(n)]
    def evaluate(self,seed,selected):
        self.evaluated.append(seed); rows=[]
        for model in ('B','C'):
            for q in self.t0_queries(): rows.append({**q,'seed':seed,'model':model,'y_true_delta_V':.1,'y_pred_delta_V':.09 if model=='C' else .11,'protocol_sha256':self.sha,'run_id':'isolated_fake'})
        return rows,{}
    def metrics(self,rows,model):
        rmse=18. if model=='C' else 20.
        if self.fail_seed42 and model=='C': rmse=21.
        return {'rmse':rmse,'fold3_rmse':30.,'p99':10.,'target_equal_rmse':10.}

class EndToEndTests(unittest.TestCase):
    def setUp(self): self.cfg,self.sha=load_protocol()
    def test_same_dispatcher_completes_isolated_fake_pipeline(self):
        fake=FakeAdapter(self.cfg,self.sha); result=execute_pipeline(cfg=self.cfg,protocol_sha=self.sha,adapter=fake,authorization_verified=True,synthetic=True)
        self.assertEqual(result['final_status'],'TFT_OPTUNA_ADVANCE'); self.assertEqual(fake.evaluated,[42,2024,3407]); self.assertEqual(fake.freeze_calls,1); self.assertIn('FINAL_DECISION',result['states'])
    def test_seed42_failure_never_starts_multiseed(self):
        fake=FakeAdapter(self.cfg,self.sha,fail_seed42=True); result=execute_pipeline(cfg=self.cfg,protocol_sha=self.sha,adapter=fake,authorization_verified=True,synthetic=True)
        self.assertEqual(result['final_status'],'FINAL_INSUFFICIENT_GAIN'); self.assertEqual(fake.evaluated,[42])
    def test_no_authorization_and_bad_trial_closure_fail_closed(self):
        with self.assertRaises(PipelineFailure): execute_pipeline(cfg=self.cfg,protocol_sha=self.sha,adapter=FakeAdapter(self.cfg,self.sha),authorization_verified=False,synthetic=True)
        with self.assertRaises(Exception): execute_pipeline(cfg=self.cfg,protocol_sha=self.sha,adapter=FakeAdapter(self.cfg,self.sha,broken='attempts'),authorization_verified=True,synthetic=True)

if __name__=='__main__': unittest.main()
