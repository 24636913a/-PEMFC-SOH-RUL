from __future__ import annotations
import copy, sys, tempfile, unittest
from pathlib import Path
import numpy as np
PHASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PHASE/'scripts'))
from t1r_common import load_protocol, load_t0_model
from t1r_real_adapter import RealAdapter

class RealAdapterSmoke(unittest.TestCase):
    def test_real_canonical_tft_fit_stopper_predict_metrics_and_artifacts(self):
        cfg,sha=load_protocol(); cfg=copy.deepcopy(cfg); cfg['search']['attempted_trials_per_outer_fold']=1; cfg['search']['minimum_completed_trials_per_outer_fold']=1; cfg['training']['max_epochs']=2; cfg['training']['patience']=1
        adapter=RealAdapter.__new__(RealAdapter); adapter.cfg=cfg; adapter.protocol_sha=sha; adapter.device=__import__('torch').device('cpu'); adapter.root=Path(tempfile.mkdtemp()); adapter.studies={}; adapter.seed_epoch_records=[]
        # Use the real frozen BS window/scaler utilities but synthetic values.
        from t1r_real_adapter import _module, T0
        adapter.common=_module('smoke_t0_common',T0/'scripts'/'tft_common.py'); adapter.bs_cfg,adapter.bs=adapter.common.load_bs_common(); adapter.Model=load_t0_model()[0]
        W=adapter.bs.WindowSet
        def ws(n,offset):
            rng=np.random.default_rng(offset); origins=np.arange(offset,offset+n); return W(x=rng.normal(size=(n,48,14)).astype('float32'),y_delta_V=rng.normal(scale=.001,size=(n,24)).astype('float32'),anchor_V=np.ones(n,dtype='float32'),origin_age_h=origins.astype(float),origin_datetime=np.array(['x']*n),target_end_age_h=(origins+1).astype(float),origin_row=origins,target_row=np.stack([np.arange(o*24+1,o*24+25) for o in origins]),target_datetime=np.array([['x']*24 for _ in origins]))
        adapter.folds={f:(ws(8,10+i),ws(2,100+i)) for i,f in enumerate(('fold_1','fold_2','fold_3'))}
        audits={f:__import__('t1r_optuna').audit_study(adapter.study(f),cfg) for f in adapter.folds}; adapter.freeze_selected(audits)
        rows,_=adapter.evaluate(42,{f:{'params':a['best_params'],'best_epoch':a['best_epoch']} for f,a in audits.items()}); metrics=adapter.metrics(rows,'B')
        self.assertEqual(len(rows),288); self.assertTrue(np.isfinite(metrics['rmse'])); self.assertTrue((adapter.root/'T1R_predictions_seed42.csv').is_file()); self.assertTrue((adapter.root/'T1R_SELECTED_CONFIGURATIONS.json').is_file())
        adapter.persist_result({'final_status':'SMOKE_ONLY'}); self.assertTrue(adapter.audit_artifacts()['pass'])
        (adapter.root/'T1R_FINAL_DECISION.json').unlink()
        with self.assertRaises(Exception): adapter.audit_artifacts()

if __name__=='__main__': unittest.main()
