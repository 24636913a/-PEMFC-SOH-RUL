"""Authorized real T1-R adapter. It is never constructed by preparation mode."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, torch
from t1r_common import PHASE, T1RFailure, load_t0_model, safe_development
from t1r_optuna import audit_study, create_authorized_study, run_search, suggest_trial_params
from t1r_training import fit

T0=PHASE.parent/'Phase_TFT_Original_Baseline'

def _module(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise T1RFailure('T0_HELPER_IMPORT_FAILURE')
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

class RealAdapter:
    """Creates only new authorized T1R studies and uses T0's frozen window semantics."""
    def __init__(self,cfg:dict[str,Any],protocol_sha:str,*,folds=None,model_cls=None,artifact_root:Path|None=None):
        self.cfg,self.protocol_sha=cfg,protocol_sha; self.common=_module('t1r_t0_common',T0/'scripts'/'tft_common.py'); self.bs_cfg,self.bs=self.common.load_bs_common()
        if folds is None:
            frame,_=safe_development(cfg); self.folds={name:(tr,val) for name,tr,val in self.bs.build_outer_folds(frame,self.bs_cfg)}
        else: self.folds=folds
        self.Model=model_cls or load_t0_model()[0]; self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); self.studies={}; self.root=artifact_root or (PHASE/'outputs'); self.root.mkdir(parents=True,exist_ok=True); self.seed_epoch_records=[]
    def _predict(self,model,x,batch,device=None):
        model.eval(); out=[]
        with torch.no_grad():
            active=self.device if device is None else device
            for i in range(0,len(x),batch): out.append(model.predict_delta(torch.from_numpy(x[i:i+batch]).to(active)).cpu().numpy())
        return np.concatenate(out)
    def _model(self,p):
        return self.Model(hidden_size=int(p['hidden_size']),hidden_continuous_size=int(p['hidden_continuous_size']),attention_head_size=4,lstm_layers=1,dropout=float(p['dropout'])).to(self.device)
    def study(self,fold:str):
        import optuna
        if fold in self.studies:return self.studies[fold]
        storage=self.root/f'optuna_T1R_{fold}.db'
        if storage.exists(): raise T1RFailure('T1R_EXISTING_STUDY_RECOVERY_NOT_AUTHORIZED')
        study=create_authorized_study(optuna_module=optuna,storage_path=storage,fold=fold,cfg=self.cfg,test_only=self.root.resolve() != (PHASE/'outputs').resolve()); outer,_=self.folds[fold]; inner,stop=self.bs.split_inner_fit_stopper(outer,float(self.cfg['training']['chronological_inner_stopper_fraction']))
        def make(_index):
            def objective(trial):
                p=suggest_trial_params(trial,self.cfg); scaler=self.bs.FoldPreprocessor().fit(inner.x); xf=scaler.transform(inner.x); yf=self.bs.standardized_delta_target(inner,scaler); xs=scaler.transform(stop.x); ys=self.bs.standardized_delta_target(stop,scaler)
                torch.manual_seed(42); result=fit(self._model(p),xf,yf,params=p,training=self.cfg['training'],device=self.device,seed=42,stop_x=xs,stop_y=ys,predict=self._predict); trial.set_user_attr('best_epoch',int(result['best_epoch'])); trial.set_user_attr('run_namespace','T1R'); return float(result['best_value'])
            return objective
        run_search(study,self.cfg,make); self.studies[fold]=study; return study
    def freeze_selected(self,audits):
        payload={'protocol_sha256':self.protocol_sha,'folds':audits}; (self.root/'T1R_SELECTED_CONFIGURATIONS.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    def evaluate(self,seed:int,selected:dict[str,dict[str,Any]]):
        rows=[]; records={}
        for fold,(outer,val) in self.folds.items():
            study=self.studies[fold]; trial0=next(t for t in study.trials if t.number==0); attrs=trial0.user_attrs
            variants={'B':{'params':attrs['resolved_params'],'epoch':attrs['best_epoch']},'C':{'params':selected[fold]['params'],'epoch':selected[fold]['best_epoch']}}
            for label,choice in variants.items():
                if seed != 42:
                    inner,stop=self.bs.split_inner_fit_stopper(outer,float(self.cfg['training']['chronological_inner_stopper_fraction'])); inner_scaler=self.bs.FoldPreprocessor().fit(inner.x)
                    selected_inner=fit(self._model(choice['params']),inner_scaler.transform(inner.x),self.bs.standardized_delta_target(inner,inner_scaler),params=choice['params'],training=self.cfg['training'],device=self.device,seed=seed,stop_x=inner_scaler.transform(stop.x),stop_y=self.bs.standardized_delta_target(stop,inner_scaler),predict=self._predict)
                    choice={**choice,'epoch':int(selected_inner['best_epoch'])}
                self.seed_epoch_records.append({'seed':seed,'fold':fold,'model':label,'selected_epoch':int(choice['epoch']),'params':choice['params']})
                torch.manual_seed(seed); scaler=self.bs.FoldPreprocessor().fit(outer.x); x=self.bs.FoldPreprocessor.transform(scaler,outer.x); y=self.bs.standardized_delta_target(outer,scaler); model=self._model(choice['params']); fit(model,x,y,params=choice['params'],training=self.cfg['training'],device=self.device,seed=seed,epochs=int(choice['epoch']),predict=self._predict); pred=self._predict(model,self.bs.FoldPreprocessor.transform(scaler,val.x),int(choice['params']['batch_size']))*scaler.voltage_scale
                for q in range(len(val.origin_row)):
                    for h in range(24): rows.append({'seed':seed,'model':label,'fold':fold,'origin_row':int(val.origin_row[q]),'target_row':int(val.target_row[q,h]),'horizon_step':h+1,'y_true_delta_V':float(val.y_delta_V[q,h]),'y_pred_delta_V':float(pred[q,h]),'protocol_sha256':self.protocol_sha,'run_id':'T1R_authorized'})
        pd.DataFrame(rows).to_csv(self.root/f'T1R_predictions_seed{seed}.csv',index=False)
        pd.DataFrame(self.seed_epoch_records).to_json(self.root/'T1R_seed_epoch_identity.json',orient='records',indent=2)
        return rows,records
    def t0_queries(self):
        from t1r_evaluation import load_frozen_t0_query_universe
        return load_frozen_t0_query_universe(T0/'outputs'/'TFT_SEED42_PREDICTIONS.csv')
    def metrics(self,rows,model):
        select=[r for r in rows if r['model']==model]; per=[]; fold_metrics=[]
        for fold in ('fold_1','fold_2','fold_3'):
            group=sorted([r for r in select if r['fold']==fold],key=lambda r:(r['origin_row'],r['horizon_step'])); origins=sorted(set(r['origin_row'] for r in group)); y=np.array([[next(r['y_true_delta_V'] for r in group if r['origin_row']==o and r['horizon_step']==h) for h in range(1,25)] for o in origins]); p=np.array([[next(r['y_pred_delta_V'] for r in group if r['origin_row']==o and r['horizon_step']==h) for h in range(1,25)] for o in origins]); targets=np.array([[next(r['target_row'] for r in group if r['origin_row']==o and r['horizon_step']==h) for h in range(1,25)] for o in origins]); fold_metrics.append(self.bs.delta_metrics(y,p,targets))
        return {'rmse':float(np.mean([m['rmse_mV'] for m in fold_metrics])),'fold3_rmse':float(fold_metrics[2]['rmse_mV']),'p99':float(np.mean([m['p99_abs_error_mV'] for m in fold_metrics])),'target_equal_rmse':float(np.mean([m['target_time_equal_rmse_mV'] for m in fold_metrics]))}
    def persist_result(self,result):
        (self.root/'T1R_FINAL_DECISION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    def audit_artifacts(self):
        needed=[self.root/'T1R_SELECTED_CONFIGURATIONS.json',self.root/'T1R_FINAL_DECISION.json',self.root/'T1R_predictions_seed42.csv']
        if not all(p.is_file() for p in needed): raise T1RFailure('T1R_PERSISTED_ARTIFACT_MISSING')
        return {'pass':True,'artifacts':[str(p) for p in needed]}
