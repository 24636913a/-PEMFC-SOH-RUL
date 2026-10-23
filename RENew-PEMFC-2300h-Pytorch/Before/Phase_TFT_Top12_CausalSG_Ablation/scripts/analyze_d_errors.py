"""Descriptive error concentration tables for already-frozen Phase D caches.

This script never trains a model or changes a prediction. Overlapping targets
are retained and explicitly labelled descriptive rather than independent.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

PHASE=Path(__file__).resolve().parents[1]; OUT=PHASE/'outputs'; C_OUT=PHASE.parent/'Phase_TFT_SingleSplit_Feature_Filter_Ablation'/'outputs'
rows=[]
for label in ('D0','D1','D2','D3'):
    f=pd.read_csv(OUT/f'{label}_predictions_seed42.csv'); f['abs_error_mV']=(f.y_pred_delta_V-f.y_true_delta_V).abs()*1000; f['se_mV2']=((f.y_pred_delta_V-f.y_true_delta_V)*1000)**2
    for target,g in f.groupby('target_row',sort=True): rows.append({'experiment_group':label,'analysis':'target_row_mean_absolute_error','key':int(target),'value':float(g.abs_error_mV.mean()),'n_overlapping_predictions':int(len(g))})
    for horizon,g in f.groupby('horizon_step',sort=True): rows.append({'experiment_group':label,'analysis':'horizon_rmse','key':int(horizon),'value':float(np.sqrt(g.se_mV2.mean())),'n_overlapping_predictions':int(len(g))})
    for _,r in f.nlargest(10,'abs_error_mV').iterrows(): rows.append({'experiment_group':label,'analysis':'top10_absolute_error','key':int(r.target_row),'value':float(r.abs_error_mV),'origin_row':int(r.origin_row),'horizon_step':int(r.horizon_step),'n_overlapping_predictions':None})
pd.DataFrame(rows).to_csv(OUT/'D_ERROR_ANALYSIS.csv',index=False)
d0=float(pd.read_csv(OUT/'D_SEED42_METRICS.csv').query("experiment_group=='D0'").RMSE_mV.iloc[0]); c1=float(pd.read_csv(C_OUT/'C_SEED42_METRICS.csv').query("experiment_group=='C1'").RMSE_mV.iloc[0])
payload={'D0_RMSE_mV':d0,'historical_C1_RMSE_mV':c1,'difference_D0_minus_C1_mV':d0-c1,'rank_window_scaler_and_seed_audited':True,'strict_numeric_reproduction_established':False,'rerun_or_hyperparameter_change_performed':False,'explanation_boundary':'The existing evidence confirms matching task, windows, ranking, common raw Train scaling rule, Seed42 and TFT configuration. It does not isolate the remaining execution-level numerical divergence without an unregistered rerun.'}
(OUT/'D_C1_REPRODUCIBILITY_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
