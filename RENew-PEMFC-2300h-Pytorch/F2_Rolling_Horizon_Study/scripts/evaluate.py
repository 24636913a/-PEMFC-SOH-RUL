"""One descriptive holdout pass after ALL six best weights are frozen.

Temporary read-only forward hooks record true encoder path scales and g. They
never replace outputs or change parameters. C1's native output is retained for
an inference-only head-off counterfactual, not a separately trained baseline.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import HERE,CONFIG,SOURCE,read,read_csv,csv,dump,sha,metrics,utc
from data import prepare,points,add_events
from models import new_model,configure
from train import state_sha

@torch.no_grad()
def infer_diagnostics(model,x,kind):
    model.eval();pred=[];native=[];corr=[];stats=[];hooks=[];captured={}
    def capture(name):
        def hook(module,inputs,output):
            # Upstream TFT shares encoder/decoder AddNorm. Retain the actual
            # 48-position ENCODER tensor; decoder callbacks must not overwrite it.
            if name=='LSTM' and output.shape[1]!=48:return
            captured[name]=output.detach()
        return hook
    if kind!='LSTM':
        hooks=[model.history_branch.register_forward_hook(capture('CNN')),
            model.post_lstm_add_norm_encoder.register_forward_hook(capture('LSTM')),
            model.regime_gate.register_forward_hook(capture('g'))]
    try:
        for i in range(0,len(x),32):
            xb=torch.from_numpy(x[i:i+32]).to(next(model.parameters()).device);y=model.predict_delta(xb);pred.append(y.cpu().numpy())
            if kind=='C1':native.append(model.last_native.cpu().numpy());corr.append(model.last_correction.cpu().numpy())
            if kind!='LSTM':
                g=captured['g'].view(-1);a=captured['LSTM'];b=captured['CNN']
                assert a.shape==b.shape and a.shape[1:]==(48,64)
                stats.extend(dict(g=float(gg),LSTM_latent_RMS=float(aa),CNN_latent_RMS=float(bb),
                    weighted_LSTM_RMS=float(gg*aa),weighted_CNN_RMS=float((1-gg)*bb))
                    for gg,aa,bb in zip(g.cpu().numpy(),a.square().mean((1,2)).sqrt().cpu().numpy(),b.square().mean((1,2)).sqrt().cpu().numpy()))
    finally:
        for hook in hooks:hook.remove()
    return np.concatenate(pred),pd.DataFrame(stats),np.concatenate(native) if native else None,np.concatenate(corr) if corr else None

def run(out):
    freeze=read(out/'ALL_SIX_BEST_CHECKPOINTS_FROZEN.json');assert len(freeze['checkpoints'])==6
    assert freeze['protocol_sha256']==sha(CONFIG) and freeze['source_sha256']==sha(SOURCE)
    if (out/'HOLDOUT_EVALUATION_COMPLETED.json').exists():raise RuntimeError('HOLDOUT_PASS_ALREADY_DONE_USE_READONLY_ANALYSIS')
    if (out/'HOLDOUT_EVALUATION_STARTED.json').exists():raise RuntimeError('INTERRUPTED_HOLDOUT_PASS_PRESERVED_REQUIRES_REVIEW')
    configure();dump(out/'HOLDOUT_EVALUATION_STARTED.json',dict(created_UTC=utc(),all_best_frozen_before_this_pass=True,
        checkpoint_manifest_sha256=sha(out/'ALL_SIX_BEST_CHECKPOINTS_FROZEN.json'),evaluation_code_sha256=sha(Path(__file__)),
        diagnostic_hook_contract='Shared AddNorm: capture 48-position historical encoder output only; ignore decoder callbacks',
        purpose='Exploratory single-seed evaluation, not blind Test'))
    rows=[];steps=[];eventrows=[];pathrows=[];toprows=[];counter=[]
    thresholds=read(out/'TRAIN_EVENT_THRESHOLDS.json')
    for item in freeze['checkpoints']:
        kind=item['model'];h=int(item['horizon_steps']);job=out/f'models/{kind}_H{h}'
        info=read(job/'COMPLETED.json');assert sha(job/'best.pt')==item['SHA256']==info['best_checkpoint_sha256']
        prep=read(out/f'preprocess/H{h}.json');frame,d,identity=prepare(out,h,True,prep)
        for part in ['TRAIN','VALIDATION']:assert prep['partitions'][part]==identity['partitions'][part]
        model=new_model(kind,h).cuda();checkpoint=torch.load(job/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True);before=state_sha(model);assert before==info['best_state_sha256']
        print(f'Exploratory holdout inference {kind} H{h}; best epoch {info["best_epoch"]}; {len(d["HOLDOUT"]["origins"])} origins',flush=True)
        yp,pathstats,base,correction=infer_diagnostics(model,d['HOLDOUT']['x'],kind)
        p=points(frame,d['HOLDOUT'],yp*prep['target_scale_V'],h,kind,'HOLDOUT');p=add_events(frame,p,thresholds)
        csv(job/'predictions_HOLDOUT.csv',p)
        common=read_csv(out/'windows/H24_HOLDOUT.csv').origin_row
        for scope,g in [('ALL_H_LEGAL',p),('COMMON_H24_ORIGINS',p[p.origin_row.isin(common)])]:
            rows.append(dict(model=kind,horizon_steps=h,partition='HOLDOUT',scope=scope,**metrics(g)))
        for step,g in p.groupby('horizon_step'):steps.append(dict(model=kind,horizon_steps=h,partition='HOLDOUT',horizon_step=int(step),**metrics(g)))
        for event in ['event_any','event_high_zero','event_high_switching','event_voltage_jump','event_start_stop']:
            for flag,g in p.groupby(event):eventrows.append(dict(model=kind,horizon_steps=h,event=event,value=bool(flag),**metrics(g)))
        p['error2']=(p.y_pred_delta_V-p.y_true_delta_V)**2
        tg=p.groupby(['target_row','target_age_h'],as_index=False).agg(queries=('error2','size'),SSE_V2=('error2','sum'),MSE_V2=('error2','mean'))
        tg['SSE_fraction']=tg.SSE_V2/tg.SSE_V2.sum();tg['model']=kind;tg['horizon_steps']=h;csv(job/'TARGET_SSE.csv',tg)
        top=tg.sort_values('SSE_V2',ascending=False).head(10);toprows.append(top)
        if kind!='LSTM':
            pathstats['origin_row']=d['HOLDOUT']['origins'];pathstats['origin_age_h']=d['HOLDOUT']['origin_age_h']
            pathstats['model']=kind;pathstats['horizon_steps']=h;pathrows.append(pathstats);csv(job/'HOLDOUT_PATH_DIAGNOSTICS.csv',pathstats)
        if kind=='C1':
            pc=p.copy();pc['y_pred_delta_V']=(base*prep['target_scale_V']).ravel()
            pc['predicted_voltage_V']=pc.origin_raw_voltage_V+pc.y_pred_delta_V
            csv(job/'predictions_HOLDOUT_HEAD_OFF.csv',pc)
            cp=p[['origin_row','target_row','horizon_step']].copy();cp['native_delta_V']=(base*prep['target_scale_V']).ravel()
            cp['residual_component_V']=(correction*prep['target_scale_V']).ravel();cp['final_delta_V']=p.y_pred_delta_V
            cp['alpha']=float(model.residual_alpha.detach());csv(job/'HOLDOUT_OUTPUT_COMPONENTS.csv',cp)
            counter.append(dict(model='C1_head_off_inference_only',horizon_steps=h,partition='HOLDOUT',**metrics(pc)))
            counter.append(dict(model='C1_full_same_weights',horizon_steps=h,partition='HOLDOUT',**metrics(p)))
        assert state_sha(model)==before and sha(job/'best.pt')==item['SHA256']
        rows.extend(read_csv(job/'TRAIN_VALIDATION_METRICS.csv').to_dict('records'))
        steps.extend(read_csv(job/'TRAIN_VALIDATION_STEP_METRICS.csv').to_dict('records'))
        del model,checkpoint;torch.cuda.empty_cache()
    csv(out/'ALL_PARTITION_METRICS.csv',rows);csv(out/'ALL_STEP_METRICS.csv',steps)
    csv(out/'HOLDOUT_EVENT_METRICS.csv',eventrows);csv(out/'HOLDOUT_TOP_TARGETS.csv',pd.concat(toprows))
    csv(out/'ALL_HOLDOUT_PATH_DIAGNOSTICS.csv',pd.concat(pathrows));csv(out/'C1_HEAD_OFF_METRICS.csv',counter)
    new=pd.DataFrame(rows);final=new[(new.partition=='HOLDOUT')&(new.scope=='ALL_H_LEGAL')].copy()
    final['history_steps']=48;final['source']='NEW_FRESH_TRAINING'
    hist=read(out/'P1_HISTORICAL_AUDIT.json')['full24']
    final=pd.concat([final,pd.DataFrame([dict(model='Historical_F2_202_selected',history_steps=48,horizon_steps=24,partition='HOLDOUT',scope='HISTORICAL_H24',source='HISTORICAL_READONLY',**{k:v for k,v in hist.items() if k not in ['source','prefix_steps']})])],ignore_index=True)
    order={('LSTM',6):0,('F2',6):1,('LSTM',12):2,('F2',12):3,('Historical_F2_202_selected',24):4,('C0',24):5,('C1',24):6}
    final['seed']=202;final['_order']=[order[(r.model,int(r.horizon_steps))] for r in final.itertuples()]
    final=final.sort_values('_order').drop(columns='_order').reset_index(drop=True)
    csv(out/'FINAL_RESULT_TABLE.csv',final)
    validation=new[(new.partition=='VALIDATION')&(new.scope=='ALL_H_LEGAL')].set_index('model')
    c0=validation.loc['C0'];c1=validation.loc['C1']
    gain=1-c1.RMSE_mV/c0.RMSE_mV
    decision=dict(validation_RMSE_gain_fraction=float(gain),validation_MAE_change_fraction=float(c1.MAE_mV/c0.MAE_mV-1),
        validation_P99_change_fraction=float(c1.P99_mV/c0.P99_mV-1),
        preliminary_validation_improvement=bool(gain>=.05 and c1.MAE_mV<=c0.MAE_mV and c1.P99_mV<=c0.P99_mV),
        stable_gain_proven=False,one_seed_only=True,all_candidates_retained=True,no_followup_search=True)
    dump(out/'C1_VALIDATION_DECISION.json',decision)
    dump(out/'HOLDOUT_EVALUATION_COMPLETED.json',dict(completed_UTC=utc(),PASS=True,completed_models=6,
        all_six_best_frozen_before_inference=True,evaluation_rounds=1,independent_replay_to_follow=True,no_model_selection_on_new_holdout=True))
    print('Single unified exploratory holdout pass completed for all six new checkpoints',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args();run(a.run or Path(read(HERE/'outputs/ACTIVE_RUN.json')['path']))
