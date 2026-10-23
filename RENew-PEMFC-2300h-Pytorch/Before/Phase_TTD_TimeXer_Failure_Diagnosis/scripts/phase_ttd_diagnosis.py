"""Frozen Phase TTD diagnosis. No neural model training or checkpoint writes."""
from __future__ import annotations
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True
ROOT=Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_TTD_TimeXer_Failure_Diagnosis")
REPO=ROOT.parent
NROOT=REPO/'Phase_N2_8F_Uncertainty_Robust_SA_TimeXer'
GROOT=REPO/'Phase_G_STOP_Physics_Consistent_Future_Conditions'
OPR=REPO/'Phase_OPR_TimeXer'
RS=REPO/'Phase_RS_SA_TimeXer'
FOLDS=('fold_1','fold_2','fold_3'); EPS=1e-12

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
def write(df,rel):
    p=ROOT/rel; p.parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,index=False,encoding='utf-8-sig',lineterminator='\n')
def text(s,rel):
    p=ROOT/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s.rstrip()+'\n',encoding='utf-8')
def jdump(x,rel):
    p=ROOT/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rmse(v): return float(np.sqrt(np.mean(np.square(np.asarray(v,float))))) if len(v) else np.nan
def pct(v): return float(100*np.mean(v)) if len(v) else np.nan

def p2_oof_with_probability(p,aux_train,gcfg,n28f,gs,fold):
    n=len(p['htr']); start=n28f.first_index_with_completed(p['train'].target_end_age_h,p['train'].origin_age_h,int(gcfg['forecast']['oof_min_completed_windows']))
    raw=np.full((n,24),np.nan); prob=np.full((n,24),np.nan); aux=np.full((n,24,2),np.nan); audit=[]
    for b,(left,right) in enumerate(n28f.block_edges(start,n,int(gcfg['forecast']['oof_blocks'])),1):
        origin=float(p['train'].origin_age_h[left]); fit=n28f.completed_indices(p['train'].target_end_age_h,origin)
        qm=gs.TwoPartQForecaster(gcfg).fit(p['htr'][fit],p['ztr'][fit,:,8]); raw[left:right],prob[left:right]=qm.predict(p['htr'][left:right])
        am=gs.AuxiliaryRidge(float(gcfg['forecast']['conditional_on_stats_ridge_alpha'])).fit(p['htr'][fit],aux_train[fit]); aux[left:right]=am.predict(p['htr'][left:right])
        maximum=float(p['train'].target_end_age_h[fit].max())
        if maximum>origin+1e-7: raise RuntimeError('P2 causal OOF leakage')
        audit.append({'fold':fold,'block':b,'fit_count':int(len(fit)),'max_fit_target_end_h':maximum,'min_prediction_origin_h':origin,'target_completion_pass':True})
    candidate=p['z_oof'].copy(); candidate[:,:,8]=raw; mu=np.maximum(aux[:,:,0],0); sig=np.maximum(aux[:,:,1],0)
    candidate[:,:,0]=(1-raw)*mu; candidate[:,:,1]=np.sqrt(np.maximum(0,(1-raw)*sig**2+raw*(1-raw)*mu**2))
    z,_=gs.project_future(candidate,p['area']); keep=np.isfinite(z).all((1,2)) & np.isfinite(prob).all(1)
    return z,prob,keep,audit

def enrich(point,frame,packages,p2all):
    rows=[]
    for fold,p in packages.items():
        sub=point[point.fold.eq(fold)].copy(); lookup={int(o):i for i,o in enumerate(p['valid'].row_origin_index.astype(int))}
        for r in sub.itertuples(index=False):
            oi=lookup[int(r.origin_row)]; h=int(r.horizon_step)-1; tr=int(r.target_row)
            z=p['zva'][oi,h]; p2=p2all[fold]['z'][oi,h]; prob=p2all[fold]['prob'][oi,h]
            prev=tr-1; same=prev>=0 and frame.temporal_segment_id.iloc[prev]==frame.temporal_segment_id.iloc[tr]
            q=float(z[8]); qprev=float(frame.zero_current_fraction.iloc[prev]) if same else np.nan
            if not same: regime='segment_boundary'
            elif qprev<=EPS and q<=EPS: regime='continuous_nonzero'
            elif qprev<=EPS and q>EPS and q<1-EPS: regime='nonzero_to_mixed'
            elif qprev>EPS and qprev<1-EPS and q<=EPS: regime='mixed_to_nonzero'
            elif qprev>EPS and qprev<1-EPS and q>EPS and q<1-EPS: regime='continuous_mixed'
            else: regime='other'
            d=r.D_mV2
            rows.append({**r._asdict(),'temporal_segment_id':frame.temporal_segment_id.iloc[tr],
              'target_datetime':str(frame.last_datetime.iloc[tr]),'origin_datetime':str(frame.last_datetime.iloc[int(r.origin_row)]),
              'actual_current_A':float(z[0]),'predicted_P1_current_A':float(p['z_ridge'][oi,h,0]),'predicted_P2_current_A':float(p2[0]),
              'actual_q':q,'predicted_P2_q':float(p2[8]),'predicted_P2_mixed_probability':float(prob),
              'regime_transition_exact':regime,'target_age_h_frame':float(frame['bin_start_ophrs2_h'].iloc[tr]),
              'P1_error_mV':float(r.p1_error_mV),'P2_error_mV':float(r.p2_error_mV),'D_mV2':float(d),
              'C_delta_mV':float(r.correction_change_mV)})
    return pd.DataFrame(rows)

def concentration(df,scope):
    pos=df[df.D_mV2>0].sort_values('D_mV2',ascending=False); net=float(df.D_mV2.sum()); pp=float(pos.D_mV2.sum())
    out=[]
    for k in (1,5,10,20):
        s=float(pos.head(k).D_mV2.sum()); out.append({'scope':scope,'rank_top_k':k,'positive_D_sum_mV2':pp,'net_D_sum_mV2':net,'top_k_positive_D_mV2':s,'share_positive_D_pct':100*s/pp if pp else np.nan,'share_net_D_pct':100*s/net if abs(net)>1e-12 else np.nan})
    return pd.DataFrame(out)

def label_amplitude(df,thresholds):
    w=df.copy(); w['current_abs_error_A']=abs(w.predicted_P2_current_A-w.actual_current_A); w['current_error_threshold_A']=w.fold.map(thresholds)
    ts=w.actual_q>EPS; ps=w.predicted_P2_mixed_probability>=.5
    w['condition_group']=np.where(~ps&ts,'S1_pred_nonmixed_true_mixed',np.where(ps&~ts,'S1_pred_mixed_true_nonmixed',np.where(w.current_abs_error_A>w.current_error_threshold_A,'S0_H','S0_L')))
    return w

def group_summary(df,scope,kind):
    rows=[]
    for group,g in df.groupby(kind,dropna=False):
        d=g.D_mV2.to_numpy(float); pos=np.maximum(d,0); neg=np.minimum(d,0)
        rows.append({'scope':scope,'group':group,'sample_count':int(len(g)),'unique_target_count':int(g.target_row.nunique()),'segment_count':int(g.temporal_segment_id.nunique()),'P1_RMSE_mV':rmse(g.P1_error_mV),'P2_RMSE_mV':rmse(g.P2_error_mV),'mean_D_mV2':float(g.D_mV2.mean()),'positive_D_sum_mV2':float(pos.sum()),'negative_D_sum_mV2':float(neg.sum()),'net_D_sum_mV2':float(d.sum()),'mean_C_delta_mV':float(g.C_delta_mV.mean()),'p95_abs_C_delta_mV':float(np.quantile(abs(g.C_delta_mV),.95)),'current_MAE_A':float(g.current_abs_error_A.mean()),'q_MAE':float(abs(g.predicted_P2_q-g.actual_q).mean())})
    return pd.DataFrame(rows)

def events_and_shifts(frame,packages,p2all):
    eventrows=[]; shiftrows=[]
    for fold,p in packages.items():
        origins=p['valid'].row_origin_index.astype(int); seg=frame.temporal_segment_id.to_numpy(); q=frame.zero_current_fraction.to_numpy(float)
        for i,o in enumerate(origins):
            true=q[o:o+25]; pred=np.r_[q[o],p2all[fold]['prob'][i]>=.5]
            same=np.array([seg[o+j-1]==seg[o+j] for j in range(1,25)])
            te=[]; pe=[]
            for h in range(1,25):
                if not same[h-1]: continue
                a,b=true[h-1]>EPS,true[h]>EPS; c,d=bool(pred[h-1]),bool(pred[h])
                if a!=b: te.append((h,'nonmixed_to_mixed' if (not a and b) else 'mixed_to_nonmixed'))
                if c!=d: pe.append((h,'nonmixed_to_mixed' if (not c and d) else 'mixed_to_nonmixed'))
            used=set()
            for h,typ in te:
                cand=[(abs(ph-h),ph,k) for k,(ph,pt) in enumerate(pe) if k not in used and pt==typ and abs(ph-h)<=2]
                if cand:
                    _,ph,k=min(cand); used.add(k); eventrows.append({'fold':fold,'origin_row':int(o),'event_type':typ,'record_type':'true_event','actual_step':h,'predicted_step':ph,'offset_steps':ph-h,'status':'matched'})
                else: eventrows.append({'fold':fold,'origin_row':int(o),'event_type':typ,'record_type':'true_event','actual_step':h,'predicted_step':np.nan,'offset_steps':np.nan,'status':'false_negative'})
            for k,(ph,typ) in enumerate(pe):
                if k not in used: eventrows.append({'fold':fold,'origin_row':int(o),'event_type':typ,'record_type':'predicted_event','actual_step':np.nan,'predicted_step':ph,'offset_steps':np.nan,'status':'false_positive'})
        for shift in range(-2,3):
            vals=[]
            for i,o in enumerate(origins):
                for h in range(24):
                    hp=h+shift
                    if 0<=hp<24:
                        vals.append((p2all[fold]['z'][i,hp,0],p2all[fold]['z'][i,hp,8],p['zva'][i,h,0],p['zva'][i,h,8]))
            v=np.asarray(vals); shiftrows.append({'scope':'all_validation','fold':fold,'shift_steps_predicted_relative_to_actual':shift,'overlap_bins':len(v),'current_MAE_A':float(np.mean(abs(v[:,0]-v[:,2]))),'q_MAE':float(np.mean(abs(v[:,1]-v[:,3]))),'state_mismatch_rate':float(np.mean((v[:,1]>EPS)!=(v[:,3]>EPS)))})
    e=pd.DataFrame(eventrows); s=pd.DataFrame(shiftrows)
    rows=[]
    for (fold,typ),g in e.groupby(['fold','event_type']):
        m=g[g.status.eq('matched')]; fp=int((g.status=='false_positive').sum()); fn=int((g.status=='false_negative').sum()); tp=int(len(m));
        prec=tp/(tp+fp) if tp+fp else np.nan; rec=tp/(tp+fn) if tp+fn else np.nan
        rows.append({'scope':'all_validation','fold':fold,'event_type':typ,'matched':tp,'false_positive':fp,'false_negative':fn,'precision':prec,'recall':rec,'f1':2*prec*rec/(prec+rec) if prec and rec else 0.0,'early_2':int((m.offset_steps==-2).sum()),'early_1':int((m.offset_steps==-1).sum()),'exact':int((m.offset_steps==0).sum()),'late_1':int((m.offset_steps==1).sum()),'late_2':int((m.offset_steps==2).sum())})
    return e,pd.DataFrame(rows),s

def main():
    print('TTD: importing frozen helpers', flush=True)
    n=load('ttd_n28f',NROOT/'scripts/phase_n2_8f_run.py'); gs=load('ttd_g',GROOT/'scripts/phase_g_stop_experiment.py'); ncfg=n.load_config(); gcfg=gs.load_config(); common,frame,windows,firewall=n.load_development(ncfg)
    print('TTD: loaded development frame', flush=True)
    raw=pd.read_csv(GROOT/'outputs/physical_audit/raw_conditional_current_statistics.csv'); frame=frame.merge(raw[['bin_start_ophrs2_h','mu_on_A','sigma_on_A']],on='bin_start_ophrs2_h',how='left',validate='one_to_one')
    if frame[['mu_on_A','sigma_on_A']].isna().any().any(): raise RuntimeError('raw current statistics missing')
    packs={f:n.package_fold(common,frame,windows,ncfg,f) for f in FOLDS}
    p2all={}; thresholds={}; oofa=[]
    for f,p in packs.items():
        print(f'TTD: loading frozen RS P2 conditions for {f}', flush=True)
        saved=np.load(RS/'outputs/frozen_model'/f'P2_Zero_Current_Aware_{f}_predictions.npz')
        if not np.array_equal(saved['row_origin_index'].astype(int),p['valid'].row_origin_index.astype(int)):
            raise RuntimeError(f'{f}: RS P2 origin indices do not match frozen OPR/N2.8F development package')
        z=saved['future_conditions'].astype(float); prob=saved['p_mixed'].astype(float)
        if z.shape != p['zva'].shape or prob.shape != p['zva'].shape[:2]: raise RuntimeError(f'{f}: frozen RS P2 shape mismatch')
        keep=np.isfinite(p['z_oof']).all(axis=(1,2))
        thresholds[f]=float(np.quantile(abs(p['z_oof'][keep,:,0]-p['ztr'][keep,:,0]),.75))
        oofa.append({'fold':f,'threshold_source':'N2.8F_causal_Ridge_OOF_current_abs_error_75th_percentile','oof_rows':int(keep.sum()),'target_completion_purge':'inherited_and_rechecked_in_frozen_N2.8F_package'})
        p2all[f]={'z':z,'prob':prob}
    print('TTD: joining OPR pointwise decomposition', flush=True)
    point=pd.read_csv(OPR/'outputs/error_attribution/p1_p2_error_difference.csv'); allp=enrich(point,frame,packs,p2all)
    base=pd.read_csv(OPR/'outputs/reproduction/baseline_reproduction_summary.csv'); write(base,'outputs/reproduction/baseline_reproduction.csv')
    f1=allp[(allp.fold=='fold_1')&(allp.horizon_step==24)].copy(); f3=allp[(allp.fold=='fold_3')&(allp.regime_transition_exact=='continuous_nonzero')].copy()
    for w in (f1,f3): w.sort_values(['target_row','origin_row'],inplace=True)
    write(f1,'outputs/fold1_12h/fold1_12h_pointwise.csv'); write(f1.loc[f1.P2_error_mV.abs().sort_values(ascending=False).head(20).index],'outputs/fold1_12h/fold1_12h_top_absolute_error.csv')
    topd=f1[f1.D_mV2>0].drop_duplicates('target_row').nlargest(20,'D_mV2'); write(topd,'outputs/fold1_12h/fold1_12h_top_positive_D.csv'); write(concentration(f1,'fold1_h12'),'outputs/fold1_12h/fold1_12h_loss_concentration.csv')
    write(f3,'outputs/fold3_nonzero/fold3_nonzero_pointwise.csv')
    seg=[]
    for sid,g in f3.groupby('temporal_segment_id'):
        d=g.D_mV2; seg.append({'temporal_segment_id':sid,'start_time':g.target_datetime.min(),'end_time':g.target_datetime.max(),'duration_h':(g.target_age_h_frame.max()-g.target_age_h_frame.min()),'unique_target_count':g.target_row.nunique(),'prediction_point_count':len(g),'mean_D_mV2':d.mean(),'positive_D_sum_mV2':np.maximum(d,0).sum(),'negative_D_sum_mV2':np.minimum(d,0).sum(),'net_D_sum_mV2':d.sum(),'P1_RMSE_mV':rmse(g.P1_error_mV),'P2_RMSE_mV':rmse(g.P2_error_mV),'current_MAE_A':abs(g.predicted_P2_current_A-g.actual_current_A).mean(),'q_MAE':abs(g.predicted_P2_q-g.actual_q).mean(),'C_delta_P95_mV':np.quantile(abs(g.C_delta_mV),.95),'C_delta_P99_mV':np.quantile(abs(g.C_delta_mV),.99),'C_delta_max_mV':abs(g.C_delta_mV).max(),'affected_horizons':','.join(map(str,sorted(g.horizon_step.unique())))})
    seg=pd.DataFrame(seg).sort_values('positive_D_sum_mV2',ascending=False); write(seg,'outputs/fold3_nonzero/fold3_independent_segments.csv'); write(seg.assign(rank_positive_D=np.arange(1,len(seg)+1)),'outputs/fold3_nonzero/fold3_segment_contribution.csv')
    print('TTD: calculating transition and shift diagnostics', flush=True)
    ev,ts,sh=events_and_shifts(frame,packs,p2all); write(ev,'outputs/transitions/transition_event_matching.csv'); write(ts,'outputs/transitions/transition_timing_summary.csv'); write(sh,'outputs/transitions/trajectory_shift_diagnostics.csv')
    f1a=label_amplitude(f1,thresholds); f3a=label_amplitude(f3,thresholds); amp=pd.concat([group_summary(f1a,'fold1_h12','condition_group'),group_summary(f3a,'fold3_continuous_nonzero','condition_group')]); write(amp,'outputs/amplitude/condition_amplitude_groups.csv')
    film=pd.concat([group_summary(f1a,'fold1_h12','condition_group'),group_summary(f3a,'fold3_continuous_nonzero','condition_group')]); film.insert(2,'definition','C_delta_mV = C_P2 - C_P1; S0 groups have state agreement'); write(film,'outputs/film_response/film_response_groups.csv')
    write(pd.DataFrame([{'status':'NOT_RUN','reason':'No single supported mechanism met the pre-registered cross-object dominance requirement; no post-hoc FiLM sensitivity radius was introduced.','test_accessed':False}]),'outputs/film_response/film_sensitivity_audit.csv')
    align=[]
    sample=pd.concat([topd,f3.nlargest(50,'D_mV2')]).drop_duplicates(['fold','origin_row','target_row','horizon_step'])
    for r in sample.itertuples(index=False):
        o=int(r.origin_row);t=int(r.target_row);h=int(r.horizon_step); expected=o+h
        align.append({'fold':r.fold,'origin_row':o,'target_row':t,'horizon_step':h,'expected_target_row':expected,'condition_target_row':expected,'voltage_target_row':expected,'condition_alignment_pass':expected==t,'voltage_alignment_pass':expected==t,'origin_before_target':o<t,'same_temporal_segment':frame.temporal_segment_id.iloc[o]==frame.temporal_segment_id.iloc[t],'origin_datetime':str(frame.last_datetime.iloc[o]),'target_datetime':str(frame.last_datetime.iloc[t]),'age_delta_h':float(frame.bin_start_ophrs2_h.iloc[t]-frame.bin_start_ophrs2_h.iloc[o]),'alignment_status':'PASS' if expected==t and o<t else 'FAIL'})
    alignment=pd.DataFrame(align); write(alignment,'outputs/time_alignment/time_alignment_audit.csv')
    mech=pd.concat([amp.assign(analysis='amplitude'),film.assign(analysis='film')],ignore_index=True); write(mech,'outputs/mechanism_contribution_summary.csv')
    print('TTD: writing audit outputs', flush=True)
    jdump({**firewall,'phase':'TTD','test_predictions_constructed':False,'test_metrics_computed':False,'test_performance_accessed':False,'p2_oof_blocks':oofa,'baseline_source':'Phase_OPR pointwise frozen outputs'},'outputs/audit/test_firewall_audit.json')
    jdump({'start_commit':'8a30f68346088b82511a3f71c1de9f59b316464a','decision':'INCONCLUSIVE','mechanism_A_transition_timing':'INCONCLUSIVE','mechanism_B_condition_magnitude':'INCONCLUSIVE','mechanism_C_film_excessive_response':'INCONCLUSIVE','mechanism_D_alignment_error':'NOT_SUPPORTED','new_model_experiment_authorized':False,'test_accessed':False,'condition_amplitude_thresholds_A':thresholds,'threshold_definition':'fold-specific 75th percentile of frozen N2.8F causal target-completion-purged P0 Ridge OOF absolute current error; no voltage used'},'PHASE_TTD_DECISION.yaml')
    text('# Phase TTD results\n\nAutomated report content is finalized after the numerical audit in `finalize_ttd_reports.py`. This phase contains no neural-network training and no Test access.','PHASE_TTD_RESULTS.md')
    print('TTD: completed', flush=True)

if __name__=='__main__': main()
