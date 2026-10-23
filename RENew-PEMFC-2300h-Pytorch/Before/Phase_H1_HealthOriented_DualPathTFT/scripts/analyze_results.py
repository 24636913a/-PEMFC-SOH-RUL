"""Allregistered results evaluated together; no selection from outerValidation."""
from common import *

def metric(d):
 e=(d.predicted_voltage_V-d.true_voltage_V).to_numpy()*1000;ae=abs(e)
 return dict(n=len(e),MAE_mV=float(ae.mean()),RMSE_mV=float(np.sqrt(np.mean(e**2))),P95_mV=float(np.quantile(ae,.95)),MAX_mV=float(ae.max()),
  MAE_SOH_pp=float(ae.mean()/(10*VREF)),RMSE_SOH_pp=float(np.sqrt(np.mean(e**2))/(10*VREF)),
  P95_SOH_pp=float(np.quantile(ae,.95)/(10*VREF)),MAX_SOH_pp=float(ae.max()/(10*VREF)))
def main():
 assert (P/'outputs/ALL_GPU_RUNS_COMPLETE.json').exists();assert not (P/'outputs/FINAL_DECISION.json').exists(),'Analysis complete'
 cfg=load_cfg();paths=sorted((P/'outputs/predictions').glob('fold_*_F*_seed*.csv'));assert len(paths)==27
 frames=[rd(q) for q in paths];baseline=rd(P/'outputs/BASELINE_PREDICTIONS.csv');pred=pd.concat(frames+[baseline],ignore_index=True)
 assert len(pred)==41*11 # 9 neural modelseed results +2baselines on same41events
 metrics=[];lag=[];ranges=[];training=[]
 samples=rd(P/'data_contracts/USED_SAMPLES.csv')
 for (fold,name,seed),g in pred.groupby(['fold','model','seed']):
  metrics.append(dict(fold=fold,model=name,seed=int(seed),**metric(g)))
  actual=1000*(g.true_voltage_V-g.origin_voltage_V);change=1000*(g.predicted_voltage_V-g.origin_voltage_V)
  lag.append(dict(fold=fold,model=name,seed=int(seed),n=len(g),actual_change_abs_median_mV=float(abs(actual).median()),predicted_change_abs_median_mV=float(abs(change).median()),
   direction_agreement=float((np.sign(actual)==np.sign(change)).mean()),actual_recoveries=int(actual.gt(0).sum()),
   prediction_closer_to_origin_fraction=float((abs(g.predicted_voltage_V-g.origin_voltage_V)<abs(g.predicted_voltage_V-g.true_voltage_V)).mean()),
   MAE_to_latest_origin_mV=float(1000*abs(g.predicted_voltage_V-g.origin_voltage_V).mean()),MAE_to_future_target_mV=metric(g)['MAE_mV'],
   lag_role='Descriptive persistence-like/change diagnostic, not a fitted calendar delay'))
  tr=samples.loc[samples.fold.eq(fold)&samples.role.eq('train')];low=tr.true_voltage_V.min();high=tr.true_voltage_V.max()
  ranges.append(dict(fold=fold,model=name,seed=int(seed),pred_min_V=g.predicted_voltage_V.min(),pred_max_V=g.predicted_voltage_V.max(),Train_min_V=low,Train_max_V=high,
   predicted_outside_Train_range=int((g.predicted_voltage_V.lt(low)|g.predicted_voltage_V.gt(high)).sum()),nonpositive_predictions=int(g.predicted_voltage_V.le(0).sum()),
   proxy_above100_count=int(g.predicted_SOH_V_percent.gt(100).sum()),clipped=False,monotonic_forced=False))
 runmetrics=pd.DataFrame(metrics);cs('outputs/ALL_POINT_PREDICTIONS.csv',pred);cs('outputs/FOLD_SEED_METRICS.csv',runmetrics);cs('outputs/LAG_AND_CHANGE_DIAGNOSTICS.csv',pd.DataFrame(lag));cs('outputs/RANGE_AND_RECOVERY_CHECKS.csv',pd.DataFrame(ranges))
 cols=['MAE_mV','RMSE_mV','P95_mV','MAX_mV','MAE_SOH_pp','RMSE_SOH_pp','P95_SOH_pp','MAX_SOH_pp']
 foldmean=runmetrics.groupby(['fold','model'])[cols].mean().reset_index();cs('outputs/FOLD_THREE_SEED_MEANS.csv',foldmean)
 equal=foldmean.groupby('model')[cols].mean().reset_index();cs('outputs/PRIMARY_EQUAL_FOLD_METRICS.csv',equal)
 seedmean=runmetrics.loc[runmetrics.seed.ge(0)].groupby(['model','seed'])[cols].mean().reset_index();cs('outputs/SEED_EQUAL_FOLD_METRICS.csv',seedmean)
 stability=seedmean.groupby('model')[cols].agg(['mean','std']).reset_index();stability.columns=['model']+[a+'_'+b for a,b in list(stability.columns)[1:]];cs('outputs/THREE_SEED_STABILITY.csv',stability)
 paired=[]
 for fold,g in runmetrics.loc[runmetrics.seed.ge(0)].groupby('fold'):
  for seed,z in g.groupby('seed'):
   index=z.set_index('model');base=runmetrics.loc[runmetrics.fold.eq(fold)].set_index('model')
   for compare in ['H-F0','H-F1','Persistence','Ridge96_control']:
    v=float(index.loc[compare,'MAE_mV']) if compare.startswith('H-') else float(base.loc[compare,'MAE_mV'])
    f2=float(index.loc['H-F2','MAE_mV']);paired.append(dict(fold=fold,seed=int(seed),comparison=compare,F2_MAE_mV=f2,comparison_MAE_mV=v,F2_improvement_percent=100*(v-f2)/v,F2_wins=f2<v))
 cs('outputs/PAIRED_COMPARISONS.csv',pd.DataFrame(paired))
 for folder in sorted((P/'outputs/training_logs').glob('fold_*')):
  if not (folder/'COMPLETION.json').exists():continue
  m=json.loads((folder/'COMPLETION.json').read_text());row=dict(run_id=m['run_id'],inner_optimizer_epochs=m['inner']['optimizer_epochs'],selected_epoch=m['inner']['selected_epoch'],
   final_refit_epochs=m['final']['optimizer_epochs'],inner_RMSE_standardized=m['inner']['best_inner_standardized_RMSE'],total_fit_seconds=m['total_training_seconds'],
   outer_used_for_selection=m['outer_used_for_selection'],final_checkpoint_SHA256=m['final']['checkpoint_sha256'],replay_available=m['replay_model_available_calendar'],earliest_origin=m['earliest_Validation_origin_calendar'])
  lines=[json.loads(z) for z in (folder/'final_epochs.jsonl').read_text().splitlines()]
  row['final_last_train_standardized_MSE']=lines[-1]['train_standardized_MSE'];training.append(row)
 cs('outputs/TRAINING_AND_OVERFIT_DIAGNOSTIC.csv',pd.DataFrame(training))
 ix=equal.set_index('model');fm=foldmean.set_index(['fold','model']);p=float(ix.loc['Persistence','MAE_mV']);f2=float(ix.loc['H-F2','MAE_mV']);gain=(p-f2)/p
 wins={}
 for compare in ['H-F0','H-F1','Persistence']:
  wins[compare]=sum(fm.loc[(fold,'H-F2'),'MAE_mV']<fm.loc[(fold,compare),'MAE_mV'] for fold in ['fold_1','fold_2','fold_3'])
 seedwins=int(seedmean.loc[seedmean.model.eq('H-F2'),'MAE_mV'].lt(p).sum())
 tail=all(fm.loc[(fold,'H-F2'),c]<=1.1*fm.loc[(fold,'Persistence'),c] for fold in ['fold_1','fold_2','fold_3'] for c in ['RMSE_mV','P95_mV','MAX_mV'])
 supported=(gain>=.05 and f2<ix.loc['H-F0','MAE_mV'] and f2<ix.loc['H-F1','MAE_mV'] and all(v>=2 for v in wins.values()) and seedwins>=2 and tail)
 status='H1_HEALTH_TFT_SUPPORTED' if supported else 'H1_HEALTH_TFT_NO_GAIN' if gain<=0 else 'H1_HEALTH_TFT_EXPLORATORY'
 js('outputs/FINAL_DECISION.json',dict(status=status,threefold_complete=True,neural_experiments=27,GPU_optimizer_fit_passes=54,CPU_Ridge_fit_passes=15,
  F2_MAE_mV=f2,Persistence_MAE_mV=p,F2_improvement_vs_Persistence_percent=100*gain,
  F2_better_F0_equal_fold_MAE=bool(f2<ix.loc['H-F0','MAE_mV']),F2_better_F1_equal_fold_MAE=bool(f2<ix.loc['H-F1','MAE_mV']),
  fold_wins={k:int(v) for k,v in wins.items()},seed_wins_vs_Persistence=seedwins,tail_gate=bool(tail),
  F2_MAE_proxy_SOH_pp=float(ix.loc['H-F2','MAE_SOH_pp']),reference_standard_SOH_qualified=False,
  historical_online_deployment_claimed=False,no_Test=True,rule_frozen_before_results=True))
 print(equal.to_string(index=False));print('DECISION',status,flush=True)
if __name__=='__main__':main()
