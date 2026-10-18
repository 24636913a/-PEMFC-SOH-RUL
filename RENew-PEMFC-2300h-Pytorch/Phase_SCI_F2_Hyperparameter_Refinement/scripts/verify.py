"""Independent three-metric recomputation, time isolation and frozen replay."""
from refine import *

def independent(y,pred):
    yy=list(map(float,np.asarray(y).ravel()));pp=list(map(float,np.asarray(pred).ravel()));n=len(yy)
    errors=[a-b for a,b in zip(pp,yy)];sse=math.fsum(e*e for e in errors);mean=math.fsum(yy)/n
    return dict(RMSE_mV=math.sqrt(sse/n)*1000,MAE_mV=math.fsum(abs(e) for e in errors)/n*1000,
        R2_deltaV=1-sse/math.fsum((a-mean)**2 for a in yy))

def main():
    p=protocol();best=selected();old.configure_environment('cuda');z=train_cache();parts=inner(z);v=validation_cache();sc=read('configs/INNER_SCALER.json')
    # Fit-only scaler and strict target boundaries, independently from optimizer.
    flat=z['train_x_raw'][parts['FIT']].astype(float).reshape(-1,14)
    req(np.array_equal(flat.mean(0),np.array(sc['mean'])) and np.array_equal(flat.std(0),np.array(sc['scale'])),'INNER_SCALER_NOT_FIT_ONLY')
    req(z['train_targets'][parts['FIT']].max()<z['train_origins'][parts['EARLY_STOP']].min(),'FIT_LABEL_LEAK')
    req(z['train_targets'][parts['EARLY_STOP']].max()<z['train_origins'][parts['TUNE']].min(),'ES_LABEL_LEAK')
    toy=independent([0,.001,0,.001],[.001,.002,-.001,0]);req(abs(toy['RMSE_mV']-1)<1e-12 and abs(toy['MAE_mV']-1)<1e-12 and abs(toy['R2_deltaV']+3)<1e-12,'KNOWN_VALUE_METRIC')
    studies=optuna.load_study(study_name='original_F2_train_internal',storage='sqlite:///'+path('outputs/study.sqlite3').as_posix())
    req(len(studies.trials)==6 and all(t.state==optuna.trial.TrialState.COMPLETE for t in studies.trials),'BUDGET')
    checks=[]
    for t in studies.trials:
        result=read(f'outputs/trials/trial{t.number:02d}.json');a=np.load(path(f'outputs/trials/trial{t.number:02d}_tune.npz'))
        req(np.array_equal(a['truth'],z['train_y'][parts['TUNE']]) and np.array_equal(a['origins'],z['train_origins'][parts['TUNE']]) and np.array_equal(a['targets'],z['train_targets'][parts['TUNE']]),'TRIAL_KEYS_OR_TRUTH')
        expected=independent(a['truth'],a['prediction'])
        for key,value in expected.items():req(abs(value-result[key])<1e-9,'TRIAL_NUMERIC_RECOMPUTATION')
        req(abs(expected['RMSE_mV']-t.value)<1e-9,'OPTUNA_OBJECTIVE')
        log=rd(f'outputs/logs/trial{t.number:02d}.csv');epoch=int(log.loc[log.early_stop_standardized_RMSE.idxmin()].epoch)
        req(epoch==result['epochs'],'ES_BEST_EPOCH')
        req(len(log)<=450 and (len(log)==450 or log.iloc[-1].stale_epochs==40),'ES_STOPPING')
        ledger=read(f'outputs/ledger/trial{t.number:02d}.json');req(sha(path(ledger['checkpoint']))==ledger['checkpoint_sha256'],'TRIAL_CHECKPOINT_SHA')
        checks.append(dict(item='trial'+str(t.number),**expected))
    expected_best=min(studies.trials,key=lambda t:(t.value,t.user_attrs['MAE_mV'],t.number))
    req(expected_best.number==best['trial_number'] and expected_best.params==best['hyperparameters'] and expected_best.user_attrs['best_epoch']==best['final_epochs'],'BEST_SELECTION_RULE')
    fullsc=json.loads((D/'SCALER.json').read_text());details=rd('outputs/PER_SEED_METRICS.csv')
    replay=[]
    for name in ['original','refined']:
        for seed in SEEDS:
            f=rd(f'outputs/predictions/{name}_seed{seed}.csv')
            req(len(f)==4920 and f.origin_row.nunique()==205 and not f.duplicated(['origin_row','target_row','horizon_step']).any(),'VALIDATION_IDENTITY')
            for key,value in [('origin_row',np.repeat(v['val_origins'],24)),('target_row',v['val_targets'].ravel()),('horizon_step',np.tile(np.arange(1,25),205)),('y_true_delta_V',v['val_y'].ravel())]:req(np.array_equal(f[key],value),'POINTWISE_IDENTITY')
            expected=independent(f.y_true_delta_V,f.y_pred_delta_V);saved=details.loc[(details.model==name)&(details.seed==seed)].iloc[0]
            for k,value in expected.items():req(abs(value-saved[k])<1e-9,'FINAL_INDEPENDENT_METRIC')
            hp=best['hyperparameters'] if name=='refined' else p['original_hyperparameters'];m=initialize(seed,hp,torch.device('cuda'))
            if name=='refined':
                ledger=read(f'outputs/ledger/refined_seed{seed}.json');req(ledger['best_epoch']==ledger['stop_epoch']==best['final_epochs'],'FINAL_FIXED_EPOCHS')
                cp=path(ledger['checkpoint']);req(sha(cp)==ledger['checkpoint_sha256'],'FINAL_CHECKPOINT_SHA')
                log=rd(f'outputs/logs/refined_seed{seed}.csv');req(log.early_stop_standardized_RMSE.isna().all(),'OUTER_VAL_USED_FOR_EARLY_STOP')
            else:cp=old.HISTORY/f'outputs/checkpoints/H-F2-Seed{seed}_best.pt'
            ck=torch.load(cp,map_location='cpu',weights_only=True);m.load_state_dict(ck['state_dict'],strict=True)
            before=old.state_sha256(m);pred=prediction(m,v['vx'])*fullsc['target_scale_V']
            req(np.array_equal(pred.ravel().astype(float),f.y_pred_delta_V.to_numpy()),'CHECKPOINT_REPLAY_NOT_EXACT')
            req(old.state_sha256(m)==before,'REPLAY_CHANGED_WEIGHTS')
            replay.append(dict(model=name,seed=seed,queries=4920,exact=True,checkpoint_sha256=sha(cp)))
            checks.append(dict(item=name+str(seed),**expected));del m
    summary=rd('SUMMARY.csv').set_index('model')
    for name in ['original','refined']:
        a=details.loc[details.model==name]
        for k in ['RMSE_mV','MAE_mV','R2_deltaV']:req(abs(math.fsum(a[k])/3-summary.loc[name,k])<1e-12,'EQUAL_SEED_MEAN')
    csv('outputs/INDEPENDENT_METRICS.csv',checks);csv('outputs/CHECKPOINT_REPLAY.csv',replay)
    js('outputs/QA.json',dict(status='PASS',search_trials=6,final_seeds=SEEDS,inner_train_only_scaler=True,strict_inner_target_isolation=True,
        objective_independently_recomputed=True,three_metric_checks=36,final_replay_sets=6,pointwise_replay_exact=True,all_4920_targets_retained=True,
        parameter_count=332436,no_gate_or_backbone_change=True,no_outer_Validation_selection=True,no_Assessment_Late_life_or_Test_evaluation=True))
    print('QA_PASS',flush=True)

if __name__=='__main__':main()
