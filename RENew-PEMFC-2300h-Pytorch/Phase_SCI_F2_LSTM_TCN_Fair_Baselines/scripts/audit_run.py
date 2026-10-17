"""Independent scalar metrics, complete best-checkpoint reload and chronology audit."""
from baseline_common import *
import argparse

def scalar(f):
    y=list(map(float,f.y_true_delta_V));e=[(float(p)-t)*1000 for p,t in zip(f.y_pred_delta_V,y)];a=sorted(map(abs,e));n=len(e);mean=math.fsum(y)/n
    def q(p):
        x=(n-1)*p;i=int(x);return a[i]+(a[min(i+1,n-1)]-a[i])*(x-i)
    groups={}
    for target,error in zip(f.target_row,e):groups.setdefault(int(target),[]).append(error*error)
    endpoint=[v*v for v,horizon in zip(e,f.horizon_step) if horizon==24]
    return dict(RMSE_mV=math.sqrt(math.fsum(v*v for v in e)/n),MAE_mV=math.fsum(map(abs,e))/n,R2=1-(math.fsum(v*v for v in e)/1e6)/math.fsum((v-mean)**2 for v in y),endpoint_RMSE_mV=math.sqrt(math.fsum(endpoint)/len(endpoint)) if endpoint else None,target_time_equal_RMSE_mV=math.sqrt(math.fsum(math.fsum(v)/len(v) for v in groups.values())/len(groups)),P95_mV=q(.95),P99_mV=q(.99),max_abs_error_mV=a[-1])

def audit(name,seed):
    c=cfg();h.configure_normal(c);run=f'{name}_seed{seed}';r=read(P/f'outputs/metrics/{run}.json');assert r['status']=='COMPLETED'
    f=rd(P/r['predictions']);assert sha(P/r['predictions'])==r['prediction_sha256'];assert_keys(f)
    independent=scalar(f)
    diffs={k:abs(v-r[k]) for k,v in independent.items()};assert max(diffs.values())<1e-10,diffs
    ck=torch.load(P/r['checkpoint'],map_location='cpu',weights_only=True);assert sha(P/r['checkpoint'])==r['checkpoint_sha256']
    assert ck['input_fingerprints']==read(P/'outputs/audits/DATA_FINGERPRINTS.json') and ck['protocol_sha256']==r['protocol_sha256']
    assert ck['model']==name and ck['seed']==seed and ck['architecture']==ARCH[name]
    m=model(name,'cuda');m.load_state_dict(ck['state_dict'],strict=True);d=data();z=predict(m,d['vx'],'cuda')
    assert np.array_equal(z,ck['full_validation_prediction'].numpy()),'RELOAD_STANDARDIZED_MISMATCH'
    assert np.array_equal((z*ck['target_scale_V']).ravel().astype('float64'),f.y_pred_delta_V.to_numpy()),'CACHE_WEIGHT_MISMATCH'
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==r['parameter_count']
    log=rd(P/f'outputs/training_logs/{run}/epochs.csv');assert np.array_equal(log.epoch,np.arange(1,len(log)+1));assert len(log)==r['stop_epoch']<=450
    best=math.inf;best_epoch=stale=0
    for row in log.itertuples():
        if row.validation_standardized_RMSE<best-1e-12:best,best_epoch,stale=row.validation_standardized_RMSE,int(row.epoch),0
        else:stale+=1
        assert row.best_epoch==best_epoch and row.stale_epochs==stale
        assert stale<40 or row.epoch==len(log)
    assert best_epoch==r['best_epoch']==ck['best_epoch'];assert len(log)==450 or stale>=40
    horizon=[]
    for step,g in f.groupby('horizon_step'):
        s=scalar(g);expected=next(v for v in r['horizon_metrics'] if v['step']==step)
        assert abs(s['RMSE_mV']-expected['RMSE_mV'])<1e-10 and abs(s['MAE_mV']-expected['MAE_mV'])<1e-10
        horizon.append(dict(step=int(step),RMSE_mV=s['RMSE_mV'],n=len(g)))
    result=dict(status='PASS',model=name,seed=seed,independent_metrics=independent,max_metric_difference=max(diffs.values()),checkpoint_all_validation_exact=True,CSV_checkpoint_exact=True,early_stopping_reconstructed=True,all24horizons_checked=True,Test=False,Assessment=False)
    js(f'outputs/audits/{run}_AUDIT.json',result);print('RUN_AUDIT_PASS',run,max(diffs.values()),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['LSTM','TCN']);p.add_argument('--seed',type=int,required=True);a=p.parse_args();audit(a.model,a.seed)
