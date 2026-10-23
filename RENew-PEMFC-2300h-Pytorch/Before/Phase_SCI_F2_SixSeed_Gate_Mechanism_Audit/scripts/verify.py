"""Independent scalar metric calculator and independent forward-hook replay."""
import math
from collections import defaultdict
from common import *
def quantile(values,q):
    z=sorted(values);pos=(len(z)-1)*q;lo=int(pos);hi=min(lo+1,len(z)-1)
    return z[lo]+(z[hi]-z[lo])*(pos-lo)
def independent(f):
    y=list(f.y_true_delta_V); p=list(f.y_pred_delta_V); n=len(y)
    err=[(b-a)*1000 for a,b in zip(y,p)];abs_e=list(map(abs,err));mu=math.fsum(y)/n
    e2=math.fsum(e*e for e in err);den=math.fsum(((v-mu)*1000)**2 for v in y)
    groups=defaultdict(list);end=[]
    for r,e in zip(f.itertuples(),err):
        groups[r.target_row].append(e*e)
        if r.horizon_step==24:end.append(e*e)
    return dict(RMSE_mV=math.sqrt(e2/n),MAE_mV=math.fsum(abs_e)/n,R2=1-e2/den,
        endpoint_RMSE_mV=math.sqrt(math.fsum(end)/len(end)),
        target_time_equal_RMSE_mV=math.sqrt(math.fsum(math.fsum(v)/len(v) for v in groups.values())/len(groups)),
        P95_mV=quantile(abs_e,.95),P99_mV=quantile(abs_e,.99),max_abs_error_mV=max(abs_e))
def main():
    c=cfg();stored=rd(P/'outputs/METRICS.csv');hh=rd(P/'outputs/HORIZON_METRICS.csv');diffs=[];replay=[]
    d=np.load(R2/'outputs/audits/FROZEN_H_DATA.npz');perms=rd(P/'configs/permutations.csv');scale=read(R2/'outputs/audits/SCALER.json')['target_scale_V']
    import torch
    from run_inference import lock_training
    lock_training();h=get_h();h.configure_normal(c)
    for row in c['models']:
        s=row['seed'];baseline=rd(REPO/row['predictions']);val_g=rd(P/f'outputs/validation_gates_seed_{s}.csv').g.to_numpy()
        tg=rd(P/f'outputs/train_gates_seed_{s}.csv');assert len(tg)==1380
        assert np.array_equal(tg.origin_row.to_numpy(),d['train_origins'])
        mean=math.fsum(tg.g)/1380;assert abs(mean-read(P/f'outputs/audits/seed_{s}.json')['train_gate_mean'])<1e-15
        ck=torch.load(REPO/row['checkpoint'],map_location='cpu',weights_only=True)
        m=h.model('F2',c['hyperparameters'],torch.device('cuda'));m.load_state_dict(ck['state_dict']);m.eval();state=h.tree_hash(m.state_dict())
        assert np.array_equal((h.e.predict(m,d['vx'],32,torch.device('cuda'))*scale).ravel().astype('float64'),baseline.y_pred_delta_V.to_numpy())
        for mr in stored[stored.seed==s].itertuples():
            mode=mr.mode;f=rd(P/f'outputs/predictions/seed_{s}_{mode}.csv');assert len(f)==4920
            for col in ['origin_row','target_row','horizon_step','y_true_delta_V']:assert np.array_equal(f[col],baseline[col]),('IDENTITY',s,mode,col)
            assert np.array_equal(f.origin_age_h,np.repeat(d['val_origin_age_h'],24))
            assert np.array_equal(f.target_age_h,d['val_target_age_h'].ravel())
            if mode=='B0': assert np.array_equal(f.y_pred_delta_V,baseline.y_pred_delta_V)
            q=independent(f)
            for k,v in q.items():diffs.append(dict(seed=s,mode=mode,metric=k,absolute_difference=abs(v-getattr(mr,k))));assert abs(v-getattr(mr,k))<1e-10,(s,mode,k)
            for step,z in f.groupby('horizon_step'):
                ee=[(b-a)*1000 for a,b in zip(z.y_true_delta_V,z.y_pred_delta_V)]
                val=math.sqrt(math.fsum(t*t for t in ee)/len(ee));old=hh[(hh.seed==s)&(hh['mode']==mode)&(hh.horizon_step==step)].RMSE_mV.iloc[0]
                assert abs(val-old)<1e-10
            assignment=None
            if mode=='B1':assignment=np.full(205,.5,dtype='float32')
            if mode=='B2':assignment=np.full(205,mean,dtype='float32')
            if mode.startswith('C_'):
                pp=perms[perms.permutation_seed==int(mode[2:])].source_window.to_numpy()
                assert np.array_equal(np.sort(pp),np.arange(205)) and np.all(pp!=np.arange(205));assignment=val_g[pp].astype('float32')
                assert np.array_equal(np.sort(assignment.astype('float64')),np.sort(val_g))
            handle=None;cursor=[0]
            if assignment is not None:
                assert np.array_equal(rd(P/f'outputs/used_gates_seed_{s}_{mode}.csv').g_applied,assignment.astype('float64'))
                def replace(module,args,out):
                    a=cursor[0];cursor[0]+=len(out)
                    return torch.as_tensor(assignment[a:cursor[0]],device=out.device,dtype=out.dtype).reshape(out.shape)
                handle=m.regime_gate.register_forward_hook(replace)
            try:pred=h.e.predict(m,d['vx'],32,torch.device('cuda'))*scale
            finally:
                if handle is not None:handle.remove()
            assert np.array_equal(pred.ravel().astype('float64'),f.y_pred_delta_V.to_numpy()),('INDEPENDENT_REPLAY',s,mode)
            assert h.tree_hash(m.state_dict())==state and not m.regime_gate._forward_hooks
            assert all(t.grad is None for t in m.parameters())
            replay.append(dict(seed=s,mode=mode,n=4920,predictions_exact=True,state_unchanged=True))
        del m,ck;torch.cuda.empty_cache();print('INDEPENDENT_SEED_VERIFIED',s,flush=True)
    j=rd(J/'outputs/fixed_gate_diagnostic/J_FIXED_GATE_COMPARISON.csv');old=j[j.seed==42].iloc[0]
    now=stored[(stored.seed==42)&(stored['mode']=='B1')].RMSE_mV.iloc[0]
    assert abs(now-old.fixed0p5_RMSE_mV)<1e-12,'J_SEED42_FIXED_PARITY'
    history=rd(P/'outputs/audits/HISTORY_BEFORE.csv')
    for r in history.itertuples():assert sha(REPO/r.path)==r.sha256,('HISTORICAL_CHANGE',r.path)
    dirty=subprocess.check_output(['git','diff','--name-only',BASE],cwd=REPO,text=True,encoding='utf-8').splitlines()
    assert all(P.name in f for f in dirty),('OUTSIDE_PHASE_CHANGE',dirty)
    cs('outputs/audits/INDEPENDENT_METRIC_DIFFERENCES.csv',diffs);cs('outputs/audits/INDEPENDENT_GPU_REPLAY.csv',replay)
    js('outputs/audits/INDEPENDENT_VERIFICATION.json',dict(pass_=True,metric_checks=len(diffs),maximum_metric_difference=max(r['absolute_difference'] for r in diffs),
        GPU_replay_conditions=len(replay),GPU_replay_points=sum(r['n'] for r in replay),history_files_unchanged=len(history),historical_checkpoints_unchanged=int(history.checkpoint.sum()),
        J_seed42_g_and_fixed05_reproduced=True,Test=False,Assessment=False,training=False))
    print('INDEPENDENT_VERIFICATION_PASS',len(replay)*4920,flush=True)
if __name__=='__main__':main()
