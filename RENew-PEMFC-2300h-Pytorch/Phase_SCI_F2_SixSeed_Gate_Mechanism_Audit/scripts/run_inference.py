"""Forward-only reversible gate interventions. No training entry point is called."""
from contextlib import contextmanager
import time, traceback
from common import *
import torch

def forbid(*args,**kwargs): raise RuntimeError('TRAINING_OR_AUTOGRAD_FORBIDDEN')
def lock_training():
    torch.set_grad_enabled(False)
    torch.Tensor.backward=forbid; torch.autograd.backward=forbid; torch.autograd.grad=forbid
    for cls in [torch.optim.Optimizer,torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]:cls.step=forbid

@contextmanager
def gate_hook(model,assigned=None):
    collected=[]; contexts=[]; offset=[0]
    def hook(module,args,out):
        assert not torch.is_grad_enabled() and not out.requires_grad
        n=len(out); contexts.append(args[0].detach().cpu().numpy().copy()); collected.append(out.detach().cpu().numpy().ravel().copy())
        lo=offset[0];offset[0]+=n
        if assigned is not None:
            a=np.asarray(assigned[lo:lo+n],dtype=np.float32);assert len(a)==n
            return torch.from_numpy(a).to(out.device).view_as(out)
        return None
    handle=model.regime_gate.register_forward_hook(hook)
    try: yield collected,contexts,offset
    finally:handle.remove()

def main():
    c=cfg();lock_training();h=get_h();env=h.configure_normal(c)
    js('outputs/audits/RUNTIME.json',dict(environment=env,started_utc=pd.Timestamp.now(tz='UTC').isoformat(),training_guards=True))
    d=np.load(R2/'outputs/audits/FROZEN_H_DATA.npz');sc=read(R2/'outputs/audits/SCALER.json');scale=sc['target_scale_V']
    fp=read(R2/'outputs/audits/DATA_FINGERPRINTS.json'); perms=rd(P/'configs/permutations.csv')
    audit=[];started=time.time()
    for row in c['models']:
        seed=row['seed'];marker=P/f'outputs/audits/seed_{seed}.json'
        if marker.exists():
            assert read(marker)['pass'];print('ALREADY_COMPLETED',seed,flush=True);continue
        try:
            ck=torch.load(REPO/row['checkpoint'],map_location='cpu',weights_only=True)
            assert ck['seed']==seed and ck['model']=='F2' and ck['input_fingerprints']==fp
            assert ck['protocol_sha256']==row['protocol_sha256'] and ck['target_scale_V']==scale
            m=h.model('F2',c['hyperparameters'],torch.device('cuda'));m.load_state_dict(ck['state_dict']);m.eval()
            state=h.tree_hash(m.state_dict()); assert all(p.grad is None for p in m.parameters())
            old=rd(REPO/row['predictions'])
            assert len(old)==4920 and np.array_equal(old.origin_row.to_numpy(),np.repeat(d['val_origins'],24))
            assert np.array_equal(old.target_row.to_numpy(),d['val_targets'].ravel())
            assert np.array_equal(old.horizon_step.to_numpy(),np.tile(np.arange(1,25),205))
            assert np.array_equal(old.y_true_delta_V.to_numpy(),d['val_y'].ravel().astype('float64'))
            for col,key in [('origin_age_h','val_origin_age_h'),('target_age_h','val_target_age_h')]:
                if col in old:
                    a=d[key]; a=np.repeat(a,24) if a.ndim==1 else a.ravel()
                    assert np.array_equal(old[col].to_numpy(),a),('TIME_IDENTITY',col)
            with gate_hook(m) as (gs,contexts,offset):base=h.e.predict(m,d['vx'],32,torch.device('cuda'))*scale
            assert offset[0]==205 and np.array_equal(base.ravel().astype('float64'),old.y_pred_delta_V.to_numpy()),'BASELINE_MISMATCH_STOP_MODEL'
            g=np.concatenate(gs);context=np.concatenate(contexts)
            with gate_hook(m) as (ts,tc,to):h.e.predict(m,d['tx'],32,torch.device('cuda'))
            tg=np.concatenate(ts);assert len(tg)==1380
            train_mean=float(tg.astype('float64').mean())
            cs(f'outputs/train_gates_seed_{seed}.csv',dict(seed=seed,origin_row=d['train_origins'],g=tg))
            gf=pd.DataFrame(dict(seed=seed,window=np.arange(205),origin_row=d['val_origins'],origin_age_h=d['val_origin_age_h'],g=g))
            for k,idx in enumerate(INDICES):
                gf[FEATURES[k]+'_standardized']=context[:,k]
                gf[FEATURES[k]+'_raw_mean']=d['val_x_raw'][:,:,idx].astype('float64').mean(axis=1)
                recovered=context[:,k].astype('float64')*sc['scale'][idx]+sc['mean'][idx]
                assert np.max(np.abs(recovered-gf[FEATURES[k]+'_raw_mean']))<1e-4,'RAW_CONTEXT_RECOVERY'
            cs(f'outputs/validation_gates_seed_{seed}.csv',gf)
            if seed==42:
                jg=rd(J/'outputs/gate_values/J_ALL_F2_GATES.csv');jg=jg[jg.seed==42]
                assert np.array_equal(jg.g.to_numpy(),g.astype('float64')),'J_SEED42_GATE_MISMATCH'
            frame=old[['origin_row','target_row','horizon_step','y_true_delta_V']].copy()
            frame['seed']=seed;frame['origin_age_h']=np.repeat(d['val_origin_age_h'],24)
            frame['target_age_h']=d['val_target_age_h'].ravel()
            frame['origin_raw_voltage_V']=np.repeat(d['val_origin_raw_voltage_V'],24)
            frame['y_pred_delta_V']=base.ravel();cs(f'outputs/predictions/seed_{seed}_B0.csv',frame)
            assignments={'B1':np.full(205,.5,dtype='float32'),'B2':np.full(205,train_mean,dtype='float32')}
            for ps in c['permutation_seeds']:
                pp=perms[perms.permutation_seed==ps].source_window.to_numpy()
                assert np.all(pp!=np.arange(205)) and np.array_equal(np.sort(pp),np.arange(205))
                assignments[f'C_{ps}']=g[pp];assert np.array_equal(np.sort(g[pp]),np.sort(g))
            checks=[]
            for mode,assigned in assignments.items():
                cs(f'outputs/used_gates_seed_{seed}_{mode}.csv',dict(window=np.arange(205),g_applied=assigned))
                with gate_hook(m,assigned) as (gg,cc,oo): pred=h.e.predict(m,d['vx'],32,torch.device('cuda'))*scale
                assert oo[0]==205 and np.array_equal(np.concatenate(gg),g),'HOOK_CHANGED_ORIGINAL_GATES'
                assert h.tree_hash(m.state_dict())==state and all(p.grad is None for p in m.parameters())
                restored=h.e.predict(m,d['vx'],32,torch.device('cuda'))*scale
                assert np.array_equal(restored,base),'RESTORATION_FAILED'
                out=frame.copy();out['y_pred_delta_V']=pred.ravel();cs(f'outputs/predictions/seed_{seed}_{mode}.csv',out)
                checks.append(dict(mode=mode,state_unchanged=True,restored_predictions_exact=True,hook_removed=not bool(m.regime_gate._forward_hooks)))
            result=dict(seed=seed,pass_=True,baseline_bit_exact=True,checkpoint_sha256=sha(REPO/row['checkpoint']),state_sha256=state,
                train_gate_mean=train_mean,train_gate_mean_float32=float(np.float32(train_mean)),train_n=1380,validation_n=205,checks=checks)
            result['pass']=result.pop('pass_');js(f'outputs/audits/seed_{seed}.json',result)
            print('SEED_COMPLETED',seed,'g_mean',float(g.mean()),'train_mean',train_mean,'seconds',round(time.time()-started,1),flush=True)
            del m,ck;torch.cuda.empty_cache()
        except Exception:
            js(f'outputs/audits/seed_{seed}_STOPPED.json',dict(seed=seed,traceback=traceback.format_exc(),formal_analysis_stopped=True))
            raise
    js('outputs/audits/INFERENCE_COMPLETE.json',dict(pass_=True,seeds=SEEDS,seconds=time.time()-started,training=False,Assessment=False,Test=False))
if __name__=='__main__':main()
