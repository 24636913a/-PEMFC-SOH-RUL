from common import *
from load_frozen_models import load,predict,state_hash,h
from preprocess_frozen_inputs import scaler
def prediction_frame(part,model,seed,pred,d):
    n=len(d['origins']);f=pd.DataFrame(dict(partition=part,model=model,seed=seed,origin_row=np.repeat(d['origins'],24),target_row=d['targets'].ravel(),horizon_step=np.tile(np.arange(1,25),n),origin_age_h=np.repeat(d['origin_age_h'],24),target_age_h=d['target_age_h'].ravel(),origin_raw_voltage_V=np.repeat(d['origin_raw_voltage_V'],24),y_true_delta_V=d['y'].ravel(),y_pred_delta_V=pred.ravel(),true_raw_target_voltage_V=d['target_raw_voltage_V'].ravel()))
    f['true_label_voltage_V']=f.origin_raw_voltage_V+f.y_true_delta_V
    f['predicted_voltage_V']=f.origin_raw_voltage_V+f.y_pred_delta_V
    return f
def validation_replay():
    env=h.configure_normal(h.load_protocol());js('outputs/audits/INFERENCE_ENVIRONMENT.json',env)
    d=data('VALIDATION');rows=[]
    for r in inventory().itertuples():
        m,ck=load(r);before=state_hash(m);std=predict(m,d['x'],r.model);pred=std*scaler()['target_scale_V']
        f=prediction_frame('VALIDATION',r.model,r.seed,pred,d);ref=rd(REPO/r.reference_prediction)
        for key in ['origin_row','target_row','horizon_step','y_true_delta_V','y_pred_delta_V']:
            assert np.array_equal(f[key].to_numpy(),ref[key].to_numpy()),('VALIDATION_POINTWISE_MISMATCH',r.model,r.seed,key,float(np.max(abs(f[key]-ref[key]))))
        if 'full_validation_prediction' in ck:assert np.array_equal(std,ck['full_validation_prediction'].numpy())
        assert before==state_hash(m)
        cs(f'outputs/predictions/VALIDATION_{r.model}_seed{r.seed}.csv',f)
        rows.append(dict(model=r.model,seed=r.seed,origins=205,queries=4920,pointwise_exact=True,max_prediction_difference_V=0.,weights_unchanged=True,state_sha256=before))
        event('VALIDATION_REPLAY_EXACT',model=r.model,seed=r.seed);print('Validation exact',r.model,r.seed,flush=True)
        del m,ck;torch.cuda.empty_cache()
    cs('outputs/audits/VALIDATION_EXACT_REPLAY.csv',rows)
def run_remaining():
    protocol();h.configure_normal(h.load_protocol());rows=[]
    for r in inventory().itertuples():
        m,ck=load(r);before=state_hash(m)
        for part in ['TRAIN','HISTORICAL_ASSESSMENT','LATE_LIFE']:
            out=f'outputs/predictions/{part}_{r.model}_seed{r.seed}.csv'
            receipt=f'outputs/execution_ledger/{part}_{r.model}_seed{r.seed}.json'
            if (P/receipt).exists():
                old=read(P/receipt);assert old['checkpoint_sha256']==r.checkpoint_sha256 and old['prediction_sha256']==sha(P/out);print('Verified resume',part,r.model,r.seed,flush=True);continue
            d=data(part);t=time.perf_counter();pred=predict(m,d['x'],r.model)*scaler()['target_scale_V']
            f=prediction_frame(part,r.model,r.seed,pred,d);assert np.isfinite(pred).all();assert state_hash(m)==before
            cs(out,f);torch.cuda.synchronize()
            js(receipt,dict(partition=part,model=r.model,seed=r.seed,checkpoint_sha256=r.checkpoint_sha256,prediction_sha256=sha(P/out),state_sha256=before,state_unchanged=True,inference_seconds=time.perf_counter()-t,queries=len(f),protocol_sha256=sha(P/'configs/FROZEN_CHRONOLOGICAL_PROTOCOL.yaml')))
            event('INFERENCE_COMPLETED',partition=part,model=r.model,seed=r.seed,queries=len(f));print('Inference',part,r.model,r.seed,len(f),flush=True)
        assert before==state_hash(m) and sha(REPO/r.checkpoint)==r.checkpoint_sha256
        rows.append(dict(model=r.model,seed=r.seed,state_sha256=before,unchanged=True))
        del m,ck;torch.cuda.empty_cache()
    cs('outputs/audits/ALL_MODEL_STATES_UNCHANGED.csv',rows)
