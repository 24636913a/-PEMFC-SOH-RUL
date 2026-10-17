"""Observe original gate/path activations only; hooks never replace outputs."""
from audit_common import *
import importlib.util
import torch
def main():
    check_protocol();spec=importlib.util.spec_from_file_location('readonly_f2_main',ROOT/'RealHyFC_F2_Main.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    environment=m.configure_environment('cuda');p=m.load_historical_hyperparameters()['parameters'];sc=read(m.FROZEN_DATA/'SCALER.json');scaler=m.TrainOnlyScaler();scaler.mean=np.array(sc['mean']);scaler.scale=np.array(sc['scale'])
    # Enforce forward-only execution, without changing any saved model or module.
    def forbidden(*args,**kwargs):raise RuntimeError('TRAINING_AUTOGRAD_FORBIDDEN')
    torch.Tensor.backward=forbidden;torch.autograd.backward=forbidden;torch.autograd.grad=forbidden
    for cls in [torch.optim.Optimizer,torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD]:cls.step=forbidden
    torch.set_grad_enabled(False);result=[];audits=[]
    for seed in SEEDS:
        row=m.historical_checkpoint(seed);file=REPO/row.checkpoint;model=m.initialize_f2(seed,p,torch.device('cuda'));checkpoint=m.load_checkpoint(model,file,scaler,seed,row);before=m.state_sha256(model)
        for part in ['VALIDATION','TRAIN','LATE_LIFE']:
            d=data(part);native=m.predict_standardized(model,d['x'],32,torch.device('cuda'))*scaler.target_scale
            reference=pred('F2',seed,part)
            for key,val in [('origin_row',np.repeat(d['origins'],24)),('target_row',d['targets'].ravel()),('horizon_step',np.tile(np.arange(1,25),len(d['origins']))),('y_true_delta_V',d['y'].ravel()),('y_pred_delta_V',native.ravel())]:assert np.array_equal(reference[key],val),('FROZEN_REPLAY_MISMATCH',seed,part,key)
            observed={'gate':[],'context':[],'lstm':[],'conv':[]}
            def gate_hook(module,args,out):observed['gate'].append(out.detach().cpu().numpy().copy());observed['context'].append(args[0].detach().cpu().numpy().copy())
            def lstm_hook(module,args,out):
                # Upstream may share encoder/decoder normalization modules.
                if out.ndim==3 and out.shape[1]==48:observed['lstm'].append(out.detach().cpu().numpy().copy())
            def conv_hook(module,args,out):observed['conv'].append(out.detach().cpu().numpy().copy())
            handles=[model.regime_gate.register_forward_hook(gate_hook),model.post_lstm_add_norm_encoder.register_forward_hook(lstm_hook),model.history_branch.register_forward_hook(conv_hook)]
            try:actual=m.predict_standardized(model,d['x'],32,torch.device('cuda'))*scaler.target_scale
            finally:
                for handle in handles:handle.remove()
            assert np.array_equal(native,actual) and m.state_sha256(model)==before
            gate=np.concatenate(observed['gate']).ravel();ctx=np.concatenate(observed['context']);left=np.concatenate(observed['lstm']).astype(float);right=np.concatenate(observed['conv']).astype(float)
            assert len(gate)==len(d['origins'])
            lrms=np.sqrt(np.mean(left*left,axis=(1,2)));crms=np.sqrt(np.mean(right*right,axis=(1,2)));diff=np.sqrt(np.mean((left-right)**2,axis=(1,2)));cos=np.sum(left*right,axis=(1,2))/np.sqrt(np.sum(left*left,axis=(1,2))*np.sum(right*right,axis=(1,2)))
            for i,o in enumerate(d['origins']):
                record=dict(seed=seed,partition=part,origin_row=int(o),origin_age_h=float(d['origin_age_h'][i]),g=float(gate[i]),LSTM_hidden_RMS=float(lrms[i]),CNN_hidden_RMS=float(crms[i]),path_difference_RMS=float(diff[i]),path_cosine=float(cos[i]))
                for j,idx in enumerate(model.regime_indices):record[m.FEATURES[idx]+'_scaled_context']=float(ctx[i,j]);record[m.FEATURES[idx]+'_raw_history_mean']=float(d['x_raw'][i,:,idx].astype(float).mean())
                result.append(record)
            audits.append(dict(seed=seed,partition=part,origins=len(d['origins']),queries=len(reference),prehook_replay_exact=True,observing_hooks_prediction_exact=True,weights_unchanged=True,state_sha256=before,checkpoint_sha256=sha(file),input_sha256=ah(d['x']),no_interventions=True))
        assert sha(file)==row.checkpoint_sha256
        del model,checkpoint;torch.cuda.empty_cache();print('GATE_OBSERVATION_COMPLETE',seed,flush=True)
    g=pd.DataFrame(result);cs('outputs/ORIGINAL_GATE_AND_PATH_OBSERVATIONS.csv',g);cs('outputs/FORWARD_IDENTITY_AUDIT.csv',audits);js('outputs/FORWARD_ENVIRONMENT.json',environment)
    summary=[]
    for (seed,part),z in g.groupby(['seed','partition']):summary.append(dict(seed=int(seed),partition=part,n=len(z),g_mean=float(z.g.mean()),g_std=float(z.g.std(ddof=0)),g_min=float(z.g.min()),g_max=float(z.g.max()),path_difference_RMS_mean=float(z.path_difference_RMS.mean()),LSTM_hidden_RMS_mean=float(z.LSTM_hidden_RMS.mean()),CNN_hidden_RMS_mean=float(z.CNN_hidden_RMS.mean())))
    cs('outputs/GATE_PATH_STATISTICS.csv',summary)
if __name__=='__main__':main()
