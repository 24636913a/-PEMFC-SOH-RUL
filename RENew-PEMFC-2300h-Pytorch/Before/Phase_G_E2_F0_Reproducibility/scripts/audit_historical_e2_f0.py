"""Historical read-only logs/checkpoints/cache audit; no model inference."""
from __future__ import annotations
import ast
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
import torch
from g_common import *

def checkpoint(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    state = ck['state_dict']
    return ck, state, {'path': str(path.relative_to(REPO)), 'file_sha256': file_hash(path),
        'metadata': {k: (str(v.dtype)+' '+str(tuple(v.shape)) if torch.is_tensor(v) else v)
                     for k,v in ck.items() if k not in ('state_dict',)},
        'parameter_count': sum(t.numel() for t in state.values()),
        'state_dictionary_sha256': tensor_tree_hash(state)}

def main():
    folder = OUT/'historical_audit'; folder.mkdir(parents=True, exist_ok=True)
    manifest = identity()
    write_json(folder/'SOURCE_MANIFEST.json', {'base_commit':BASE,'source_sha256':manifest,'test_accessed':False})
    ej = read_json(E/'outputs/training_logs/E2_seed42.json')
    f = pd.read_csv(F/'outputs/training_logs/F0_epochs_seed42.csv', float_precision='round_trip')
    e = pd.DataFrame(ej['fit']['history']).rename(columns={'train_standardized_mse':'E2_train_MSE',
        'validation_standardized_rmse':'E2_validation_RMSE'})
    f = f.rename(columns={'train_standardized_MSE':'F0_train_MSE', 'validation_standardized_RMSE':'F0_validation_RMSE'})
    table = e.merge(f[['epoch','F0_train_MSE','F0_validation_RMSE']], on='epoch', how='outer')
    table['train_MSE_absolute_difference'] = abs(table.E2_train_MSE-table.F0_train_MSE)
    table['validation_RMSE_absolute_difference'] = abs(table.E2_validation_RMSE-table.F0_validation_RMSE)
    table.to_csv(folder/'HISTORICAL_EPOCH_COMPARISON.csv',index=False,float_format='%.17g')
    common = table.dropna()
    divergences = common[(common.train_MSE_absolute_difference>0)|(common.validation_RMSE_absolute_difference>0)]
    first = None if divergences.empty else divergences.iloc[0].to_dict()
    ec,es,ea = checkpoint(E/'outputs/checkpoints/E2_Top10_EWMA_seed42.pt')
    fc,fs,fa = checkpoint(F/'outputs/checkpoints/F0_best_seed42.pt')
    if es.keys()!=fs.keys() or any(es[k].shape!=fs[k].shape for k in es):
        raise RuntimeError('HISTORICAL_CHECKPOINT_PARAMETER_SCHEMA_MISMATCH')
    diffs=[]
    for k in es:
        x,y=es[k].double(),fs[k].double(); d=(x-y).abs()
        diffs.append({'parameter':k,'module':k.split('.')[0],'shape':str(tuple(x.shape)),
                      'numel':x.numel(),'max_abs_difference':d.max().item(),
                      'l2_difference':torch.linalg.vector_norm(x-y).item(),'exactly_equal':torch.equal(es[k],fs[k])})
    pd.DataFrame(diffs).to_csv(folder/'HISTORICAL_CHECKPOINT_PARAMETER_DIFFERENCES.csv',index=False)
    pd.DataFrame(diffs).groupby('module').agg(parameters=('numel','sum'),
        max_abs_difference=('max_abs_difference','max'),l2_sum=('l2_difference','sum')).to_csv(folder/'HISTORICAL_MODULE_DIFFERENCES.csv')
    write_json(folder/'HISTORICAL_CHECKPOINT_AUDIT.json',{'E2':ea,'F0':fa,'schema_equal':True,
        'initialization_equal':ec['initialization_sha256']==fc['initialization_sha256'],
        'target_scale_equal':ec['common_target_scale_V']==fc['target_scale_V'],
        'note':'Different best epochs imply different final weights; not evidence of a code defect.'})
    left=pd.read_csv(E/'outputs/E2_predictions_seed42.csv',float_precision='round_trip').sort_values(['origin_row','horizon_step'])
    right=pd.read_csv(F/'outputs/predictions/F0_assessment_seed42.csv',float_precision='round_trip').sort_values(['origin_row','horizon_step'])
    keys=['origin_row','horizon_step','target_row']
    if len(left)!=9072 or len(right)!=9072 or left.duplicated(keys).any() or right.duplicated(keys).any():
        raise RuntimeError('HISTORICAL_CACHE_KEY_FAILURE')
    if not np.array_equal(left[keys].to_numpy(),right[keys].to_numpy()) or not np.array_equal(left.y_true_delta_V,right.y_true_delta_V):
        raise RuntimeError('HISTORICAL_TRUE_LABEL_MISMATCH')
    write_json(folder/'HISTORICAL_SAVED_ASSESSMENT_COMPARISON.json',{
        'saved_cache_only':True,'new_assessment_inference':False,'truth_and_keys_equal':True,'records':len(left),
        'E2_RMSE_mV':float(np.sqrt(np.mean((left.y_pred_delta_V-left.y_true_delta_V)**2))*1000),
        'F0_RMSE_mV':float(np.sqrt(np.mean((right.y_pred_delta_V-right.y_true_delta_V)**2))*1000),
        'max_prediction_difference_V':float(np.max(abs(left.y_pred_delta_V.to_numpy()-right.y_pred_delta_V.to_numpy())))})
    env_e=read_json(E/'outputs/E_PRETRAIN_AUDIT.json'); env_f=read_json(F/'outputs/audits/F_ENVIRONMENT.json')
    rows=[]
    emap={'torch':'torch_version','device':'device','cudnn_deterministic':'cudnn_deterministic','cudnn_benchmark':'cudnn_benchmark'}
    for field in ['python','torch','CUDA','cuDNN','pytorch_forecasting','CPU_threads','GPU','device',
                  'cudnn_deterministic','cudnn_benchmark','torch_deterministic_algorithms','TF32_matmul','TF32_cudnn']:
        ev=env_e.get(emap.get(field,field),'HISTORICAL_RUNTIME_EVIDENCE_UNAVAILABLE')
        fv=env_f.get(field,'HISTORICAL_RUNTIME_EVIDENCE_UNAVAILABLE')
        rows.append({'setting':field,'E2':ev,'F0':fv,'interpretation':'not a verified historical cause'})
    pd.DataFrame(rows).to_csv(folder/'HISTORICAL_ENVIRONMENT_COMPARISON.csv',index=False)
    operations=[
        ('model','e.make_model → c.load_model_module/c.make_model','F.make_model F0 → same canonical class','same mathematical architecture'),
        ('seed','e.set_seed(42)','e.set_seed(42)','same Python/NumPy/Torch seeds'),
        ('dataset','TensorDataset(x,y)','TensorDataset(x,y,ids)','extra ids do not enter model'),
        ('loader','generator42 shuffle workers0 batch32','generator42 shuffle workers0 batch32','same sampler settings'),
        ('forward_loss','MSE(model.predict_delta(x),y)','MSE(model.predict_delta(x),y)','same operations'),
        ('backward_clip_step','backward → clip_grad_norm_1 → AdamW.step','same, plus finite gradient check','same math, added host synchronization'),
        ('validation','e.predict → c.predict; numpy RMSE','e.predict → c.predict; numpy RMSE','same eval/no_grad batch32'),
        ('best_state','CPU clone held in memory; save once after training','CPU state saved on every improvement','different control/I/O timing, not yet causal'),
        ('early_stop','strict best−1e-12 patience40','strict best−1e-12 patience40','same criterion'),
        ('CPU_threads','no explicit torch.set_num_threads','torch.set_num_threads(1)','historical E2 actual value unknown'),
        ('initial_hash','one state digest','state digest plus initialization/first batch audits','hash/timing differences, not training math'),
    ]
    pd.DataFrame(operations,columns=['stage','E2_execution','F0_execution','classification']).to_csv(folder/'SOURCE_EXECUTION_COMPARISON.csv',index=False)
    blob={}
    for path in [E/'scripts/run_e_comparison.py',F/'scripts/run_f_training.py',C/'scripts/run_c_ablation.py',T0]:
        rel=str(path.relative_to(REPO)).replace('\\','/')
        blob[rel]={'current_blob':subprocess.check_output(['git','rev-parse',f'{BASE}:{rel}'],cwd=REPO,text=True).strip()}
    write_json(folder/'HISTORICAL_SUMMARY.json',{'first_divergence':first,'E2_epochs':len(e),'F0_epochs':len(f),
        'E2_best_epoch':ec['best_epoch'],'F0_best_epoch':fc['best_epoch'],
        'first_three_epochs_exact':bool((common.iloc[:3][['train_MSE_absolute_difference','validation_RMSE_absolute_difference']]==0).all().all()),
        'E2_first_batch_evidence':'HISTORICAL_FIRST_BATCH_FINGERPRINT_UNAVAILABLE','source_git_blobs':blob})
    reports=PHASE/'reports';reports.mkdir(exist_ok=True)
    (reports/'G_HISTORICAL_EXECUTION_COMPARISON.md').write_text(
        '# Historical E2/F0 execution comparison\n\n'
        +json.dumps(read_json(folder/'HISTORICAL_SUMMARY.json'),indent=2)+'\n\n'
        +'Source stage comparison and available runtime evidence are in the CSV tables. Source order '
        +'has been reviewed: mathematical forward/loss/backward/clip/step and validation expressions agree. '
        +'Dataset ID tracking, finite checks, in-memory best clones versus frequent serialization, '
        +'logs and explicit CPU thread control differ. None is presently a proven historical cause.\n\n'
        +'E2 historical CPU threads, Python, CUDA/cuDNN, GPU, TF32, and deterministic-algorithm flag '
        +'were not separately saved. Absence of an explicit setting does not identify its actual value. '
        +'The E2 first-batch hash is unavailable; reconstructed hashes cannot substitute for historical evidence.\n\n'
        +'Only existing Assessment caches were compared. No new Assessment windows or inference were run. '
        +'Final checkpoint differences are expected at best epochs 287 versus 165. Historical files are preserved.\n',encoding='utf-8')
    print(json.dumps({'historical_audit':'PASS','first_divergence':first},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
