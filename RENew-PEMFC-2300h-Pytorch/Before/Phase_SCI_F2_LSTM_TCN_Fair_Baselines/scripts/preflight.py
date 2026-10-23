"""One-shot strict-prefix reconstruction and freeze, before any formal training."""
from baseline_common import *
import ast,shutil
from unittest.mock import patch

def main():
    assert not (P/'configs/FROZEN_BASELINE_PROTOCOL.yaml').exists(),'NO_REFREEZE'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','diff','--name-only',BASE],cwd=REPO,text=True).strip()
    old=h.load_protocol();env=h.configure_normal(old)
    assert shutil.disk_usage(P).free>10*1024**3
    calls=[];captured={};original=pd.read_csv
    def guarded(f,*args,**kw):
        if Path(f).resolve()==h.e.c.N1.resolve():
            assert not args and kw.get('nrows')==3007,'TEST_FIREWALL'
            prefix=original(f,**kw);assert len(prefix)==3007;captured['prefix']=prefix.copy();calls.append(dict(nrows=3007,rows_returned=len(prefix)));return prefix
        return original(f,*args,**kw)
    with patch.object(pd,'read_csv',guarded):tr,va,scaler,tx,vx,fp=h.prepare(old)
    assert len(calls)==1
    dev=captured['prefix'].loc[captured['prefix'].bin_end_ophrs2_h<1819.997738].reset_index(drop=True);assert len(dev)==3005
    assert fp==read(R2/'outputs/audits/DATA_FINGERPRINTS.json'),'H_FINGERPRINT_MISMATCH'
    arrays=dict(tx=tx,vx=vx,train_x_raw=tr.x,val_x_raw=va.x,train_y=tr.y,val_y=va.y,train_origins=tr.origins,val_origins=va.origins,train_targets=tr.targets,val_targets=va.targets,val_origin_age_h=va.ages,val_target_age_h=dev.bin_end_ophrs2_h.to_numpy()[va.targets],val_origin_raw_voltage_V=dev.cell_voltage_mean.to_numpy()[va.origins],val_target_raw_voltage_V=dev.cell_voltage_mean.to_numpy()[va.targets])
    prior=np.load(R2/'outputs/audits/FROZEN_H_DATA.npz',allow_pickle=False)
    assert set(arrays)==set(prior.files) and all(np.array_equal(v,prior[k]) for k,v in arrays.items()),'FULL_ARRAY_IDENTITY_MISMATCH'
    assert tx.shape==(1380,48,14) and vx.shape==(205,48,14)
    assert (tr.targets>tr.origins[:,None]).all() and (va.targets>va.origins[:,None]).all()
    values=dev[old['task']['raw_f2_feature_order']].to_numpy(float)
    for windows in [tr,va]:
        for i,origin in enumerate(windows.origins):
            assert np.array_equal(windows.x[i],values[origin-47:origin+1].astype(np.float32)),'NON_HISTORY_INPUT'
            assert np.array_equal(windows.targets[i],np.arange(origin+1,origin+25)),'FUTURE_TARGET_IDENTITY'
    # Existing window-builder source supplies histories [origin-47,origin]; target row identities pinned above.
    source=(N2/'scripts/phase_n2_models.py').read_text(encoding='utf-8');copy=(P/'models/baseline_models.py').read_text(encoding='utf-8')
    original_classes={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.ClassDef)}
    for n in ast.parse(copy).body:
        if isinstance(n,ast.ClassDef):assert ast.dump(n,include_attributes=False)==original_classes[n.name],'MODEL_CLASS_CHANGED'
    # Synthetic engineering backward only, no optimizer update and no trained weights reused.
    counts={};smoke={}
    for name in ['LSTM','TCN']:
        h.e.set_seed(999);m=model(name,'cuda');m.train();x=torch.randn(2,48,14,device='cuda');out=m(x);assert out.shape==(2,24) and torch.isfinite(out).all()
        out.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        counts[name]=sum(p.numel() for p in m.parameters() if p.requires_grad);smoke[name]=dict(shape=list(out.shape),finite=True,backward=True,formal_optimizer_steps=0)
        del m
    f2=h.model('F2',old['hyperparameters'],torch.device('cpu'));counts['F2']=sum(p.numel() for p in f2.parameters() if p.requires_grad);assert counts['F2']==332436;del f2
    torch.cuda.empty_cache()
    for name in ['FROZEN_H_DATA.npz','VALIDATION_KEY_TRUTH.csv','SCALER.json','DATA_FINGERPRINTS.json']:shutil.copyfile(R2/'outputs/audits'/name,P/'outputs/audits'/name)
    before={r.path:r.sha256 for r in rd(PAPER/'outputs/audits/HISTORY_BEFORE.csv').itertuples()}
    tracked=subprocess.check_output(['git','ls-files','-z','--',str(PAPER.relative_to(REPO)),str(N2.relative_to(REPO)),str(H.relative_to(REPO)),str(h.E.relative_to(REPO)),str(h.F.relative_to(REPO))],cwd=REPO).decode('utf-8').strip('\0').split('\0')
    for q in tracked:
        if q and Path(q).name!='causal_features_30min.csv':before[q]=sha(REPO/q)
    cs('outputs/audits/HISTORY_BEFORE.csv',[dict(path=q,sha256=v) for q,v in sorted(before.items())]);protect()
    sources=[H/'configs/H_FROZEN_PROTOCOL.yaml',N2/'scripts/phase_n2_models.py',H/'scripts/h_common.py',h.E/'scripts/run_e_comparison.py',h.C/'scripts/run_c_ablation.py',PAPER/'outputs/condition_analysis/TRAIN_THRESHOLDS.csv',PAPER/'outputs/condition_analysis/ORIGIN_CONDITIONS.csv',PAPER/'tables/T04_FORMAL_MODEL_COMPARISON.csv',H/'outputs/metrics/H_VALIDATION_METRICS.csv']
    sources+=sorted((h.F/'models').glob('*.py'))+[h.T0]
    for r in rd(H/'outputs/metrics/H_VALIDATION_METRICS.csv').query("role=='MULTISEED_PRIMARY'").itertuples():
        pred=H/r.validation_prediction;assert sha(pred)==r.prediction_file_sha256;sources.append(pred)
    critical=['scripts/baseline_common.py','scripts/preflight.py','scripts/train_baseline.py','scripts/run_baselines.py','scripts/audit_run.py','models/baseline_models.py']
    c=dict(phase=P.name,base_commit=BASE,status='FROZEN_BEFORE_FORMAL_RUNS',task=old['task'],preprocessing=old['preprocessing'],hyperparameters=old['hyperparameters'],normal_runtime=old['normal_runtime'],data=old['data'],runs=[dict(model=m,seed=s) for m,s in RUNS],architectures=ARCH,parameter_counts=counts,optimizer='AdamW',early_stopping='validation_standardized_RMSE < best - 1e-12; patience40; max450',dataloader=dict(batch_size=32,shuffle=True,num_workers=0,generator_seed='run seed'),capacity_matching=False,learning_rate_tuned_for_baselines=False,TCN_padding='N2 symmetric within observed history; not strictly causal within48steps; no future actual input',no_Test=True,no_Assessment=True,one_shot=True,maximum_formal_runs=6,pilot_order=['LSTM42','TCN42'],continue_only_after_both_pilot_audits=True,case_origins=dict(fixed=2205,difficult=2224),diagnostic_targets=[2247,2248],critical_sha256={q:sha(P/q) for q in critical},sources_sha256={q.relative_to(REPO).as_posix():sha(q) for q in sources},data_sha256={q:sha(P/q) for q in ['outputs/audits/FROZEN_H_DATA.npz','outputs/audits/VALIDATION_KEY_TRUTH.csv','outputs/audits/SCALER.json','outputs/audits/DATA_FINGERPRINTS.json']})
    write('configs/FROZEN_BASELINE_PROTOCOL.yaml',yaml.safe_dump(c,sort_keys=False,allow_unicode=True));write('configs/FROZEN_BASELINE_PROTOCOL.sha256',sha(P/'configs/FROZEN_BASELINE_PROTOCOL.yaml')+'\n')
    cfg();js('outputs/audits/PREFLIGHT_REPORT.json',dict(status='PASS',environment=env,original_prefix_reads=calls,development_rows=3005,train_windows=1380,validation_windows=205,validation_queries=4920,all_h_fingerprints_exact=True,all_snapshot_arrays_bit_exact=True,model_class_AST_exact=True,smoke=smoke,parameter_counts=counts,historical_files_protected=len(before),protocol_sha256=sha(P/'configs/FROZEN_BASELINE_PROTOCOL.yaml'),Test=False,Assessment=False))
    event('PREFLIGHT_PASS');print('PREFLIGHT_PASS',counts,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:js('outputs/audits/PREFLIGHT_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
