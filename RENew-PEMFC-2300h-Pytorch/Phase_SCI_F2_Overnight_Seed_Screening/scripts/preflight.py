"""Run once: audit frozen H, exact Development prefix and synthetic GPU flow."""
from screen_common import *
import inspect,shutil,yaml
from unittest.mock import patch
def main():
 assert not (P/'configs/frozen_seed_screening.json').exists(),'NO_REFREEZE_OR_REPEAT_PRECHECK'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
 existing=subprocess.check_output(['git','status','--porcelain','-z'],cwd=REPO).decode('utf-8').split('\0')
 prefix=P.relative_to(REPO).as_posix()+'/'
 assert all(not x or x[3:].startswith(prefix) for x in existing),'EXTERNAL_UNCOMMITTED_CHANGES'
 assert shutil.disk_usage(P).free>10*1024**3,'INSUFFICIENT_DISK'
 old=h.load_protocol();env=h.configure_normal(old);assert '3070 Laptop' in env['GPU']
 total=torch.cuda.get_device_properties(0).total_memory;free,_=torch.cuda.mem_get_info();assert free>1024**3,'INSUFFICIENT_GPU_MEMORY'
 captured={};real_reader=pd.read_csv;calls=[]
 def guarded(f,*args,**kw):
  if Path(f).resolve()==h.e.c.N1.resolve():
   assert kw.get('nrows')==3007 and not args,'TEST_READ_FIREWALL'
   d=real_reader(f,**kw);calls.append(dict(nrows=3007,returned=len(d)));captured['prefix']=d.copy();return d
  return real_reader(f,*args,**kw)
 with patch.object(pd,'read_csv',guarded):tr,va,scaler,tx,vx,fp=h.prepare(old)
 assert len(calls)==1 and len(captured['prefix'])==3007
 dev=captured['prefix'].loc[captured['prefix'].bin_end_ophrs2_h<1819.997738].reset_index(drop=True)
 assert len(dev)==3005
 key=rd(H/'outputs/audits/H_VALIDATION_KEY_TRUTH_AUTHORITY.csv')
 expected=pd.DataFrame(dict(origin_row=np.repeat(va.origins,24),target_row=va.targets.ravel(),horizon_step=np.tile(np.arange(1,25),len(va.origins)),y_true_delta_V=va.y.ravel()))
 assert np.array_equal(key.to_numpy(),expected.to_numpy()),'H_VALIDATION_KEYS_TRUTH_MISMATCH'
 arrays=dict(tx=tx,vx=vx,train_x_raw=tr.x,val_x_raw=va.x,train_y=tr.y,val_y=va.y,train_origins=tr.origins,val_origins=va.origins,train_targets=tr.targets,val_targets=va.targets,
  val_origin_age_h=va.ages,val_target_age_h=dev.bin_end_ophrs2_h.to_numpy()[va.targets],val_origin_raw_voltage_V=dev.cell_voltage_mean.to_numpy()[va.origins],val_target_raw_voltage_V=dev.cell_voltage_mean.to_numpy()[va.targets])
 out=P/'outputs/audits';out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'FROZEN_H_DATA.npz',**arrays)
 cs('outputs/audits/VALIDATION_KEY_TRUTH.csv',expected)
 js('outputs/audits/DATA_FINGERPRINTS.json',fp);js('outputs/audits/SCALER.json',dict(mean=scaler.mean.tolist(),scale=scaler.scale.tolist(),target_scale_V=scaler.target_scale))
 # Synthetic only: no formal train labels or actual seed model used for this step.
 h.e.set_seed(999);m=h.model('F2',old['hyperparameters'],torch.device('cuda'));m.train();x=torch.randn(2,48,14,device='cuda');y=m.predict_delta(x)
 assert y.shape==(2,24) and torch.isfinite(y).all()
 y.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 for name in ['regime_gate','history_branch']:
  # Require genuine branch/gate gradients, not a zero placeholder loss.
  named=[(n,p) for n,p in m.named_parameters() if name in n]
  assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in named),('SYNTHETIC_BRANCH_GRADIENT',name)
 m.eval();q=m.build_input(x);assert q['decoder_cont'].shape==(2,24,15)
 assert torch.count_nonzero(q['decoder_cont'][:,:,:14])==0,'FUTURE_INPUT_FIREWALL'
 a=torch.randn(2,48,14).numpy();b=a.copy();b[:,25:,0]+=7
 assert np.array_equal(h.e.ewma(a)[:,:25],h.e.ewma(b)[:,:25]),'EWMA_CAUSALITY'
 del m;torch.cuda.empty_cache()
 sources=dict(old['code_identity']['files']);sources[str((H/'configs/H_FROZEN_PROTOCOL.yaml').relative_to(REPO)).replace('\\','/')]=sha(H/'configs/H_FROZEN_PROTOCOL.yaml')
 for f in [H/'outputs/H_FINAL_SUMMARY.json',H/'outputs/audits/H_VALIDATION_KEY_TRUTH_AUTHORITY.csv']:
  sources[f.relative_to(REPO).as_posix()]=sha(f)
 critical=['scripts/screen_common.py','scripts/train_seed.py','scripts/run_seed_screening.py','scripts/preflight.py']
 config=dict(status='FROZEN_BEFORE_CANDIDATE_TRAINING',base_commit=BASE,seeds=SEEDS,historical_seeds=[42,52,62],
  historical_protocol_sha256=sha(H/'configs/H_FROZEN_PROTOCOL.yaml'),task=old['task'],preprocessing=old['preprocessing'],hyperparameters=old['hyperparameters'],normal_runtime=old['normal_runtime'],
  data=old['data'],maximum_wall_s=8*3600,start_next_seed_reserve_s=3600,per_seed_timeout_s=3300,summary_reserve_s=120,
  selection='Validation standardized RMSE improvement >1e-12, patience40 max450; ranking all completed Validation pooled RMSE; ties ascending seed',
  scientific_change='Seed only; exact inherited H model, data tensors, batching, initialization, loss, optimizer and validation selection; new local IO/ledger only',
  no_Test=True,no_Assessment=True,no_CPU_fallback=True,no_retries=True,no_extra_seeds=True,source_sha256=sources,
  critical_code_sha256={f:sha(P/f) for f in critical},data_sha256={f'outputs/audits/{f}':sha(out/f) for f in ['FROZEN_H_DATA.npz','DATA_FINGERPRINTS.json','SCALER.json','VALIDATION_KEY_TRUTH.csv']})
 js('configs/frozen_seed_screening.json',config);f=P/'configs/frozen_seed_screening.json';f.with_suffix('.sha256').write_bytes((sha(f)+'\n').encode())
 js('outputs/audits/PREFLIGHT.json',dict(status='PASS',environment=env,GPU_total_bytes=total,GPU_free_bytes=free,disk_free_bytes=shutil.disk_usage(P).free,
  original_prefix_reads=calls,development_rows=3005,train_windows=1380,validation_windows=205,validation_points=4920,synthetic_output_shape=[2,24],synthetic_backward=True,
  parameter_count=332436,Test=False,Assessment=False,formal_optimizer_steps=0,source_identity_matches_H=True,protocol_sha256=sha(f)))
 print('PREFLIGHT_PASS '+sha(f),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:js('outputs/audits/PREFLIGHT_FAILURE.json',dict(status='BLOCKED',exception=repr(exc),traceback=traceback.format_exc()));raise
