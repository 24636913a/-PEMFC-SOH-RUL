"""One-time exact-prefix data reconstruction, historical replay, protocol freeze."""
from screen_common import *
from reliable_io import js
from unittest.mock import patch
import shutil,inspect

def inherited_worker():
 s=(OLD/'scripts/train_seed_v3.py').read_text(encoding='utf-8')
 s=s.replace('from reliable_io import js,verify_IO_repair','from reliable_io import js,atomic_replace')
 s=s.replace(' from repair_identity import verify_repair\n verify_repair()\n verify_IO_repair()\n','')
 s=s.replace('os.replace(tmp,checkpoint)','atomic_replace(tmp,checkpoint)')
 s=s.replace("  js(f'outputs/metrics/seed_{seed}.json',result);", "  result['step1_RMSE_mV']=float(1000*np.sqrt(np.mean((saved.loc[saved.horizon_step.eq(1),'y_pred_delta_V'].to_numpy()-saved.loc[saved.horizon_step.eq(1),'y_true_delta_V'].to_numpy())**2)))\n  result['horizon_metrics']=[dict(horizon_step=int(step),horizon_h=float(step*.5),RMSE_mV=float(1000*np.sqrt(np.mean((g.y_pred_delta_V.to_numpy()-g.y_true_delta_V.to_numpy())**2))),MAE_mV=float(1000*np.mean(abs(g.y_pred_delta_V.to_numpy()-g.y_true_delta_V.to_numpy()))),n=len(g)) for step,g in saved.groupby('horizon_step')]\n  js(f'outputs/metrics/seed_{seed}.json',result);")
 return s

def main():
 assert not (P/'configs/frozen_seed_screening.json').exists(),'NO_REFREEZE'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
 assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='main'
 assert (P/'scripts/train_seed.py').read_text(encoding='utf-8')==inherited_worker(),'WORKER_MATH_INHERITANCE_MISMATCH'
 old=json.loads((OLD/'configs/frozen_seed_screening.json').read_text());assert sha(OLD/'configs/frozen_seed_screening.json')==(OLD/'configs/frozen_seed_screening.sha256').read_text().strip()
 assert json.loads((OLD/'outputs/audits/INDEPENDENT_AUDIT.json').read_text())['status']=='PASS'
 for path,digest in old['source_sha256'].items():assert sha(Path(path) if Path(path).is_absolute() else REPO/path)==digest
 for path,digest in old['data_sha256'].items():assert sha(OLD/path)==digest
 env=h.configure_normal(old);assert shutil.disk_usage(P).free>15*1024**3
 free,total=torch.cuda.mem_get_info();assert free>1024**3
 captured={};calls=[];real=pd.read_csv
 def guarded(f,*args,**kw):
  if Path(f).resolve()==h.e.c.N1.resolve():
   assert not args and kw.get('nrows')==3007,'TEST_FIREWALL'
   result=real(f,**kw);captured['prefix']=result.copy();calls.append(dict(nrows=3007,returned=len(result)));return result
  return real(f,*args,**kw)
 with patch.object(pd,'read_csv',guarded):tr,va,scaler,tx,vx,fp=h.prepare(old)
 assert len(calls)==1 and len(captured['prefix'])==3007
 dev=captured['prefix'].loc[captured['prefix'].bin_end_ophrs2_h<1819.997738].reset_index(drop=True);assert len(dev)==3005
 assert fp==json.loads((OLD/'outputs/audits/DATA_FINGERPRINTS.json').read_text()),'FULL_FINGERPRINT_DICTIONARY_MISMATCH'
 arrays=dict(tx=tx,vx=vx,train_x_raw=tr.x,val_x_raw=va.x,train_y=tr.y,val_y=va.y,train_origins=tr.origins,val_origins=va.origins,train_targets=tr.targets,val_targets=va.targets,val_origin_age_h=va.ages,val_target_age_h=dev.bin_end_ophrs2_h.to_numpy()[va.targets],val_origin_raw_voltage_V=dev.cell_voltage_mean.to_numpy()[va.origins],val_target_raw_voltage_V=dev.cell_voltage_mean.to_numpy()[va.targets])
 prior=np.load(OLD/'outputs/audits/FROZEN_H_DATA.npz',allow_pickle=False)
 assert set(prior.files)==set(arrays)
 assert all(np.array_equal(prior[k],v) for k,v in arrays.items())
 for path in old['data_sha256']:shutil.copyfile(OLD/path,P/path)
 # Old-stage deliverables and ignored full batch journals are byte-protected.
 history=[]
 for f in sorted(OLD.rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':history.append(dict(path=f.relative_to(REPO).as_posix(),sha256=sha(f),bytes=f.stat().st_size))
 cs('outputs/audits/ROUND1_PROTECTION.csv',pd.DataFrame(history))
 # Synthetic only: demonstrate the unchanged forward/backward and logger fix.
 h.e.set_seed(999);m=h.model('F2',old['hyperparameters'],torch.device('cuda'));m.train();x=torch.randn(2,48,14,device='cuda');out=m.predict_delta(x)
 assert out.shape==(2,24) and torch.isfinite(out).all();out.square().mean().backward()
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 for branch in ['regime_gate','history_branch']:assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if branch in n)
 assert torch.count_nonzero(m.build_input(x)['decoder_cont'][:,:,:14])==0
 del m;torch.cuda.empty_cache()
 # Read-only validation inference for historical 42 (never refit).
 from historical_candidates import build_historical
 build_historical(old,arrays,scaler.target_scale)
 sources=dict(old['source_sha256'])
 for f in [OLD/'README.md',OLD/'configs/frozen_seed_screening.json',OLD/'configs/WINDOWS_IO_REPAIR_AMENDMENT.json',OLD/'scripts/train_seed_v3.py',OLD/'scripts/reliable_io.py',OLD/'scripts/resume_after_IO_repair.py',OLD/'outputs/audits/INDEPENDENT_AUDIT.json']:
  sources[f.relative_to(REPO).as_posix()]=sha(f)
 critical=['scripts/screen_common.py','scripts/train_seed.py','scripts/reliable_io.py','scripts/run_round2.py','scripts/prepare_round2.py','scripts/historical_candidates.py']
 c={k:old[k] for k in ['task','preprocessing','hyperparameters','normal_runtime','data']}
 c.update(status='FROZEN_BEFORE_CANDIDATE_TRAINING',base_commit=BASE,seeds=SEEDS,historical_seeds=[42,103,193],round1_protocol_sha256=sha(OLD/'configs/frozen_seed_screening.json'),thresholds=dict(RMSE_mV=6.3,endpoint_RMSE_mV=10.0,max_abs_error_mV=50.0),qualified_stop_count=3,maximum_wall_s=43200,start_next_seed_reserve_s=3420,per_seed_timeout_s=3300,
  selection=old['selection'],scientific_change='Seed only; IO guard relocated to new phase, bounded checkpoint atomic rename retry added; metrics extended only after fitting',fixed_figure_origin_row=int(va.origins[len(va.origins)//2]),source_sha256=sources,critical_code_sha256={p:sha(P/p) for p in critical},data_sha256=old['data_sha256'],no_Test=True,no_Assessment=True,no_CPU_fallback=True,no_retries=True,no_extra_seeds=True)
 js('configs/frozen_seed_screening.json',c);f=P/'configs/frozen_seed_screening.json';f.with_suffix('.sha256').write_bytes((sha(f)+'\n').encode())
 js('outputs/audits/PREFLIGHT.json',dict(status='PASS',environment=env,GPU_free_bytes=free,GPU_total_bytes=total,disk_free_bytes=shutil.disk_usage(P).free,original_prefix_reads=calls,development_rows=3005,train_windows=1380,validation_windows=205,validation_points=4920,all_round1_arrays_bit_exact=True,all_round1_fingerprints_exact=True,worker_mathematics_exact=True,synthetic_backward=True,formal_optimizer_steps=0,Test=False,Assessment=False,protocol_sha256=sha(f)))
 cfg();print('ROUND2_PREFLIGHT_PASS '+sha(f),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:js('outputs/audits/PREFLIGHT_FAILURE.json',dict(status='BLOCKED',exception=repr(exc),traceback=traceback.format_exc()));raise
