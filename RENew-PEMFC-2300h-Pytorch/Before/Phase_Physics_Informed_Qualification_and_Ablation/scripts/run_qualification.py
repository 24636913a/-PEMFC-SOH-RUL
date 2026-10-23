"""Two frozen candidates: physical data qualification, CPU contracts, no optimizer."""
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from physics_math import nernst,polarization,upper_loss,amplitude_loss,empty_domain_loss,S
P=Path(__file__).resolve().parents[1];ROOT=P.parent;A=ROOT/'Phase_TFT_DynamicVoltage_to_ReferenceSOH'
SOURCES=[
 'Phase_P_Final_Innovation_and_Experiment_Planning/P_PHYSICS_LITE_CANDIDATE_REVIEW.md',
 'Phase_P_Final_Innovation_and_Experiment_Planning/P_PHYSICS_LITE_EQUATION_AND_INFORMATION_AUDIT.md',
 'Phase_DDP_SoftPhysics_Overnight_Experiment/01_SOFT_PHYSICS_METHOD.md',
 'Phase_DDP_SoftPhysics_Overnight_Experiment/outputs/TRAIN_PHYSICS_ACTIVITY.json',
 'Phase_M_F2_PINN_Physics_Feasibility/reports/M_PHYSICAL_VARIABLE_AVAILABILITY.md',
 'Phase_M_F2_PINN_Physics_Feasibility/reports/M_EQUATION_IDENTIFIABILITY_AUDIT.md',
 'Phase_M1_P3_TrainOnly_Polarization_Support_Qualification/outputs/M1_SUPPORT_SUMMARY.json',
 'Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/outputs/M2_SUPPORT_SUMMARY.json',
 'Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/outputs/M2_STAGE1_DOMAIN_COUNTS.csv',
 'Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/reports/M2_RAW_DWELL_STABILITY_REPORT.md',
 'Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/reports/M2_SENSOR_QUALITY_AND_PHYSICAL_DOMAIN.md',
 'Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/reports/M2_AUXILIARY_IDENTIFIABILITY.md',
 'Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/reports/M2_STATE_AND_HYSTERESIS_AUDIT.md',
 'Phase_N_Physics_Informed_F2_Method_Convergence/N_PHYSICS_EQUATION_AND_PARAMETER_AUDIT.md',
 'Phase_N_Physics_Informed_F2_Method_Convergence/N_FINAL_TRAINING_READINESS_DECISION.md',
 'Phase_TFT_DynamicVoltage_to_ReferenceSOH/inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv',
 'Phase_TFT_DynamicVoltage_to_ReferenceSOH/outputs/FINAL_DECISION.json',
 'Phase_TFT_P1S96_HealthAware_Fusion/inputs/MATCHED_INPUTS.npz',
 'Phase_TFT_P1S96_HealthAware_Fusion/inputs/INPUT_FINGERPRINTS.json',
 'Phase_TFT_P1S96_HealthAware_Fusion/configs/FROZEN_PROTOCOL.json',
 'Phase_TFT_P1S96_HealthAware_Fusion/models/health_fusion.py',
 'Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F2-Seed42_best.pt',
]+[str(f.relative_to(ROOT)).replace('\\','/') for folder in ['Phase_F_TFT_DualPath_RegimeFusion/models','Phase_TFT_Original_Baseline/models'] for f in (ROOT/folder).glob('*.py')]
PDF=ROOT.parent/'参考论文/1-s2.0-S0378775326006919-main.pdf'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes());return h.hexdigest()
def js(rel,obj):
 q=P/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def cs(rel,d):d.to_csv(P/rel,index=False,float_format='%.17g',lineterminator='\n')
def md(rel,s):(P/rel).write_bytes((s.strip()+'\n').encode('utf-8'))
def freeze():
 assert not (P/'configs/FROZEN_PROTOCOL.json').exists()
 config=dict(base_commit='b9e4ff7f81deea193daf0549c03d701bb00b9dc2',candidate_count=2,
  candidates=['C1_STATE_DEPENDENT_NERNST_UPPER_ENVELOPE','C2_QUANTITATIVE_ELECTROCHEMICAL_AMPLITUDE_RESIDUAL'],
  source_sha256={f:sha(ROOT/f) for f in sorted(set(SOURCES))},local_physics_pdf_sha256=sha(PDF),
  execution_code_sha256={f.name:sha(f) for f in (P/'scripts').glob('*.py')},
  rule='No physical evaluation without certified reaction state, measurement operator and independently identifiable required parameters',
  Train_range='Frozen F2 table rows [0,2103), last bin end1310h; imported stage-A bounded snapshot only',
  certification='No invented tolerances. Existing independently certified physical records only; numeric finite is not certification',
  no_future_actual_inference=True,Train_privileged_interface='Only conditional if independent teacher, temporal crossfit, same-information nonphysical control all proven',
  no_SOH_residual_without_A_bridge=True,no_old_zero_or_sign_retry=True,
  loss_scale_V=S,formal_lambda=None,formal_GPU_budget=0,
  GPU_rule='Conditional on all B6 critical gates; with missing state/parameters no real residual and no GPU even if synthetic gradient passes',
  no_Validation_for_selection=True,Test=False,Raw_reparse=False,optimizer_steps=0)
 js('configs/FROZEN_PROTOCOL.json',config);md('configs/FROZEN_PROTOCOL.sha256',sha(P/'configs/FROZEN_PROTOCOL.json'))
 md('.gitignore','__pycache__/\n*.pyc\nruntime/\noutputs/GIT_RECEIPT.local.json')
 md('.gitattributes','* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.png binary\n*.pdf binary')
 print('FROZEN',sha(P/'configs/FROZEN_PROTOCOL.json'))

def run():
 assert not (P/'outputs/DECISION.json').exists(),'Already completed'
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text());assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 for f,h in cfg['source_sha256'].items():assert sha(ROOT/f)==h,f
 for f,h in cfg['execution_code_sha256'].items():assert sha(P/'scripts'/f)==h,f
 assert sha(PDF)==cfg['local_physics_pdf_sha256']
 # From this point repository writes outside this new phase are denied.
 allowed={(ROOT/f).resolve() for f in cfg['source_sha256']}
 def guard(event,args):
  if event!='open' or isinstance(args[0],int):return
  q=Path(os.fsdecode(args[0])).resolve();m=args[1];flags=args[2]
  writing=(isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if q.is_relative_to(ROOT.parent) and not q.is_relative_to(P):
   if writing or q not in allowed:raise PermissionError('Unregistered historical access: '+str(q))
 sys.addaudithook(guard)
 b=pd.read_csv(A/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv',nrows=2103,float_precision='round_trip')
 assert len(b)==2103 and b.bin_end_ophrs2_h.max()==1310
 cs('inputs/TRAIN_BIN_CONDITIONS.csv',b)
 m2=json.loads((ROOT/'Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/outputs/M2_SUPPORT_SUMMARY.json').read_text())
 domain=pd.read_csv(ROOT/'Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/outputs/M2_STAGE1_DOMAIN_COUNTS.csv')
 assert domain.holds.sum()==m2['unique_holds_ge120s']==7273 and domain.relative_quiet.sum()==8
 assert m2['independent_physically_certified_dwells']==0
 cs('inputs/INHERITED_RAW_DWELL_DOMAIN_COUNTS.csv',domain);js('inputs/INHERITED_RAW_DWELL_QUALIFICATION.json',m2)
 stats=[]
 for name in b:
  if name.startswith('actual_'):
   v=b[name].to_numpy();stats.append(dict(variable=name,train_bins=len(v),finite_bins=int(np.isfinite(v).sum()),observed_min=float(np.nanmin(v)),observed_max=float(np.nanmax(v)),
    historical_observed=True,future_known_at_origin=False,interface_state_identity=False))
 cs('outputs/TRAIN_VARIABLE_COVERAGE.csv',pd.DataFrame(stats))
 eligibility=[]
 for candidate in cfg['candidates']:
  q=b[['bin_start_ophrs2_h','bin_end_ophrs2_h']].copy();q['candidate']=candidate
  q['reaction_interface_state_observed']=False;q['independently_certified_domain']=False;q['measurement_operator_certified']=False
  q['physical_parameters_identified']=False;q['physics_residual_dimensionless']=np.nan;q['physics_loss_dimensionless']=np.nan
  q['reason']='No interface activities / membrane state / certified observation operator'+('; required polarization parameters not identified' if candidate.startswith('C2') else '; E0(T) and water phase / standard state not registered for individual measurements')
  eligibility.append(q)
 cs('outputs/REAL_TRAIN_PHYSICS_DOMAIN_AND_REJECTIONS.csv',pd.concat(eligibility,ignore_index=True))
 # Independent analytical, finite-difference and autograd checks (synthetic only).
 torch.set_num_threads(1);dtype=torch.float64;torch.manual_seed(42)
 fixture={'T_K':348.15,'a_H2':1.5,'a_O2':.2,'a_H2O':.5,'E0_V':1.18,'b_V':.035,'log_j0':-7.,'ASR_ohm_cm2':.12,'B_V':.015,'jlim_A_cm2':2.0}
 js('configs/SYNTHETIC_FIXTURE_NOT_REAL_PARAMETERS.json',fixture)
 T=torch.tensor(fixture['T_K'],dtype=dtype,requires_grad=True);activities=[torch.tensor(fixture[k],dtype=dtype) for k in ['a_H2','a_O2','a_H2O']]
 E=nernst(T,*activities,torch.tensor(1.18,dtype=dtype));dE=torch.autograd.grad(E,T)[0]
 analytic=(E.detach()-1.18)/T.detach();assert torch.allclose(dE,analytic,atol=1e-15,rtol=0)
 u=torch.tensor([0.,4.],dtype=dtype,requires_grad=True);origin=torch.tensor([.7,.7],dtype=dtype)
 loss=upper_loss(u,origin,E.detach());g=torch.autograd.grad(loss,u)[0]
 expected=2*torch.relu((origin+S*u.detach()-E.detach())/S)/len(u)
 assert torch.allclose(g,expected,atol=1e-13,rtol=0)
 def finite(f,x,eps=1e-6):
  return torch.stack([(f(x.detach()+torch.eye(len(x),dtype=dtype)[i]*eps)-f(x.detach()-torch.eye(len(x),dtype=dtype)[i]*eps))/(2*eps) for i in range(len(x))])
 fd=finite(lambda x:upper_loss(x,origin,E.detach()),u);assert torch.allclose(g,fd,atol=1e-8,rtol=1e-8)
 j=torch.tensor([.2,.5,.8,1.,1.2,1.5],dtype=dtype)
 theta=torch.tensor([1.18,.035,-7.,.12,.015,2.],dtype=dtype,requires_grad=True)
 fn=lambda th:polarization(j,*th)
 jac=torch.autograd.functional.jacobian(fn,theta);sing=torch.linalg.svdvals(jac);rank=int(torch.linalg.matrix_rank(jac))
 shift=theta.detach().clone();shift[0]-=shift[1]*.7;shift[2]+=.7
 symmetry=float(torch.max(abs(fn(theta.detach())-fn(shift))));assert symmetry<5e-16 and rank<6
 teacher=fn(theta.detach());u2=torch.zeros(len(j),dtype=dtype,requires_grad=True);v0=torch.full_like(u2,.7)
 l2=amplitude_loss(u2,v0,teacher);g2=torch.autograd.grad(l2,u2)[0]
 expected2=2*(v0+S*u2.detach()-teacher)/S/len(j);fd2=finite(lambda x:amplitude_loss(x,v0,teacher),u2)
 assert torch.allclose(g2,expected2,atol=1e-13,rtol=0) and torch.allclose(g2,fd2,atol=1e-8,rtol=1e-8)
 # Parameter sensitivity and aggregation are mathematical probes, not parameter fitting.
 lowj=torch.tensor([.2,1.2],dtype=dtype);jensen=float(polarization(lowj,*theta.detach()).mean()-polarization(lowj.mean().reshape(1),*theta.detach())[0])
 math=dict(synthetic_only=True,nernst_fixture_E_V=float(E.detach()),nernst_dEdT_V_K=float(dE),
  upper_loss=float(loss),upper_output_gradient=g.tolist(),upper_finite_difference_max_error=float(torch.max(abs(fd-g))),
  amplitude_loss=float(l2),amplitude_output_gradient=g2.tolist(),amplitude_fd_max_error=float(torch.max(abs(fd2-g2))),
  fixture_jacobian_singular_values=sing.tolist(),fixture_parameter_count=6,fixture_jacobian_rank=rank,
  E_logj0_symmetry_max_difference_V=symmetry,nonlinear_aggregation_difference_V=jensen,
  no_real_parameters_fitted=True,RealHyFC_physical_gain_proven=False)
 js('outputs/SYNTHETIC_EQUATION_AND_GRADIENT_RESULTS.json',math)
 cs('outputs/SYNTHETIC_PARAMETER_SENSITIVITY.csv',pd.DataFrame(jac.detach().numpy(),columns=['E','b','log_j0','ASR','B','jlim']).assign(j_A_cm2=j.numpy()))
 # One real CPU gradient contribution probe, with no optimizer and no new forward information.
 start=time.perf_counter();hp=json.loads((ROOT/'Phase_TFT_P1S96_HealthAware_Fusion/configs/FROZEN_PROTOCOL.json').read_text())['H']['hyperparameters']
 q=ROOT/'Phase_TFT_P1S96_HealthAware_Fusion/models/health_fusion.py';spec=importlib.util.spec_from_file_location('_b_original_import',q);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 model=mod.OriginalF2(hidden_size=hp['hidden_size'],hidden_continuous_size=hp['hidden_continuous_size'],attention_head_size=4,lstm_layers=1,dropout=hp['dropout'])
 ck=torch.load(ROOT/'Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F2-Seed42_best.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(ck['state_dict'],strict=True);model.eval()
 fp=json.loads((ROOT/'Phase_TFT_P1S96_HealthAware_Fusion/inputs/INPUT_FINGERPRINTS.json').read_text())
 with np.load(ROOT/'Phase_TFT_P1S96_HealthAware_Fusion/inputs/MATCHED_INPUTS.npz',allow_pickle=False) as z:
  x=z['train_x'];y=z['train_y'];origins=z['train_origins']
  # Do not index or decode any validation arrays from the ZIP archive.
  assert array_sha(x)==fp['train_x']
  assert array_sha(y)==fp['train_y']
  assert array_sha(origins)==fp['train_origins']
  assert len(x)==870 and (origins<2103).all()
 out=model.predict_delta(torch.tensor(x[:8]));data_loss=((out-torch.tensor(y[:8])/S)**2).mean()
 gd=torch.autograd.grad(data_loss,tuple(model.parameters()),retain_graph=True,allow_unused=True)
 gated=empty_domain_loss(out);gp=torch.autograd.grad(gated,tuple(model.parameters()),allow_unused=True)
 data_norm=float(torch.sqrt(sum(v.square().sum() for v in gd if v is not None)))
 phy_norm=float(torch.sqrt(sum(v.square().sum() for v in gp if v is not None)))
 assert data_norm>0 and phy_norm==0
 probe=dict(device='cpu',real_train_origins=origins[:8].tolist(),forecast_outputs=192,optimizer_steps=0,
  data_MSE_dimensionless=float(data_loss.detach()),data_gradient_L2=data_norm,
  fail_closed_empty_domain_loss=0.,fail_closed_empty_domain_gradient_L2=phy_norm,
  real_physics_residual_evaluated=False,real_physics_loss=None,real_physics_activation_fraction=None,
  weighted_real_physics_to_data_ratio=None,eligible_train_domain_records=0,
  note='Zero gated contribution is engineering empty-domain behavior, NOT a measured zero physical residual or measured inactivity',
  inference_future_actual_inputs=False,cpu_elapsed_s=time.perf_counter()-start,
  loaded_checkpoint_source='Frozen original H-F2 Seed42; no weight updates')
 js('outputs/REAL_TRAIN_CPU_GRADIENT_CONTRIBUTION.json',probe)
 summary=dict(Train_bins=2103,Train_legal_origins=1380,Train_last_bin_end_h=1310,
  Train_j_near_1_bins=int(b.actual_current_density_A_cm2_mean.between(.95,1.05).sum()),
  Train_RH_anode_mean_gt100_bins=int((b.actual_anode_inlet_rh_pct_mean>100).sum()),
  historical_T_RH_pressure_present=True,interface_reactant_activity_present=False,membrane_state_present=False,
  independent_polarization_parameter_identity=False,certified_observation_operator=False,
  inherited_raw_holds=7273,inherited_quiet_holds=8,inherited_evidence_domain_quiet_holds=0,independently_certified_raw_dwells=0,
  pure_physical_parameter_fits=0,sources_unchanged=all(sha(ROOT/f)==h for f,h in cfg['source_sha256'].items()))
 js('outputs/REAL_TRAIN_DATA_QUALIFICATION.json',summary)
 decision=dict(status='PHYSICS_ROUTE_NOT_QUALIFIED',candidates_reviewed=cfg['candidates'],certified_candidates=0,
  candidate_C1='Missing certified reaction state / target-time state; cannot calculate defensible Nernst envelope',
  candidate_C2='Quantitative polarization parameters and common state not identified; no independent Train-only physical teacher',
  synthetic_equation_and_gradient_checks='PASS',real_admissible_parameter_gradient=False,
  synthetic_gradient_not_real_gain=True,formal_GPU_runs=0,small_GPU_probes=0,
  GPU_not_started_reason='Physical state / parameter / measurement gates fail before GPU; CPU engineering pass cannot override',
  actual_physics_loss=None,activation_fraction=None,physical_consistency_improvement_percent=None,prediction_accuracy_improvement_percent=None,
  formal_lambda=None,extra_model_parameters=0,optimizer_steps=0,no_future_inference_inputs=True,Test=False,
  recommendation='First identify a pure electrochemical model with independent synchronized Train measurements, then reassess training-only physics supervision',
  stopped=True,protocol_sha256=sha(P/'configs/FROZEN_PROTOCOL.json'))
 js('outputs/DECISION.json',decision)
 print(json.dumps(summary,ensure_ascii=False));print(json.dumps(math,ensure_ascii=False));print(json.dumps(probe,ensure_ascii=False));print(json.dumps(decision,ensure_ascii=False))

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
