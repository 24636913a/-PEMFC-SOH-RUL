"""Resume only the previously blocked CPU probe; equations/data checks are not rerun."""
import hashlib,importlib.util,json,sys,time,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from physics_math import S,empty_domain_loss
from run_qualification import P,ROOT,sha,array_sha,js,md
DEPENDENCY='Phase_TFT_FullFeature_CausalFilter_Comparison/configs/PREREGISTERED_E_COMPARISON.yaml'
if sys.argv[1]=='freeze':
 assert not (P/'configs/CPU_PROBE_DEPENDENCY_AMENDMENT.json').exists()
 js('configs/CPU_PROBE_DEPENDENCY_AMENDMENT.json',dict(parent_protocol_sha256=sha(P/'configs/FROZEN_PROTOCOL.json'),
  resume_code_sha256=sha(__file__),static_dependency_sha256={DEPENDENCY:sha(ROOT/DEPENDENCY)},
  completed_artifact_sha256={f:sha(P/f) for f in ['outputs/SYNTHETIC_EQUATION_AND_GRADIENT_RESULTS.json','inputs/TRAIN_BIN_CONDITIONS.csv','outputs/REAL_TRAIN_PHYSICS_DOMAIN_AND_REJECTIONS.csv']},
  reason='Original read guard correctly denied an unlisted static feature-schema YAML dependency before model construction/forward',
  science_rules_changed=False,original_freeze_preserved=True,previous_real_model_forwards=0,optimizer_steps=0))
 js('outputs/PREPROBE_ACCESS_FAILURE.json',dict(error='Historical read not allowlisted: '+DEPENDENCY,
  completed='Data coverage and synthetic gradient/identifiability checks',failed='CPU model import',
  forward_executed=False,weight_updates=0,repair='Register exact read-only static schema dependency and resume CPU step only'))
 print('CPU static dependency amendment frozen')
else:
 assert not (P/'outputs/DECISION.json').exists()
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text());amend=json.loads((P/'configs/CPU_PROBE_DEPENDENCY_AMENDMENT.json').read_text())
 assert amend['resume_code_sha256']==sha(__file__) and amend['parent_protocol_sha256']==sha(P/'configs/FROZEN_PROTOCOL.json')
 for f,h in amend['completed_artifact_sha256'].items():assert sha(P/f)==h
 sources=dict(cfg['source_sha256']);sources.update(amend['static_dependency_sha256'])
 for f,h in sources.items():assert sha(ROOT/f)==h,f
 allowed={(ROOT/f).resolve() for f in sources}
 def guard(event,args):
  if event!='open' or isinstance(args[0],int):return
  q=Path(os.fsdecode(args[0])).resolve();m=args[1];fl=args[2]
  writing=(isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(fl,int) and fl&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if q.is_relative_to(ROOT.parent) and not q.is_relative_to(P) and (writing or q not in allowed):raise PermissionError(str(q))
 sys.addaudithook(guard);torch.set_num_threads(1);start=time.perf_counter()
 hp=json.loads((ROOT/'Phase_TFT_P1S96_HealthAware_Fusion/configs/FROZEN_PROTOCOL.json').read_text())['H']['hyperparameters']
 q=ROOT/'Phase_TFT_P1S96_HealthAware_Fusion/models/health_fusion.py';spec=importlib.util.spec_from_file_location('_b_original_import',q);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 model=mod.OriginalF2(hidden_size=hp['hidden_size'],hidden_continuous_size=hp['hidden_continuous_size'],attention_head_size=4,lstm_layers=1,dropout=hp['dropout'])
 ck=torch.load(ROOT/'Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F2-Seed42_best.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(ck['state_dict'],strict=True);model.eval()
 fp=json.loads((ROOT/'Phase_TFT_P1S96_HealthAware_Fusion/inputs/INPUT_FINGERPRINTS.json').read_text())
 with np.load(ROOT/'Phase_TFT_P1S96_HealthAware_Fusion/inputs/MATCHED_INPUTS.npz',allow_pickle=False) as z:
  x=z['train_x'];y=z['train_y'];origins=z['train_origins']
 for key,v in [('train_x',x),('train_y',y),('train_origins',origins)]:assert array_sha(v)==fp[key]
 assert len(x)==870 and (origins<2103).all()
 out=model.predict_delta(torch.tensor(x[:8]));data_loss=((out-torch.tensor(y[:8])/S)**2).mean()
 gd=torch.autograd.grad(data_loss,tuple(model.parameters()),retain_graph=True,allow_unused=True)
 gated=empty_domain_loss(out);gp=torch.autograd.grad(gated,tuple(model.parameters()),allow_unused=True)
 dn=float(torch.sqrt(sum(v.square().sum() for v in gd if v is not None)));pn=float(torch.sqrt(sum(v.square().sum() for v in gp if v is not None)))
 assert dn>0 and pn==0
 probe=dict(device='cpu',real_train_origins=origins[:8].tolist(),forecast_outputs=192,optimizer_steps=0,
  data_MSE_dimensionless=float(data_loss.detach()),data_gradient_L2=dn,fail_closed_empty_domain_loss=0.,fail_closed_empty_domain_gradient_L2=pn,
  real_physics_residual_evaluated=False,real_physics_loss=None,real_physics_activation_fraction=None,
  weighted_real_physics_to_data_ratio=None,eligible_train_domain_records=0,
  note='Zero gated contribution is engineering empty-domain behavior, NOT a measured physical residual/inactivity',
  inference_future_actual_inputs=False,cpu_elapsed_s=time.perf_counter()-start,loaded_checkpoint_source='Original H-F2 Seed42; no weight updates')
 js('outputs/REAL_TRAIN_CPU_GRADIENT_CONTRIBUTION.json',probe)
 b=pd.read_csv(P/'inputs/TRAIN_BIN_CONDITIONS.csv');m2=json.loads((P/'inputs/INHERITED_RAW_DWELL_QUALIFICATION.json').read_text())
 summary=dict(Train_bins=2103,Train_legal_origins=1380,Train_last_bin_end_h=1310,
  Train_j_near_1_bins=int(b.actual_current_density_A_cm2_mean.between(.95,1.05).sum()),
  Train_RH_anode_mean_gt100_bins=int((b.actual_anode_inlet_rh_pct_mean>100).sum()),
  historical_T_RH_pressure_present=True,interface_reactant_activity_present=False,membrane_state_present=False,
  independent_polarization_parameter_identity=False,certified_observation_operator=False,
  inherited_raw_holds=m2['unique_holds_ge120s'],inherited_quiet_holds=m2['relative_quiet_candidates'],
  inherited_evidence_domain_quiet_holds=m2['relative_quiet_evidence_domain'],independently_certified_raw_dwells=0,
  pure_physical_parameter_fits=0,sources_unchanged=all(sha(ROOT/f)==h for f,h in sources.items()))
 js('outputs/REAL_TRAIN_DATA_QUALIFICATION.json',summary)
 decision=dict(status='PHYSICS_ROUTE_NOT_QUALIFIED',candidates_reviewed=cfg['candidates'],certified_candidates=0,
  candidate_C1='Missing certified reaction state / target-time state; no defensible Nernst envelope',
  candidate_C2='Polarization parameters and common state not identified; no independent physical teacher',
  synthetic_equation_and_gradient_checks='PASS',real_admissible_parameter_gradient=False,
  synthetic_gradient_not_real_gain=True,formal_GPU_runs=0,small_GPU_probes=0,
  GPU_not_started_reason='Physical state / parameter / measurement gates fail before GPU',
  actual_physics_loss=None,activation_fraction=None,physical_consistency_improvement_percent=None,prediction_accuracy_improvement_percent=None,
  formal_lambda=None,extra_model_parameters=0,optimizer_steps=0,no_future_inference_inputs=True,Test=False,
  recommendation='First identify a pure electrochemical model with independent synchronized Train measurements; reassess training-only supervision afterward',
  stopped=True,protocol_sha256=sha(P/'configs/FROZEN_PROTOCOL.json'),CPU_probe_dependency_amended=True)
 js('outputs/DECISION.json',decision)
 print(json.dumps(summary,ensure_ascii=False));print(json.dumps(probe,ensure_ascii=False));print(json.dumps(decision,ensure_ascii=False))
