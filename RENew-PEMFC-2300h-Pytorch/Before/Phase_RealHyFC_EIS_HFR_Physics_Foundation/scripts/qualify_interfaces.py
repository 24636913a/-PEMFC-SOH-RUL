"""Measurement identity, LFS object provenance, and no-Test Train interface audit."""
import hashlib,json,subprocess
import numpy as np
import pandas as pd
from common import P,PROJECT,REPO,BASE,sha,js,cs,md,firewall
N1=PROJECT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
def run():
 assert not (P/'outputs/RESEARCH_DECISION.json').exists(),'Completed qualification'
 # Git objects of exact allowed paths only. No git-lfs fetch/pull/checkout.
 manifest=pd.read_csv(P/'inputs/SOURCE_CONTENT_MANIFEST.csv')
 cmdinput=''.join('HEAD:'+p+'\n' for p in manifest.path).encode('utf-8')
 output=subprocess.check_output(['git','cat-file','--batch'],input=cmdinput,cwd=REPO)
 offset=0;lfs=[]
 for r in manifest.itertuples():
  end=output.index(b'\n',offset);h=output[offset:end].decode();offset=end+1
  if h.endswith(' missing'):lfs.append(dict(path=r.path,tracked=False,LFS_pointer_in_Git=False,oid_matches=None,size_matches=None));continue
  size=int(h.split()[-1]);blob=output[offset:offset+size];offset+=size+1
  pointer=blob.startswith(b'version https://git-lfs.github.com/spec/v1')
  if pointer:
   lines=blob.decode('ascii').splitlines();oid=next(x.split('sha256:')[1] for x in lines if x.startswith('oid '));expected=int(next(x.split()[1] for x in lines if x.startswith('size ')))
   lfs.append(dict(path=r.path,tracked=True,LFS_pointer_in_Git=True,oid_sha256=oid,expected_bytes=expected,oid_matches=oid==r.content_sha256,size_matches=expected==r.byte_size))
  else:lfs.append(dict(path=r.path,tracked=True,LFS_pointer_in_Git=False,oid_matches=hashlib.sha256(blob).hexdigest()==r.content_sha256,size_matches=len(blob)==r.byte_size))
 cs('outputs/GIT_LFS_OBJECT_PROVENANCE.csv',pd.DataFrame(lfs));assert all(r.get('oid_matches',True) is not False and r.get('size_matches',True) is not False for r in lfs)
 metadata=pd.read_csv(P/'outputs/EIS_MEASUREMENT_METADATA.csv')
 identity=[
  dict(pair='January nominal BoL vs April1150 nominal EIS',status='CONFIGURATION_CONDITIONAL',same_stack_verified=False,
   evidence='January10 positive cell channels;paper prior nominal10-cell setup and durability10-cell;April10 channels;no immutable stack ID',
   limitation='Matching hardware/current/instrument IDs are insufficient for a serial-level longitudinal claim'),
  dict(pair='Fresh BoL sensitivity vs durability nominal measurements',status='DIFFERENT_STACK_CONFIRMED',same_stack_verified=False,
   evidence='Source paper explicitly separates8-cell fresh sensitivity and10-cell durability;headers Oct/Nov2025 vs Jan/Apr2025',
   limitation='Per-file wiring/serial not exported; fresh background cannot calibrate past durability origins'),
  dict(pair='April1150 EIS vs1150 polarization reference',status='CONFIGURATION_CONDITIONAL',same_stack_verified=False,
   evidence='Nominal durability configuration supported;EIS April24/25 versus polarization April18',
   limitation='Different calendar event/current1.8 versus fixed j*=1.0, no precise durability OpHrs2/release'),
  dict(pair='EIS cell channels to U1-U11',status='STACK_IDENTITY_UNKNOWN',same_stack_verified=False,
   evidence='Actual instrument CH01-CH05/CH07-CH11 recorded;U6 excluded in polarization',
   limitation='Instrument channel labels/instrument unit IDs are not an independently supplied raw-voltage wiring map'),
  dict(pair='Spatial file00 measurement object',status='CONFIGURATION_CONDITIONAL',same_stack_verified=False,
   evidence='ReadMe states00 reference cell;paper states cell above CDDM and global reference voltage',
   limitation='Raw U-channel link and cable/de-embedding certificate absent'),
  dict(pair='Spatial file01-16 to exact physical zones',status='STACK_IDENTITY_UNKNOWN',same_stack_verified=False,
   evidence='Paper defines16 zones:7 inlet,2 middle,7 outlet;local-current transfer measurement',
   limitation='Figure-zone geography known;filename/instrument-channel-to-zone wiring not independently specified')]
 cs('outputs/EIS_STACK_IDENTITY_MATRIX.csv',pd.DataFrame(identity))
 mapping=[]
 for r in metadata.loc[metadata.role.eq('CORE_SPECTRUM')].itertuples():
  mapping.append(dict(source_path=r.source_path,export_index=r.export_index,instrument_channels=r.instrument_channels,
   measurement_object=r.measurement_object,physical_U_channel='UNKNOWN',filename_zone_to_paper_zone='NOT_VERIFIED',
   instrument_ids_are_stack_ids=False,area_header_cm2=r.area_cm2,
   apply280_to_zone=False,divide_composite_by_cellcount=False))
 cs('outputs/MEASUREMENT_CHANNEL_AND_UNIT_MAP.csv',pd.DataFrame(mapping))
 extra=[N1];protected=json.loads((P/'inputs/PROTECTED_HISTORY_SHA.json').read_text());extra += [PROJECT/f for f in protected]
 firewall(extra)
 # Use current corrected metadata-first firewall: no Test numerical record parsing.
 meta=pd.read_csv(N1,nrows=3007,usecols=['bin_start_ophrs2_h','bin_end_ophrs2_h','first_datetime','last_datetime','split','bin_valid_for_windows'])
 ids=np.flatnonzero(meta.bin_end_ophrs2_h.to_numpy()<1819.997738)
 assert np.array_equal(ids,np.arange(3005)) and not meta.iloc[ids]['split'].str.contains('test',case=False).any()
 cs('inputs/DEVELOPMENT_ROW_SELECTION_METADATA.csv',meta.assign(source_row=np.arange(len(meta)),retained=np.isin(np.arange(len(meta)),ids)))
 train=pd.read_csv(N1,nrows=2103,usecols=['bin_start_ophrs2_h','bin_end_ophrs2_h','first_datetime','last_datetime','split',
  'actual_current_density_A_cm2_mean','actual_current_density_A_cm2_std','actual_current_density_A_cm2_max','load_transition_count','zero_current_fraction',
  'actual_anode_inlet_temperature_C_mean','actual_cathode_inlet_temperature_C_mean','actual_anode_inlet_rh_pct_mean','actual_cathode_inlet_rh_pct_mean',
  'actual_anode_outlet_pressure_kPag_mean','actual_cathode_outlet_pressure_kPag_mean'])
 assert train.bin_end_ophrs2_h.max()==1310 and not train['split'].str.contains('test',case=False).any()
 cs('inputs/REAL_TRAIN_CONDITIONS_ONLY.csv',train)
 coverage=dict(Train_bins=2103,Train_max_actual_h=1310,mean_j_near_EIS_1_8_bins=int(train.actual_current_density_A_cm2_mean.between(1.75,1.85).sum()),
  mean_j_near_BoL_low0_18_bins=int(train.actual_current_density_A_cm2_mean.between(.13,.23).sum()),
  instantaneous_max_j_ge1_75_bins=int((train.actual_current_density_A_cm2_max>=1.75).sum()),
  mean_j_max=float(train.actual_current_density_A_cm2_mean.max()),RH_anode_mean_gt100_bins=int((train.actual_anode_inlet_rh_pct_mean>100).sum()),
  certified_EIS_to_Train_joint_state_bins=0,reference_parameter_release_OpHrs2='UNKNOWN',
  raw_EIS_measured_current_range_A=[float(metadata.loc[metadata.role.eq('CORE_SPECTRUM'),'current_A'].min()),float(metadata.loc[metadata.role.eq('CORE_SPECTRUM'),'current_A'].max())],
  Test_numeric_records=0,Test_metadata_records_for_boundary_only=1,physical_states=['interface activities','membrane hydration','cable-deembedding','same-stack temporal identity'])
 js('outputs/REAL_TRAIN_PHYSICAL_SUPPORT.json',coverage)
 # One conditional candidate: EIS-anchored ohmic component of a full physical-voltage residual.
 # The missing other overpotentials are NOT set to zero; no real teacher is fabricated.
 units=dict(ASR_unit='ohm cm2',j_unit='A/cm2',j_ASR_unit='V',scale_V=.13680318714138442)
 u=np.array([.1,-.2]);v0=.7;j=np.array([.2,.8]);R=.04;other=np.array([.65,.60]);scale=units['scale_V']
 residual=(v0+scale*u-(other-j*R))/scale;loss=float(np.mean(residual**2))
 analytical_u=2*residual/len(u);analytical_R=float(np.mean(2*residual*j/scale));eps=1e-6
 def fn(uu,rr):return np.mean(((v0+scale*uu-(other-j*rr))/scale)**2)
 numerical_u=np.array([(fn(u+np.eye(2)[k]*eps,R)-fn(u-np.eye(2)[k]*eps,R))/(2*eps) for k in range(2)])
 numerical_R=float((fn(u,R+eps)-fn(u,R-eps))/(2*eps))
 assert np.allclose(analytical_u,numerical_u,rtol=0,atol=1e-8) and abs(analytical_R-numerical_R)<1e-8
 js('outputs/SYNTHETIC_OHMIC_RESIDUAL_GRADIENT.json',dict(synthetic_only=True,**units,
  loss=loss,output_gradient=analytical_u.tolist(),ASR_gradient=analytical_R,numerical_output_gradient=numerical_u.tolist(),numerical_ASR_gradient=numerical_R,
  fixture_other_losses_registered_as_synthetic=True,no_measured_ASR_or_real_Train_physics_loss_used=True))
 domain=train[['bin_start_ophrs2_h','bin_end_ophrs2_h']].copy();domain['physical_domain_eligible']=False;domain['real_physics_loss']=np.nan
 domain['reason']='No qualified same-stack ASR/time/state;E_rev and activation/concentration loss unknown;future actual conditions unavailable'
 cs('outputs/REAL_TRAIN_PHYSICS_ELIGIBILITY_AND_FAILURES.csv',domain)
 gates=pd.DataFrame([dict(gate=g,status='FAIL',reason=r) for g,r in [
  ('Parameter identity','Measured HFR/ASR not certified;equivalent-circuit R0 is model-dependent;some boundary solutions'),
  ('Equation operating domain','Nominal sparse EIS versus dynamic whole-bin mean;internal state and deembedding unresolved'),
  ('Origin-known information','EIS lacks exact durability OpHrs2/release;future actual j/T/RH/P are not forecast inputs'),
  ('Real physical residual','ASR alone cannot determine E_rev/activation/concentration;not computable,NA'),
  ('Actual model gradient contribution','No valid real residual reachesF2/shared parameters;synthetic nonzero does not certify'),
  ('Matched control readiness','Additional-info physical teacher not established;no instantiated fair training design')]])
 cs('outputs/CPU_PHYSICS_QUALIFICATION_GATES.csv',gates)
 d=pd.read_csv(P/'outputs/EQUIVALENT_CIRCUIT_DIAGNOSTICS.csv');details=json.loads((P/'outputs/EQUIVALENT_CIRCUIT_FULL_PARAMETERS.json').read_text())
 boundary=sum(any(v!=0 for v in item['active_bounds']) for item in details.values())
 js('outputs/PARAMETER_IDENTIFIABILITY_SUMMARY.json',dict(fits=33,parameters_per_fit=7,fixed_multistarts=3,
  fit_with_active_bounds=boundary,relative_complex_RMSE_percent_min=float(d.relative_complex_RMSE_percent.min()),relative_complex_RMSE_percent_max=float(d.relative_complex_RMSE_percent.max()),
  maximum_logparameter_jac_condition=float(d.jac_condition.max()),profile_sources=5,profiles_are_confidence_intervals=False,
  measured_HFR_qualified=0,independently_qualified_ASR=0,diagnostic_R0_is_electrochemical_damage=False))
 js('outputs/RESEARCH_DECISION.json',dict(source_status='REAL_EIS_AVAILABLE',EIS_level='EIS_ONLY_DESCRIPTIVE',
  nominal_BoL_vs1150_stack_level='CONFIGURATION_CONDITIONAL',same_stack_verified=False,
  fresh_sensitivity_vs_durability_level='DIFFERENT_STACK_CONFIRMED',
  physical_parameter_level='NO_CERTIFIED_HFR_OR_ASR;MODEL_DEPENDENT_R0_DIAGNOSTICS_AVAILABLE',
  physics_training_level='PHYSICS_TRAINING_NOT_QUALIFIED',SOH_bridge_qualified=False,
  core_spectra=65,real_points=3620,equivalent_circuit_fits=33,measured_HFR_values=0,certified_ASR_values=0,
  GPU_runs=0,actual_Train_physics_loss=None,actual_gradient_contribution=None,new_prediction_accuracy=None,
  no_EIS_features_concatenated_to_TFT=True,no_future_actual_input=True,no_Test_numeric=True,RUL_trained=False,
  useful_result='Real spectra,object/condition distinctions,series-R0 fitting diagnostics and identity/artifact blockers;not a validated SOH/physics enhancement',
  stopped_after_this_stage=True,history_unchanged=all(sha(PROJECT/f)==h for f,h in protected.items())))
 print(json.dumps(coverage,ensure_ascii=False));print('PHYSICS_TRAINING_NOT_QUALIFIED; no GPU started')
if __name__=='__main__':run()
