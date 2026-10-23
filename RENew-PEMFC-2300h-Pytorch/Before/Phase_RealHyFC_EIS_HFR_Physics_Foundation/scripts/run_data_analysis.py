"""Allowlisted real spectra and Development polarization; fixed CPU diagnostics."""
import csv,json,re,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from common import P,PROJECT,REPO,BASE,sha,js,cs,md,safe_raw,firewall
from eis_math import header,parse,crosses,split_scans,fit,profile
OLD=[
 'Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/polarization_configuration_compatibility.md',
 'Phase_SOH_LongHorizon_Future_Forecasting/inputs/DEVELOPMENT_SOH_REFERENCES.csv',
 'Phase_TFT_DynamicVoltage_to_ReferenceSOH/outputs/FINAL_DECISION.json',
 'Phase_Physics_Informed_Qualification_and_Ablation/outputs/DECISION.json',
 'Phase_Development_ReadRule_Independent_Consistency_Audit/outputs/FINAL_AUDIT.json',
 'Phase_TFT_P1S96_HealthAware_Fusion/outputs/RESEARCH_DECISION.json',
 'Phase_P1S_96h_ThreeFold_Comparison/outputs/FINAL_RESEARCH_JUDGMENT.json',
]
def freeze():
 assert not (P/'configs/FROZEN_PROTOCOL.json').exists()
 allowed=json.loads((P/'configs/SOURCE_PATH_WHITELIST.json').read_text(encoding='utf-8'));sources=[];lfs=[]
 for r in allowed['exact_paths']:
  path=safe_raw(REPO/r['path'])
  with path.open('rb') as stream:h=stream.read(200)
  pointer=h.startswith(b'version https://git-lfs.github.com/spec/v1')
  sources.append(dict(**r,byte_size=path.stat().st_size,content_sha256=sha(path),LFS_pointer=pointer))
  lfs.append(dict(path=r['path'],local_bytes=path.stat().st_size,pointer=pointer,status='SOURCE_CONTENT_UNAVAILABLE' if pointer else 'REAL_LOCAL_CONTENT'))
 cs('inputs/SOURCE_CONTENT_MANIFEST.csv',pd.DataFrame(sources));cs('outputs/GIT_LFS_FILE_STATUS.csv',pd.DataFrame(lfs))
 js('outputs/GIT_LFS_STATUS.json',dict(version=subprocess.check_output(['git','lfs','version'],text=True).strip(),
  permitted_files=len(sources),local_real_files=sum(not r['LFS_pointer'] for r in sources),pointer_files=sum(r['LFS_pointer'] for r in sources),
  fetch_executed=False,pull_executed=False,whole_archive_download=False,reason='All allowlisted files already contain real local content'))
 assert not any(r['LFS_pointer'] for r in sources),'Exact-path retrieval must be performed before numerical analysis'
 protected={}
 for phase in ['Phase_TFT_DynamicVoltage_to_ReferenceSOH','Phase_Physics_Informed_Qualification_and_Ablation','Phase_Development_ReadRule_Independent_Consistency_Audit']:
  for f in (PROJECT/phase).rglob('*'):
   if f.is_file() and 'runtime' not in f.relative_to(PROJECT/phase).parts and f.name!='GIT_RECEIPT.local.json':protected[str(f.relative_to(PROJECT)).replace('\\','/')]=sha(f)
 for x in OLD:protected[x]=sha(PROJECT/x)
 js('inputs/PROTECTED_HISTORY_SHA.json',protected)
 js('configs/FROZEN_PROTOCOL.json',dict(base_commit='1af4176283c9bdeb0350071bc142919dddafc68a',
  source_manifest_sha256=sha(P/'inputs/SOURCE_CONTENT_MANIFEST.csv'),source_whitelist_sha256=sha(P/'configs/SOURCE_PATH_WHITELIST.json'),
  code_sha256={f.name:sha(f) for f in (P/'scripts').glob('*.py')},
  scope='65 core spectra full numerical parsing;201 fresh-BoL sensitivity files header-only;7 exact Development nominal curves',
  HFR_rule='Highest-frequency real value is never HFR. Report observed >=1000Hz sign-crossing intercepts as apparent only; parasitics not certified',
  circuit='One literature-defined R0 + R1/(1+(i*w*tau1)^alpha1) + R2/(1+(i*w*tau2)^alpha2), log-resistance/log-time parameterization',
  fit_frequency='Descending nonduplicate branch,0.1<=f<=1026Hz; published single-cell artifact exclusion; no new model or band search',
  fit_role='Effective circuit diagnostics, not independent identification of irreversible physical parameters or measured HFR',
  fixed_initial_starts=3,max_nfev_per_start=2000,objective='Unchanged relative complex least squares, divided by measured |Z|',
  CPE_alpha_bounds=[.1,1.],bound_role='Broad numerical parameter domain, not measured RealHyFC parameter values',
  profile_sources='Filename01 at BoL low/high load and1150 high load;spatial reference00 at BoL/1150',profile_factors=[.5,.75,.9,1.,1.1,1.25,1.5],
  profile_interpretation='Nuisance-optimized sensitivity, no statistical confidence interval / independent repeated lifetime claims',
  polarization='Descriptive stable-setpoint middle50 dwell >=120 records;keep hysteresis; frozen referenceSOH imported only; no reference-current selection',
  identity='Unknown serial/wiring remains UNKNOWN; instrument unit IDs do not identify a stack',
  Test_numeric_access=False,old_models_modified=False,no_automatic_RUL=True))
 md('configs/FROZEN_PROTOCOL.sha256',sha(P/'configs/FROZEN_PROTOCOL.json'))
 md('.gitattributes','* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.png binary\n*.pdf binary')
 print('FROZEN real source identities',len(sources),flush=True)

def run():
 assert not (P/'outputs/CORE_ANALYSIS_COMPLETE.json').exists(),'Analysis completed; no duplicate fitting'
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text());assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 manifest=pd.read_csv(P/'inputs/SOURCE_CONTENT_MANIFEST.csv');assert sha(P/'inputs/SOURCE_CONTENT_MANIFEST.csv')==cfg['source_manifest_sha256']
 for f,h in cfg['code_sha256'].items():assert sha(P/'scripts'/f)==h
 for r in manifest.itertuples():assert sha(safe_raw(REPO/r.path))==r.content_sha256,r.path
 protected=json.loads((P/'inputs/PROTECTED_HISTORY_SHA.json').read_text());assert all(sha(PROJECT/f)==h for f,h in protected.items())
 firewall([PROJECT/f for f in protected])
 inventory=[];metadata=[];points=[];quality=[];intercepts=[];fits=[];fitpoints=[];profiles=[];fullfitdetails={};start=time.perf_counter()
 for n,r in enumerate(manifest.itertuples()):
  path=REPO/r.path;role=r.role;relative=str(path.relative_to(BASE)).replace('\\','/')
  if role=='FRESH_BOL_SENSITIVITY_HEADER_ONLY':
   # Read only until the numerical section marker, never fit fresh sensitivity as ageing data.
   texts=[]
   with path.open('r',encoding='ascii') as stream:
    for line in stream:
     texts.append(line)
     if 'Impedance samples' in line:break
   info=header(''.join(texts));metadata.append(dict(source_path=r.path,role=role,**info))
   inventory.append(dict(path=r.path,type='EIS_SENSITIVITY',read_status='HEADER_ONLY_REAL_CONTENT',byte_size=r.byte_size,sha256=r.content_sha256,
    calendar_start=info['calendar_start'],physical_stack='Separate8-cell fresh sensitivity stack by paper; exact file binding conditional',development_status='INDEPENDENT_FRESH_BACKGROUND_NOT_CHRONOLOGICAL_TRAIN',
    precise_OpHrs2='UNKNOWN',actual_temperature='NOT_CERTIFIED',humidity='Not in spectrum',pressure='Not in spectrum',columns='Numerical table not parsed',
    cell_or_zone='Reference/zone channel, exact wiring unknown'))
   continue
  if role!='CORE_SPECTRUM':continue
  text=path.read_text(encoding='ascii');info,f,z,q,t,v,i,fmt,names=parse(text)
  spatial='spatiallyResolvedEIS' in r.path;idx=int(re.search(r'(\d{2})\.txt$',path.name).group(1))
  family=('Spatial' if spatial else 'Cell')+('_BoL_Jan2025' if '/BoL' in r.path else '_1150nom_Apr2025')
  load='low_0.18' if '10A_2-5A' in r.path else 'high_1.8'
  obj='REFERENCE_SINGLE_CELL' if spatial and idx==0 else ('LOCAL_CURRENT_VOLTAGE_TRANSFER_ZONE' if spatial else ('FIVE_CELL_REFERENCE_COMPOSITE' if idx==0 else 'SINGLE_CELL_CHANNEL'))
  m=dict(source_path=r.path,relative_path=relative,family=family,role=role,export_index=idx,measurement_object=obj,nominal_load=load,format=fmt,**info)
  metadata.append(m)
  inventory.append(dict(path=r.path,type='SPATIALLY_RESOLVED_EIS' if spatial else 'CELL_EIS',read_status='REAL_SPECTRUM_PARSED',byte_size=r.byte_size,sha256=r.content_sha256,
   calendar_start=info['calendar_start'],physical_stack='10-cell configuration supported; longitudinal serial binding UNKNOWN',development_status='DEVELOPMENT_NOMINAL_DIAGNOSTIC_CONDITIONAL',
   precise_OpHrs2='UNKNOWN',actual_temperature='NOT_CERTIFIED',humidity='Not in spectrum',pressure='Not in spectrum',columns='|'.join(names),cell_or_zone=obj))
  peak=int(np.argmax(f));dup=len(f)-len(np.unique(f));repeat=[]
  for freq in sorted(set(f[:peak])&set(f[peak+1:])):
   pair=np.flatnonzero(f==freq);repeat.append(abs(z[pair[0]]-z[pair[-1]])/max(abs(z[pair[0]]),1e-20))
  quality.append(dict(source_path=r.path,family=family,object=obj,points=len(f),unique_frequencies=len(np.unique(f)),f_min_Hz=float(f.min()),f_max_Hz=float(f.max()),
   nonfinite=int((~np.isfinite(z)).sum()),duplicate_frequencies=dup,duplicate_role='Initial up-scan then descending scan, not independent replicate cells',
   significance_min=float(q.min()),negative_real_points=int((z.real<0).sum()),positive_imag_points=int((z.imag>0).sum()),negative_imag_points=int((z.imag<0).sum()),
   max_intrascan_revisit_relative_difference=float(max(repeat)) if repeat else np.nan,
   header_Columns_declared7_actual8=True,phase_convention='Literal export angle;Z=modulus*exp(i*phase);rectangular preserved; no sign flip',
   highest_frequency_real_ohm=float(z[peak].real),highest_frequency_imag_ohm=float(z[peak].imag),highest_frequency_is_HFR=False))
  for k in range(len(f)):points.append(dict(source_path=r.path,family=family,object=obj,export_index=idx,point=k+1,frequency_Hz=float(f[k]),Z_real_ohm=float(z[k].real),Z_imag_ohm=float(z[k].imag),
   magnitude_ohm=float(abs(z[k])),phase_deg=float(np.angle(z[k],deg=True)),significance=float(q[k]),elapsed_s=float(t[k]),voltage_export_V=float(v[k]),current_export_A=float(i[k])))
  for cross in crosses(f,z,q):intercepts.append(dict(source_path=r.path,object=obj,**cross))
  if not spatial or idx==0:
   fitted=fit(f,z)
   if fitted:
    summary,ff,zz,zfit,residual,opts=fitted
    record=dict(source_path=r.path,family=family,object=obj,load=load,**{k:x for k,x in summary.items() if k not in ['parameters','singular_values','active_bounds']})
    record['ASR_effective_model_candidate_ohm_cm2']=summary['R0_effective_model_ohm']*280 if obj in ['SINGLE_CELL_CHANNEL','REFERENCE_SINGLE_CELL'] else np.nan
    record['ASR_certified']=False;fits.append(record);fullfitdetails[r.path]=summary
    for h in range(len(ff)):fitpoints.append(dict(source_path=r.path,frequency_Hz=float(ff[h]),actual_real_ohm=float(zz[h].real),actual_imag_ohm=float(zz[h].imag),fitted_real_ohm=float(zfit[h].real),fitted_imag_ohm=float(zfit[h].imag)))
    if idx in ([0] if spatial else [1]):
     for item in profile(fitted):profiles.append(dict(source_path=r.path,**item))
  print('EIS',relative,'points',len(f),'object',obj,flush=True)
 cs('outputs/EIS_NORMALIZED_POINTS.csv',pd.DataFrame(points));cs('outputs/EIS_MEASUREMENT_METADATA.csv',pd.DataFrame(metadata));cs('outputs/EIS_DATA_QUALITY.csv',pd.DataFrame(quality))
 cs('outputs/APPARENT_HIGH_FREQUENCY_INTERCEPTS_NOT_HFR.csv',pd.DataFrame(intercepts));cs('outputs/EQUIVALENT_CIRCUIT_DIAGNOSTICS.csv',pd.DataFrame(fits));cs('outputs/EQUIVALENT_CIRCUIT_POINT_RESIDUALS.csv',pd.DataFrame(fitpoints))
 cs('outputs/R0_PROFILE_SENSITIVITY.csv',pd.DataFrame(profiles));js('outputs/EQUIVALENT_CIRCUIT_FULL_PARAMETERS.json',fullfitdetails)
 # Polarization is descriptive only. Original SOH labels are imported unchanged.
 pol_meta=[];curves=[]
 for r in manifest.loc[manifest.role.eq('DEVELOPMENT_NOMINAL_POLARIZATION')].itertuples():
  path=REPO/r.path
  with path.open(encoding='utf-8-sig',newline='') as stream:columns=next(csv.reader(stream,delimiter=';'))
  selected=[x for x in columns if x in ['DateTime','OpHrs [h]','CalHrs [h]','Number of cells','Membrane active area [cm2]','I.S.set [A]','I.S.act [A]','j.S.act [A/cm2]','U.S.AveCell [V]',
   'T.Si.A [°C]','T.Si.C [°C]','T.Si.CL [°C]','p.So.A [kPag]','p.So.C [kPag]','Stoic.S.A [-]','Stoic.S.C [-]'] or re.fullmatch(r'U\d+ \[V\]',x) or x.startswith('RH.Si.')]
  d=pd.read_csv(path,sep=';',usecols=selected,thousands=',');ts=pd.to_datetime(d.DateTime)
  nominal=path.stem.split('RHFC_')[1];active=[f'U{k} [V]' for k in range(1,12) if k!=6];V=d[active].mean(axis=1)
  info=dict(source_path=r.path,nominal_checkpoint=nominal,records=len(d),first_calendar=str(ts.min()),last_calendar=str(ts.max()),
   exported_cell_counts='|'.join(str(x) for x in d['Number of cells'].unique()),active_area_values='|'.join(str(x) for x in d['Membrane active area [cm2]'].unique()),
   exported_OpHrs_min=float(d['OpHrs [h]'].min()),exported_OpHrs_max=float(d['OpHrs [h]'].max()),
   exported_OpHrs_is_durability_OpHrs2=False,max_active_cell_average_difference_V=float(abs(V-d['U.S.AveCell [V]']).max()),
   independent_repeat_stacks=0,precise_durability_clock='UNKNOWN',raw_voltage_channels='|'.join(active))
  for x in selected:
   if x.startswith(('T.Si.','p.So.','Stoic.','RH.Si.')):
    vals=pd.to_numeric(d[x],errors='coerce');info[x+'_median']=float(vals.median());info[x+'_min']=float(vals.min());info[x+'_max']=float(vals.max())
  pol_meta.append(info)
  token=d['I.S.set [A]'].round(2);group=token.ne(token.shift()).cumsum();last_j=None
  for g,rows in d.groupby(group,sort=False):
   if len(rows)<120:continue
   ids=rows.index.to_numpy();mid=ids[len(ids)//4:3*len(ids)//4];j=float(d.loc[mid,'j.S.act [A/cm2]'].mean())
   branch='INITIAL_OR_CONDITIONING' if last_j is None else 'ASCENDING' if j>last_j else 'DESCENDING' if j<last_j else 'SAME_LOAD';last_j=j
   curves.append(dict(source_path=r.path,nominal_checkpoint=nominal,dwell_group=int(g),set_current_A=float(token.loc[ids[0]]),samples=len(rows),
    start_calendar=str(ts.loc[ids[0]]),end_calendar=str(ts.loc[ids[-1]]),j_A_cm2=j,V_active10_mean=float(V.loc[mid].mean()),V_temporal_std=float(V.loc[mid].std(ddof=0)),branch=branch,
    independent_replicate=False,new_SOH_label=False))
  inventory.append(dict(path=r.path,type='NOMINAL_POLARIZATION',read_status='DEVELOPMENT_SELECTED_COLUMNS_PARSED',byte_size=r.byte_size,sha256=r.content_sha256,
   calendar_start=str(ts.min()),physical_stack='10 active-cell mean, nominal durability configuration supported',development_status='DEVELOPMENT_REFERENCE',precise_OpHrs2='ABSENT;OpHrs different clock',
   actual_temperature='Present gas/coolant external values',humidity='Present only if export RH columns exist; otherwise dewpoint/setpoint distinct',pressure='Outlet gauge values',columns='|'.join(selected),cell_or_zone='TEN_ACTIVE_CELL_MEAN_EXCLUDING_U6'))
  print('POLARIZATION',nominal,'records',len(d),flush=True)
 cs('outputs/POLARIZATION_SOURCE_AND_CONDITIONS.csv',pd.DataFrame(pol_meta));cs('outputs/POLARIZATION_DESCRIPTIVE_DWELL_CURVES.csv',pd.DataFrame(curves));cs('outputs/DATA_RESOURCE_INVENTORY.csv',pd.DataFrame(inventory))
 refs=pd.read_csv(PROJECT/OLD[1]);cs('inputs/FROZEN_REFERENCE_SOH_UNCHANGED.csv',refs)
 # Metadata-only inventory of excluded late files. Their contents are never opened.
 blocked=[]
 for f in BASE.rglob('*'):
  if f.is_file() and any(x in str(f.relative_to(BASE)).lower() for x in ['2300h','2000h']):blocked.append(dict(path=str(f.relative_to(REPO)).replace('\\','/'),bytes_metadata_only=f.stat().st_size,content_read=False,sha256_not_computed=True,status='TEST_CONTENT_BLOCKED'))
 cs('outputs/EXCLUDED_TEST_FILE_METADATA_ONLY.csv',pd.DataFrame(blocked))
 js('outputs/CORE_ANALYSIS_COMPLETE.json',dict(core_spectra=65,spectrum_points=len(points),header_only_sensitivity_files=201,
  equivalent_circuit_fits=len(fits),pure_CPU=True,elapsed_s=time.perf_counter()-start,formal_HFR_values=0,formal_ASR_values=0,
  apparent_HF_crossings=len(intercepts),diagnostic_R0_profiles_sources=len(set(r['source_path'] for r in profiles)),polarization_files=7,Test_contents_read=False,
  old_history_unchanged=all(sha(PROJECT/f)==h for f,h in protected.items()),physics_training_started=False))
 print('CORE_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':import sys;{'freeze':freeze,'run':run}[sys.argv[1]]()
