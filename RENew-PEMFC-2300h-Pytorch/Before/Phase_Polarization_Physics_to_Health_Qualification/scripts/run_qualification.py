"""CPU-only, checkpointed polarization diagnostics; no health-network fitting."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv,sys
from scipy.optimize import least_squares
from common import *
WINDOWS={'middle50':(.25,.75),'middle25':(.375,.625),'last25':(.75,1),'last50':(.5,1)}
PARAMS=['Eeff_V','b_V','Reff_ohm_cm2','c_V','jlim_A_cm2']
LOWER=[.3,0,0,0,1.85]; UPPER=[1.2,.2,.8,.2,10.]
STARTS=[[.8,.025,.05,.02,2.2],[.9,.05,.1,.05,3.],[.75,.01,.03,.01,6.]]
def polar_model(j,p):
 E,b,R,c,L=p
 return E-b*np.log(j/1.)-R*j+c*np.log1p(-j/L)
def freeze():
 assert not (P/'configs/FROZEN_PROTOCOL.json').exists()
 for d in ['configs','data_contracts','scripts','models','outputs','figures','reports','tests']:out(d+'/.keep').touch(exist_ok=True)
 c={'base_commit':'5df54d174e7aad2bde3eb1d212971e2cbe3aebd4','CPU_only':True,'Test_read':False,
 'source_sha256':{k:sha(q) for k,q in SOURCES.items()},'source_paths':{k:q.relative_to(REPO).as_posix() for k,q in SOURCES.items()},
 'code_sha256':{q.relative_to(P).as_posix():sha(q) for q in (P/'scripts').glob('*.py') if q.name in ['common.py','run_qualification.py','run_h2.py']},
 'checkpoints':CHECKPOINTS,'SOH_ref':'Frozen V_pol(j*=1)/0.6989533305168152; no replacement labels',
 'raw_columns':'Exact7 files only; active10 mean excludes missingU6; all raw row IDs and clocks retained in selected-channel CSV',
 'dwell':'Contiguous Iset rounded0.02A, >=120records and >=119s. First max504A to first0 descending; first0 to next504 ascending; conditioning/return excluded.',
 'windows':WINDOWS,'domain':'setpoint density0.36..1.80 A/cm2; actualj used in slopes/fits; 5min holds expected, gates duration>=299s',
 'stability_gates':{'j_std_A_cm2_max':.005,'V_abs_drift_mV_min_max':2,'T_span_C_max':2,'dewpoint_span_C_max':2},
 'gate_scope':'Exploratory within-dwell diagnostics, not certification of matched gas/state or independent replicas. All failed dwells retained.',
 'candidate_A':'Branch-specific finite differences; window sensitivity plus propagated temporal SD are descriptive, not independent-sample CIs.',
 'candidate_B':'Rdiff=-dV/dj [ohm cm2], composite not HFR/ASR. Check window sensitivity, branch hysteresis, j=1-centered shapes; no new3event calibration.',
 'candidate_C':{'model':'Eeff-b*ln(j/1)-Reff*j+c*ln(1-j/jlim)','parameters':PARAMS,'lower':LOWER,'upper':UPPER,'starts':STARTS,'max_nfev':5000,'profile_jlim':[1.85,2.,2.5,3.,5.,10.],
 'identifiability':'Column-normalized analytic Jacobian condition<1e4, full rank, all parameters interior, near-best multistart(relativeSSE<=1% or absolute1e-10) spread<=20%of bounds. Diagnostic criterion, not confidence interval.',
 'stop':'Exactly3starts +6fixedjlim profiles per checkpoint/branch/middle50; no alternative formula, no outcome-driven range changes; >=8 stable points.'},
 'H2':'First inspect all3 folds without fitting. If all outer and inner eligible, run one fixed96h five-feature Ridge CPU experiment to supply missing1-3cycle task; no H1/H3 fitting.',
 'folds':sj('native_protocol')['folds'],'alpha_grid':[.0001,.01,1.,100.],
 'H2_inner':'lastceil15% chronologicalTrain, purge labels after firstselectionorigin; scaler onlyinnerTrain; finalscaleronlyouterTrain; innerMAE_R tie loweralpha',
 'H2_primary':'threefold equal MAE_mVeq vs same-samplePersistence; allfoldsrequired; no separate claims of statisticalsignificance; no tuning afterscores',
 'availability':'Measurement completion is necessary; publication/release timestamp unknown, therefore no certified strict-online physical parameter. Report completion-only retrospective counts separately.',
 'P_A':'FrozenM2_NC control only. No new calibration fit; only3eligible events. Physicalshape diagnostics not generalization evidence.',
 'P_B':'Cross-current derivative undefined on scalar fixedj health output; do not fabricate extra outputs. Invalid realtraining loss/gradient NA, no forced healthmonotonicity.',
 'stop_rule':'No GPU; no H6/H12; no newbackbone/SOH/RUL model; no EIS refit; candidatefailure does not trigger search'}
 js('configs/FROZEN_PROTOCOL.json',c);md('configs/FROZEN_PROTOCOL.sha256',sha(P/'configs/FROZEN_PROTOCOL.json'))
 cs('data_contracts/SOURCE_SHA256.csv',pd.DataFrame([{'key':k,'path':c['source_paths'][k],'sha256':v,'bytes':SOURCES[k].stat().st_size} for k,v in c['source_sha256'].items()]))
 md('.gitignore','__pycache__/\n*.pyc\noutputs/GIT_RECEIPT.local.json')
 md('.gitattributes','* -text\n*.png binary\n*.pdf binary')
 print('FROZEN',sha(P/'configs/FROZEN_PROTOCOL.json'),flush=True)
def raw_columns(q):
 with q.open(encoding='utf-8-sig') as f:header=next(csv.reader(f,delimiter=';'))
 exact=['DateTime','OpHrs [h]','CalHrs [h]','Number of cells','Membrane active area [cm2]','I.S.set [A]','I.S.act [A]','j.S.act [A/cm2]','U.S.AveCell [V]']+[f'U{i} [V]' for i in [1,2,3,4,5,7,8,9,10,11]]
 return [c for c in header if c in exact or c.startswith(('T.Si.','DPT.Si.','p.So.','Stoic.S.'))]
def extract(k):
 q=SOURCES['polar_'+k];d=pd.read_csv(q,sep=';',usecols=raw_columns(q),thousands=',',float_precision='round_trip')
 d['raw_row_id']=np.arange(len(d));d['DateTime']=pd.to_datetime(d.DateTime);d['V_active10']=d[[f'U{i} [V]' for i in [1,2,3,4,5,7,8,9,10,11]]].mean(axis=1)
 assert not d.DateTime.isna().any() and d.DateTime.is_monotonic_increasing
 assert (d['Membrane active area [cm2]']==280).all() and (d['Number of cells']==10).all()
 sc=(d['I.S.set [A]']/.02).round()*.02;grp=sc.ne(sc.shift()).cumsum();groups=[]
 for gid,z in d.groupby(grp,sort=False):
  duration=(z.DateTime.iloc[-1]-z.DateTime.iloc[0]).total_seconds()
  if len(z)>=120 and duration>=119:groups.append((int(gid),z,float(sc.loc[z.index[0]])))
 peak=next(i for i,(_,z,s) in enumerate(groups) if abs(s-504)<.1);zero=next(i for i in range(peak+1,len(groups)) if abs(groups[i][2])<.1)
 peak2=next(i for i in range(zero+1,len(groups)) if abs(groups[i][2]-504)<.1)
 rows=[]
 for i,(gid,z,s) in enumerate(groups):
  branch='descending' if peak<=i<zero else 'ascending' if zero<i<=peak2 else 'zero_turnaround' if i==zero else 'conditioning_or_return'
  for w,(lo,hi) in WINDOWS.items():
   a=z.iloc[int(len(z)*lo):int(len(z)*hi)];seconds=(a.DateTime-a.DateTime.iloc[0]).dt.total_seconds().to_numpy();v=a.V_active10.to_numpy()
   drift=float(np.polyfit(seconds,v,1)[0]*60000)
   ts=[c for c in a if c.startswith('T.Si.') and '.set' not in c];dp=[c for c in a if c.startswith('DPT.Si.') and '.set' not in c]
   tspan=float(max(a[c].max()-a[c].min() for c in ts));dpspan=float(max(a[c].max()-a[c].min() for c in dp));jstd=float(a['j.S.act [A/cm2]'].std(ddof=0))
   reasons=[]
   if branch not in ['descending','ascending']:reasons.append('NOT_SCAN_BRANCH')
   if not .36-1e-10<=s/280<=1.8+1e-10:reasons.append('OUTSIDE_PREDECLARED_CURRENT_DOMAIN')
   duration=(z.DateTime.iloc[-1]-z.DateTime.iloc[0]).total_seconds()
   if duration<299:reasons.append('SHORT_HOLD')
   if jstd>.005:reasons.append('CURRENT_UNSTABLE')
   if abs(drift)>2:reasons.append('V_DRIFT')
   if tspan>2:reasons.append('T_SPAN')
   if dpspan>2:reasons.append('DEWPOINT_SPAN')
   row={'checkpoint':k,'dwell_group':gid,'branch':branch,'window':w,'set_current_A':s,'set_j_A_cm2':s/280,'duration_s':duration,'n':len(a),
    'raw_first':int(z.raw_row_id.iloc[0]),'raw_last':int(z.raw_row_id.iloc[-1]),'window_raw_first':int(a.raw_row_id.iloc[0]),'window_raw_last':int(a.raw_row_id.iloc[-1]),
    'start_calendar':str(z.DateTime.iloc[0]),'end_calendar':str(z.DateTime.iloc[-1]),'window_start_calendar':str(a.DateTime.iloc[0]),'window_end_calendar':str(a.DateTime.iloc[-1]),
    'exported_OpHrs_first':float(z['OpHrs [h]'].iloc[0]),'exported_OpHrs_last':float(z['OpHrs [h]'].iloc[-1]),'durability_OpHrs2_h':np.nan,
    'j_A_cm2':float(a['j.S.act [A/cm2]'].mean()),'j_std':jstd,'V_V':float(v.mean()),'V_std_V':float(v.std(ddof=0)),'V_drift_mV_min':drift,
    'T_span_C':tspan,'dewpoint_span_C':dpspan,'stable_domain':not reasons,'exclusion_reason':'|'.join(reasons)}
   for c in ts+dp+[c for c in a if c.startswith(('p.So.','Stoic.S.'))]:row[c+'_mean']=float(a[c].mean());row[c+'_min']=float(a[c].min());row[c+'_max']=float(a[c].max())
   rows.append(row)
 cs(f'outputs/raw_selected/{k}.csv',d);cs(f'outputs/dwells/{k}.csv',pd.DataFrame(rows))
 meta={'checkpoint':k,'source_key':'polar_'+k,'rows':len(d),'start_calendar':str(d.DateTime.iloc[0]),'end_calendar':str(d.DateTime.iloc[-1]),
 'exported_OpHrs_start':float(d['OpHrs [h]'].iloc[0]),'exported_OpHrs_end':float(d['OpHrs [h]'].iloc[-1]),'durability_OpHrs2_h':None,
 'publication_time':None,'actual_RH_column_present':False,'dewpoint_measured':True,'area_cm2':280,'cells':10,'scan_descending_holds':zero-peak,'scan_ascending_holds':peak2-zero,
 'max_abs_exported_mean_minus_active10_mV':float(abs(d['U.S.AveCell [V]']-d.V_active10).max()*1000)}
 js(f'outputs/dwells/{k}_metadata.json',meta);print('EXTRACTED',k,len(d),flush=True)
def slopes(dw):
 rows=[]
 for (k,b,w),g in dw[dw.stable_domain].groupby(['checkpoint','branch','window']):
  g=g.sort_values('set_current_A');a=list(g.itertuples(index=False))
  for l,r in zip(a[:-1],a[1:]):
   delta=r.j_A_cm2-l.j_A_cm2;s=(r.V_V-l.V_V)/delta
   rows.append({'checkpoint':k,'branch':b,'window':w,'low_set_A':l.set_current_A,'high_set_A':r.set_current_A,'j_mid':(l.j_A_cm2+r.j_A_cm2)/2,'dV_dj':s,'Rdiff_ohm_cm2':-s,
    'temporal_SD_propagation_ohm_cm2':float(np.hypot(l.V_std_V,r.V_std_V)/delta),'monotonic_violation':bool(s>0),'not_independent_CI':True})
 return pd.DataFrame(rows)
def fit_one(k,b,g):
 token=f'{k}_{b}';done=P/f'models/{token}.json'
 if done.exists():return jl(f'models/{token}.json')
 g=g.sort_values('j_A_cm2');x=g.j_A_cm2.to_numpy();y=g.V_V.to_numpy()
 if len(g)<8:
  z={'checkpoint':k,'branch':b,'status':'NOT_ENOUGH_STABLE_POINTS','n':len(g),'identifiable':False};js(f'models/{token}.json',z);return z
 rec=[]
 for n,start in enumerate(STARTS):
  f=least_squares(lambda p:polar_model(x,p)-y,start,bounds=(LOWER,UPPER),max_nfev=5000,ftol=1e-12,xtol=1e-12,gtol=1e-12)
  rec.append({'start':n,'parameters':f.x.tolist(),'SSE':float(f.fun@f.fun),'success':bool(f.success),'nfev':int(f.nfev)})
  js(f'models/checkpoints/{token}_start{n}.json',rec[-1])
 best=min(rec,key=lambda r:r['SSE']);p=np.array(best['parameters']);E,bb,R,c,L=p
 J=np.column_stack([np.ones(len(x)),-np.log(x),-x,np.log1p(-x/L),c*x/(L*(L-x))]);norm=np.linalg.norm(J,axis=0);sv=np.linalg.svd(J/np.maximum(norm,1e-300),compute_uv=False);condition=float(sv[0]/sv[-1]) if sv[-1]>0 else 1e300
 near=[r for r in rec if r['SSE']<=best['SSE']*1.01+1e-10];spread=np.ptp([r['parameters'] for r in near],axis=0)/(np.array(UPPER)-LOWER)
 bound=bool(np.any((p-np.array(LOWER))/(np.array(UPPER)-LOWER)<1e-4) or np.any((np.array(UPPER)-p)/(np.array(UPPER)-LOWER)<1e-4))
 profile=[]
 for fixed in [1.85,2.,2.5,3.,5.,10.]:
  f=least_squares(lambda a:polar_model(x,[*a,fixed])-y,p[:4],bounds=(LOWER[:4],UPPER[:4]),max_nfev=5000,ftol=1e-12,xtol=1e-12,gtol=1e-12)
  profile.append({'checkpoint':k,'branch':b,'fixed_jlim':fixed,'SSE':float(f.fun@f.fun),'RMSE_mV':float(np.sqrt(np.mean(f.fun**2))*1000),'parameters':[*f.x.tolist(),fixed],'success':bool(f.success)})
 cs(f'outputs/fits/{token}_residuals.csv',pd.DataFrame({'checkpoint':k,'branch':b,'j_A_cm2':x,'observed_V':y,'fitted_V':polar_model(x,p),'residual_mV':(polar_model(x,p)-y)*1000}))
 z={'checkpoint':k,'branch':b,'status':'FITTED_DESCRIPTIVE','n':len(x),'parameters':dict(zip(PARAMS,p.tolist())),'multistarts':rec,'RMSE_mV':float(np.sqrt(best['SSE']/len(x))*1000),
 'jacobian_normalized_condition':condition,'jacobian_normalized_singular_values':sv.tolist(),'boundary_parameter':bound,'nearbest_spread_fraction_bounds':spread.tolist(),
 'identifiable':bool(condition<1e4 and not bound and max(spread)<=.2),'profiles':profile,'parameter_release_time':None}
 js(f'models/{token}.json',z);print('FIT',token,z['RMSE_mV'],condition,'identified',z['identifiable'],flush=True);return z
def run():
 config();assert not (P/'outputs/POLARIZATION_COMPLETED.json').exists(),'Completed one-shot analysis must not be repeated'
 for k in CHECKPOINTS:
  if not (P/f'outputs/dwells/{k}_metadata.json').exists():extract(k)
 dw=pd.concat([pd.read_csv(P/f'outputs/dwells/{k}.csv',float_precision='round_trip') for k in CHECKPOINTS],ignore_index=True)
 cs('outputs/POLARIZATION_DWELLS.csv',dw);meta=[jl(f'outputs/dwells/{k}_metadata.json') for k in CHECKPOINTS];cs('outputs/POLARIZATION_QUALIFICATION.csv',pd.DataFrame(meta))
 s=slopes(dw);cs('outputs/NUMERICAL_SLOPES.csv',s)
 sensitivity=s.groupby(['checkpoint','branch','low_set_A','high_set_A']).agg(windows=('window','nunique'),slope_min=('Rdiff_ohm_cm2','min'),slope_max=('Rdiff_ohm_cm2','max')).reset_index();cs('outputs/SLOPE_WINDOW_SENSITIVITY.csv',sensitivity)
 hs=[];shapes=[]
 for k in CHECKPOINTS:
  a=dw[(dw.checkpoint==k)&dw.window.eq('middle50')&dw.stable_domain];down=a[a.branch.eq('descending')];up=a[a.branch.eq('ascending')]
  for r in down.itertuples():
   u=up[abs(up.set_current_A-r.set_current_A)<1e-8]
   if len(u):hs.append({'checkpoint':k,'set_current_A':r.set_current_A,'j_A_cm2':r.j_A_cm2,'up_minus_down_mV':float((u.V_V.iloc[0]-r.V_V)*1000)})
  for b,g in a.groupby('branch'):
   g=g.sort_values('j_A_cm2')
   if g.j_A_cm2.min()<1<g.j_A_cm2.max():
    v1=float(np.interp(1,g.j_A_cm2,g.V_V))
    for r in g.itertuples():shapes.append({'checkpoint':k,'branch':b,'j_A_cm2':r.j_A_cm2,'V_V':r.V_V,'diagnostic_interpolated_V1':v1,'V_minus_V1':r.V_V-v1,'not_replacement_SOH':True})
 cs('outputs/HYSTERESIS.csv',pd.DataFrame(hs));cs('outputs/J1_CENTERED_SHAPES.csv',pd.DataFrame(shapes))
 domain=dw[dw.window.eq('middle50')].groupby(['branch','set_current_A']).agg(checkpoints=('checkpoint','nunique'),stable_checkpoints=('stable_domain','sum'),duration_min_s=('duration_s','min'),V_drift_abs_max_mV_min=('V_drift_mV_min',lambda x:float(abs(x).max())),T_span_max=('T_span_C','max'),dewpoint_span_max=('dewpoint_span_C','max')).reset_index();domain['all7_stable']=(domain.checkpoints==7)&(domain.stable_checkpoints==7)&domain.branch.isin(['descending','ascending']);cs('outputs/COMPARABLE_CURRENT_DOMAIN.csv',domain)
 fits=[]
 for k in CHECKPOINTS:
  for b in ['descending','ascending']:fits.append(fit_one(k,b,dw[(dw.checkpoint==k)&dw.branch.eq(b)&dw.window.eq('middle50')&dw.stable_domain]))
 cs('outputs/PARAMETER_IDENTIFIABILITY.csv',pd.DataFrame([{k:v for k,v in r.items() if k not in ['multistarts','profiles','jacobian_normalized_singular_values','nearbest_spread_fraction_bounds','parameters']}|r.get('parameters',{}) for r in fits]))
 cs('outputs/JLIM_PROFILE.csv',pd.DataFrame([r for f in fits for r in f.get('profiles',[])]))
 # Frozen H1/H3 prediction replay, no fitting and no model mutation.
 pred=read('native_predictions');pars=sj('native_parameters');replay=[]
 fn=['L96_origin_R','L96_history_mean_R','L96_history_std_R','L96_history_slope_R_per_h','L96_origin_minus_median_R']
 for (h,fold,model),g in pred.groupby(['horizon','fold','model']):
  p=next(p for p in pars if p['horizon']==h and p['fold']==fold);z=g.origin_R_P1S_causal.to_numpy()
  if model=='RIDGE_96H':z=z+((g[fn].to_numpy()-p['train_scaler_mean'])/p['train_scaler_scale'])@p['coef_standardized_X']+p['intercept_deltaR']
  difference=float(abs(z-g.predicted_R_P1S_causal.to_numpy()).max());assert difference<1e-12
  replay.append({'horizon':h,'fold':fold,'method':model,'max_difference_R':difference,**metric(g.true_R_P1S_causal,z)})
 cs('outputs/FROZEN_H1_H3_REPLAY.csv',pd.DataFrame(replay))
 meas=read('measurement');meas=meas[meas.stable_window.eq('middle_50')].set_index('cycle_interval_id');avail=[]
 # Completion-only availability is conditional, publication time never silently assumed equal to completion.
 native=read('native_legal')
 for row in native.itertuples():
  t=pd.Timestamp(meas.loc[row.origin_cycle_interval_id,'measurement_end_datetime']);past=[m for m in meta if pd.Timestamp(m['end_calendar'])<=t]
  avail.append({'horizon':row.horizon,'fold':row.fold,'role':row.role,'origin_raw_cycle_id':row.origin_raw_cycle_id,'target_raw_cycle_id':row.target_raw_cycle_id,'origin_calendar':str(t),
   'completion_only_checkpoint_count':len(past),'completion_only_checkpoints':'|'.join(m['checkpoint'] for m in past),'certified_publication_available_count':0,'publication_unknown':True,'strict_online_qualified':False})
 cs('outputs/H1_H3_PARAMETER_AVAILABILITY.csv',pd.DataFrame(avail))
 cs('outputs/AVAILABILITY_COUNTS.csv',pd.DataFrame(avail).groupby(['horizon','fold','role']).agg(n=('origin_raw_cycle_id','size'),completion_min=('completion_only_checkpoint_count','min'),completion_max=('completion_only_checkpoint_count','max'),certified_count_max=('certified_publication_available_count','max')).reset_index())
 links=read('links');cs('outputs/FROZEN_SOH_LINKS_READONLY.csv',links)
 js('outputs/FROZEN_CALIBRATION_READONLY.json',sj('calibration'))
 js('outputs/POLARIZATION_COMPLETED.json',{'status':'COMPLETED','raw_files':7,'raw_rows':sum(m['rows'] for m in meta),'candidate_C_branches':len(fits),'new_EIS_fits':0,'H1_H3_refits':0,'GPU_runs':0,'Test_read':False})
 js('outputs/READ_LEDGER_POLARIZATION.json',sorted(set(ACCESS)));config()
if __name__=='__main__':
 firewall();{'freeze':freeze,'run':run}[sys.argv[1]]()
