"""No optimizer/model fitting: independent raw parser, algebra and clock verification."""
import csv
from common import *
def close(a,b,tol=1e-12):
 a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape
 assert np.array_equal(np.isnan(a),np.isnan(b))
 diff=float(np.max(abs(a-b),initial=0,where=np.isfinite(a)&np.isfinite(b)));assert diff<=tol,(diff,tol);return diff
def independent_ridge(train,par):
 x=train[FEATURES].to_numpy();y=(train.target_R-train.origin_R).to_numpy();mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale==0]=1
 close(mean,par['scaler_mean']);close(scale,par['scaler_scale']);z=(x-mean)/scale;zc=z-z.mean(axis=0);yc=y-y.mean()
 beta=np.linalg.solve(zc.T@zc+par['alpha']*np.eye(5),zc.T@yc);intercept=float(y.mean()-z.mean(axis=0)@beta)
 return max(close(beta,par['coef'],1e-10),close(intercept,par['intercept'],1e-12))
def pred(frame,par):return frame.origin_R.to_numpy()+((frame[FEATURES].to_numpy()-par['scaler_mean'])/par['scaler_scale'])@par['coef']+par['intercept']
def run():
 cfg=config();rawrows=0;rawcells=0;maxraw=0.;maxdwell=0.;raw_identity=[]
 dw=pd.read_csv(P/'outputs/POLARIZATION_DWELLS.csv',float_precision='round_trip');dwells=0
 for k in CHECKPOINTS:
  saved=pd.read_csv(P/f'outputs/raw_selected/{k}.csv',float_precision='round_trip');cols=[c for c in saved if c not in ['raw_row_id','V_active10']];numerical=[c for c in cols if c!='DateTime'];values=[];times=[]
  with SOURCES['polar_'+k].open(encoding='utf-8-sig',newline='') as stream:
   reader=csv.reader(stream,delimiter=';');header=next(reader);ix=[header.index(c) for c in numerical];dt=header.index('DateTime')
   for i,row in enumerate(reader):
    times.append(pd.Timestamp(row[dt]));values.append([float(row[j].replace(',','')) if row[j].strip() else float('nan') for j in ix])
  a=np.array(values);assert len(a)==len(saved);assert saved.raw_row_id.tolist()==list(range(len(a)))
  assert np.array_equal(pd.DatetimeIndex(times).asi8,pd.DatetimeIndex(saved.DateTime).asi8)
  maxraw=max(maxraw,close(a,saved[numerical].to_numpy(),0.));cells=[numerical.index(f'U{i} [V]') for i in [1,2,3,4,5,7,8,9,10,11]];maxraw=max(maxraw,close(np.nanmean(a[:,cells],axis=1),saved.V_active10.to_numpy()))
  rawrows+=len(a);rawcells+=a.size;raw_identity.append({'checkpoint':k,'raw_rows':len(a),'numeric_cells':a.size,'numeric_max_abs_difference':0.,'row_and_calendar_identity_exact':True})
  sc=np.round(saved['I.S.set [A]'].to_numpy()/.02)*.02;cuts=np.r_[0,np.flatnonzero(sc[1:]!=sc[:-1])+1,len(sc)];groupmap={n+1:np.arange(l,r) for n,(l,r) in enumerate(zip(cuts[:-1],cuts[1:]))}
  for r in dw[dw.checkpoint.eq(k)].to_dict('records'):
   ids=groupmap[r['dwell_group']];assert int(ids[0])==r['raw_first'] and int(ids[-1])==r['raw_last'];lo,hi=cfg['windows'][r['window']];ixs=ids[int(len(ids)*lo):int(len(ids)*hi)];assert len(ixs)==r['n'] and ixs[0]==r['window_raw_first'] and ixs[-1]==r['window_raw_last']
   vv=saved.iloc[ixs].V_active10.to_numpy();jj=saved.iloc[ixs]['j.S.act [A/cm2]'].to_numpy();maxdwell=max(maxdwell,close([vv.mean(),vv.std(),jj.mean(),jj.std()],[r['V_V'],r['V_std_V'],r['j_A_cm2'],r['j_std']],1e-12));dwells+=1
 print('INDEPENDENT_RAW_PASS',rawrows,'rows',flush=True)
 slope=pd.read_csv(P/'outputs/NUMERICAL_SLOPES.csv',float_precision='round_trip');maxs=0
 for r in slope.itertuples():
  g=dw[(dw.checkpoint==r.checkpoint)&(dw.branch==r.branch)&(dw.window==r.window)&dw.stable_domain];l=g[np.isclose(g.set_current_A,r.low_set_A)].iloc[0];u=g[np.isclose(g.set_current_A,r.high_set_A)].iloc[0];v=-(u.V_V-l.V_V)/(u.j_A_cm2-l.j_A_cm2);maxs=max(maxs,close(v,r.Rdiff_ohm_cm2))
 # Reconstruct true feature histories without importing run_h2 or its feature code.
 ident=read('identity');measure=read('measurement');measure=measure[measure.stable_window.eq('middle_50')].set_index('cycle_interval_id');cat=read('catalog').set_index('raw_cycle_id');maxfeat=0;maxcoef=0;maxpred=0;fitrecords=0
 used=pd.read_csv(P/'outputs/H2_USED_SAMPLES.csv',float_precision='round_trip');audit=pd.read_csv(P/'outputs/H2_ELIGIBILITY.csv',float_precision='round_trip')
 for r in used.itertuples():
  o=ident[ident.raw_cycle_id.eq(r.origin_raw_cycle_id)].iloc[0];t=ident[ident.raw_cycle_id.eq(r.target_raw_cycle_id)].iloc[0];g=ident[(ident.measurement_end_h>=o.measurement_end_h-96)&(ident.measurement_end_h<=o.measurement_end_h)&(ident.conservative_available_end_h<=o.conservative_available_end_h)]
  assert len(g)>=5 and np.ptp(g.measurement_end_h)>=67.2 and r.target_raw_cycle_id==r.origin_raw_cycle_id+2
  assert all(bool(cat.loc[i,'complete_cycle']) for i in range(r.origin_raw_cycle_id,r.target_raw_cycle_id+1))
  x=g.measurement_end_h.to_numpy();y=g.R.to_numpy();s=np.linalg.lstsq(np.column_stack([x-x.mean(),np.ones(len(x))]),y,rcond=None)[0][0]
  maxfeat=max(maxfeat,close([o.R,y.mean(),y.std(),s,o.R-np.median(y)],[getattr(r,k) for k in FEATURES],1e-12))
  lo,mid,hi=cfg['folds'][r.fold];assert (lo<=t.measurement_start_h and t.conservative_available_end_h<mid) if r.role=='train' else (mid<=t.measurement_start_h and t.conservative_available_end_h<hi)
  assert pd.Timestamp(measure.loc[o.cycle_interval_id,'measurement_end_datetime'])==pd.Timestamp(r.origin_calendar)
  assert pd.Timestamp(measure.loc[t.cycle_interval_id,'measurement_end_datetime'])==pd.Timestamp(r.target_calendar)
 for fold in cfg['folds']:
  train=pd.read_csv(P/f'data_contracts/H2_{fold}_TRAIN.csv',float_precision='round_trip');val=pd.read_csv(P/f'data_contracts/H2_{fold}_VALIDATION.csv',float_precision='round_trip');inner=pd.read_csv(P/f'data_contracts/H2_{fold}_INNER_FIT.csv',float_precision='round_trip');selection=pd.read_csv(P/f'data_contracts/H2_{fold}_INNER_SELECTION.csv',float_precision='round_trip');par=jl(f'models/H2_{fold}.json')
  assert train.target_available_h.max()<=val.origin_available_h.min();assert inner.target_available_h.max()<=selection.origin_available_h.min()
  assert pd.to_datetime(train.target_calendar).max()<=pd.to_datetime(val.origin_calendar).min();assert pd.to_datetime(inner.target_calendar).max()<=pd.to_datetime(selection.origin_calendar).min()
  assert par['sample_ids']==train.sample_id.tolist();maxcoef=max(maxcoef,independent_ridge(train,par));fitrecords+=1
  for rec in par['inner_grid']:
   assert rec['sample_ids']==inner.sample_id.tolist();maxcoef=max(maxcoef,independent_ridge(inner,rec));fitrecords+=1
   close(abs(pred(selection,rec)-selection.target_R).mean(),rec['selection_MAE_R'],1e-13)
  best=min(par['inner_grid'],key=lambda x:(x['selection_MAE_R'],x['alpha']));assert best['alpha']==par['alpha']
  pp=pd.read_csv(P/f'outputs/H2_{fold}_PREDICTIONS.csv',float_precision='round_trip')
  for method in ['PERSISTENCE','RIDGE96']:
   g=pp[pp.method.eq(method)];assert g.sample_id.tolist()==val.sample_id.tolist();p=val.origin_R.to_numpy() if method=='PERSISTENCE' else pred(val,par);maxpred=max(maxpred,close(p,g.predicted_R,1e-12))
 # Independently recompute metrics, not read rendered report values.
 pp=pd.read_csv(P/'outputs/H2_PREDICTIONS.csv',float_precision='round_trip');met=pd.read_csv(P/'outputs/H2_FOLD_METRICS.csv',float_precision='round_trip')
 for (fold,method),g in pp.groupby(['fold','method']):
  e=(g.predicted_R-g.target_R).to_numpy()*713.83023635;a=abs(e);r=met[(met.fold==fold)&(met.method==method)].iloc[0];close([a.mean(),np.sqrt((e*e).mean()),np.quantile(a,.95),a.max()],[r.MAE_mVeq,r.RMSE_mVeq,r.P95_mVeq,r.max_mVeq],1e-10)
 # Exact origin+target availability record count and completion-only events.
 av=pd.read_csv(P/'outputs/H1_H3_PARAMETER_AVAILABILITY.csv');native=read('native_legal');assert len(av)==len(native)
 meta=pd.read_csv(P/'outputs/POLARIZATION_QUALIFICATION.csv')
 for r in av.itertuples():
  expected=int((pd.to_datetime(meta.end_calendar)<=pd.Timestamp(r.origin_calendar)).sum());assert expected==r.completion_only_checkpoint_count;assert r.certified_publication_available_count==0 and not r.strict_online_qualified
 # Physical fit residuals and analytic-gradient finite difference check, no optimization.
 maxgrad=0.;fitted=0
 for q in (P/'models').glob('*.json'):
  if q.name.startswith('H2_'):continue
  f=jl(q.relative_to(P).as_posix())
  if f['status']!='FITTED_DESCRIPTIVE':continue
  z=pd.read_csv(P/f"outputs/fits/{f['checkpoint']}_{f['branch']}_residuals.csv",float_precision='round_trip');j=z.j_A_cm2.to_numpy();p=f['parameters'];E,b,R,c,L=[p[n] for n in ['Eeff_V','b_V','Reff_ohm_cm2','c_V','jlim_A_cm2']]
  def fn(bb):return E-bb*np.log(j)-R*j+c*np.log(1-j/L)
  close(fn(b),z.fitted_V,1e-12);close((fn(b)-z.observed_V)*1000,z.residual_mV,1e-9)
  def loss(bb):return float(np.mean((fn(bb)-z.observed_V)**2))
  analytical=float(2*np.mean((fn(b+.001)-z.observed_V)*(-np.log(j))));numeric=(loss(b+.001+1e-7)-loss(b+.001-1e-7))/2e-7;maxgrad=max(maxgrad,close(analytical,numeric,1e-9));fitted+=1
 # Frozen H1/H3 are independently reconstructed with original saved coefficient fields.
 old=read('native_predictions');pars=sj('native_parameters');fn=['L96_origin_R','L96_history_mean_R','L96_history_std_R','L96_history_slope_R_per_h','L96_origin_minus_median_R'];native_max=0
 for (h,fold,model),g in old.groupby(['horizon','fold','model']):
  p=next(p for p in pars if p['horizon']==h and p['fold']==fold);y=g.origin_R_P1S_causal.to_numpy()
  if model=='RIDGE_96H':y=y+((g[fn].to_numpy()-p['train_scaler_mean'])/p['train_scaler_scale'])@p['coef_standardized_X']+p['intercept_deltaR']
  native_max=max(native_max,close(y,g.predicted_R_P1S_causal,1e-12))
 config();cs('outputs/INDEPENDENT_RAW_IDENTITY.csv',pd.DataFrame(raw_identity));js('outputs/INDEPENDENT_VERIFICATION.json',{'status':'PASS','raw_files':7,'raw_rows':rawrows,'raw_numeric_cells':rawcells,'raw_original_numeric_max_difference':maxraw,'raw_row_order_and_calendar_exact':True,'dwells_windows_checked':dwells,'dwell_max_difference':maxdwell,'slope_records_checked':len(slope),'slope_max_difference':maxs,'H2_sample_records':len(used),'H2_feature_max_difference':maxfeat,'ridge_parameter_records_algebra_verified':fitrecords,'ridge_coef_max_difference':maxcoef,'H2_prediction_max_difference':maxpred,'native_prediction_rows':len(old),'native_prediction_max_difference':native_max,'curve_fits_checked':fitted,'curve_gradient_fd_max_difference':maxgrad,'source_SHA256_unchanged':True,'strict_online_publication_not_certified':True,'no_fitting_in_verifier':True,'Test_read':False})
 js('outputs/READ_LEDGER_VERIFIER.json',sorted(set(ACCESS)));print('INDEPENDENT_VERIFICATION_PASS',rawrows,fitrecords,len(used))
if __name__=='__main__':firewall();run()
