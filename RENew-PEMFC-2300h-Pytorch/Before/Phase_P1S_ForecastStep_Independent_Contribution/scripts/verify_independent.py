"""Independent stdlib CSV/Decimal verifier. Does not import production formulas."""
import csv,json,hashlib,datetime
from pathlib import Path
from decimal import Decimal,localcontext
P=Path(__file__).resolve().parents[1];ROOT=P.parent
D=Decimal;TOL=D('1e-12')
def rows(name):
 with (P/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def jread(name):return json.loads((P/name).read_text(encoding='utf-8'),parse_float=D)
def digest(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def asdate(s):return datetime.datetime.fromisoformat(s)
def close(a,b,tol=TOL):
 diff=abs(D(str(a))-D(str(b)));assert diff<tol,(str(a),str(b),str(diff));return diff
def write(name,obj):
 (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(name,records):
 with (P/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def run():
 with localcontext() as ctx:
  ctx.prec=60
  protocol=jread('configs/FROZEN_PROTOCOL.json')
  assert digest(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
  for path,h in protocol['source_sha256'].items():assert digest(ROOT/path)==h
  for path,h in protocol['audit_code_sha256'].items():assert digest(P/path)==h
  for s in rows('inputs/SOURCE_SHA256.csv'):
   extension=Path(s['source']).suffix;assert digest(P/f"inputs/{s['key']}{extension}")==s['sha256']
  links=rows('inputs/links.csv');cache=rows('inputs/cache.csv');params=jread('inputs/calibration.json');identity={r['cycle_interval_id']:r for r in rows('inputs/identity.csv')}
  measurement={r['cycle_interval_id']:r for r in rows('inputs/measurement.csv') if r['stable_window']=='middle_50'}
  truth={int(r['nominal_h']):D(r['Y']) for r in rows('inputs/refs.csv')};out=rows('outputs/POINT_PREDICTIONS.csv');decomp=rows('outputs/ERROR_DECOMPOSITION.csv')
  independent=[];maxdiff=D(0);replay=[]
  for link in links:
   n=int(link['nominal_h']);h=link['horizon'];key=(n,h)
   model=next(m for m in params if (m['nominal_h'],m['horizon'])==key)
   selected=[r for r in cache if all(r[k]==link[k] for k in ['horizon','fold','origin_raw_cycle_id','target_raw_cycle_id'])]
   assert len(selected)==2
   rr=next(r for r in selected if r['model']=='RIDGE_96H');pr=next(r for r in selected if r['model']=='PERSISTENCE')
   for field in ['origin_cycle_interval_id','target_cycle_interval_id']:assert rr[field]==link[field]==pr[field]
   assert int(link['target_raw_cycle_id'])-int(link['origin_raw_cycle_id'])=={'H1':1,'H3':3}[h]
   origin=identity[link['origin_cycle_interval_id']];target=identity[link['target_cycle_interval_id']];mr=measurement[link['origin_cycle_interval_id']]
   assert D(origin['measurement_end_h'])<=D(origin['conservative_available_end_h'])<=D(link['origin_conservative_available_h'])
   assert asdate(mr['measurement_end_datetime'])==asdate(link['origin_calendar'])
   assert asdate(link['prior_reference_end_max_calendar'])<asdate(link['origin_calendar'])
   assert len(model['training_events'])>=3 and all(e<n for e in model['training_events'])
   a=model['a'];b=model['b'];ro=D(origin['R']);rp=D(rr['predicted_R_P1S_causal']);rt=D(target['R']);y=truth[n]
   close(ro,D(mr['P1S_voltage_V'])/D('.71383023635'));close(ro,pr['predicted_R_P1S_causal']);close(rt,rr['true_R_P1S_causal']);close(rp,model['R_input_forecast'])
   previous=truth[max(e for e in truth if e<n)]
   answers={'M0_REFERENCE_PERSISTENCE':previous,'M_ORIGIN_P1S_CALIBRATED':a*ro+b,'M_RIDGE_P1S_CALIBRATED':a*rp+b}
   for method,value in answers.items():
    saved=next(r for r in out if (int(r['nominal_h']),r['horizon'],r['method'])==(n,h,method));diff=close(value,saved['predicted_SOH']);maxdiff=max(maxdiff,diff)
    error=(value-y)*100;close(error,saved['error_SOH_pp']);close(abs(error),saved['absolute_error_SOH_pp'])
    independent.append({'nominal_h':n,'horizon':h,'method':method,'Decimal_predicted_SOH':str(value),'Decimal_error_SOH_pp':str(error),'saved_float_difference_R':str(diff)})
   calerr=(a*rt+b-y)*100;proxyerr=a*(rp-rt)*100;orproxy=a*(ro-rt)*100
   dec=next(r for r in decomp if (int(r['nominal_h']),r['horizon'])==key)
   for name,v in [('calibration_at_actual_target_SOH_pp',calerr),('ridge_proxy_error_translated_SOH_pp',proxyerr),('origin_proxy_error_translated_SOH_pp',orproxy),
    ('ridge_final_error_SOH_pp',calerr+proxyerr),('origin_final_error_SOH_pp',calerr+orproxy),('ridge_step_delta_SOH_pp',a*(rp-ro)*100),
    ('ridge_cancellation_magnitude_SOH_pp',abs(calerr)+abs(proxyerr)-abs(calerr+proxyerr))]:close(v,dec[name])
   assert (dec['ridge_error_cancels_calibration']=='True')==(calerr*proxyerr<0)
  for metric in rows('outputs/SOH_METRICS.csv'):
   g=[r for r in independent if r['horizon']==metric['horizon'] and r['method']==metric['method']];errs=[D(r['Decimal_error_SOH_pp']) for r in g]
   mae=sum(abs(e) for e in errs)/len(errs);rmse=(sum(e*e for e in errs)/len(errs)).sqrt()
   close(mae,metric['MAE_SOH_pp']);close(rmse,metric['RMSE_SOH_pp'])
  for r in rows('outputs/PAIRWISE_COMPARISONS.csv'):
   diffs=[]
   for n in [1150,1500,1750]:
    def error(m):return abs(D(next(v for v in independent if (v['nominal_h'],v['horizon'],v['method'])==(n,r['horizon'],m))['Decimal_error_SOH_pp']))
    diffs.append(error(r['baseline'])-error(r['candidate']))
   assert sum(d>TOL for d in diffs)==int(r['candidate_wins']);assert sum(d<-TOL for d in diffs)==int(r['candidate_losses'])
   close(sum(diffs)/len(diffs),r['MAE_reduction_SOH_pp'])
  for r in rows('outputs/SEQUENTIAL_CONTRIBUTION.csv'):
   def mae(method):
    v=[abs(D(p['Decimal_error_SOH_pp'])) for p in independent if p['horizon']==r['horizon'] and p['method']==method];return sum(v)/len(v)
   m0=mae('M0_REFERENCE_PERSISTENCE');origin=mae('M_ORIGIN_P1S_CALIBRATED');ridge=mae('M_RIDGE_P1S_CALIBRATED')
   close(origin-ridge,r['Ridge_forecast_step_MAE_reduction_SOH_pp'])
   close((origin-ridge)/origin*100,r['Ridge_step_relative_to_origin_percent'],D('1e-9'))
   close((origin-ridge)/(m0-ridge)*100,r['Ridge_step_share_of_total_MAE_reduction_percent'],D('1e-9'))
  extra=P/'outputs/H3_INCREMENT_CONCENTRATION.csv'
  if extra.exists():
   values=rows('outputs/H3_INCREMENT_CONCENTRATION.csv');total=sum(D(r['absolute_error_reduction_SOH_pp']) for r in values)
   for r in values:close(D(r['absolute_error_reduction_SOH_pp'])/total*100,r['share_of_H3_total_AE_reduction_percent'],D('1e-10'))
  # All94 historical Ridge coefficient evaluations, plus all188 errors, are checked afresh.
  rp=jread('inputs/ridge_parameters.json');fm=rows('inputs/ridge_fold_metrics.csv');foldreplay=[]
  for horizon in ['H1','H3']:
   for fold in ['fold_1','fold_2','fold_3']:
    for model in ['PERSISTENCE','RIDGE_96H']:
     g=[r for r in cache if (r['horizon'],r['fold'],r['model'])==(horizon,fold,model)];errors=[]
     par=next(m for m in rp if (m['horizon'],m['fold'])==(horizon,fold))
     for r in g:
      prediction=D(r['origin_R_P1S_causal'])
      if model=='RIDGE_96H':
       prediction+=par['intercept_deltaR']+sum((D(r[name])-mean)/scale*coef for name,mean,scale,coef in zip(par['features'],par['train_scaler_mean'],par['train_scaler_scale'],par['coef_standardized_X']))
      close(prediction,r['predicted_R_P1S_causal'])
      errors.append((prediction-D(r['true_R_P1S_causal']))*D('.71383023635')*1000)
     mae=sum(abs(e) for e in errors)/len(errors);rmse=(sum(e*e for e in errors)/len(errors)).sqrt();saved=next(r for r in fm if (r['horizon'],r['fold'],r['model'])==(horizon,fold,model))
     dm=close(mae,saved['MAE_mV_equivalent'],D('1e-9'));dr=close(rmse,saved['RMSE_mV_equivalent'],D('1e-9'))
     foldreplay.append({'horizon':horizon,'fold':fold,'model':model,'n':len(g),'independent_MAE_mV_equivalent':str(mae),'independent_RMSE_mV_equivalent':str(rmse),'historical_MAE_difference':str(dm),'historical_RMSE_difference':str(dr)})
  # Past historical SOH predictions are checked independently from old values, including M0.
  for old in rows('inputs/old_predictions.csv'):
   if old['horizon'] not in ['H1','H3']:continue
   method={'M0_REFERENCE_PERSISTENCE':'M0_REFERENCE_PERSISTENCE','M1_RIDGE96_M2_NC':'M_RIDGE_P1S_CALIBRATED'}[old['model']]
   q=next(r for r in independent if (r['nominal_h'],r['horizon'],r['method'])==(int(old['nominal_h']),old['horizon'],method))
   diff=close(q['Decimal_predicted_SOH'],old['predicted_SOH']);replay.append({'nominal_h':old['nominal_h'],'horizon':old['horizon'],'model':old['model'],'prediction_difference_R':str(diff)})
  csvwrite('outputs/INDEPENDENT_DECIMAL_PREDICTIONS.csv',independent);csvwrite('outputs/INDEPENDENT_HISTORICAL_FOLD_REPLAY.csv',foldreplay);csvwrite('outputs/INDEPENDENT_OLD_SOH_REPLAY.csv',replay)
  write('outputs/INDEPENDENT_VERIFICATION.json',{'PASS':True,'implementation':'stdlib csv, JSON, Decimal60 digits; no production calculation imports','max_prediction_float_vs_Decimal_difference_R':str(maxdiff),
   'method_predictions_verified':len(independent),'historical_P1S_rows_verified':len(cache),'historical_SOH_rows_verified':len(replay),'unique_events':3,'no_fits':True,'no_p_values_or_independence_assumption':True})
  print('INDEPENDENT_DECIMAL_VERIFICATION_PASS',len(independent),'predictions;',len(cache),'historical P1S rows; max float diff',maxdiff)
if __name__=='__main__':run()
