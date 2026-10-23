"""Primary project metrics versus independent scalar CSV re-calculation."""
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from i_common import c

def project_metrics(frame):
 f=frame.sort_values(['origin_row','horizon_step'])
 y=f.y_true_delta_V.to_numpy(np.float64).reshape(-1,24);p=f.y_pred_delta_V.to_numpy(np.float64).reshape(-1,24)
 result=c.metric(y,p,f.target_row.to_numpy().reshape(-1,24))
 return result

def linear_quantile(values,q):
 ordered=sorted(values);position=(len(ordered)-1)*q;lo=math.floor(position);hi=math.ceil(position)
 return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)

def independent_metrics(frame):
 truth=frame.y_true_delta_V.tolist();pred=frame.y_pred_delta_V.tolist();errors=[p-y for p,y in zip(pred,truth)];n=len(errors)
 mean=math.fsum(truth)/n;ss=math.fsum(e*e for e in errors);den=math.fsum((y-mean)**2 for y in truth)
 targets=defaultdict(list);endpoint=[]
 for target,step,error in zip(frame.target_row,frame.horizon_step,errors):
  targets[int(target)].append(error*error)
  if step==24:endpoint.append(error*error)
 absv=[abs(e)*1000 for e in errors]
 return {'RMSE_mV':math.sqrt(ss/n)*1000,'MAE_mV':math.fsum(absv)/n,'R2':1-ss/den,
 'endpoint_RMSE_mV':math.sqrt(math.fsum(endpoint)/len(endpoint))*1000,
 'P95_mV':linear_quantile(absv,.95),'P99_mV':linear_quantile(absv,.99),'max_abs_error_mV':max(absv),
 'target_time_equal_RMSE_mV':math.sqrt(math.fsum(math.fsum(es)/len(es) for es in targets.values())/len(targets))*1000}

def diagnostics(frame):
 f=frame.copy();f['squared_error_mV2']=((f.y_pred_delta_V-f.y_true_delta_V)*1000)**2
 f['absolute_error_mV']=np.abs(f.y_pred_delta_V-f.y_true_delta_V)*1000
 target=f.groupby('target_row',sort=True).agg(target_age_h=('target_age_h','first'),multiplicity=('horizon_step','size'),SSE_mV2=('squared_error_mV2','sum'),MSE_mV2=('squared_error_mV2','mean'))
 target['SSE_share']=target.SSE_mV2/target.SSE_mV2.sum();top=target.sort_values('SSE_mV2',ascending=False,kind='stable')
 if not math.isclose(target.SSE_mV2.sum(),f.squared_error_mV2.sum(),rel_tol=1e-12):raise RuntimeError('TARGET_SSE_DECOMPOSITION_FAILURE')
 horizon=f.groupby('horizon_step',sort=True).agg(predictions=('target_row','size'),MSE_mV2=('squared_error_mV2','mean'))
 horizon['RMSE_mV']=np.sqrt(horizon.MSE_mV2);horizon['forecast_hours']=horizon.index*.5
 summary={'max_SSE_target_row':int(top.index[0]),'max_SSE_target_age_h':float(top.iloc[0].target_age_h),
 'max_SSE_target_multiplicity':int(top.iloc[0].multiplicity),'max_SSE_target_share':float(top.iloc[0].SSE_share),
 'top5_target_SSE_share':float(top.SSE_share.iloc[:5].sum()),'independent_target_rows':len(target)}
 return summary,target.reset_index(),horizon.reset_index()

def scientific_decision(paired):
 a=int((paired.D20_F2_minus_F0_mV<0).sum());b=int((paired.D21_F2_minus_F1_mV<0).sum())
 if a==3 and b==3:primary='F2_DEVELOPMENT_ASSESSMENT_MULTISEED_GAIN_OBSERVED'
 elif b==3:primary='F2_GATE_GAIN_WITHOUT_NATIVE_SUPERIORITY'
 elif a in (1,2) or b in (1,2):primary='F2_ASSESSMENT_GAIN_SEED_DEPENDENT'
 else:primary='F2_VALIDATION_GAIN_NOT_TRANSFERRED_TO_ASSESSMENT'
 descriptive=[]
 if a in (1,2) or b in (1,2):descriptive.append('F2_ASSESSMENT_GAIN_SEED_DEPENDENT')
 if a==0 or paired.D20_F2_minus_F0_mV.mean()>=0:descriptive.append('F2_VALIDATION_GAIN_NOT_TRANSFERRED_TO_ASSESSMENT')
 return primary,descriptive
