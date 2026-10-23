"""Query-equal dynamic DeltaV errors; distinct-target weighting is explicit."""
import numpy as np
import pandas as pd
def measure(frame):
 e=(frame.y_pred_delta_V-frame.y_true_delta_V).to_numpy(float)*1000;ae=np.abs(e)
 target=pd.DataFrame({'target_row':frame.target_row.to_numpy(),'squared_error_mV2':e**2}).groupby('target_row').squared_error_mV2.mean()
 endpoint=e[frame.horizon_step.to_numpy()==24]
 summary=dict(RMSE_mV=float(np.sqrt(np.mean(e**2))),MAE_mV=float(ae.mean()),endpoint_12h_RMSE_mV=float(np.sqrt(np.mean(endpoint**2))),
  target_time_equal_RMSE_mV=float(np.sqrt(target.mean())),P95_abs_error_mV=float(np.quantile(ae,.95)),P99_abs_error_mV=float(np.quantile(ae,.99)),
  max_abs_error_mV=float(ae.max()),validation_origins=int(frame.origin_row.nunique()),prediction_queries=len(e),distinct_target_rows=len(target))
 horizons=[]
 for h,g in frame.groupby('horizon_step'):
  er=(g.y_pred_delta_V-g.y_true_delta_V).to_numpy()*1000;horizons.append(dict(horizon_step=int(h),lead_h=float(h*.5),RMSE_mV=float(np.sqrt(np.mean(er**2))),MAE_mV=float(np.abs(er).mean())))
 return summary,pd.DataFrame(horizons),target.rename('mean_squared_error_mV2').reset_index()
