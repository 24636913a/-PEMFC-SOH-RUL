"""Frozen deterministic slope envelope; no new model fit or late measurements."""
from common import *
def run():
 config();assert not (P/'outputs/TRAJECTORY_SENSITIVITY_COMPLETE.json').exists()
 js('configs/TRAJECTORY_SENSITIVITY_CODE_IDENTITY.json',{'code_sha256':sha(__file__),'rule':'Use already fitted slopes and leave-one-out slopes plus zero; deterministic min/max, not confidence interval','fitting':False})
 s=pd.read_csv(P/'outputs/CONDITIONAL_TRAJECTORY_SLOPES.csv');loo=pd.read_csv(P/'outputs/CONDITIONAL_SLOPE_SENSITIVITY.csv');curves=pd.read_csv(P/'outputs/CONDITIONAL_CALENDAR_TRAJECTORIES.csv')
 slopes=np.r_[s.slope_SOH_per_calendar_h.to_numpy(),loo.slope_SOH_per_calendar_h.to_numpy()]
 start=float(curves.iloc[0].projected_SOH);elapsed=np.sort(curves.elapsed_calendar_h.unique())
 cs('outputs/CONDITIONAL_TRAJECTORY_SENSITIVITY_ENVELOPE.csv',pd.DataFrame({'elapsed_calendar_h':elapsed,'low_SOH':start+slopes.min()*elapsed,'high_SOH':start+slopes.max()*elapsed,'is_confidence_interval':False}))
 b=pd.read_csv(P/'outputs/LONG_TRAJECTORY_CALENDAR_BACKTEST.csv');cs('outputs/LONG_TRAJECTORY_CALENDAR_BACKTEST_METRICS.csv',b.groupby('model',as_index=False).agg(n=('abs_error_SOH_pp','size'),MAE_SOH_pp=('abs_error_SOH_pp','mean'),max_abs_SOH_pp=('abs_error_SOH_pp','max')))
 js('outputs/TRAJECTORY_SENSITIVITY_COMPLETE.json',{'slope_models_including_leave_one_out':len(slopes),'minimum_slope':float(slopes.min()),'maximum_slope':float(slopes.max()),'physical_RUL_model_qualified':False,'reason':'Scenario slopes strongly depend on past subset and recovery assumption; no failure/OpHrs2 labels or calibrated uncertainty'})
 print('Deterministic trajectory sensitivity saved')
if __name__=='__main__':firewall();run()
