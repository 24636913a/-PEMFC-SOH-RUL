from pathlib import Path
import numpy as np
import pandas as pd

root=Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_TTD_TimeXer_Failure_Diagnosis")
out=root/'outputs/fold3_nonzero/fold3_segment_contribution.csv'
segments=pd.read_csv(out)
points=pd.read_csv(root/'outputs/fold3_nonzero/fold3_nonzero_pointwise.csv')
ranked=segments.sort_values('positive_D_sum_mV2',ascending=False).temporal_segment_id.tolist()
records=[]
for k in (1,2,5):
    removed=set(ranked[:k]); keep=points[~points.temporal_segment_id.isin(removed)]
    if keep.empty:
        records.append({'temporal_segment_id':f'SENSITIVITY_REMOVE_TOP_{k}','start_time':'DIAGNOSTIC_ONLY','end_time':'DIAGNOSTIC_ONLY','duration_h':np.nan,'unique_target_count':0,'prediction_point_count':0,'mean_D_mV2':np.nan,'positive_D_sum_mV2':np.nan,'negative_D_sum_mV2':np.nan,'net_D_sum_mV2':np.nan,'P1_RMSE_mV':np.nan,'P2_RMSE_mV':np.nan,'current_MAE_A':np.nan,'q_MAE':np.nan,'C_delta_P95_mV':np.nan,'C_delta_P99_mV':np.nan,'C_delta_max_mV':np.nan,'affected_horizons':'NOT_ESTIMABLE_ALL_FOUR_SEGMENTS_REMOVED','rank_positive_D':np.nan})
        continue
    records.append({'temporal_segment_id':f'SENSITIVITY_REMOVE_TOP_{k}','start_time':'DIAGNOSTIC_ONLY','end_time':'DIAGNOSTIC_ONLY','duration_h':np.nan,'unique_target_count':keep.target_row.nunique(),'prediction_point_count':len(keep),'mean_D_mV2':keep.D_mV2.mean(),'positive_D_sum_mV2':np.maximum(keep.D_mV2,0).sum(),'negative_D_sum_mV2':np.minimum(keep.D_mV2,0).sum(),'net_D_sum_mV2':keep.D_mV2.sum(),'P1_RMSE_mV':np.sqrt(np.mean(keep.P1_error_mV**2)),'P2_RMSE_mV':np.sqrt(np.mean(keep.P2_error_mV**2)),'current_MAE_A':np.mean(abs(keep.predicted_P2_current_A-keep.actual_current_A)),'q_MAE':np.mean(abs(keep.predicted_P2_q-keep.actual_q)),'C_delta_P95_mV':np.quantile(abs(keep.C_delta_mV),.95),'C_delta_P99_mV':np.quantile(abs(keep.C_delta_mV),.99),'C_delta_max_mV':abs(keep.C_delta_mV).max(),'affected_horizons':'DIAGNOSTIC_ONLY','rank_positive_D':np.nan})
pd.concat([segments,pd.DataFrame(records)],ignore_index=True).to_csv(out,index=False,encoding='utf-8-sig',lineterminator='\n')
