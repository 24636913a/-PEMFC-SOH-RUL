"""Fixed 96h residual Ridge; adapted from the frozen prior stage, never imports it."""
import math
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
FEATURES=['origin_R','history_mean_R','history_std_R','history_slope_R_per_h','origin_minus_median_R']
ALPHAS=[.0001,.01,1.,100.]
def train_validation(frame,fold):
    original=frame.loc[frame.fold==fold]
    train=original.loc[(original.role=='train')&original.L96_eligible].sort_values('target_p1s_measurement_end_ophrs2_h').copy()
    validation=original.loc[(original.role=='validation')&original.L96_eligible].sort_values('target_p1s_measurement_end_ophrs2_h').copy()
    # The final model's training labels must already exist at its first forecast origin.
    candidates=validation if len(validation) else original.loc[original.role=='validation']
    cutoff=float(candidates.origin_p1s_measurement_end_ophrs2_h.min())
    purge=train.target_p1s_measurement_end_ophrs2_h>cutoff+1e-9
    rejected=train.loc[purge].copy();train=train.loc[~purge].copy()
    return train,validation,rejected,cutoff

def inner_split(train):
    n=math.ceil(.15*len(train));selection=train.iloc[-n:].copy();fit=train.iloc[:-n].copy()
    cutoff=float(selection.origin_p1s_measurement_end_ophrs2_h.min())
    purge=fit.target_p1s_measurement_end_ophrs2_h>cutoff+1e-9
    rejected=fit.loc[purge].copy();fit=fit.loc[~purge].copy()
    if len(fit)<2 or len(selection)<1:raise ValueError('INSUFFICIENT_INNER_TRAIN')
    return fit,selection,rejected,cutoff

def ridge_fit(train,validation,length):
    names=[f'L{length}_{f}' for f in FEATURES];fit,selection,purged,cutoff=inner_split(train)
    inner_scaler=StandardScaler().fit(fit[names].to_numpy(float))
    x=inner_scaler.transform(fit[names].to_numpy(float));xval=inner_scaler.transform(selection[names].to_numpy(float))
    y=(fit.target_R_P1S_causal-fit.origin_R_P1S_causal).to_numpy(float)
    yval=(selection.target_R_P1S_causal-selection.origin_R_P1S_causal).to_numpy(float)
    scores=[]
    for alpha in ALPHAS:
        estimator=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(x,y)
        scores.append({'alpha':alpha,'inner_MAE_R':float(np.mean(np.abs(estimator.predict(xval)-yval)))})
    alpha=min(scores,key=lambda r:(r['inner_MAE_R'],r['alpha']))['alpha']
    scaler=StandardScaler().fit(train[names].to_numpy(float))
    target=(train.target_R_P1S_causal-train.origin_R_P1S_causal).to_numpy(float)
    model=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(scaler.transform(train[names].to_numpy(float)),target)
    pred=model.predict(scaler.transform(validation[names].to_numpy(float))) if len(validation) else np.array([])
    record={'length_h':length,'alpha':alpha,'features':names,'inner_selection_n':len(selection),'inner_fit_n':len(fit),
        'inner_fit_samples':fit.sample_index.astype(int).tolist(),'inner_selection_samples':selection.sample_index.astype(int).tolist(),
        'inner_purged_samples':purged.sample_index.astype(int).tolist(),'inner_label_availability_cutoff_h':cutoff,
        'inner_scaler_mean':inner_scaler.mean_.tolist(),'inner_scaler_scale':inner_scaler.scale_.tolist(),
        'alpha_grid_scores':scores,'train_samples':train.sample_index.astype(int).tolist(),'validation_samples':validation.sample_index.astype(int).tolist(),
        'train_scaler_mean':scaler.mean_.tolist(),'train_scaler_scale':scaler.scale_.tolist(),
        'coef_standardized_X':model.coef_.tolist(),'intercept_deltaR':float(model.intercept_),
        'target_standardized':False,'solver':'cholesky','ridge_fits':5}
    return pred,record

def error_metrics(truth,pred,VREF):
    e=np.asarray(pred)-np.asarray(truth);ae=np.abs(e)
    if not len(e):return {'n':0,**{name:None for name in ['MAE_R','RMSE_R','MAE_mV_equivalent','RMSE_mV_equivalent','P95_abs_error_R','P95_abs_error_mV_equivalent','max_abs_error_R','max_abs_error_mV_equivalent']}}
    return {'n':len(e),'MAE_R':float(ae.mean()),'RMSE_R':float(np.sqrt(np.mean(e**2))),
        'MAE_mV_equivalent':float(ae.mean()*VREF*1000),'RMSE_mV_equivalent':float(np.sqrt(np.mean(e**2))*VREF*1000),
        'P95_abs_error_R':float(np.quantile(ae,.95)),'P95_abs_error_mV_equivalent':float(np.quantile(ae,.95)*VREF*1000),
        'max_abs_error_R':float(ae.max()),'max_abs_error_mV_equivalent':float(ae.max()*VREF*1000)}
