"""冻结阶段2A的纯数值预测核心。无原始数据准备、神经网络、搜索或顶层运行。"""
import json
import numpy as np
import pandas as pd


def linear(x, y, nonnegative=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    den = np.sum((x-xm)**2)
    b = np.sum((x-xm)*(y-ym))/den if den > 0 else 0.
    if nonnegative:
        b = max(0., b)
    return float(ym-b*xm), float(b)


def score(y, p, base=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p-y
    r = dict(n=len(e), RMSE_mV=float(np.sqrt(np.mean(e**2))*1000),
             MAE_mV=float(np.mean(np.abs(e))*1000), P95_mV=float(np.quantile(abs(e), .95)*1000),
             max_mV=float(max(abs(e))*1000))
    if base is not None:
        se = e/np.asarray(base)*100
        r.update(RMSE_SOH_pp=float(np.sqrt(np.mean(se**2))), MAE_SOH_pp=float(np.mean(abs(se))))
    return r


def metrics(pred, keys):
    rows = []
    if pred.empty:
        return pd.DataFrame(columns=keys+['n','RMSE_mV','MAE_mV','RMSE_SOH_pp','MAE_SOH_pp'])
    for group, d in pred.groupby(keys, dropna=False, sort=True):
        if not isinstance(group, tuple):
            group = (group,)
        rows.append(dict(zip(keys, group), **score(d.true_v, d.pred_v, d.base_V)))
    return pd.DataFrame(rows)


def continuous(obs, dataset, lookback, min_history, horizons):
    """每个起点重新拟合已可用局部历史；未来真实时间仅在输出审计中。"""
    pred, exclusions, models = [], [], []
    for scheme, d in obs.groupby('scheme', sort=True):
        d = d.sort_values('id').reset_index(drop=True)
        ix = {int(r.id): r for r in d.itertuples()}
        baseline = d[d.valid & (d.partition=='Train')].head(10 if dataset=='FC1' else 5)
        base_ready_id = int(baseline.id.max()) if len(baseline)==(10 if dataset=='FC1' else 5) else np.inf
        for o in d.itertuples():
            if not o.valid or not np.isfinite(o.base_V) or o.id < base_ready_id:
                for h in horizons:
                    exclusions.append(dict(dataset=dataset, scheme=scheme, origin_id=o.id, horizon=h,
                                           reason='ORIGIN_INVALID_OR_BASE_NOT_YET_AVAILABLE'))
                continue
            hist = d[(d.id <= o.id) & (d.id > o.id-lookback) & d.valid]
            # A cycle is available at cycle end, FC1 hourly mean at hour end.
            if len(hist) < min_history:
                for h in horizons:
                    exclusions.append(dict(dataset=dataset, scheme=scheme, origin_id=o.id, horizon=h,
                                           reason='HISTORY_TOO_SHORT', n_history=len(hist)))
                continue
            a, b = linear(hist.time, hist.v)
            models.append(dict(dataset=dataset, scheme=scheme, origin_id=o.id, n_history=len(hist),
                               history_ids=json.dumps(hist.id.tolist()), history_last_id=int(hist.id.max()),
                               a=a, b=b, coordinate='hour' if dataset=='FC1' else 'physical_cycle_index'))
            for h in horizons:
                ids = list(range(int(o.id)+1, int(o.id)+h+1))
                targets = [ix.get(i) for i in ids]
                reasons = []
                if any(t is None for t in targets):
                    reasons.append('TARGET_OUTSIDE_RECORD')
                elif any(not t.valid for t in targets):
                    reasons.append('MISSING_OR_REJECTED_TARGET_IN_BLOCK')
                if all(t is not None for t in targets):
                    parts = {t.partition for t in targets}
                    if len(parts) != 1 or 'CROSS_BOUNDARY' in parts:
                        reasons.append('TARGET_BLOCK_CROSSES_PARTITION')
                if reasons:
                    exclusions.append(dict(dataset=dataset, scheme=scheme, origin_id=o.id, horizon=h,
                                           reason='|'.join(reasons), n_history=len(hist)))
                    continue
                for t in targets:
                    step = int(t.id-o.id)
                    for method, p in [('Persistence', o.v), ('LocalLinear', a+b*t.time)]:
                        pred.append(dict(dataset=dataset, scheme=scheme, partition=t.partition,
                            origin_id=int(o.id), target_id=int(t.id), horizon=h, step=step, method=method,
                            origin_time=float(o.time), target_time=float(t.time),
                            origin_operating_h=float(o.time if dataset=='FC1' else o.available_h),
                            target_operating_h=float(t.time if dataset=='FC1' else t.available_h),
                            actual_lead_h=float(t.time-o.time if dataset=='FC1' else t.available_h-o.available_h),
                            true_v=float(t.v), pred_v=float(p), base_V=float(o.base_V),
                            true_D=float(o.base_V-t.v), pred_D=float(o.base_V-p),
                            true_SOH=float(t.v/o.base_V), pred_SOH=float(p/o.base_V), n_history=len(hist)))
    return pd.DataFrame(pred), pd.DataFrame(exclusions), pd.DataFrame(models)


def ridge_fit(X, y, alpha, minimum):
    X, y = np.asarray(X, float), np.asarray(y, float)
    rec = dict(n=len(y), minimum=minimum, alpha=alpha)
    if len(y) < minimum:
        return None, dict(rec, reason='INSUFFICIENT_PAST_TRANSITIONS')
    mu, sd = X.mean(0), X.std(0)
    if np.any(sd <= 1e-14):
        return None, dict(rec, reason='ZERO_SCALE_OR_RANK_DEFICIENT')
    Z = (X-mu)/sd
    A = np.column_stack([np.ones(len(X)), Z])
    rank, cond = np.linalg.matrix_rank(A), np.linalg.cond(A)
    rec.update(mean=mu.tolist(), scale=sd.tolist(), rank=int(rank), condition=float(cond))
    if rank < A.shape[1] or cond > 100:
        return None, dict(rec, reason='RANK_OR_CONDITION_REJECTED')
    penalty = np.diag([0.]+[alpha]*X.shape[1])
    coef = np.linalg.solve(A.T@A+penalty, A.T@y)
    rec.update(coef=coef.tolist(), reason='OK')
    return (mu, sd, coef), rec


def sparse(obs, alpha=1.):
    records, logs = [], []
    for (dataset, scheme), d in obs[obs.primary].groupby(['电堆', 'scheme'], sort=True):
        d = d.sort_values('标称小时' if dataset=='FC1' else '实际测试结束').reset_index(drop=True)
        for i in range(len(d)-1):
            o, t = d.iloc[i], d.iloc[i+1]
            basic = dict(dataset=dataset, scheme=scheme, origin_id=i, target_id=i+1,
                origin_file=o['文件名'], target_file=t['文件名'], origin_nominal_h=o['标称小时'],
                target_nominal_h=t['标称小时'], origin_completed=o['实际测试结束'],
                target_completed=t['实际测试结束'], clock=o['时钟资格'],
                origin_operating_upper_h=o['累计小时上界'], target_operating_upper_h=t['累计小时上界'],
                available_assumption=o.available_assumption, release_confirmed=False,
                true_D=t.D_ref_V, base_V=o.base_V)
            preds = {}
            valid_target = bool(o.valid and t.valid and np.isfinite(o.D_ref_V) and np.isfinite(t.D_ref_V))
            if valid_target:
                preds['Persistence'] = (float(o.D_ref_V), 'OK')
                last = d.iloc[max(0,i-2):i+1]
                if len(last)==3 and last.valid.all():
                    a, b = linear(np.arange(max(0,i-2),i+1), last.D_ref_V)
                    preds['LocalLinear3'] = (a+b*(i+1), 'OK')
                else:
                    preds['LocalLinear3'] = (np.nan, 'LESS_THAN_3_CONSECUTIVE_DIAGNOSES')
            # Both regressions use the SAME complete 2-feature historical transition set.
            X, y, transitions = [], [], []
            for k in range(i):
                u, v = d.iloc[k], d.iloc[k+1]
                if u.valid and v.valid and np.isfinite([u.D_ref_V,u.D_low_V,v.D_ref_V]).all():
                    X.append([u.D_ref_V,u.D_low_V]);y.append(v.D_ref_V-u.D_ref_V);transitions.append([k,k+1])
            for method, dims, minimum in [('Ridge1D',1,4),('Ridge2D',2,6)]:
                fit, rec = ridge_fit(np.asarray(X).reshape(-1,2)[:,:dims], y, alpha, minimum)
                rec.update(basic, method=method, historical_transitions=transitions)
                logs.append(rec)
                if valid_target and fit is not None:
                    mu, sd, coef = fit
                    new = np.array([o.D_ref_V,o.D_low_V])[:dims]
                    p = o.D_ref_V + np.r_[1., (new-mu)/sd]@coef
                    preds[method] = (float(p), 'OK')
                else:
                    preds[method] = (np.nan, 'ORIGIN_OR_TARGET_UNSUPPORTED' if not valid_target else rec['reason'])
            for method in ['Persistence','LocalLinear3','Ridge1D','Ridge2D']:
                p, reason = preds.get(method, (np.nan,'ORIGIN_OR_TARGET_UNSUPPORTED'))
                records.append(dict(basic, method=method, pred_D=p, reason=reason,
                    true_v=o.base_V-t.D_ref_V, pred_v=o.base_V-p,
                    true_SOH=1-t.D_ref_V/o.base_V, pred_SOH=1-p/o.base_V,
                    n_past_transitions=len(y), training_label_last_id=max([k[1] for k in transitions],default=-1)))
    rows = pd.DataFrame(records)
    if rows.empty:
        return rows, rows.copy(), logs
    good = rows[rows.pred_D.notna()].copy()
    counts = good.groupby(['dataset','scheme','origin_id']).method.nunique()
    fairkeys = set(counts[counts==4].index)
    fair = good[[tuple(x) in fairkeys for x in good[['dataset','scheme','origin_id']].itertuples(index=False,name=None)]].copy()
    return rows, fair, logs


def rul(obs, origins, drops, windows, max_h=1000):
    d = obs[obs.valid].sort_values('time').copy()
    results, curves, selections = [], [], []
    base = float(d.base_V.iloc[0])
    for origin in origins:
        hist = d[d.time<=origin]
        cut = int(len(hist)*.8)
        train, val = hist.iloc[:cut], hist.iloc[cut:]
        inner_origin = float(train.time.max())
        candidates = []
        for w in windows:
            sub = train[train.time>inner_origin-w]
            if len(sub) < 24:
                selections.append(dict(origin_h=origin, window_h=w, reason='INNER_HISTORY_TOO_SHORT', n=len(sub)))
                continue
            a,b = linear(sub.time,sub.D,True)
            err = a+b*val.time.to_numpy()-val.D.to_numpy()
            rec = dict(origin_h=origin, window_h=w, inner_origin_h=inner_origin,
                       inner_train_ids=sub.id.tolist(),inner_validation_ids=val.id.tolist(),
                       n=len(sub), RMSE_V=float(np.sqrt(np.mean(err**2))),
                       MAE_V=float(np.mean(abs(err))), reason='OK')
            selections.append(rec);candidates.append(rec)
        selected = min(candidates,key=lambda r:(r['RMSE_V'],r['MAE_V'],r['window_h'])) if candidates else None
        pars = {'Persistence':(float(hist.D.iloc[-1]),0.,0)}
        if selected:
            w = selected['window_h'];sub=hist[hist.time>origin-w]
            if len(sub)>=24:
                a,b=linear(sub.time,sub.D,True);pars['LocalLinear']=(a,b,w)
        for method,(a,b,w) in pars.items():
            future = np.arange(origin+1,origin+max_h+1,dtype=float)
            yh = a+b*future
            for x,p in zip(future,yh):
                curves.append(dict(origin_h=origin,method=method,time_h=x,pred_D_V=p,pred_v=base-p,
                                   base_V=base,selected_window_h=w,a=a,b=b,history_last_h=hist.time.max()))
            for drop in drops:
                crossing=d[d.D>=drop*base]
                observed=float(crossing.time.iloc[0]) if len(crossing) else np.nan
                before=d[d.time<observed] if np.isfinite(observed) else d
                last_above=float(before.time.max()) if len(before) else np.nan
                pred=future[yh>=drop*base]
                pc=float(pred[0]) if len(pred) else np.nan
                eligible=bool(hist.D.iloc[-1]<drop*base and (not np.isfinite(observed) or observed>origin))
                both=eligible and np.isfinite(observed) and np.isfinite(pc)
                results.append(dict(origin_h=origin,method=method,drop=drop,main=drop==.04,
                    threshold_V=base*(1-drop),base_V=base,selected_window_h=w,history_last_h=float(hist.time.max()),
                    actual_first_cross_h=observed,actual_previous_qualified_h=last_above,
                    actual_right_censored=not np.isfinite(observed),actual_observation_end_h=float(d.time.max()),
                    pred_first_cross_h=pc,pred_right_censored=not np.isfinite(pc),prediction_end_h=origin+max_h,
                    origin_eligible=eligible,status='CONDITIONAL_THRESHOLD_RUL' if both else
                    ('ORIGIN_ALREADY_CROSSED_OR_BELOW' if not eligible else 'CENSORED'),
                    actual_RUL_h=observed-origin if eligible and np.isfinite(observed) else np.nan,
                    pred_RUL_h=pc-origin if eligible and np.isfinite(pc) else np.nan,
                    abs_error_h=abs(pc-observed) if both else np.nan,
                    relative_error_pct=abs(pc-observed)/(observed-origin)*100 if both else np.nan,
                    meaning='OBSERVED_CONDITION_THRESHOLD_NOT_CERTIFIED_EOL'))
    return pd.DataFrame(results),pd.DataFrame(curves),selections
