from common import *
def build(f,start,end):
    """H-compatible origin convention: origin belongs to partition, then 24 targets.

    Future covariates are inspected only for historical eligibility parity, NEVER
    returned as forecast inputs. All inputs end exactly at origin.
    """
    v=f[FEATURES].to_numpy(float); y=f.cell_voltage_mean.to_numpy(float)
    ages=f.bin_end_ophrs2_h.to_numpy(float); valid=f.bin_valid_for_windows.astype(bool).to_numpy();seg=f.temporal_segment_id.to_numpy()
    xs=[];ys=[];origins=[];targets=[];excluded=[];legal=[]
    for o in range(start,end):
        first=o-47;last=o+24;reasons=[]
        if first<0:reasons.append('HISTORY_BEFORE_DATASET')
        if last>=end:reasons.append('24_TARGETS_OUTSIDE_PARTITION')
        if not reasons:
            block=slice(first,last+1)
            if not valid[block].all():reasons.append('INVALID_BIN')
            if np.any(seg[block]!=seg[o]):reasons.append('TEMPORAL_SEGMENT_BREAK')
            if not np.allclose(np.diff(ages[block]),.5,atol=1e-6,rtol=0):reasons.append('OPERATING_TIME_GAP')
            if not np.isfinite(v[block]).all():reasons.append('NONFINITE_RAW14_OR_VOLTAGE')
        row=dict(origin_row=o,history_first_row=first,history_last_row=o,target_first_row=o+1,target_last_row=last,origin_age_h=float(ages[o]))
        if reasons:excluded.append(dict(**row,primary_reason=reasons[0],all_reasons='|'.join(reasons)));continue
        legal.append(row);xs.append(v[first:o+1]);ys.append(y[o+1:last+1]-y[o]);origins.append(o);targets.append(np.arange(o+1,last+1))
    origins=np.asarray(origins,dtype=np.int64);targets=np.asarray(targets,dtype=np.int64).reshape(-1,24)
    d=dict(x_raw=np.asarray(xs,np.float32).reshape(-1,48,14),y=np.asarray(ys,np.float32).reshape(-1,24),origins=origins,targets=targets,origin_age_h=ages[origins],target_age_h=ages[targets],origin_raw_voltage_V=y[origins],target_raw_voltage_V=y[targets])
    return d,pd.DataFrame(legal),pd.DataFrame(excluded)
def save_windows(name,d,legal,excluded):
    cs(f'outputs/audits/{name}_LEGAL_WINDOWS.csv',legal);cs(f'outputs/audits/{name}_EXCLUDED_WINDOWS.csv',excluded)
    np.savez_compressed(path(f'outputs/audits/{name}_WINDOWS.npz'),**d)
    return dict(partition=name,records=len(legal)+len(excluded),legal_origins=len(legal),prediction_queries=24*len(legal),unique_targets=len(np.unique(d['targets'])),excluded_origins=len(excluded),primary_reason_counts=excluded.primary_reason.value_counts().to_dict())
