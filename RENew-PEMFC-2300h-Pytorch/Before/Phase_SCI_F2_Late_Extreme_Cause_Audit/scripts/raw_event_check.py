"""Recheck only two already-exposed event regions from their existing raw files.

Two CSVs are streamed once (selected columns); only four bins per event are
retained. No new evaluation region or model input is constructed.
"""
from audit_common import *
ACTIVE=[f'U{i} [V]' for i in range(1,12) if i!=6]
def numeric(s):
    if pd.api.types.is_numeric_dtype(s):return pd.to_numeric(s,errors='coerce')
    v=s.astype('string').str.replace('\ufeff','',regex=False).str.strip();comma=v.str.contains(',',regex=False,na=False);dot=v.str.contains('.',regex=False,na=False)
    v=v.mask(comma&dot,v.str.replace(',','',regex=False));v=v.mask(comma&~dot,v.str.replace(',','.',regex=False))
    return pd.to_numeric(v,errors='coerce')
def main():
    check_protocol();f=raw();summary=[];runs=[];minute=[];identities=[]
    parent=REPO/'数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据/Loadcycle_data/Loadcycle_data'
    for event,(a,b) in EVENTS.items():
        rows=f.loc[a-1:b+1];names=set(rows.source_file_names);assert len(names)==1;source=parent/next(iter(names))
        minimum=float(rows.bin_start_ophrs2_h.min());maximum=float(rows.bin_end_ophrs2_h.max());pieces=[];offset=0
        use=['DateTime','OpHrs2 [h]','I.S.act [A]','operation_point']+ACTIVE
        for chunk in pd.read_csv(source,sep=';',encoding='utf-8-sig',usecols=use,chunksize=50000,low_memory=False):
            ages=numeric(chunk['OpHrs2 [h]']);binstart=np.round(np.floor((ages.to_numpy(float)+1e-10)/.5)*.5,10)
            keep=(binstart>=minimum)&(binstart<maximum)
            if keep.any():
                z=chunk.loc[keep].copy();z['raw_row_index']=offset+np.flatnonzero(keep);z['age_h']=ages.loc[keep].to_numpy();z['bin_start_h']=binstart[keep]
                pieces.append(z)
            offset+=len(chunk)
        z=pd.concat(pieces,ignore_index=True);z['calendar_time']=pd.to_datetime(z.DateTime);z=z.sort_values(['age_h','calendar_time'],kind='stable').reset_index(drop=True)
        for c in ACTIVE+['I.S.act [A]','operation_point']:z[c]=numeric(z[c])
        z['active_10cell_mean_V']=z[ACTIVE].mean(axis=1);z['current_A']=z['I.S.act [A]'];z['zero_current']=abs(z.current_A)<=.1
        z['event']=event
        columns=['event','raw_row_index','calendar_time','age_h','bin_start_h','current_A','zero_current','active_10cell_mean_V','operation_point']+ACTIVE
        cs(f'outputs/RAW_EVENT_SAMPLES_{event}.csv',z[columns])
        for r in rows.itertuples():
            g=z.loc[z.bin_start_h==r.bin_start_ophrs2_h];assert len(g)==r.raw_sample_count
            voltage=float(g.active_10cell_mean_V.mean());off=g.zero_current.to_numpy();onV=float(g.loc[~g.zero_current,'active_10cell_mean_V'].mean());offV=float(g.loc[g.zero_current,'active_10cell_mean_V'].mean()) if off.any() else None
            diff=voltage-r.cell_voltage_mean;assert abs(diff)<1e-14
            assert float(off.mean())==r.zero_current_fraction
            assert pd.Timestamp(g.calendar_time.iloc[0])==pd.Timestamp(r.first_datetime) and pd.Timestamp(g.calendar_time.iloc[-1])==pd.Timestamp(r.last_datetime)
            summary.append(dict(event=event,row_id=r.row_id,age_h=r.bin_end_ophrs2_h,raw_samples=len(g),zero_current_samples=int(off.sum()),on_samples=int((~off).sum()),zero_current_fraction=float(off.mean()),active_samples_mean_V=onV,zero_current_samples_mean_V=offV,whole_bin_mean_V=voltage,frozen_bin_mean_V=r.cell_voltage_mean,numeric_difference_V=diff,raw_calendar_span_h=(g.calendar_time.iloc[-1]-g.calendar_time.iloc[0]).total_seconds()/3600,op_clock_range_h=float(g.age_h.max()-g.age_h.min())))
        ordered=z.sort_values('calendar_time').copy();ordered['run_id']=(ordered.zero_current!=ordered.zero_current.shift()).cumsum()
        for _,g in ordered.groupby('run_id'):
            runs.append(dict(event=event,state='ZERO_CURRENT' if bool(g.zero_current.iloc[0]) else 'POWERED',first_datetime=str(g.calendar_time.iloc[0]),last_datetime=str(g.calendar_time.iloc[-1]),samples=len(g),voltage_mean_V=float(g.active_10cell_mean_V.mean()),current_mean_A=float(g.current_A.mean()),op_age_min_h=float(g.age_h.min()),op_age_max_h=float(g.age_h.max())))
        ordered['minute']=ordered.calendar_time.dt.floor('min')
        for dt,g in ordered.groupby('minute'):minute.append(dict(event=event,calendar_time=str(dt),n=len(g),voltage_V=float(g.active_10cell_mean_V.mean()),current_A=float(g.current_A.mean()),zero_fraction=float(g.zero_current.mean()),age_min_h=float(g.age_h.min()),age_max_h=float(g.age_h.max())))
        identities.append(dict(event=event,source_file=source.relative_to(REPO).as_posix(),source_sha256=sha(source),scanned_rows=offset,retained_raw_samples=len(z),selection='four existing bins a-1..b+1 only; raw row IDs retained; no model inputs changed'))
        print('RAW_EVENT_RECHECKED',event,len(z),flush=True)
    cs('outputs/RAW_BIN_MIXTURE_RECOMPUTATION.csv',summary);cs('outputs/RAW_STATE_RUNS.csv',runs);cs('outputs/RAW_MINUTE_PLOT_DATA.csv',minute);cs('outputs/RAW_SOURCE_IDENTITY.csv',identities)
if __name__=='__main__':main()
