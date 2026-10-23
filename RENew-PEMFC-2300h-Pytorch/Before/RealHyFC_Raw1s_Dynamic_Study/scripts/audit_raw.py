"""P0: stream original CSVs, verify LFS OIDs, preserve raw timestamps, build mmap.

No interpolation or original/legacy writes. Run with python -B scripts/audit_raw.py.
"""
from __future__ import annotations
import collections
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
RESEARCH = STUDY.parent
REPO = RESEARCH.parent
RAW = REPO / '数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据/Loadcycle_data/Loadcycle_data'
CACHE = STUDY / 'cache'
OUT = STUDY / 'outputs'
ACTIVE = [f'U{i} [V]' for i in range(1, 12) if i != 6]
FROZEN_FEATURES = [
    'mean10_V','std10_V','range10_V','min10_V','max10_V',
    'actual_current_A','current_past30_std_A','current_past30_ptp_A',
    'current_density_A_cm2','density_past30_std_A_cm2','density_past30_ptp_A_cm2',
    'past1s_load_change','past1s_operation_point_change','zero_current_flag',
    'set_current_A','internal_set_current_A','anode_inlet_T_C','cathode_inlet_T_C',
    'coolant_inlet_T_C','coolant_outlet_T_C','anode_inlet_p_kPag',
    'cathode_inlet_p_kPag','operation_point','OpHrs2_h']
EXTRA = ['I.S.set [A]','I.S.set.internal [A]','T.Si.A [°C]','T.Si.C [°C]',
         'T.Si.CL [°C]','T.So.CL [°C]','p.Si.A [kPag]','p.Si.C [kPag]',
         'operation_point','OpHrs2 [h]']

def save(name, obj):
    (STUDY / name).parent.mkdir(parents=True, exist_ok=True)
    (STUDY / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                        allow_nan=False)+'\n', encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def numeric(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce')
    x = s.astype('string').str.strip().str.replace('\ufeff','',regex=False)
    both = x.str.contains(',',regex=False,na=False) & x.str.contains('.',regex=False,na=False)
    x = x.mask(both, x.str.replace(',','',regex=False))
    return pd.to_numeric(x.str.replace(',','.',regex=False),errors='coerce')

def main():
    started = time.time()
    CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'raw_audit.json').exists():
        raise RuntimeError('Audit already exists; reuse recorded cache instead of overwriting.')
    files = sorted(RAW.glob('Loadcycle_*.csv'),key=lambda p:int(re.search(r'_(\d+)-',p.name)[1]))
    assert len(files)==17, len(files)
    base_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    streams = {k:(CACHE/f'{k}.bin').open('wb') for k in
               ['x','time_ms','age_h','file_id','source_row','segment','valid']}
    identities=[]; cols=None; all_cols=[]; file_schemas=[]; total=0; segment_id=-1; previous=None
    dt_hist=collections.Counter(); breaks=[]; state_runs=[]; run=None
    stats={}; changes=collections.Counter(); examples={}; clock=collections.Counter()
    # Tail only contains original observations. Rolling features reset at every segment break.
    tail=pd.DataFrame(); segment_meta=[]; seg_record=None
    try:
      for fid,path in enumerate(files):
        with path.open('rb') as f: assert not f.read(200).startswith(b'version https://git-lfs.github.com/spec/v1')
        pointer=subprocess.check_output(['git','show','HEAD:'+path.relative_to(REPO).as_posix()],cwd=REPO)
        match=re.search(rb'oid sha256:([0-9a-f]{64})\nsize (\d+)',pointer)
        assert match, 'Expected tracked Git LFS pointer in committed tree'
        actual_sha=sha(path)
        assert actual_sha==match[1].decode() and path.stat().st_size==int(match[2]), 'Incomplete LFS payload'
        offset=0; first_time=None; file_first_age=None; file_last_age=None
        file_dt=collections.Counter(); file_breaks=0; first_global=total
        for frame in pd.read_csv(path,sep=';',encoding='utf-8-sig',chunksize=50000,low_memory=False):
          if cols is None:
            cols=frame.columns.tolist()
            save('outputs/field_schema.json', [dict(index=i,name=c,unit=(re.search(r'\[([^]]+)\]',c)[1] if re.search(r'\[([^]]+)\]',c) else 'categorical/time')) for i,c in enumerate(cols)])
          if offset==0:
            file_schemas.append(dict(file=path.name,columns=frame.columns.tolist(),missing_vs_first=sorted(set(cols)-set(frame.columns)),extra_vs_first=sorted(set(frame.columns)-set(cols))))
            all_cols=list(dict.fromkeys(all_cols+frame.columns.tolist()))
          assert set(ACTIVE+EXTRA+['DateTime','I.S.act [A]','j.S.act [A/cm2]']).issubset(frame.columns), 'Missing essential model instruments'
          dates=pd.to_datetime(frame['DateTime'],errors='raise')
          raw_ns=dates.to_numpy(dtype='datetime64[ns]').astype(np.int64)
          assert np.all(raw_ns%1_000_000==0), 'Source has submillisecond timestamps; cache schema must be extended, not truncated'
          ts=raw_ns//1_000_000
          assert np.all(ts>0)
          values=pd.DataFrame({c:numeric(frame[c]).to_numpy(dtype=float,na_value=np.nan) for c in frame.columns if c!='DateTime'})
          a=values.to_numpy(); ages=values['OpHrs2 [h]'].to_numpy()
          for j,c in enumerate(values.columns):
            v=a[:,j]; finite=np.isfinite(v); clean=v[finite]
            st=stats.setdefault(c,dict(count=0,missing=0,min=None,max=None,sum=0.,sum2=0.,zero=0))
            st['count']+=int(finite.sum()); st['missing']+=int((~finite).sum())
            if clean.size:
              lo=float(clean.min()); hi=float(clean.max())
              st['min']=lo if st['min'] is None else min(st['min'],lo)
              st['max']=hi if st['max'] is None else max(st['max'],hi)
              st['sum']+=float(clean.sum()); st['sum2']+=float(np.dot(clean,clean)); st['zero']+=int((np.abs(clean)<=.001).sum())
            pairs=np.isfinite(v[1:])&np.isfinite(v[:-1]); changed=pairs&(v[1:]!=v[:-1])
            changes[c]+=int(changed.sum())
            if c not in examples: examples[c]=v[np.linspace(0,len(v)-1,min(1000,len(v))).astype(int)].tolist()
          cells=values[ACTIVE].to_numpy(); mean=cells.mean(1)
          curr=values['I.S.act [A]'].to_numpy(); density=values['j.S.act [A/cm2]'].to_numpy()
          op=values['operation_point'].to_numpy()
          x=np.zeros((len(frame),24),np.float32)
          x[:,:5]=np.column_stack([mean,cells.std(1),np.ptp(cells,axis=1),cells.min(1),cells.max(1)])
          x[:,5]=curr; x[:,8]=density; x[:,13]=(np.abs(curr)<=.1)
          x[:,14:]=values[EXTRA].to_numpy(dtype=float)
          # Exclude only invalid/missing instruments, never data-dependent large jumps.
          valid=np.isfinite(x).all(1)&np.isfinite(cells).all(1)&(cells>=-.1).all(1)&(cells<=1.5).all(1)
          dts=np.diff(ts,prepend=previous['time'] if previous else ts[0]-1000)
          dages=np.diff(ages,prepend=previous['age'] if previous else ages[0])
          hist=collections.Counter(dts[1 if previous is None else 0:].tolist())
          dt_hist.update(hist); file_dt.update(hist)
          adjacency=(dts>=950)&(dts<=1050)&(dages>=-1e-7)
          if previous and not previous['valid']: adjacency[0]=False
          adjacency[1:] &= valid[:-1]
          adjacency &= valid
          # Conservative file seam firewall even if a seam happens to be 1 second.
          adjacency[0] &= offset>0
          new=~adjacency
          if previous is None: new[0]=True
          ids=segment_id+np.cumsum(new).astype(np.int32)
          real_start=1 if previous is None else 0
          real_dt=dts[real_start:]; real_age_delta=dages[real_start:]
          clock['OpHrs2_frozen_pairs']+=int((np.abs(real_age_delta)<1e-9).sum())
          clock['OpHrs2_backward_pairs']+=int((real_age_delta < -1e-7).sum())
          clock['DateTime_backward_or_duplicate_pairs']+=int((real_dt<=0).sum())
          clock['near_1second_pairs']+=int(((real_dt>=950)&(real_dt<=1050)).sum())
          for k in np.flatnonzero(new):
            breaks.append(dict(global_row=total+int(k),file=path.name,source_row=offset+int(k),DateTime=str(dates.iloc[k]),age_h=float(ages[k]),gap_s=float(dts[k]/1000),age_delta_h=float(dages[k]),file_boundary=bool(k==0 and offset==0),instrument_valid=bool(valid[k])))
          file_breaks+=int(new.sum())
          # Vectorized causal 30-observation (=approx 30 s) rolling, per actual segment.
          roll=pd.DataFrame({'current':curr,'density':density,'op':op,'seg':ids})
          combined=pd.concat([tail,roll],ignore_index=True)
          group=combined.groupby('seg',sort=False)
          for col,jstd,jptp in [('current',6,7),('density',9,10)]:
            r=group[col].rolling(30,min_periods=30)
            std=r.std(ddof=0).reset_index(level=0,drop=True).sort_index().to_numpy()
            rng=(r.max()-r.min()).reset_index(level=0,drop=True).sort_index().to_numpy()
            x[:,jstd]=std[-len(frame):]; x[:,jptp]=rng[-len(frame):]
          change_i=group['current'].diff().abs().gt(.1).to_numpy()
          change_op=group['op'].diff().abs().gt(0).to_numpy()
          x[:,11]=change_i[-len(frame):]; x[:,12]=change_op[-len(frame):]
          valid &= np.isfinite(x).all(1)
          tail=combined.loc[combined.seg==ids[-1]].tail(29).copy()
          segment_id=int(ids[-1])
          # Segments/statistics retain sample identities rather than synthetic clock rows.
          for sid,g in pd.DataFrame({'sid':ids,'t':ts,'age':ages,'valid':valid}).groupby('sid',sort=False):
            rec=dict(segment_id=int(sid),first_row=total+int(g.index[0]),last_row=total+int(g.index[-1]),rows=len(g),first_ms=int(g.t.iloc[0]),last_ms=int(g.t.iloc[-1]),first_age_h=float(g.age.iloc[0]),last_age_h=float(g.age.iloc[-1]),file_id=fid)
            if segment_meta and segment_meta[-1]['segment_id']==int(sid):
              segment_meta[-1].update(last_row=rec['last_row'],rows=segment_meta[-1]['rows']+len(g),last_ms=rec['last_ms'],last_age_h=rec['last_age_h'])
            else: segment_meta.append(rec)
          # Run-length state audit (zero current is a proxy, not an instrument run flag).
          off=np.abs(curr)<=.1
          state_new=np.r_[True,off[1:]!=off[:-1]]|new
          cuts=np.r_[np.flatnonzero(state_new),len(frame)]
          for left,right in zip(cuts[:-1],cuts[1:]):
            same=run and run['zero_current']==bool(off[left]) and not new[left]
            if same:
              run.update(last_ms=int(ts[right-1]),last_age_h=float(ages[right-1]),samples=run['samples']+int(right-left))
            else:
              if run: state_runs.append(run)
              run=dict(first_ms=int(ts[left]),last_ms=int(ts[right-1]),first_age_h=float(ages[left]),last_age_h=float(ages[right-1]),zero_current=bool(off[left]),samples=int(right-left))
          output_arrays=dict(x=x,time_ms=ts.astype('<i8'),age_h=ages.astype('<f8'),file_id=np.full(len(frame),fid,'<u2'),source_row=np.arange(offset,offset+len(frame),dtype='<i4'),segment=ids.astype('<i4'),valid=valid.astype('u1'))
          for key,array in output_arrays.items(): streams[key].write(np.ascontiguousarray(array).tobytes())
          if first_time is None: first_time=str(dates.iloc[0]); file_first_age=float(ages[0])
          previous=dict(time=int(ts[-1]),age=float(ages[-1]),valid=bool(valid[-1]))
          file_last_age=float(ages[-1]); last_time=str(dates.iloc[-1])
          total+=len(frame); offset+=len(frame)
        identities.append(dict(file=path.name,path=path.relative_to(REPO).as_posix(),sha256=actual_sha,lfs_oid=match[1].decode(),bytes=path.stat().st_size,lfs_size=int(match[2]),lfs_verified=True,rows=offset,first_row=first_global,first_datetime=first_time,last_datetime=last_time,first_age_h=file_first_age,last_age_h=file_last_age,segments_started=file_breaks))
        print(json.dumps(dict(stage='P0',file=path.name,rows=offset,total_rows=total,seconds=round(time.time()-started,1))),flush=True)
    finally:
      for stream in streams.values(): stream.close()
    if run: state_runs.append(run)
    for c,st in stats.items():
      n=st['count']; st['mean']=st.pop('sum')/n if n else None
      st['std']=float(np.sqrt(max(0,st.pop('sum2')/n-st['mean']**2))) if n else None
      st['changed_pairs_within_chunk']=changes[c]
      st['missing_or_absent_total']=total-n
    pd.DataFrame(identities).to_csv(OUT/'raw_file_inventory.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(field=c,**v) for c,v in stats.items()]).to_csv(OUT/'raw_field_stats.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(interval_ms=k,count=v) for k,v in sorted(dt_hist.items())]).to_csv(OUT/'sampling_interval_counts.csv',index=False)
    pd.DataFrame(breaks).to_csv(OUT/'discontinuities.csv',index=False)
    pd.DataFrame(state_runs).to_csv(OUT/'current_state_runs.csv',index=False)
    pd.DataFrame(segment_meta).to_csv(OUT/'continuous_segments.csv',index=False)
    save('outputs/file_schemas.json',file_schemas)
    save('outputs/field_schema.json', [dict(index_in_union=i,name=c,unit=(re.search(r'\[([^]]+)\]',c)[1] if re.search(r'\[([^]]+)\]',c) else 'categorical/time')) for i,c in enumerate(all_cols)])
    save('outputs/cache_schema.json',dict(rows=total,n_features=24,features=FROZEN_FEATURES,arrays=dict(x='float32 [N,24]',time_ms='int64 epoch_ms (unaltered DateTime)',age_h='float64',file_id='uint16',source_row='int32 zero-based excluding header',segment='int32',valid='uint8'),source_files=identities,continuity_rule='within one file; DateTime delta 950..1050ms; nondecreasing OpHrs2; instruments finite; no interpolation',rolling_rule='30 original consecutive observations, past only; first29 invalid for model'))
    audit=dict(status='RAW_AUDIT_COMPLETE',base_commit=base_commit,rows=total,files=len(files),all_lfs_verified=True,fields_union=len(all_cols),file_schema_variations=[v for v in file_schemas if v['missing_vs_first'] or v['extra_vs_first']],clock_counts=dict(clock),segments=len(segment_meta),valid_long_segments=sum(s['rows']>=450 for s in segment_meta),interval_top=dt_hist.most_common(15),validity_rule='all10 cell channels finite and [-0.1,1.5]V; all model raw instruments finite; causal30 stats available; large real jumps retained',u6=stats['U6 [V]'],active_channels=ACTIVE,seconds=time.time()-started)
    save('outputs/raw_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)

def rebuild_cache_only():
    """Reconstruct ignored memmaps in a fresh clone without changing frozen outputs."""
    global STUDY,OUT
    from datetime import datetime
    original=STUDY
    expected=json.loads((original/'outputs/cache_schema.json').read_text(encoding='utf-8'))
    if CACHE.exists() and any(CACHE.iterdir()):
        raise RuntimeError('Rebuild is only for an absent/empty cache in a fresh copy; refuses to overwrite an existing cache.')
    scratch=original/'work'/('cache_rebuild_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    STUDY=scratch; OUT=scratch/'outputs'
    try:
        main()
        observed=json.loads((OUT/'cache_schema.json').read_text(encoding='utf-8'))
        assert observed['rows']==expected['rows'] and observed['features']==expected['features']
        assert [(s['sha256'],s['rows']) for s in observed['source_files']]==[(s['sha256'],s['rows']) for s in expected['source_files']]
    finally:
        STUDY=original;OUT=original/'outputs'
    # Separate full precision target authority; no protocol fitting or new training.
    from freeze_protocol import load_cache,preserve_voltage_precision
    schema,arrays=load_cache();preserve_voltage_precision(schema,arrays)
    print('CACHE_REBUILD_VERIFIED; frozen protocol, origins and original data unchanged',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--rebuild-cache',action='store_true');args=p.parse_args()
    if args.rebuild_cache: rebuild_cache_only()
    else: main()
