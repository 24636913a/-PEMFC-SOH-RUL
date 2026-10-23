"""Clock-only full-file access; outcomes are restricted to TRAIN + VALIDATION.

No model imports, CUDA calls, weights, optimizers or inference in this audit.
Run once per new output directory; no historical output can be overwritten.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
REPO = ROOT.parent
SOURCE = ROOT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
CONFIG = HERE/'configs/PREREGISTERED_QUALIFICATION.json'
META = ['bin_start_ophrs2_h', 'bin_end_ophrs2_h', 'first_datetime', 'last_datetime',
        'temporal_segment_id', 'bin_valid_for_windows', 'gap_before_flag',
        'gap_before_duration_s', 'gap_after_flag', 'low_coverage_flag',
        'internal_datetime_gap_count', 'maximum_internal_datetime_gap_s',
        'coverage_ratio', 'raw_sample_count', 'startup_flag', 'shutdown_flag']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csv(path, frame):
    frame.to_csv(path,index=False,float_format='%.17g',encoding='utf-8-sig')

def load_meta():
    f=pd.read_csv(SOURCE,usecols=META)
    assert len(f)==3843
    f.insert(0,'row',np.arange(len(f)))
    return f

def continuity(frame, cfg):
    age=frame.bin_end_ophrs2_h.to_numpy(float)
    first=pd.to_datetime(frame.first_datetime).astype('int64').to_numpy()/1e9
    last=pd.to_datetime(frame.last_datetime).astype('int64').to_numpy()/1e9
    delta=np.r_[np.nan,np.diff(age)]
    gap=np.r_[np.nan,first[1:]-last[:-1]]
    seg=frame.temporal_segment_id.to_numpy()
    valid=frame.bin_valid_for_windows.to_numpy()==1
    edges=np.r_[False,(seg[1:]==seg[:-1]) & np.isclose(delta[1:],.5,rtol=0,atol=cfg['clock_atol_h']) & (gap[1:]>=0)&(gap[1:]<=cfg['calendar_gap_max_s'])]
    blocks=np.cumsum(~edges | ~valid | np.r_[False,~valid[:-1]])
    return age,valid,edges,blocks,delta,gap

def windows(frame, cfg, history, horizon, start, end):
    age,valid,edge,block,_,_=continuity(frame,cfg)
    seg=frame.temporal_segment_id.to_numpy()
    accepted=[]; rejected=[]
    for o in range(start,end):
        lo=o-history+1; hi=o+horizon
        reasons=[]
        if lo<0: reasons.append('HISTORY_BEFORE_SOURCE')
        if hi>=end: reasons.append('TARGET_OUTSIDE_PARTITION')
        if lo>=0 and hi<len(frame):
            if not valid[lo:hi+1].all(): reasons.append('INVALID_BIN')
            if not np.all(seg[lo:hi+1]==seg[o]): reasons.append('SEGMENT_BREAK')
            if not np.isclose(np.diff(age[lo:hi+1]),.5,rtol=0,atol=cfg['clock_atol_h']).all(): reasons.append('OPERATING_CLOCK_BREAK')
            # Inspect calendar separately, even if a segment field happens to be wrong.
            first=pd.to_datetime(frame.first_datetime.iloc[lo:hi+1]).astype('int64').to_numpy()/1e9
            last=pd.to_datetime(frame.last_datetime.iloc[lo:hi+1]).astype('int64').to_numpy()/1e9
            dg=first[1:]-last[:-1]
            if ((dg<0)|(dg>cfg['calendar_gap_max_s'])).any(): reasons.append('CALENDAR_CLOCK_BREAK')
        if reasons:
            rejected.append(dict(origin_row=o,reasons='|'.join(reasons)))
        else:
            accepted.append(dict(origin_row=o,history_first_row=lo,target_first_row=o+1,target_last_row=hi,
                origin_operating_h=age[o],target_last_operating_h=age[hi],actual_lead_h=age[hi]-age[o],
                temporal_segment_id=int(seg[o]),continuous_block=int(block[o])))
    return pd.DataFrame(accepted,columns=['origin_row','history_first_row','target_first_row','target_last_row','origin_operating_h','target_last_operating_h','actual_lead_h','temporal_segment_id','continuous_block']),pd.DataFrame(rejected,columns=['origin_row','reasons'])

def disjoint_count(w):
    last=-1; count=0
    for row in w.itertuples():
        if row.target_first_row>last: count+=1; last=row.target_last_row
    return count

def history_snapshot(out):
    paths=subprocess.check_output(['git','ls-files','-z','--','RENew-PEMFC-2300h-Pytorch'],cwd=REPO).decode().split('\0')
    files={}; missing=[]
    for p in paths:
        if not p or '/F2_LongTerm_Degradation_Study/' in p: continue
        if not (REPO/p).is_file():
            missing.append(p); continue
        files[p]=sha(REPO/p)
        if len(files)%10000==0: print('Historical SHA snapshot:',len(files),flush=True)
    dump(out/'HISTORY_BEFORE.json',files)
    dump(out/'PREEXISTING_MISSING_TRACKED_FILES.json',missing)
    return len(files)

def run(out):
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    out.mkdir(parents=True,exist_ok=False)
    print('Registering qualification, before source reads',flush=True)
    dump(out/'REGISTRATION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(CONFIG),
        source_sha256=sha(SOURCE),git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        python=sys.version,argv=sys.argv))
    history_snapshot(out)
    f=load_meta(); age,valid,edges,blocks,delta,gap=continuity(f,cfg)
    f['continuous_block']=blocks; f['previous_operating_delta_h']=delta; f['previous_calendar_gap_s']=gap
    csv(out/'CLOCK_METADATA_ALL_ROWS.csv',f)
    segments=[]
    for seg,g in f.groupby('temporal_segment_id',sort=False):
        segments.append(dict(temporal_segment_id=seg,records=len(g),valid_records=int(g.bin_valid_for_windows.sum()),
            first_row=int(g.row.min()),last_row=int(g.row.max()),first_h=float(g.bin_end_ophrs2_h.min()),last_h=float(g.bin_end_ophrs2_h.max())))
    csv(out/'SEGMENT_SUMMARY.csv',pd.DataFrame(segments))
    pieces=[]
    for block,g in f[valid].groupby('continuous_block',sort=False):
        pieces.append(dict(continuous_block=block,first_row=int(g.row.min()),last_row=int(g.row.max()),records=len(g),
            first_h=float(g.bin_end_ophrs2_h.min()),last_h=float(g.bin_end_ophrs2_h.max()),
            max_forecast_h_after_48_history=max(0,(len(g)-48)*.5),temporal_segment_id=int(g.temporal_segment_id.iloc[0])))
    csv(out/'CONTINUOUS_VALID_PIECES.csv',pd.DataFrame(pieces))
    summaries=[]; allowed=[]
    for h in cfg['history_steps']:
        for k in cfg['horizon_steps']:
            bypart={}
            for name,(start,end) in cfg['partition_rows'].items():
                w,x=windows(f,cfg,h,k,start,end)
                prefix=f'L{h}_H{k}_{name}'
                csv(out/(prefix+'_LEGAL.csv'),w); csv(out/(prefix+'_EXCLUDED.csv'),x)
                targets=set()
                for r in w.itertuples(): targets.update(range(r.target_first_row,r.target_last_row+1))
                item=dict(history_steps=h,horizon_steps=k,horizon_h=k*.5,partition=name,partition_records=end-start,
                    legal_origins=len(w),queries=len(w)*k,unique_target_rows=len(targets),disjoint_target_trajectories=disjoint_count(w),
                    excluded_origins=len(x),source_segments=w.temporal_segment_id.nunique(),continuous_pieces=w.continuous_block.nunique())
                summaries.append(item); bypart[name]=item
                print('Window qualification',prefix,'origins',len(w),'disjoint',disjoint_count(w),flush=True)
            s=cfg['sufficiency']; t=bypart['TRAIN']; v=bypart['VALIDATION']
            checks=dict(train_origins=t['legal_origins']>=s['min_train_origins'],validation_origins=v['legal_origins']>=s['min_validation_origins'],
                train_disjoint=t['disjoint_target_trajectories']>=s['min_disjoint_train_target_trajectories'],validation_disjoint=v['disjoint_target_trajectories']>=s['min_disjoint_validation_target_trajectories'])
            allowed.append(dict(history_steps=h,horizon_steps=k,horizon_h=k*.5,qualified=all(checks.values()),**checks))
    csv(out/'WINDOW_COUNTS.csv',pd.DataFrame(summaries)); csv(out/'P0_QUALIFICATION.csv',pd.DataFrame(allowed))
    summary=dict(records=len(f),valid_records=int(valid.sum()),first_operating_h=float(age.min()),last_operating_h=float(age.max()),
        segment_count=int(f.temporal_segment_id.nunique()),valid_continuous_piece_count=len(pieces),
        max_valid_piece_records=max(p['records'] for p in pieces),max_valid_piece_h=max((p['records']-1)*.5 for p in pieces),
        operating_breaks=int((~np.isclose(delta[1:],.5,rtol=0,atol=cfg['clock_atol_h'])).sum()),
        calendar_breaks=int(((gap[1:]<0)|(gap[1:]>cfg['calendar_gap_max_s'])).sum()),
        p0_any_qualified=any(x['qualified'] for x in allowed),
        holdout_outcome_columns_read=False,training_runs=0)
    dump(out/'P0_SUMMARY.json',summary); print(json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); run(args.out)
