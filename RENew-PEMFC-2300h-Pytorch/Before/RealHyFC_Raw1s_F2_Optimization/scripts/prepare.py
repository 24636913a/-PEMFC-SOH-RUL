"""P0-P2: freeze legal grids and predeclared sampling BEFORE any new model result."""
from __future__ import annotations
import json
import math
import subprocess
import numpy as np
import pandas as pd
from core import STUDY,OLD,REPO,OUT,PARTS,load_cache,save,sha

def uniform(values,n):
    values=np.asarray(values,dtype=np.int64)
    return values if len(values)<=n else values[np.linspace(0,len(values)-1,n).round().astype(np.int64)]

def legal(a,origins,h,partitions,invalid_prefix):
    o=np.asarray(origins,dtype=np.int64);begin=o-299;raw_begin=o-328;end=o+h
    valid=(raw_begin>=0)&(end<len(partitions));o=o[valid];begin=begin[valid];raw_begin=raw_begin[valid];end=end[valid]
    keep=(a['segment'][raw_begin]==a['segment'][end])&(a['file_id'][raw_begin]==a['file_id'][end])
    keep&=partitions[raw_begin]==partitions[end]
    keep&=(invalid_prefix[end+1]-invalid_prefix[begin])==0
    keep&=np.abs(a['time_ms'][o]-a['time_ms'][begin]-299000)<=100
    o=o[keep]
    out=[]
    for start in range(0,len(o),10000):
        block=o[start:start+10000];offs=a['time_ms'][block[:,None]+np.arange(1,h+1)]-a['time_ms'][block,None]
        out.append(block[(np.max(np.abs(offs-np.arange(1,h+1)*1000),axis=1)<=100)])
    return np.concatenate(out) if out else np.array([],dtype=np.int64)

def main():
    if (OUT/'frozen_protocol.json').exists():raise RuntimeError('Already frozen; refuse overwrite')
    OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((OLD/'outputs/frozen_protocol.json').read_text(encoding='utf-8'))
    schema,a=load_cache();n=schema['rows'];ts=a['time_ms'];v=a['voltage_V']
    # Preserve exact identities of ALL tracked previous-study artifacts and the raw payloads.
    historical={f.relative_to(OLD).as_posix():sha(f) for f in OLD.rglob('*') if f.is_file()
        and not set(f.relative_to(OLD).parts).intersection({'cache','work','__pycache__'})}
    raw=[]
    for item in schema['source_files']:
        p=REPO/item['path'];oid=subprocess.check_output(['git','show','HEAD:'+item['path']],cwd=REPO)
        assert not p.open('rb').read(100).startswith(b'version https://git-lfs.github.com/spec/v1')
        assert ('oid sha256:'+item['sha256']).encode() in oid
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
        raw.append(dict(path=item['path'],sha256=item['sha256'],bytes=item['bytes'],rows=item['rows'],lfs_payload_verified=True))
        print('LFS_VERIFIED',item['file'],flush=True)
    save('outputs/raw_integrity.json',dict(files=raw,rows=n,DateTime='original int64 milliseconds, no rounding',legacy_cache_mode='r'))
    cache_manifest={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in (OLD/'cache').glob('*.bin')}
    save('outputs/legacy_integrity.json',dict(study_files=historical,cache_files=cache_manifest,
        code_sha256=sha(STUDY.parent/'RealHyFC_F2_Main.py')))
    cuts=[s['cut_ms'] for s in previous['seams']];part=np.searchsorted(cuts,ts,side='right')
    invalid=np.r_[0,np.cumsum(~a['valid'].astype(bool),dtype=np.int64)]
    anchor=previous['sampling']['anchor_ms'];segments=pd.read_csv(OLD/'outputs/continuous_segments.csv')
    candidate=[]
    for seg in segments.itertuples():
        first=int(seg.first_row)+328;last=int(seg.last_row)-30
        if last<first:continue
        start=anchor+math.ceil((int(ts[first])-anchor)/60000)*60000
        grid=np.arange(start,int(ts[last])+1,60000,dtype=np.int64)
        index=np.searchsorted(ts[first:last+1],grid)+first
        keep=index<=last;index=index[keep];grid=grid[keep]
        candidate.extend(index[np.abs(ts[index]-grid)<=50].tolist())
    candidate=np.asarray(candidate,dtype=np.int64)
    candidate=np.unique(candidate)
    origins={}
    sampling=[]
    for h in [30,120]:
        allowed=legal(a,candidate,h,part,invalid)
        arrays={}
        for pi,pname in enumerate(PARTS[:2]):
            selected=allowed[part[allowed]==pi]
            arrays[pname]=selected
            sampling.append(dict(horizon=h,partition=pname,grid='60s',candidate_origins=len(selected),queries=len(selected)*h))
        origins[h]=arrays
    # No future model outputs or Validation labels enter the fixed candidate list.
    common=np.intersect1d(origins[30]['TRAIN'],origins[120]['TRAIN'])
    all_train=origins[30]['TRAIN']
    meta=pd.DataFrame(dict(origin=all_train,DateTime=pd.to_datetime(ts[all_train],unit='ms'),origin_time_ms=ts[all_train],
        segment_id=a['segment'][all_train],file_id=a['file_id'][all_train],source_row=a['source_row'][all_train],
        partition='TRAIN',legal_H30=True,legal_H120=np.isin(all_train,common),
        last_target_H30=all_train+30,last_target_H120=all_train+120))
    meta.to_csv(OUT/'train_candidate_origins_60s.csv',index=False)
    # Episode operational definition: adjacent trigger gaps <=120 real seconds, never across partition/segment.
    records=[];triggers=[];eid=0
    for seg in segments.itertuples():
        lo=int(seg.first_row);hi=int(seg.last_row)+1
        if lo>=int(previous['seams'][1]['after_raw_row']):break
        hi=min(hi,int(previous['seams'][1]['after_raw_row']))
        jumps=lo+1+np.flatnonzero(np.abs(np.diff(v[lo:hi]))>previous['voltage_event']['threshold_V'])
        if not len(jumps):continue
        starts=np.r_[0,1+np.flatnonzero((np.diff(ts[jumps])>120000)|(part[jumps[1:]]!=part[jumps[:-1]]))]
        ends=np.r_[starts[1:],len(jumps)]
        for begin,end in zip(starts,ends):
            rows=jumps[begin:end];eid+=1
            records.append(dict(event_id=eid,first_row=int(rows[0]),last_row=int(rows[-1]),first_ms=int(ts[rows[0]]),
                last_ms=int(ts[rows[-1]]),segment_id=int(seg.segment_id),partition=PARTS[int(part[rows[0]])],jumps=len(rows)))
            triggers.extend((int(row),eid) for row in rows)
    catalogue=pd.DataFrame(records);catalogue.to_csv(OUT/'event_catalog.csv',index=False)
    trigger_frame=pd.DataFrame(triggers,columns=['trigger_row','event_id']);trigger_frame.to_csv(OUT/'event_triggers.csv',index=False)
    trigger_rows=trigger_frame.trigger_row.to_numpy();trigger_eid=trigger_frame.event_id.to_numpy()
    base=uniform(common,min(6144,len(common)));base_set=set(base.tolist())
    representatives={}
    for origin in common:
        if int(origin) in base_set:continue
        left=np.searchsorted(trigger_rows,origin+1);right=np.searchsorted(trigger_rows,origin+31)
        for event in np.unique(trigger_eid[left:right]):
            own=trigger_rows[left:right][trigger_eid[left:right]==event]
            # A shared legal history; supplemental event really lies within H30, hence also H120.
            score=(float(np.min(np.abs(ts[own]-ts[origin]-15000))),int(origin))
            if int(event) not in representatives or score<representatives[int(event)][0]:
                representatives[int(event)]=(score,int(origin))
    available=np.array(sorted(representatives),dtype=np.int64)
    wanted=uniform(available,min(2048,len(available)))
    extra=[];used=set(base_set)
    for event in wanted:
        origin=representatives[int(event)][1]
        if origin not in used:extra.append(origin);used.add(origin)
    # Fixed no-label fallback if episode de-duplication leaves quota unavailable.
    fallback=uniform(np.setdiff1d(common,np.array(sorted(used),dtype=np.int64)),max(0,min(8192,len(common))-len(used)))
    selected=np.sort(np.r_[base,np.asarray(extra,dtype=np.int64),fallback])
    assert len(np.unique(selected))==len(selected) and len(selected)<=8192
    selection_kind={int(o):'UNIFORM_75_PERCENT' for o in base}
    selection_kind.update({int(o):'EVENT_SUPPLEMENT' for o in extra})
    selection_kind.update({int(o):'UNIFORM_SHORTFALL_FALLBACK' for o in fallback})
    meta.loc[meta.origin.isin(selected)].assign(selection=lambda z:z.origin.map(selection_kind)).to_csv(OUT/'selected_train_origins.csv',index=False)
    aux_summaries=[]
    for h in [30,120]:
        origins[h]['TRAIN']=selected
        np.savez_compressed(OUT/f'origins_H{h}.npz',**origins[h])
        old_orig=np.load(OLD/f'outputs/origins_H{h}.npz')['VALIDATION']
        kept=legal(a,old_orig,h,part,invalid)
        np.savez_compressed(OUT/f'origins_H{h}_600s.npz',VALIDATION=kept)
        aux_summaries.append(dict(horizon=h,previous_validation_origins=len(old_orig),kept=len(kept),
            excluded_by_full_support_partition_rule=len(old_orig)-len(kept)))
    pd.DataFrame(sampling).to_csv(OUT/'window_sampling_counts.csv',index=False)
    # Record exact episode coverage / repetition under both horizons, never count overlapping queries as trials.
    coverage=[]
    for h in [30,120]:
        old_train=np.load(OLD/f'outputs/origins_H{h}.npz')['TRAIN']
        for label,values in [('OLD_2048',old_train),('NEW_TRAIN',selected)]:
            counts={}
            for o in values:
                left=np.searchsorted(trigger_rows,o+1);right=np.searchsorted(trigger_rows,o+h+1)
                for event in np.unique(trigger_eid[left:right]):counts[int(event)]=counts.get(int(event),0)+1
            for event,count in counts.items():coverage.append(dict(horizon=h,sampling=label,event_id=event,origins_covering_event=count))
    pd.DataFrame(coverage).to_csv(OUT/'event_origin_coverage.csv',index=False)
    cfg={k:previous[k] for k in ['dataset','seed','history_steps','horizons_seconds','input_features','n_features','voltage_channels',
        'target','output','target_timestamp_tolerance_ms','raw_sampling_tolerance_ms','partitions','seams','partition_raw_rows',
        'scaler','voltage_event','current_event','historical_code_sha256']}
    cfg.update(version=2,status='FROZEN_BEFORE_NEW_GPU_TRAINING',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        old_protocol_sha256=sha(OLD/'outputs/frozen_protocol.json'),historical_artifact_sha256=historical,
        cache='read-only previous-study memmaps, raw CSV remains authority',history_information_support_rows=329,
        partition_firewall='origin-328 through origin+H all in same partition; file/segment never crossed',
        old_comparison=dict(rule='same surviving previous 600s queries; old NPZ subset, no retraining',counts=aux_summaries),
        sampling=dict(anchor_ms=anchor,candidate_grid_seconds=60,origin_match='legacy searchsorted next original row, <=50ms',
            train_common_H30_H120=True,train_max_origins=8192,uniform_requested=6144,event_requested=2048,
            actual_uniform=len(base),actual_event_supplement=len(extra),actual_fallback_uniform=len(fallback),actual_total=len(selected),
            event_definition='first 30 future targets have at least one Train threshold jump; full H120 legal',
            episode_rule='same segment and partition; merge triggers separated by <=120000 real ms; recorded episodes, not hardware truth',
            event_representative='closest covered jump to origin+15s; tie earliest origin; one extra origin per episode; chronological uniform ranks of episode IDs',
            fallback='if insufficient unique event origins, fill fixed chronological uniform unused origins without labels',
            evaluation='all legal fixed60s VALIDATION origins, no label/error filtering; old600s auxiliary filtered by same full-support rule',
            deduplicated_auxiliary='per real target row/DateTime retain latest origin / shortest lead actual query; no averaging; H120 auxiliary changes lead mixture'),
        training={**previous['training'],'max_epochs':50,'patience':10,'batch_size':16,'gradient_accumulation':2,
            'effective_batch_size':32,'loss':'A: MSE; B: sum(weight*SE)/sum(weight), weights1/3 over entire effective batch',
            'validation_loss':'unweighted standardized MSE on all60s query targets, including B',
            'initialization':'fresh Seed202 separately for each A/B; A2/B1 and A4/B2 identical initialization/data order',
            'failure_policy':'save log and stop on exception; never overwrite completed checkpoint or automatically retry positive-step run'},
        runs=[dict(id='A1',model='LSTM',horizon=30,loss='MSE'),dict(id='A2',model='F2',horizon=30,loss='MSE'),
            dict(id='A3',model='LSTM',horizon=120,loss='MSE'),dict(id='A4',model='F2',horizon=120,loss='MSE'),
            dict(id='B1',model='F2',horizon=30,loss='EVENT_WEIGHTED_MSE'),dict(id='B2',model='F2',horizon=120,loss='EVENT_WEIGHTED_MSE',conditional=True)],
        B2_gate=dict(relative_all_RMSE_reduction_at_least=.03,relative_event_RMSE_reduction_at_least=.05,
            relative_normal_MAE_increase_at_most=.10,validation_grid='60s',metric_scope='ALL_QUERIES',independent_integrity_pass_required=True),
        max_trained_instances=6,holdout_policy='no new holdout inference in this phase; known events are historical explanations only',
        single_seed_exploratory=True,AI_assisted=True)
    frozen=['outputs/raw_integrity.json','outputs/legacy_integrity.json','outputs/train_candidate_origins_60s.csv',
        'outputs/selected_train_origins.csv','outputs/event_catalog.csv','outputs/event_triggers.csv','outputs/event_origin_coverage.csv']
    frozen += [f'outputs/origins_H{h}{suffix}.npz' for h in [30,120] for suffix in ['', '_600s']]
    cfg['frozen_file_sha256']={name:sha(STUDY/name) for name in frozen}
    # Diagnostic rules must also be finalized before ANY new training.
    rules=OUT/'diagnostic_rules.json'
    if not rules.exists():
        save('outputs/protocol_draft.json',cfg)
        print('SAMPLING_READY_WAITING_FOR_DIAGNOSTIC_RULE_FREEZE',flush=True)
        return
    cfg['diagnostic_rules_sha256']=sha(rules)
    save('outputs/frozen_protocol.json',cfg)
    print(json.dumps(dict(status=cfg['status'],sampling=cfg['sampling'],auxiliary=aux_summaries),ensure_ascii=False),flush=True)

def finalize():
    target=OUT/'frozen_protocol.json'
    if target.exists():raise RuntimeError('Protocol already frozen; refuse overwrite')
    cfg=json.loads((OUT/'protocol_draft.json').read_text(encoding='utf-8'))
    rules=OUT/'diagnostic_rules.json';assert rules.exists()
    assert not list((STUDY/'runs').glob('*/training_state.json'))
    for name,expected in cfg['frozen_file_sha256'].items():assert sha(STUDY/name)==expected,name
    cfg['diagnostic_rules_sha256']=sha(rules)
    cfg['implementation_sha256']={f'scripts/{name}':sha(STUDY/'scripts'/name) for name in ['core.py','prepare.py','train.py']}
    cfg['frozen_file_sha256']['outputs/diagnostic_rules.json']=sha(rules)
    save('outputs/frozen_protocol.json',cfg)
    print('PROTOCOL_FROZEN_BEFORE_ANY_NEW_GPU',sha(target),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');args=p.parse_args()
    if args.finalize:finalize()
    else:main()
