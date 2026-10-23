"""P2: verify historical physical seams, freeze Train-only scaling and sampling."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from audit_raw import STUDY, RESEARCH, REPO, CACHE, OUT, ACTIVE, FROZEN_FEATURES, save, sha, numeric

DTYPES={'x':'float32','time_ms':'int64','age_h':'float64','file_id':'uint16',
        'source_row':'int32','segment':'int32','valid':'uint8'}
PARTS=['TRAIN','VALIDATION','HISTORICAL_ASSESSMENT','LATE_LIFE']
BOUNDARY_ROWS=[2103,2404,3005]

def load_cache():
    schema=json.loads((OUT/'cache_schema.json').read_text(encoding='utf-8'))
    n=schema['rows']
    arrays={k:np.memmap(CACHE/f'{k}.bin',mode='r',dtype=d,
                    shape=(n,24) if k=='x' else (n,)) for k,d in DTYPES.items()}
    if (CACHE/'voltage_V.bin').exists():
        arrays['voltage_V']=np.memmap(CACHE/'voltage_V.bin',mode='r',dtype='float64',shape=(n,))
    return schema,arrays

def preserve_voltage_precision(schema,a):
    path=CACHE/'voltage_V.bin'
    assert not path.exists(), 'Refuse to overwrite voltage authority'
    offset=0; max_rounding=0.
    with path.open('wb') as dest:
      for identity in schema['source_files']:
        source=REPO/identity['path']; file_rows=0
        for chunk in pd.read_csv(source,sep=';',encoding='utf-8-sig',usecols=['DateTime']+ACTIVE,chunksize=100000,low_memory=False):
            cells=np.column_stack([numeric(chunk[c]).to_numpy(dtype=float,na_value=np.nan) for c in ACTIVE])
            mean=cells.mean(1); times=pd.to_datetime(chunk.DateTime).to_numpy(dtype='datetime64[ns]').astype(np.int64)//1_000_000
            assert np.array_equal(times,a['time_ms'][offset:offset+len(mean)])
            diff=np.abs(mean-a['x'][offset:offset+len(mean),0]); clean=diff[np.isfinite(diff)]
            if clean.size: max_rounding=max(max_rounding,float(clean.max()))
            dest.write(mean.astype('<f8').tobytes()); offset+=len(mean); file_rows+=len(mean)
        assert file_rows==identity['rows']
    assert offset==schema['rows'] and max_rounding<1e-6
    save('outputs/voltage_precision_audit.json',dict(rows=offset,target_dtype='float64 raw ten-cell mean',input_dtype='float32',max_input_rounding_V=max_rounding,all_original_timestamps_match=True))

def metrics(y,p):
    error=np.asarray(p,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    err=error.ravel(); truth=np.asarray(y,dtype=np.float64).ravel()
    if not len(err): return dict(queries=0,RMSE_V=None,MAE_V=None,R2_delta=None,P95_abs_V=None,P99_abs_V=None)
    mse=float(np.mean(err**2)); denom=float(np.sum((truth-truth.mean())**2))
    return dict(queries=len(err),RMSE_V=math.sqrt(mse),MAE_V=float(np.mean(np.abs(err))),
                R2_delta=(1-float(np.sum(err**2))/denom) if denom>0 else None,
                P95_abs_V=float(np.quantile(np.abs(err),.95)),P99_abs_V=float(np.quantile(np.abs(err),.99)))

def main():
    if (OUT/'frozen_protocol.json').exists(): raise RuntimeError('Protocol already frozen; do not overwrite.')
    audit=json.loads((OUT/'raw_audit.json').read_text(encoding='utf-8'))
    assert audit['all_lfs_verified'] and audit['files']==17
    schema,a=load_cache()
    if 'voltage_V' not in a:
        preserve_voltage_precision(schema,a); schema,a=load_cache()
    n=schema['rows']; ts=a['time_ms']; ages=a['age_h']; x=a['x']; valid=a['valid'].astype(bool)
    old=pd.read_csv(RESEARCH/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv')
    seams=[]; cutms=[]
    for row in BOUNDARY_ROWS:
        older=old.iloc[row-1]; newer=old.iloc[row]
        cut=int(pd.Timestamp(newer.first_datetime).value//1_000_000); cutms.append(cut)
        before=np.flatnonzero(ts<cut); after=np.flatnonzero(ts>=cut)
        lo=int(before[-1]); hi=int(after[0])
        assert int(ts[lo])==int(pd.Timestamp(older.last_datetime).value//1_000_000)
        assert int(ts[hi])==cut and hi==lo+1
        assert float(ages[lo])<float(newer.bin_start_ophrs2_h) and float(ages[hi])>=float(newer.bin_start_ophrs2_h)
        seams.append(dict(legacy_first_row=row,age_boundary_h=float(newer.bin_start_ophrs2_h),cut_datetime=str(newer.first_datetime),cut_ms=cut,before_raw_row=lo,after_raw_row=hi,before_raw_age_h=float(ages[lo]),after_raw_age_h=float(ages[hi]),verified=True))
    partitions=np.searchsorted(cutms,ts,side='right').astype('int8')
    assert np.all(np.diff(partitions)>=0), 'Chronological partitions must not interleave'
    split_counts={p:int((partitions==i).sum()) for i,p in enumerate(PARTS)}
    train=(partitions==0)&valid
    sums=np.zeros(24); sums2=np.zeros(24); count=0
    for start in range(0,n,100000):
        block=np.array(x[start:start+100000],dtype=np.float64); keep=train[start:start+len(block)]; block=block[keep]
        sums+=block.sum(0); sums2+=(block*block).sum(0); count+=len(block)
    mean=sums/count; scale=np.sqrt(np.maximum(sums2/count-mean*mean,0)); scale=np.where(scale<1e-8,1.,scale)
    assert np.isfinite(mean).all() and np.isfinite(scale).all() and count>0
    pair=train[1:]&train[:-1]&(a['segment'][1:]==a['segment'][:-1])
    train_dv=np.abs(np.diff(a['voltage_V']))[pair]
    train_di=np.abs(np.diff(x[:,5].astype(np.float64)))[pair]
    threshold=float(np.quantile(train_dv,.995)); current_threshold=float(np.quantile(train_di,.995))
    assert threshold>0
    cfg=dict(version=1,status='FROZEN_BEFORE_GPU_TRAINING',dataset='RealHyFC Loadcycle only',
        seed=202,history_steps=300,horizons_seconds=[30,120],future_candidate_seconds=600,
        input_features=FROZEN_FEATURES,n_features=24,voltage_channels=audit['active_channels'],
        target='future original-second mean10 voltage minus last observed mean10 voltage',
        output='direct all H steps; original DateTime targets; no recursive feedback',
        target_timestamp_tolerance_ms=100,raw_sampling_tolerance_ms=[950,1050],
        partitions=PARTS,seams=seams,partition_raw_rows=split_counts,
        holdout_status='both post-validation periods historically exposed, exploratory only, never tuned',
        scaler=dict(mean=mean.tolist(),scale=scale.tolist(),fit_rows=count,fit='all valid TRAIN rows only, no validation/holdout'),
        voltage_event=dict(threshold_V=threshold,train_quantile=.995,definition='target query is event if any absolute original 1-step voltage change from origin+1 through that target exceeds frozen Train threshold'),
        current_event=dict(threshold_A=current_threshold,train_quantile=.995),
        sampling=dict(origin_spacing_seconds=600,anchor_ms=int(ts[0])+328000,train_max_origins=2048,
            train_cap='if needed, deterministic evenly spaced ranks of chronology-only lattice before labels; identical models within each horizon; horizon-specific eligibility',
            validation_cap=None,heldout_cap=None,phase='strict timestamp lattice; at most 50ms to original origin; no error-driven choices'),
        training=dict(batch_size=16,gradient_accumulation=2,effective_batch_size=32,max_epochs=12,patience=3,
            optimizer='AdamW',learning_rate=.0009605449687795866,weight_decay=6.95942232671908e-06,
            loss='MSE of standardized DeltaV',gradient_clip_norm=1.0,target_scale_V=float(scale[0]),
            scheduler=None,amp=False,initialization='fresh Seed202; no warm-start',
            selection='best VALIDATION all-step RMSE only',gpu_budget_GiB=8,
            oom_policy='reduce microbatch and preserve effective batch by accumulation; no task changes; log failure, no unbounded retry'),
        max_trained_instances=4,models=['Persistence','LSTM','F2'],
        ordering='H30 Persistence,LSTM,F2,verify then H120 Persistence,LSTM,F2; single GPU serial',
        historical_code_sha256=sha(RESEARCH/'RealHyFC_F2_Main.py'),
        source_inventory_sha256=sha(OUT/'raw_file_inventory.csv'))
    # Evaluate sampling eligibility, never target errors, before freezing the protocol.
    segments=pd.read_csv(OUT/'continuous_segments.csv')
    origin_summaries=[]
    for horizon in cfg['horizons_seconds']:
        origins={p:[] for p in PARTS}
        for seg in segments.itertuples():
            first=int(seg.first_row)+328; last=int(seg.last_row)-horizon
            if last<first: continue
            anchor=cfg['sampling']['anchor_ms']; spacing=600000
            start_grid=anchor+math.ceil((int(ts[first])-anchor)/spacing)*spacing
            grid=np.arange(start_grid,int(ts[last])+1,spacing,dtype='int64')
            if not len(grid): continue
            local=np.searchsorted(ts[first:last+1],grid)+first
            for origin,g in zip(local,grid):
                origin=int(origin); end=origin+horizon; begin=origin-299
                if abs(int(ts[origin])-int(g))>50: continue
                if not valid[begin:end+1].all(): continue
                if partitions[origin]!=partitions[end]: continue
                if abs(int(ts[origin])-int(ts[begin])-299000)>100: continue
                offsets=ts[origin+1:end+1]-int(ts[origin])
                if np.max(np.abs(offsets-np.arange(1,horizon+1)*1000))>100: continue
                origins[PARTS[int(partitions[origin])]].append(origin)
        arrays={}
        for name,indices in origins.items():
            indices=np.array(indices,dtype=np.int64); before=len(indices)
            if name=='TRAIN' and len(indices)>2048:
                indices=indices[np.linspace(0,len(indices)-1,2048).round().astype(int)]
            assert len(indices)>0, 'No legal windows '+name
            assert len(np.unique(indices))==len(indices)
            arrays[name]=indices
            origin_summaries.append(dict(horizon=horizon,partition=name,lattice_eligible_origins=before,sampled_origins=len(indices),all_queries=len(indices)*horizon,first_raw_row=int(indices[0]),last_raw_row=int(indices[-1])))
        np.savez_compressed(OUT/f'origins_H{horizon}.npz',**arrays)
    cfg['origin_summaries']=origin_summaries
    cfg['sampling_hashes']={str(h):sha(OUT/f'origins_H{h}.npz') for h in [30,120]}
    save('outputs/frozen_protocol.json',cfg)
    pd.DataFrame(seams).to_csv(OUT/'partition_boundary_verification.csv',index=False)
    pd.DataFrame(origin_summaries).to_csv(OUT/'window_sampling_counts.csv',index=False)
    old_names=['voltage_mean','voltage_std','voltage_range','voltage_min','voltage_max','current_mean','current_std','current_ptp','density_mean','density_std','density_ptp','load_transition_count','operation_point_transitions','zero_current_fraction']
    top10=[0,3,13,11,4,9,6,7,10,8]
    definitions=['ten valid cell channels, instantaneous mean','instantaneous spatial std10','instantaneous spatial ptp10','instantaneous min10','instantaneous max10','I.S.act [A], no temporal averaging','causal past30-sample std (ddof0)','causal past30-sample ptp','j.S.act [A/cm2], instantaneous','causal past30-sample density std','causal past30-sample density ptp','abs(I[t]-I[t-1])>0.1A (instrument-resolution change flag)','operation_point[t]!=operation_point[t-1]','abs(I[t])<=0.1A, state proxy']
    pd.DataFrame([dict(new_index=i,new_name=FROZEN_FEATURES[i],old_aggregate_name=old_names[i] if i<14 else None,was_old_top10=i in top10,raw_derivation=definitions[i] if i<14 else 'current raw instrument: '+FROZEN_FEATURES[i]) for i in range(24)]).to_csv(OUT/'feature_mapping.csv',index=False)
    report0=f'''# 原始数据与采样真实性审计\n\n## 结论与数据身份\n\n已逐字节核对17份原始CSV与Git LFS SHA256 OID及声明文件大小，均是真实CSV内容；共{n:,}行。数据未重采样、未填补断点，原始DateTime和文件内行号保留在可重建memmap中。原始文件是权威，缓存不提交Git。\n\n文件逐项行数、时间范围和SHA见 `outputs/raw_file_inventory.csv`；字段、单位和文件间schema变化见 `field_schema.json`、`file_schemas.json`。并非17文件字段完全相同：51–70缺少MIN(U.S)而多CalHrs，134–150也多CalHrs；24维必要仪器字段逐文件确认存在。\n\n## 采样和双时钟\n\n真实DateTime相邻间隔的完整频数见 `sampling_interval_counts.csv`，而不是根据总行数猜测1秒。近1秒判定在查看预测误差前定义为950–1050ms；OpHrs2不倒退且所有必要仪器有效时才构成候选连续段，所有文件接缝保守断开。共有{audit['segments']:,}段，其中至少450条原始记录的段有{audit['valid_long_segments']}段。时间间隔最常见项：{audit['interval_top']}。完整断点含原文件行号见 `discontinuities.csv`。\n\nOpHrs2冻结相邻对{audit['clock_counts']['OpHrs2_frozen_pairs']:,}，说明运行小时不能替代真实秒钟。启停期间它可冻结；还可能量化到0.01h。近1秒真实时间记录上的原始实际/设定电流、温度、压力、operation_point均有记录，但记录频率不等于传感器独立刷新频率；字段完全重复计数与变化次数见 `raw_field_stats.csv`，不声称所有传感器都真正1Hz更新。\n\n## 电压与异常\n\nU6已独立统计：{json.dumps(audit['u6'],ensure_ascii=False)}。目标使用U1–U5和U7–U11十个有效通道的逐行均值，U6不进入目标或输入；原始设备AveCell只作审计字段。十通道必须有限且处于[-0.1,1.5]V的宽物理仪器范围，因果滚动量可计算；不会因ΔV很大而删掉启停。多通道同时变化、与电流/气体/温度相符是判断真实工况的证据，但不足排除共同测量开关，不能仅凭幅度断言退化。状态表使用|I|≤0.1A代理，既不把operation_point误称独立运行开关，也不把OpHrs2冻结秒直接当作启停秒。\n\n## 特征与可信范围\n\n旧Top10=[0,3,13,11,4,9,6,7,10,8]是30分钟统计表索引。`feature_mapping.csv`逐项说明瞬时空间电压统计、仅过去30条连续样本的电流统计、过去一步变化标志和原始控制/热/压力字段。本轮固定24维，共享同一输入，不重新选择Top10、不EWMA平滑电压、不跨段滚动。\n\n原始数据足以开展“存在真实连续秒级观测区间内”的动态预测，不足以宣称无缺失的全寿命连续1Hz序列。物理状态/开关时序存在观测限制，留出时期已经被历史研究暴露。AI协助生成代码和文本，数值来自本目录运行产物。\n'''
    (STUDY/'00_原始数据与采样真实性审计.md').write_text(report0,encoding='utf-8',newline='\n')
    table=pd.DataFrame(origin_summaries).to_csv(index=False,lineterminator='\n')
    report3=f'''# 原始1秒动态预测正式协议\n\n协议状态：首次GPU训练前冻结。配置权威为 `outputs/frozen_protocol.json`；Seed202；4个全新可训练实例上限（2模型×2跨度），Persistence不训练。\n\n## 任务与输入\n\n300条连续原始近1秒历史观测→未来30或120条原始真实秒级单体平均电压变化ΔV。实际历史跨度约299秒，目标时间分别约起点后1…H秒，允许采样时间抖动但总跨度误差≤100ms。原始目标DateTime不舍入、不插值，不把不规则时间强行变成严格1Hz。完整300秒候选历史末点是已获得的起点观测。在线更新只能在获得下一次真实测量后进行，不递归回填。600秒尚未试验。\n\n24个输入见 `feature_mapping.csv`。过去30点滚动只用同连续段过去信息。F2保留完整原生VSN/LSTM/CNN/工况门控/注意力，输入schema由14改为24，窗口及直接解码长度作必要适配；旧checkpoint不加载。普通LSTM宽64、单层，与F2的LSTM宽度层数相同，直接线性多步头；并非参数量完全匹配。两模型采用共同输入、起点、scaler、损失、优化器和训练预算。未来设定电流和状态不输入解码器，只有确定的相对预测步坐标。\n\n## 物理分区\n\n独立核对旧F2的聚合行切口后，以真实原始DateTime分区，边界分别为{[s['cut_datetime'] for s in seams]}，对应OpHrs2 { [s['age_boundary_h'] for s in seams] }h。四区原始行数{split_counts}。目标必须全部在所属分区内；历史允许读取之前分区的过去观测。后两区均已被历史研究暴露，报告为探索性留出，不能叫从未查看的独立Test。其数据不用来定事件阈值、改超参数或选checkpoint。\n\n## 采样与资源预算\n\n在每个真实连续段内，起点候选采用固定600秒日历时间网格；Train如超过2048个，按预定均匀秩上限抽取，绝不按标签或误差挑选。Validation和两个留出使用全部合法网格起点。起点清单在训练前保存并SHA冻结。它是小规模覆盖性工程试验，不是全量窗口训练。\n\n```csv\n{table}```\n\nTrain所有有效原始行拟合共同均值/标准差（{count:,}行）；目标以Train电压标准差{scale[0]:.9g}V缩放。突变阈值为Train连续有效对|一步ΔV|的99.5%分位数，即{threshold:.9g}V；电流阈值为{current_threshold:.9g}A。评价时，一个目标查询若起点之后到该目标之间已出现超过阈值的电压变化，标为事件查询。事件/普通指标是事后分层，事件真值不作为输入或采样依据。\n\nGPU预算8GiB，单GPU串行。microbatch16、累计2步、有效batch32；AdamW lr={cfg['training']['learning_rate']}、weight_decay={cfg['training']['weight_decay']}；MSE，clip1，12epoch上限、Validation全步RMSE最佳checkpoint、patience3。固定小预算不保证充分收敛，不得据此断言架构的最终上限；没有超参搜索或多种子筛选。OOM仅允许记录后降低microbatch并保持有效batch，不改变任务。\n\n## 指标、完整输出及验收\n\n每个起点输出全部H个真实目标秒，NPZ含原始起点/目标行号、DateTime毫秒、ΔV真值与预测、起点电压。主指标对所有起点×步数查询计算RMSE、MAE、ΔV R²及绝对误差P95/P99；同时报告事件和普通查询及Persistence。重叠目标重复计权在查询级主指标中明确保留，不能只挑最后一步或好看的秒。电压RMSE/MAE不受加回起点电压影响，ΔV R²不冒称绝对电压R²。\n\nP0/P1/P2通过后才运行 `python -B scripts/train_pilot.py --horizon 30`，完成真实性验收再运行 `--horizon 120`。图为300dpi PNG、矢量PDF和绘图原始CSV。不得将本轮30秒误差与旧12小时误差相比较来证明改进。所有数值结论限定于单种子、采样起点、已观测工况和本次固定训练预算。AI辅助实现与核查已披露。\n'''
    (STUDY/'03_原始1秒动态预测正式协议.md').write_text(report3,encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=cfg['status'],rows=n,scaler_rows=count,events_V=threshold,origins=origin_summaries),ensure_ascii=False),flush=True)

if __name__=='__main__': main()
