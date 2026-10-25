"""阶段2A数据准备：只读旧缓存/Excel/冻结特征；不读取旧预测。"""
from pathlib import Path
from datetime import datetime,timezone
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import hashlib,json,subprocess,sys
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent.parent;GUIDE=WORK.parent;REPO=GUIDE.parent
OUT=WORK/'输出结果';TAB=OUT/'表格';PREP=WORK/'数据准备'
OLD=GUIDE/'健康特征提取/共同负载特征研究'
CACHE=REPO/'RENew-PEMFC-2300h-Pytorch/RealHyFC_Raw1s_Rolling_300to120/audits/Startup_Precursor_Audit/runs/20261022T071957_960946Z_b1590612'
RCR=REPO/'RENew-PEMFC-2300h-Pytorch/Before/Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs'
sys.path.insert(0,str(GUIDE/'数据处理/程序'))
from 核验基础 import numeric,mean_cells,CELLS_FC1

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def js(p,obj):Path(p).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(p,rows):pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig')

def freeze():
    p=WORK/'配置/执行前冻结.json'
    if p.exists():
        old=json.loads(p.read_text(encoding='utf-8'));assert old['协议SHA256']==sha(WORK/'配置/阶段2A冻结协议.json');return
    files={str(q.relative_to(GUIDE)):sha(q) for q in GUIDE.rglob('*') if q.is_file() and not q.is_relative_to(WORK) and '__pycache__' not in q.parts and '绘图缓存' not in q.parts and not q.name.startswith('~$')}
    js(p,dict(UTC=datetime.now(timezone.utc).isoformat(),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
       协议SHA256=sha(WORK/'配置/阶段2A冻结协议.json'),历史SHA256=files,允许变更='正式版本验收后仅完整代码/使用说明.md可更新，须先备份；阶段0–1.3一律只读'))
    with ZipFile(GUIDE/'step1/燃料电池三档研究方案.docx') as z:root=ET.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    text='\n'.join(''.join(x.text or '' for x in e.iter() if x.tag.endswith('}t')) for e in root.findall('.//w:body/w:p',ns))
    (WORK/'配置/Word原方案逐段文本.txt').write_text(text,encoding='utf-8')

def role(start,end,bounds):
    for label,lo,hi in [('Train',bounds[0],bounds[1]),('Validation',bounds[1],bounds[2]),('Assessment',bounds[2],bounds[3])]:
        if start>=lo-1e-10 and end<=hi+1e-10:return label
    return 'CROSS_BOUNDARY'

def fc1(cfg):
    root=REPO/'数据/FC1_FC2_Excel/FC1_FC2_Excel/FC1_Without_Ripples_Excel';allrows=[];sources=[]
    for p in sorted(root.glob('FC1_Ageing_part*.xlsx')):
        f=pd.read_excel(p,engine='openpyxl');v,_=mean_cells(f,CELLS_FC1);time=numeric(f['Time (h)']);jn=next(c for c in f if c.startswith('J (A/cm'))
        allrows.append(pd.DataFrame(dict(source=p.name,row=np.arange(len(f)),time=time,j=numeric(f[jn]),v=v)))
        sources.append(dict(source=str(p.relative_to(REPO)),SHA256=sha(p),rows=len(f)))
    raw=pd.concat(allrows,ignore_index=True);assert len(raw)==143862
    raw['duplicate_time']=raw.time.duplicated(keep=False)
    raw['nominal_test_guard']=False
    for t in [0,48,185,348,515,658,823,991]:raw.loc[raw.time.ge(t-1)&raw.time.lt(t+3),'nominal_test_guard']=True
    c=cfg['FC1'];raw['valid']=np.isfinite(raw[['time','j','v']]).all(axis=1)&raw.j.between(.7-c['容差'],.7+c['容差'])&~raw.nominal_test_guard
    raw['reason']=np.where(raw.nominal_test_guard,'NOMINAL_DIAGNOSTIC_GUARD_ASSUMED',np.where(~raw.j.between(.68,.72),'OFF_NOMINAL_CURRENT',np.where(~np.isfinite(raw.v),'MISSING_CELL','')))
    raw['hour']=np.floor(raw.time).astype(int);hourly=[]
    tmin=float(raw.time.min());tmax=float(raw.time.max());bounds=[tmin,tmin+.6*(tmax-tmin),tmin+.8*(tmax-tmin),tmax]
    for h,g in raw.groupby('hour',sort=True):
        good=g.loc[g.valid];reasons=[];span=good.time.max()-good.time.min() if len(good) else np.nan
        if len(good)<c['小时最少记录']:reasons.append('TOO_FEW_RECORDS')
        if len(good)/len(g)<c['额定记录占比至少']:reasons.append('LOW_NOMINAL_FRACTION')
        if not np.isfinite(span) or span<c['小时最少观测跨度_h']:reasons.append('SPAN_LT_0_5H')
        if g.nominal_test_guard.any():reasons.append('NOMINAL_DIAGNOSTIC_GUARD_ASSUMED')
        if len(good)>1 and good.time.diff().max()*3600>c['相邻间断上限_s']:reasons.append('INTERNAL_GAP_GT_120S')
        hourly.append(dict(id=int(h)+1,time=float(h+1),start_time=float(h),end_time=float(h+1),partition=role(h,h+1,bounds),scheme='FC1_NOMINAL',
          valid=not reasons,v=float(good.v.mean()) if len(good) and not reasons else np.nan,stack_V=float(5*good.v.mean()) if len(good) and not reasons else np.nan,
          raw_records=len(g),retained_records=len(good),actual_first_h=float(g.time.min()),actual_last_h=float(g.time.max()),observed_span_h=span,
          duplicated_records=int(g.duplicate_time.sum()),reason=';'.join(reasons),clock='RECORDED_TIME_H_NOT_1HZ',mean_rule='raw-record equal mean at actual timestamps; duplicate values retained and audited'))
    data=pd.DataFrame(hourly);basehours=data.loc[data.valid].head(10);assert len(basehours)==10 and (basehours.partition=='Train').all()
    Ubase=float(basehours.stack_V.median());base=Ubase/5
    data['base_V']=base;data['D']=base-data.v;data['SOH']=data.v/base
    csv(PREP/'FC1小时观测.csv',data);csv(TAB/'FC1原始记录资格.csv',raw);csv(TAB/'FC1基准十小时.csv',basehours)
    js(OUT/'FC1数据准备记录.json',dict(sources=sources,raw_rows=len(raw),bounds=bounds,Ubase=Ubase,base_cell_V=base,qualified_hours=int(data.valid.sum()),duplicates=int(raw.duplicate_time.sum()),nominal_guard_is_actual_clock=False))
    print('FC1小时观测准备完成',int(data.valid.sum()),flush=True)
    return dict(bounds=bounds,base_V=base,Ubase=Ubase)

def lcs_bit(left,right):
    masks={}
    for i,t in enumerate(right):masks[t]=masks.get(t,0)|(1<<i)
    state=0
    for t in left:
        x=masks.get(t,0)|state;state=x&~(x-((state<<1)|1))
    return state.bit_count()

def real(cfg):
    meta=json.loads((CACHE/'source_manifest.json').read_text(encoding='utf-8'));N=meta['rows'];cols=meta['columns'];idx={k:i for i,k in enumerate(cols)}
    checks=[]
    for key in ['values','time_ns','segment','file_id','source_row']:
        p=CACHE/meta['cache_files'][key]['path'];digest=sha(p);assert digest==meta['cache_files'][key]['sha256'];checks.append(dict(cache=key,SHA256=digest,bytes=p.stat().st_size))
    csv(TAB/'原始缓存身份.csv',checks)
    x=np.memmap(CACHE/'cache/values.bin',mode='r',dtype='float64',shape=(N,len(cols)))
    ns=np.memmap(CACHE/'cache/time_ns.bin',mode='r',dtype='int64',shape=(N,));sg=np.memmap(CACHE/'cache/segment.bin',mode='r',dtype='int32',shape=(N,))
    fi=np.memmap(CACHE/'cache/file_id.bin',mode='r',dtype='int16',shape=(N,));sr=np.memmap(CACHE/'cache/source_row.bin',mode='r',dtype='int64',shape=(N,))
    age=x[:,idx['OpHrs2 [h]']];op=x[:,idx['operation_point']];sets=np.round(x[:,idx['I.S.set [A]']],3);act=x[:,idx['I.S.act [A]']];j=x[:,idx['j.S.act [A/cm2]']]
    tmin=float(age[0]);tmax=float(age[-1]);bounds=[tmin,tmin+.6*(tmax-tmin),tmin+.8*(tmax-tmin),tmax]
    breaks=(op[1:]!=op[:-1])|(sets[1:]!=sets[:-1])|(np.diff(ns)>5e9)|(np.diff(ns)<=0)|(np.diff(age)<0)
    starts=np.r_[0,np.flatnonzero(breaks)+1];ends=np.r_[starts[1:]-1,N-1]
    zeros=np.add.reduceat((abs(act)<=.1).astype('int32'),starts)>0
    runs=pd.DataFrame(dict(run_id=np.arange(len(starts))+1,a=starts,b=ends,start_h=age[starts],end_h=age[ends],start_ns=ns[starts],end_ns=ns[ends],
       duration_s=(ns[ends]-ns[starts])/1e9,op=op[starts].astype(int),set_A=sets[starts],stop=zeros))
    runs['token']=[f'{a}||{b:.3f}' for a,b in zip(runs.op,runs.set_A)]
    template=json.loads((RCR/'CYCLE_TEMPLATE.json').read_text(encoding='utf-8'));tokens=runs.token.to_numpy();anchors=np.flatnonzero((tokens[:-1]==template['anchor'][0])&(tokens[1:]==template['anchor'][1]))
    cycles=[]
    for cid,(a,b) in enumerate(zip(anchors[:-1],anchors[1:]),1):
        group=runs.iloc[a:b];start=runs.iloc[a];end=runs.iloc[b];loaded=end.start_h-start.start_h
        seq=[str(v) for v in group.loc[~group.stop,'op']];fraction=lcs_bit(template['template_positive_operation_tokens'],seq)/len(template['template_positive_operation_tokens'])
        reasons=[]
        if not 8<=loaded<=11:reasons.append('LOADED_DURATION_OUTSIDE_8_11H')
        if fraction<.90:reasons.append('CORE_SEQUENCE_LT_0_90')
        cycles.append(dict(id=cid,left_run=int(a),right_run=int(b),start_h=start.start_h,end_h=end.start_h,start_ns=int(start.start_ns),end_ns=int(end.start_ns),
            loaded_duration_h=loaded,wall_duration_h=(end.start_ns-start.start_ns)/3.6e12,sequence_fraction=fraction,complete=not reasons,reason=';'.join(reasons),partition=role(start.start_h,end.start_h,bounds)))
    cyc=pd.DataFrame(cycles);old=pd.read_csv(RCR/'RAW_CYCLE_CATALOG.csv')
    for r in old.itertuples():
        now=cyc.loc[cyc.id==r.raw_cycle_id].iloc[0]
        assert now.start_ns==pd.Timestamp(r.cycle_start).value and now.end_ns==pd.Timestamp(r.cycle_end).value
        assert abs(now.start_h-r.start_ophrs2_h)<1e-9 and abs(now.end_h-r.end_ophrs2_h)<1e-9
    csv(TAB/'完整运行周期身份.csv',cyc)
    csv(TAB/'未闭合首尾运行片段.csv',[dict(kind='PREFIX',start_row=0,end_row=int(runs.iloc[anchors[0]].a)-1,complete=False),dict(kind='SUFFIX',start_row=int(runs.iloc[anchors[-1]].a),end_row=N-1,complete=False)])
    long=runs.loc[(runs.duration_s>=180)&~runs.stop].copy();long['j_median']=[float(np.median(j[int(r.a):int(r.b)+1])) for r in long.itertuples()]
    previous_set=np.r_[np.nan,runs.set_A.to_numpy()[:-1]]
    lastzero=np.maximum.accumulate(np.where(runs.stop,runs.end_ns,-1));priorzero=np.r_[-1,lastzero[:-1]]
    runs['entry']=np.where(previous_set<runs.set_A,'UP',np.where(previous_set>runs.set_A,'DOWN','SAME'))
    runs['recovery']=np.where((priorzero>=0)&((runs.start_ns-priorzero)/1e9<=1800),'WITHIN_1800S','LATER_OR_UNKNOWN')
    train=long.loc[long.end_h<=bounds[1]]
    counts=train.groupby('token',as_index=False).agg(count=('token','size'),duration=('duration_s','sum'),j=('j_median','median'))
    preferred=counts.loc[counts.j.between(.88,.92)]
    pool=preferred if len(preferred) else counts
    selected=pool.sort_values(['count','duration','token'],ascending=[False,False,True]).iloc[0]
    token=selected.token
    group_train=runs.loc[runs.token.eq(token)&runs.duration_s.ge(180)&runs.end_h.le(bounds[1])]
    classes=group_train.groupby(['entry','recovery']).size().reset_index(name='n').sort_values(['n','entry','recovery'],ascending=[False,True,True]);groupchoice=classes.iloc[0]
    csv(TAB/'Train运行平台候选.csv',counts);csv(TAB/'Train平台方向恢复分组.csv',classes)
    selection=dict(token=token,j=float(selected.j),near_0_9_train_raw_count=int(np.count_nonzero((j[:np.searchsorted(age,bounds[1],side='right')]>=.88)&(j[:np.searchsorted(age,bounds[1],side='right')]<=.92))),
       train_count=int(selected['count']),entry=groupchoice.entry,recovery=groupchoice.recovery,rule='Train-only count selection, not forecast performance',bounds=bounds)
    js(WORK/'配置/训练期平台冻结选择.json',selection)
    candidates=runs.loc[runs.token.eq(token)&runs.duration_s.ge(180)].copy();platforms=[]
    fields={'j':'j.S.act [A/cm2]','v':'mean_voltage_V','temp':'T.Si.CL [°C]','tout':'T.So.CL [°C]','pa':'p.Si.A [kPag]','pc':'p.Si.C [kPag]','current':'I.S.act [A]'}
    for r in candidates.itertuples():
        a=max(int(r.a),int(np.searchsorted(ns,int(r.end_ns)-60_000_000_000,side='right')));b=int(r.b)+1;times=(ns[a:b]-ns[a])/1e9
        rec=dict(run_id=r.run_id,a=a,b=b,entry=r.entry,recovery=r.recovery,start_h=r.start_h,end_h=r.end_h,measurement_end_h=float(age[b-1]),start_ns=int(ns[a]),end_ns=int(ns[b-1]),
           file_start=int(fi[a]),row_start=int(sr[a]),file_end=int(fi[b-1]),row_end=int(sr[b-1]),samples=b-a)
        reasons=[]
        if b-a<54 or sg[a]!=sg[b-1] or (len(times)>1 and np.diff(times).max()>1.05):reasons.append('TAIL_INCOMPLETE_OR_GAP')
        for label,field in fields.items():
            v=np.asarray(x[a:b,idx[field]]);rec[label+'_mean']=float(np.mean(v));rec[label+'_span']=float(np.ptp(v));rec[label+'_p95_p5']=float(np.quantile(v,.95)-np.quantile(v,.05))
            if not np.isfinite(v).all():reasons.append('MISSING_'+label)
        slope=float(np.polyfit(times,x[a:b,0],1)[0]) if len(times)>2 else np.nan;rec['v_slope']=slope
        if rec['j_span']>.01:reasons.append('CURRENT_UNSTABLE')
        if rec['v_p95_p5']>.005:reasons.append('VOLTAGE_SPREAD')
        if not np.isfinite(slope) or abs(slope)>.00002:reasons.append('VOLTAGE_DRIFT')
        if rec['pa_span']>3:reasons.append('PRESSURE_A')
        if rec['pc_span']>3:reasons.append('PRESSURE_C')
        if r.entry!=selection['entry'] or r.recovery!=selection['recovery']:reasons.append('OTHER_ENTRY_RECOVERY_GROUP')
        rec['B60']=not reasons;rec['A60']=not reasons and rec['temp_span']<=1;rec['reason_B60']=';'.join(reasons);rec['reason_A60']=';'.join(reasons+(['TEMP_RANGE_GT_1C'] if rec['temp_span']>1 else []))
        platforms.append(rec)
    pl=pd.DataFrame(platforms);csv(TAB/'Real运行平台逐段资格.csv',pl);observations=[];bases={}
    for scheme in ['A60','B60']:
        series=[]
        for c in cyc.itertuples():
            q=pl.loc[pl.run_id.between(c.left_run+1,c.right_run)&pl[scheme]]
            # 右端锚点不属于本周期；run_id为零基位置+1。
            q=q.loc[q.run_id<c.right_run+1]
            good=c.complete and len(q)>0
            series.append(dict(id=c.id,time=c.id,start_time=c.id-1,end_time=c.id,partition=c.partition,scheme=scheme,valid=bool(good),
                v=float(q.v_mean.median()) if good else np.nan,raw_records=int(q.samples.sum()) if good else 0,platform_count=len(q),
                operating_start_h=c.start_h,operating_end_h=c.end_h,available_h=c.end_h,available_ns=c.end_ns,
                measurement_end_h=float(q.measurement_end_h.max()) if len(q) else np.nan,
                reason=c.reason if not c.complete else '' if good else 'NO_QUALIFIED_TARGET_PLATFORM',clock='ACTUAL_CYCLE_ID; future step=nominal12h; actual hours audit only'))
        s=pd.DataFrame(series);first=s.loc[s.valid&s.partition.eq('Train')].head(5)
        base=float(first.v.median()) if len(first)==5 else np.nan;bases[scheme]=base
        s['base_V']=base;s['D']=base-s.v;s['SOH']=s.v/base;observations.append(s)
        csv(TAB/f'Real_{scheme}_基准五循环.csv',first)
    data=pd.concat(observations,ignore_index=True);csv(PREP/'RealHyFC循环观测.csv',data)
    js(OUT/'Real数据准备记录.json',dict(cache_rows=N,manifest_SHA256=sha(CACHE/'source_manifest.json'),bounds=bounds,cycle_count=len(cyc),complete_cycles=int(cyc.complete.sum()),
       old156_identity_match=True,selected=selection,bases=bases,raw_CSV_reparsed=False,source_files=meta['source_files']))
    print('Real循环观测准备完成',len(cyc),'周期；Train选择',selection,flush=True)
    return dict(bounds=bounds,bases=bases)

def diagnostic():
    d=pd.read_csv(OLD/'输出结果/表格/路线B双参考特征.csv');t=pd.read_csv(OLD/'输出结果/表格/实际时间资格.csv')
    d=d.loc[d.方案.isin(['FC1_10','A60','B60'])].merge(t.drop(columns=['电堆','标称小时','启停后']),on='文件名')
    d['scheme']=np.where(d.电堆.eq('FC1'),'FC1_10',d.方案);d['valid']=d.双参考支持;d['primary']=~d.启停后
    d['available_assumption']='DETECTION_COMPLETED_THEN_AVAILABLE_OFFLINE_ONLY';d['release_confirmed']=False
    csv(PREP/'稀疏诊断观测.csv',d)
    support=pd.read_csv(OLD/'输出结果/表格/共同负载实际支持表.csv')
    csv(TAB/'共同负载健康特征核对表.csv',support.loc[support.方案.isin(['FC1_10','A60','B60'])&support.j.isin([.45,.9])])
    cuts=[]
    for r in t.loc[t.电堆.eq('RealHyFC')].itertuples():
        for cut in [300,600]:
            upper=r.累计小时上界;lower=r.累计小时下界
            phys=bool(np.isfinite(upper) and upper<=cut)
            status='CONFIRMED_PHYSICAL_COMPLETE' if phys else 'AFTER_CUTOFF' if np.isfinite(lower) and lower>cut else 'CROSSES_OR_UNKNOWN_EXCLUDED'
            cuts.append(dict(文件名=r.文件名,标称小时=r.标称小时,启停后=r.启停后,实际检测开始=r.实际测试开始,实际检测完成=r.实际测试结束,
                累计小时下界=lower,累计小时上界=upper,截止_h=cut,物理完成资格=status,本轮离线假设下可用=phys,
                实际发布时间='UNKNOWN',可用性='DETECTION_COMPLETION_ASSUMPTION_NOT_VERIFIED_ONLINE',
                使用该曲线的后移起点保守上界_h=float(upper) if np.isfinite(upper) and upper>cut else np.nan,
                后移起点说明='需该运行小时上界且日历已过检测结束；不是精确完成OpHrs2，不与严格截止混合'))
    csv(TAB/'300和600小时诊断可用性表.csv',cuts)
    # 引用真实支持的BoL电压恢复诊断SOH；不用运行基准混代极化基准。
    refs=support.loc[support.方案.isin(['FC1_10','A60','B60'])&support.j.eq(.9),['电堆','文件名','方案','BoL_V']].rename(columns={'BoL_V':'base_V'})
    d=d.merge(refs,on=['电堆','文件名','方案']);csv(PREP/'稀疏诊断观测.csv',d)

def temperature():
    cache=GUIDE/'健康特征提取/方法复核/输出结果/表格/长平台时序绘图数据.csv';d=pd.read_csv(cache);rows=[];views=[]
    for key,g in d.groupby('平台键',sort=True):
        g=g.sort_values('row').copy();s=g.平台内秒.to_numpy();tc=g.temp.to_numpy();v=g.v.to_numpy()
        z=s-s.mean();basis=np.column_stack([np.ones(len(s)),z,np.sin(2*np.pi*s/4),np.cos(2*np.pi*s/4)])
        bt=np.linalg.lstsq(basis,tc,rcond=None)[0];bv=np.linalg.lstsq(basis,v,rcond=None)[0]
        slowT=pd.Series(tc).rolling(16,min_periods=16,center=True).mean().to_numpy();slowV=pd.Series(v).rolling(16,min_periods=16,center=True).mean().to_numpy()
        rows.append(dict(文件名=g.文件名.iloc[0],平台键=key,主扫描=bool(g.主扫描.iloc[0]),j=float(g.j.median()),原始温度峰峰_C=float(np.ptp(tc)),
           温度线性漂移_C_min=float(bt[1]*60),四秒温度投影幅值_C=float(np.hypot(bt[2],bt[3])),电压线性漂移_mV_min=float(bv[1]*60000),
           四秒电压投影幅值_mV=float(np.hypot(bv[2],bv[3])*1000),慢变温度首末差_C=float(np.nanmean(slowT[-60:])-np.nanmean(slowT[:60])),
           阳极压力峰峰_kPa=float(np.ptp(g.pa)),阴极压力峰峰_kPa=float(np.ptp(g.pc)),定义='固定4s投影与16s滚动均值仅回顾解释，不用于资格修订或预测'))
        g['slow_temp_16s']=slowT;g['fast_temp_residual']=tc-slowT;g['slow_v_16s']=slowV;g['fast_v_residual']=v-slowV;views.append(g)
    csv(TAB/'温压周期波动与工况漂移对比.csv',rows);csv(TAB/'周期漂移时序绘图数据.csv',pd.concat(views,ignore_index=True))
    csv(TAB/'筛选前后参考负载支持.csv',pd.read_csv(OLD/'输出结果/表格/共同负载实际支持表.csv').query("方案 in ['A60','B60'] and j in [0.45,0.9]"))

def main():
    freeze()
    if (OUT/'数据准备完成.json').exists():raise RuntimeError('数据准备已完成，不重复扫描或覆盖；直接运行预测入口')
    cfg=json.loads((WORK/'配置/阶段2A冻结协议.json').read_text(encoding='utf-8'))
    a=fc1(cfg);b=real(cfg);diagnostic();temperature()
    js(OUT/'数据准备完成.json',dict(UTC=datetime.now(timezone.utc).isoformat(),FC1=a,RealHyFC=b,protocol_SHA256=sha(WORK/'配置/阶段2A冻结协议.json'),inputs={p.name:sha(p) for p in PREP.glob('*.csv')}))

if __name__=='__main__':main()
