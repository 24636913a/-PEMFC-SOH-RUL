"""阶段1.2准备研究：只读19份极化源，拒绝外推、缺失填补及训练。"""
from pathlib import Path
import copy
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
import yaml

WORK = Path(__file__).resolve().parent.parent
GUIDE = WORK.parent.parent
REPO = GUIDE.parent
OUT = WORK / '输出结果'
TAB = OUT / '表格'
sys.path[:0] = [str(GUIDE/'数据处理/程序'), str(GUIDE/'健康特征提取/程序')]
from 核验基础 import sha, save_json, table, col, scan_directions
from 极化特征计算 import prepare, representatives, interpolate


def snapshot():
    """先登记规则时间和旧文件SHA；本任务外任何内容均不可被修改。"""
    marker = WORK/'配置/执行前证据.json'
    if marker.exists():
        return json.loads(marker.read_text(encoding='utf-8'))
    rules = WORK/'配置/候选表示预登记.json'
    old = {str(p.relative_to(GUIDE)): sha(p) for p in GUIDE.rglob('*')
           if p.is_file() and not p.is_relative_to(WORK) and '__pycache__' not in p.parts}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    data = dict(登记UTC=datetime.now(timezone.utc).isoformat(), 本地HEAD=git,
                规则SHA256=sha(rules), 历史文件SHA256=old,
                远端核对='两次失败：Recv failure: Connection was reset；curl 28 Failed to connect to github.com:443')
    save_json(marker, data)
    # 只读原指导方案，不依赖Word，不改文档。
    path=GUIDE/'step1/燃料电池三档研究方案.docx'
    with zipfile.ZipFile(path) as z:
        root=ET.fromstring(z.read('word/document.xml'))
        text='\n'.join(''.join(p.itertext()) for p in root.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'))
    (OUT/'指导方案只读文本.txt').write_text(text, encoding='utf-8')
    return data


def guarded(reps, audit, queries, gap=.2, values=None):
    """原局部插值加拒绝平台屏障；不能跨过压力失败平台。"""
    result, support=interpolate(reps, np.asarray(queries), gap, values)
    for i,s in enumerate(support):
        blocked=[]
        if s['支持']:
            for a in audit:
                if (not a.get('保留',False) and a.get('平台秒',0)>=180 and
                    np.isfinite(a.get('实测j',np.nan)) and
                    s['左j'] < a['实测j'] < s['右j']):
                    blocked.append(str(a.get('组号'))+':'+a.get('拒绝原因',''))
            if blocked:
                result[i]=np.nan; s['支持']=False; s['拒绝原因']='REJECTED_PLATFORM_BARRIER:'+'|'.join(blocked)
        s['括距']=s['右j']-s['左j']
        s['阻断平台']='|'.join(blocked)
        # 供共享原始支持连通分组，不能把插值格数当独立负载数。
        s['支持组']=';'.join(str(p['group']) for p in reps if s['支持'] and p['j'] in (s['左j'],s['右j']))
    return result,support


def centered(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    X=np.column_stack([np.ones(len(x)),x-.9]); beta=np.linalg.lstsq(X,y,rcond=None)[0]
    gram=X.T@X; inv=np.linalg.inv(gram);pred=X@beta
    # 比较原参数化时变换第二列单位，不能把有单位的条件数当固有物理性质。
    Z=np.column_stack([np.ones(len(x)),x/.9]);zi=np.linalg.inv(Z.T@Z)
    return dict(h_ref_V=float(beta[0]),s_V_per_A_cm2=float(beta[1]),
                一维c_V=float(y.mean()),一维RMSE_V=float(np.sqrt(np.mean((y-y.mean())**2))),
                二维RMSE_V=float(np.sqrt(np.mean((y-pred)**2))),
                设计矩阵条件数=float(np.linalg.cond(X)),
                参数相关=float(inv[0,1]/np.sqrt(inv[0,0]*inv[1,1])),
                原参数相关=float(zi[0,1]/np.sqrt(zi[0,0]*zi[1,1])),
                最大绝对残差_V=float(np.max(abs(y-pred))),
                l1等价_V=float(beta[0]-.9*beta[1]),l2等价_V=float(.9*beta[1]),
                pred=pred)


def grouped_holdout(x,y,current_support,bol_support):
    """按共享平台连通分组；未受负载包络/局部间隙支持的留出不预测。"""
    n=len(x); links=[]
    for a,b in zip(current_support,bol_support):
        links.append(set('cur:'+k for k in a['支持组'].split(';') if k) |
                     set('bol:'+k for k in b['支持组'].split(';') if k))
    groups=[];unused=set(range(n))
    while unused:
        ids={unused.pop()}; prev=-1
        while len(ids)!=prev:
            prev=len(ids);union=set().union(*(links[i] for i in ids))
            add={i for i in unused if links[i]&union};ids|=add;unused-=add
        groups.append(sorted(ids))
    rows=[]
    for gi,ids in enumerate(groups):
        keep=np.array([i for i in range(n) if i not in ids],dtype=int);reason=''
        if len(keep)<2 or np.ptp(x[keep])<1e-12:reason='TRAIN_SUPPORT_TOO_FEW'
        beta=np.linalg.lstsq(np.column_stack([np.ones(len(keep)),x[keep]-.9]),y[keep],rcond=None)[0] if not reason else None
        for i in ids:
            why=reason
            if not why:
                p=np.searchsorted(x[keep],x[i])
                if p==0 or p==len(keep):why='OUTSIDE_TRAIN_LOAD_ENVELOPE_NO_EXTRAPOLATION'
                elif x[keep][p]-x[keep][p-1]>.2+1e-12:why='TRAIN_LOAD_GAP_GT_0_20'
            rows.append(dict(留出组=gi,j=float(x[i]),训练j=';'.join(map(str,x[keep])),
                 可评价=not why,拒绝原因=why,
                 一维预测_V=float(y[keep].mean()) if not why else np.nan,
                 二维预测_V=float(np.array([1,x[i]-.9])@beta) if not why else np.nan,
                 真实损失_V=float(y[i]),独立性='按共享原始平台连通分组，非寿命预测'))
    return rows


def raw_neighborhood(raw, sid, q):
    g=raw.loc[(raw.segment==sid)&np.isfinite(raw.j)&np.isfinite(raw.v)]
    left=g.loc[g.j<=q].sort_values(['j','row']).tail(1)
    right=g.loc[g.j>=q].sort_values(['j','row']).head(1)
    r=dict(扫描分段=int(sid),j=q,邻域半径=.01,邻域原始点数=int((abs(g.j-q)<=.01).sum()),
           原始可括定=bool(len(left) and len(right)),原始支持不等于稳态=True)
    for label,side in [('左',left),('右',right)]:
        r[label+'原始行']=int(side.row.iloc[0]) if len(side) else np.nan
        r[label+'实测j']=float(side.j.iloc[0]) if len(side) else np.nan
        r[label+'电压_V']=float(side.v.iloc[0]) if len(side) else np.nan
        r[label+'时间']=str(side.datetime.iloc[0]) if len(side) else ''
    r['原始括距']=r['右实测j']-r['左实测j']
    near=g.loc[abs(g.j-q)<=.01]
    r['邻域原始行号']=';'.join(map(str,near.row.to_list()))
    return r


def unit_tests():
    p=[dict(j=.4,v=.7,row_start=1,group=1),dict(j=.6,v=.6,row_start=2,group=3)]
    a=[dict(保留=False,平台秒=300,实测j=.5,组号=2,拒绝原因='CONDITION_UNSTABLE_pa')]
    assert np.isnan(guarded(p,a,[.5])[0][0])
    assert np.isnan(guarded(p,[],[.3,.7])[0]).all()
    assert np.isfinite(guarded(p,[],[.5])[0][0])
    x=np.array([.45,.5,.8,.85,.9,.95]);y=-.01+.03*(x-.9)
    f=centered(x,y);assert abs(f['h_ref_V']+.01)<1e-14 and abs(f['s_V_per_A_cm2']-.03)<1e-14
    assert centered(x,np.zeros(6))['h_ref_V']==0
    e=centered(np.array([.45,.9]),np.array([.01,.02]));assert abs(e['s_V_per_A_cm2']-(.02-.01)/.45)<1e-14
    su=[{'支持组':'1;2'} for _ in range(len(x))]
    assert not any(r['可评价'] for r in grouped_holdout(x,y,su,su))
    return ['压力拒绝平台屏障','禁止外推','合法局部插值','负损失保留','中心参数闭式解','BoL恒等零','双参考数学等价','留出整组后空训练集合返回NA']


def run():
    evidence=snapshot();cfg=json.loads((WORK/'配置/候选表示预登记.json').read_text(encoding='utf-8'))
    assert sha(WORK/'配置/候选表示预登记.json')==evidence['规则SHA256']
    if (OUT/'数值运行完成.json').exists():raise RuntimeError('已有完成结果；禁止重复运行覆盖，请使用独立复核入口')
    old=yaml.safe_load((GUIDE/cfg['原始规则']).read_text(encoding='utf-8-sig'))
    identity=pd.read_csv(GUIDE/'健康特征提取/输出结果/历史核验/20261025T153526_430847/表格/极化源身份复核.csv')
    meta=pd.read_csv(GUIDE/'数据处理/输出结果/表格/运行数据与极化测试对应表.csv').set_index('文件名')
    queries=np.array(cfg['探索性候选j']);profiles={};raws={};sources=[];audits=[];members=[];rep_rows=[];neighborhood=[];times=[]
    conditions=[];segments=[]
    for info in identity.to_dict('records'):
        path=REPO/info['原始路径'];assert sha(path)==info['SHA256']
        stack=info['电堆'];name=info['文件名']
        frame=pd.read_excel(path,engine='openpyxl') if stack=='FC1' else pd.read_csv(path,sep=';',encoding='utf-8-sig',low_memory=False)
        assert len(frame)==info['实际行数'];raw,chosen=prepare(frame,stack,old);assert chosen==info['扫描分段']
        direction,sg=scan_directions(raw.j.to_numpy(),.05);raw['direction']=direction
        for label,field in [('current','I.S.act [A]'),('tout','T.So.CL [°C]'),('tgas_a','T.Si.A [°C]'),('tgas_c','T.Si.C [°C]'),('pout_a','p.So.A [kPag]'),('pout_c','p.So.C [kPag]')]:
            raw[label]=col(frame,field)
        raws[name]=raw
        sources.append(dict(**info,本次SHA256=sha(path),源身份一致=True,实际字段=';'.join(frame.columns)))
        t=meta.loc[name]
        times.append(dict(电堆=stack,文件名=name,标称小时=t.标称老化小时,启停后=bool(t.启停后),
            实际测试开始=t.实际测试开始,实际测试结束=t.实际测试结束,时钟资格=t.时钟资格,
            累计小时下界=t.累计小时下界,累计小时上界=t.累计小时上界,
            前300h物理结束资格=t.前300h物理结束资格,前600h物理结束资格=t.前600h物理结束资格,
            结果发布时间='UNKNOWN',严格在线资格='BLOCKED_RELEASE_UNKNOWN' if stack=='RealHyFC' else 'BLOCKED_NOMINAL_ONLY'))
        for sid,part in raw.groupby('segment'):
            di=int(part.direction.iloc[-1]);label='降载' if di<0 else '升载' if di>0 else '未定'
            segments.append(dict(电堆=stack,文件名=name,扫描分段=int(sid),扫描方向=label,主扫描=sid==chosen,点数=len(part),最小j=part.j.min(),最大j=part.j.max()))
            for q in queries:neighborhood.append(dict(电堆=stack,文件名=name,扫描方向=label,主扫描=sid==chosen,**raw_neighborhood(raw,sid,q)))
        cases=[('FC1_10',chosen,'主规则',copy.deepcopy(old)),('FC1_05',chosen,'FC1窄分组',copy.deepcopy(old)),('FC1_20',chosen,'FC1宽分组',copy.deepcopy(old))] if stack=='FC1' else []
        if stack=='RealHyFC':
            ups=[int(s) for s,p in raw.groupby('segment') if p.direction.iloc[-1]>0 and p.j.min()<=.3 and p.j.max()>=1]
            for prefix,sid in [('A',chosen),('B',chosen)]+([('UP_B',ups[0])] if ups else []):
                for tail in cfg['RealHyFC末尾秒']:
                    c=copy.deepcopy(old);c['RealHyFC']['末尾窗口秒']=tail
                    if prefix!='A':c['RealHyFC']['冷却液温度峰峰上限_C']=np.inf
                    cases.append((prefix+str(tail),sid,'主规则',c))
        for label,sid,variant,c in cases:
            reps,audit,mem=representatives(raw,sid,stack,variant,c)
            profiles[(name,label)]=(reps,audit)
            audits.extend(dict(电堆=stack,文件名=name,方案=label,**a) for a in audit)
            members.extend(dict(电堆=stack,文件名=name,方案=label,**m) for m in mem)
            rep_rows.extend(dict(电堆=stack,文件名=name,方案=label,**{k:v for k,v in p.items() if k!='block'},分块代表值=';'.join(format(float(v),'.17g') for v in p['block'])) for p in reps)
            if stack=='RealHyFC' and label=='B60':
                for p in reps:
                    g=raw.loc[raw.row.between(p['row_start'],p['row_end'])]
                    r=dict(文件名=name,扫描分段=sid,组号=p['group'],代表j=p['j'],原始首行=p['row_start'],原始末行=p['row_end'],时间开始=str(g.datetime.iloc[0]),时间结束=str(g.datetime.iloc[-1]))
                    for f in ['j','v','temp','tout','tgas_a','tgas_c','pa','pc','pout_a','pout_c','current','hydrogen','air']:
                        v=g[f].to_numpy(float);r[f+'_mean']=float(np.mean(v));r[f+'_range']=float(np.ptp(v));r[f+'_p95_p5']=float(np.quantile(v,.95)-np.quantile(v,.05))
                    sec=(g.datetime-g.datetime.iloc[0]).dt.total_seconds().to_numpy();r['v_slope']=float(np.polyfit(sec,g.v,1)[0]);r['temp_v_r']=float(np.corrcoef(g.temp,g.v)[0,1]) if np.ptp(g.temp)>0 and np.ptp(g.v)>0 else np.nan
                    conditions.append(r)
        print('只读极化及支持审计:',stack,name,flush=True)
    for filename,rows in [('源文件身份复核.csv',sources),('实际时间资格.csv',times),('全部扫描身份.csv',segments),('逐扫描原始邻域支持.csv',neighborhood),('平台及分组筛选.csv',audits),('代表点原始成员.csv',members),('合格代表点.csv',rep_rows),('RealHyFC代表平台温压与电压.csv',conditions)]:table(TAB/filename,rows)
    supportrows=[];gridrows=[];values={};supports={}
    for (name,label),(reps,audit) in profiles.items():
        stack=meta.loc[name,'电堆'];bol=old['基准'][stack];br,ba=profiles[(bol,label)]
        v,su=guarded(reps,audit,queries);bv,bs=guarded(br,ba,queries)
        values[(name,label)]=(v,bv);supports[(name,label)]=(su,bs)
        legacy=interpolate(reps,queries,.2)[0]
        for i,q in enumerate(queries):
            s=su[i];b=bs[i];sid=audit[0]['扫描分段'];rawnb=next(r for r in neighborhood if r['文件名']==name and r['扫描分段']==sid and r['j']==q)
            supportrows.append(dict(电堆=stack,文件名=name,方案=label,扫描方向='升载' if label.startswith('UP') else '降载',j=q,
              原始可括定=rawnb['原始可括定'],本曲线支持=s['支持'],BoL支持=b['支持'],共同支持=bool(s['支持'] and b['支持']),
              本曲线V=v[i],BoL_V=bv[i],左j=s['左j'],右j=s['右j'],括距=s['括距'],左原始行=s['左原始行'],右原始行=s['右原始行'],支持组=s['支持组'],
              BoL左j=b['左j'],BoL右j=b['右j'],BoL左原始行=b['左原始行'],BoL右原始行=b['右原始行'],BoL支持组=b['支持组'],
              拒绝原因=s['拒绝原因'],BoL拒绝原因=b['拒绝原因'],旧插值可计算=bool(np.isfinite(legacy[i])),
              压力规则_kPa=3,温度资格='原1C规则' if label.startswith('A') else '未获认证；FC1缺条件或B仅诊断'))
            gridrows.append(dict(电堆=stack,文件名=name,方案=label,j=q,V=v[i],BoL_V=bv[i],D_V=bv[i]-v[i],支持=bool(np.isfinite(v[i]) and np.isfinite(bv[i]))))
    table(TAB/'共同负载实际支持表.csv',supportrows);table(TAB/'候选网格损失.csv',gridrows)
    main_names=[n for n in identity.文件名 if not bool(meta.loc[n,'启停后'])]
    common=np.ones(len(queries),bool)
    for n in main_names:
        label='FC1_10' if meta.loc[n,'电堆']=='FC1' else 'B60'
        v,b=values[(n,label)];common&=np.isfinite(v)&np.isfinite(b)
    domains={'每曲线实际支持':None,'主曲线跨堆共同域':common}
    features=[];points=[];cv=[];dual=[];dualeval=[];perturb=[];sensitivity=[];draws_by={}
    for (name,label),(reps,audit) in profiles.items():
        stack=meta.loc[name,'电堆'];bol=old['基准'][stack];v,b=values[(name,label)];loss=b-v;su,bs=supports[(name,label)]
        for domain,mask in domains.items():
            ok=np.isfinite(loss) if mask is None else np.isfinite(loss)&mask
            idx=np.flatnonzero(ok);x=queries[ok];y=loss[ok]
            r=dict(电堆=stack,文件名=name,方案=label,支持域=domain,标称小时=float(meta.loc[name,'标称老化小时']),启停后=bool(meta.loc[name,'启停后']),
              支持点数=len(x),支持j=';'.join(map(str,x)),跨度=float(np.ptp(x)) if len(x) else np.nan,
              最大中间缺口=float(np.diff(x).max()) if len(x)>1 else np.nan,
              代数可计算=len(x)>=3,历史阶段1结论='BLOCKED保持不变',正式资格=False)
            if len(x)>=3:
                f=centered(x,y);r.update({k:a for k,a in f.items() if k!='pred'})
                cv.extend(dict(电堆=stack,文件名=name,方案=label,支持域=domain,**a) for a in grouped_holdout(x,y,[su[i] for i in idx],[bs[i] for i in idx]))
                points.extend(dict(电堆=stack,文件名=name,方案=label,支持域=domain,j=float(q),D_V=float(d),一维重构_V=f['一维c_V'],二维重构_V=float(pred),权重=1/len(x)) for q,d,pred in zip(x,y,f['pred']))
            features.append(r)
        lo=np.where(abs(queries-.45)<1e-12)[0][0];hi=np.where(abs(queries-.9)<1e-12)[0][0]
        valid=bool(np.isfinite(loss[[lo,hi]]).all());d=dict(电堆=stack,文件名=name,方案=label,
            标称小时=float(meta.loc[name,'标称老化小时']),启停后=bool(meta.loc[name,'启停后']),
            双参考支持=valid,D_low_V=float(loss[lo]),D_ref_V=float(loss[hi]),S_V_per_A_cm2=float((loss[hi]-loss[lo])/.45) if valid else np.nan,
            拒绝原因='' if valid else 'LOW_OR_REFERENCE_OR_BOL_NOT_SUPPORTED',正式资格=False)
        if valid:
            f=centered(np.array([.45,.9]),loss[[lo,hi]])
            d['两点A与B最大系数差']=float(max(abs(f['h_ref_V']-d['D_ref_V']),abs(f['s_V_per_A_cm2']-d['S_V_per_A_cm2'])))
            for i,q in enumerate(queries):
                if .45<q<.9 and np.isfinite(loss[i]):
                    pred=d['D_low_V']+d['S_V_per_A_cm2']*(q-.45)
                    dualeval.append(dict(电堆=stack,文件名=name,方案=label,j=q,真实损失_V=loss[i],单参考重构_V=d['D_ref_V'],双参考重构_V=pred,
                        同信息两点A重构_V=f['h_ref_V']+f['s_V_per_A_cm2']*(q-.9),
                        资格='描述性留查询，不是独立留平台；共享原始插值平台，不是预测'))
        dual.append(d)
        if label not in ['FC1_10','B60']:continue
        # 同一个BoL扰动矩阵对全部日期共享；只模拟平台内记录变化。
        def draw(key):
            if key in draws_by:return draws_by[key]
            rr,aa=profiles[key];rng=np.random.default_rng(20261025+int(hashlib.sha256(key[0].encode()).hexdigest()[:8],16))
            vals=[]
            for _ in range(100):
                vv=[float(rng.choice(p['block'],len(p['block']),replace=True).mean()) for p in rr]
                vals.append(guarded(rr,aa,queries,values=vv)[0])
            draws_by[key]=np.array(vals);return draws_by[key]
        db=draw((bol,label));dt=db if name==bol else draw((name,label));dl=db-dt
        ok=common&np.isfinite(loss)
        for k,row in enumerate(dl):
            if ok.sum()<3:continue
            f=centered(queries[ok],row[ok]);perturb.append(dict(电堆=stack,文件名=name,扰动=k,来源='共享BoL分块扰动，非重复测量置信区间',h_ref_V=f['h_ref_V'],s_V_per_A_cm2=f['s_V_per_A_cm2'],
               D_low_V=row[lo],D_ref_V=row[hi],S_V_per_A_cm2=(row[hi]-row[lo])/.45 if valid else np.nan,
               仅BoL_Dref扰动_V=db[k,hi]-b[hi]))
    feature=pd.DataFrame(features);dualdf=pd.DataFrame(dual)
    # 固定跨所有方案交集，拒绝用不同支持域造成假的参数敏感性。
    for name in identity.文件名:
        labels=['FC1_10','FC1_05','FC1_20'] if meta.loc[name,'电堆']=='FC1' else ['B60','B30','B90']
        masks=[np.isfinite(values[(name,l)][1]-values[(name,l)][0]) for l in labels]
        ok=common&np.logical_and.reduce(masks);base=None
        for label in labels:
            v,b=values[(name,label)];r=dict(电堆=meta.loc[name,'电堆'],文件名=name,方案=label,同域j=';'.join(map(str,queries[ok])),同域点数=int(ok.sum()))
            if ok.sum()>=3:
                f=centered(queries[ok],(b-v)[ok]);now=np.array([f['h_ref_V'],f['s_V_per_A_cm2']]);base=now if base is None else base
                r.update(h_ref_V=now[0],s_V_per_A_cm2=now[1],h_ref相对主方案差_V=now[0]-base[0],s相对主方案差_V_per_A_cm2=now[1]-base[1])
                a=dualdf.loc[(dualdf.文件名==name)&(dualdf.方案==label)].iloc[0]
                r.update(D_low_V=a.D_low_V,D_ref_V=a.D_ref_V,S_V_per_A_cm2=a.S_V_per_A_cm2)
            sensitivity.append(r)
    for filename,rows in [('路线A中心参数.csv',features),('路线A逐点重构.csv',points),('共享平台分组留出.csv',cv),('路线B双参考特征.csv',dual),('单参考与双参考留查询重构.csv',dualeval),('共享BoL条件扰动.csv',perturb),('同域处理敏感性.csv',sensitivity)]:table(TAB/filename,rows)
    # 不重叠末段，只比较同一已保留平台，不视为独立诊断实验。
    within=[]
    for name in identity.loc[identity.电堆=='RealHyFC','文件名']:
        raw=raws[name];rr,aa=profiles[(name,'B90')]
        for p in rr:
            end=raw.loc[raw.row==p['row_end'],'datetime'].iloc[0]
            for a,b in [(90,60),(60,30),(30,0)]:
                # 主扫描身份从实际profile audit取得，不能假定所有文件segment恒定。
                sid=aa[0]['扫描分段'];g=raw.loc[(raw.datetime>end-pd.Timedelta(seconds=a))&(raw.datetime<=end-pd.Timedelta(seconds=b))&(raw.segment==sid)]
                r=dict(文件名=name,组号=p['group'],代表j=p['j'],非重叠时间段=f'末{a}至{b}秒',点数=len(g),原始首行=int(g.row.iloc[0]),原始末行=int(g.row.iloc[-1]))
                for f in ['j','v','temp','tout','pa','pc']:r[f+'_mean']=float(g[f].mean());r[f+'_range']=float(np.ptp(g[f]))
                within.append(r)
    table(TAB/'平台末段不重叠窗口.csv',within)
    # 同方向BoL、升降载和启停前后单列；只在两侧均有支持的位置比较。
    pairs=[]
    for name in identity.loc[identity.电堆=='RealHyFC','文件名']:
        for kind,a,b in [('升降载','B60','UP_B60')]:
            if (name,b) not in values:continue
            va,ba=values[(name,a)];vb,bb=values[(name,b)]
            for i,q in enumerate(queries):
                valid=np.isfinite([va[i],ba[i],vb[i],bb[i]]).all()
                pairs.append(dict(文件名=name,对照=kind,j=q,双方支持=valid,降载V=va[i],升载V=vb[i],升减降_V=vb[i]-va[i] if valid else np.nan,
                    同方向BoL损失差_V=(bb[i]-vb[i])-(ba[i]-va[i]) if valid else np.nan))
    table(TAB/'升降载描述性对照.csv',pairs)
    recovery=[]
    for h in [1500,2300]:
        n=f'Polcurve_Nominal_RHFC_{h}h.csv';after=f'Polcurve_Nominal_RHFC_{h}h_afterAirAirStart.csv'
        va,ba=values[(n,'B60')];vb,bb=values[(after,'B60')]
        for i,q in enumerate(queries):
            r=dict(标称小时=h,j=q,双方支持=bool(np.isfinite(va[i]) and np.isfinite(vb[i])),前V=va[i],后V=vb[i],后减前_V=vb[i]-va[i],
                前日历=meta.loc[n,'实际测试结束'],后日历=meta.loc[after,'实际测试开始'],解释='描述性配对，不认定不可逆损伤修复')
            recovery.append(r)
    table(TAB/'启停恢复配对.csv',recovery)
    cond=pd.DataFrame(conditions);co=[]
    for n in identity.loc[identity.电堆=='RealHyFC','文件名']:
        a=cond.loc[cond.文件名==n];bo=cond.loc[cond.文件名==old['基准']['RealHyFC']]
        for p in a.to_dict('records'):
            match=bo.loc[abs(bo.代表j-p['代表j'])<.01]
            if len(match)!=1:continue
            b=match.iloc[0];r=dict(文件名=n,j=p['代表j'],本次组=p['组号'],BoL组=b.组号,匹配='仅用于描述原始平台条件差，未认证物理等价')
            for f in ['temp','tout','tgas_a','tgas_c','pa','pc','pout_a','pout_c','current','hydrogen','air']:
                r[f+'_本次']=p[f+'_mean'];r[f+'_BoL']=b[f+'_mean'];r[f+'_差']=p[f+'_mean']-b[f+'_mean']
            co.append(r)
    table(TAB/'与BoL平台条件差.csv',co)
    # 独立文献证据读取已有核对文字，保留SHA与定位，不访问网络或其他Test。
    literature=GUIDE/'健康特征提取/方法复核/输出结果/表格/实验说明与字段证据.csv'
    table(TAB/'原论文实验依据.csv',pd.read_csv(literature))
    save_json(OUT/'数值运行完成.json',dict(完成UTC=datetime.now(timezone.utc).isoformat(),预登记SHA256=evidence['规则SHA256'],
        源数=len(identity),原始极化行数=int(identity.实际行数.sum()),主曲线数=len(main_names),
        跨堆主曲线共同j=queries[common].tolist(),算法测试=unit_tests(),正式训练=False,历史阶段1='BLOCKED_UNCHANGED'))
    print('一次候选诊断完成；跨堆主曲线交集:',queries[common],flush=True)


if __name__=='__main__':
    run()
