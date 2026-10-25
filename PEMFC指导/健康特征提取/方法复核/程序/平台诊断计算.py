"""阶段1.1探索性诊断算法，复用冻结数值函数，仅B对照取消单个温度峰峰判据。"""
from pathlib import Path
import copy,json,sys
import numpy as np
import pandas as pd
import yaml

GUIDE=next(p for p in Path(__file__).resolve().parents if p.name=='PEMFC指导')
sys.path[:0]=[str(GUIDE/'数据处理/程序'),str(GUIDE/'健康特征提取/程序')]
from 核验基础 import col,mean_cells,CELLS_REAL,sha,table,save_json,scan_directions
from 极化特征计算 import prepare,representatives,interpolate,grid,solve,block_draws


def pearson(x,y):
    a=np.asarray(x,float);b=np.asarray(y,float);ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
    return float(np.corrcoef(a,b)[0,1]) if len(a)>3 and np.ptp(a)>1e-12 and np.ptp(b)>1e-12 else np.nan


def metrics(group):
    fields=['j','v','temp','pa','pc','tout','tgas_a','tgas_c','pout_a','pout_c','current']
    result={}
    for f in fields:
        v=group[f].to_numpy(float);v=v[np.isfinite(v)]
        for k,value in [('mean',np.mean(v) if len(v) else np.nan),('range',np.ptp(v) if len(v) else np.nan),
                        ('p95_p5',np.quantile(v,.95)-np.quantile(v,.05) if len(v) else np.nan)]:result[f+'_'+k]=float(value)
    sec=(group.datetime-group.datetime.iloc[0]).dt.total_seconds().to_numpy()
    result['v_slope']=float(np.polyfit(sec,group.v,1)[0]) if len(group)>2 and np.ptp(sec)>0 else np.nan
    result['temp_v_r']=pearson(group.temp,group.v)
    result['temp_v_diff_r']=pearson(np.diff(group.temp),np.diff(group.v))
    result['temp_out_r']=pearson(group.temp,group.tout)
    result['samples']=len(group)
    return result


def periodogram(t,v):
    good=np.isfinite(v)&np.isfinite(t);t=t[good];v=v[good]
    if len(v)<32 or np.ptp(t)==0:return np.nan,np.nan
    dt=np.median(np.diff(t));res=v-np.polyval(np.polyfit(t,v,1),t)
    power=np.abs(np.fft.rfft(res))**2;freq=np.fft.rfftfreq(len(res),dt)
    ok=(freq>=1/300)&(freq<=1/4)
    if not ok.any() or power[ok].sum()==0:return np.nan,np.nan
    peak=np.flatnonzero(ok)[np.argmax(power[ok])]
    return float(1/freq[peak]),float(power[peak]/power[ok].sum())


def raw_diagnostics(raw,name,selected,out,plot_frames,windows,platforms,lags):
    for sid,part in raw.groupby('segment',sort=True):
        delta=part.datetime.diff().dt.total_seconds()
        ids=((part['set'].diff().abs()>.1)|part['set'].isna()|(delta>1.05)|(delta<=0)).cumsum()
        for gid,whole in part.groupby(ids,sort=True):
            duration=(whole.datetime.iloc[-1]-whole.datetime.iloc[0]).total_seconds()
            if duration<180:continue
            key=f'{name}__s{sid}_p{gid}';sec=(whole.datetime-whole.datetime.iloc[0]).dt.total_seconds().to_numpy()
            direction=int(whole.direction.iloc[-1]);state=dict(文件名=name,平台键=key,扫描分段=int(sid),组号=int(gid),
                段末方向='降载' if direction<0 else '升载' if direction>0 else '未定',主扫描=bool(sid==selected),
                原始首行=int(whole.row.iloc[0]),原始末行=int(whole.row.iloc[-1]),开始时间=str(whole.datetime.iloc[0]),结束时间=str(whole.datetime.iloc[-1]),
                平台秒=float(duration),j中位=float(whole.j.median()))
            periods=periodogram(sec,whole.temp.to_numpy());full=metrics(whole)
            platforms.append(dict(**state,**full,温度主周期秒=periods[0],温度谱峰占比=periods[1]))
            view=whole.copy();view['文件名']=name;view['平台键']=key;view['平台内秒']=sec;view['主扫描']=sid==selected
            plot_frames.append(view)
            subsets={'全平台':whole,'前60秒':whole.loc[whole.datetime<whole.datetime.iloc[0]+pd.Timedelta(seconds=60)]}
            for tail in [30,60,90]:subsets[f'末{tail}秒']=whole.loc[whole.datetime>whole.datetime.iloc[-1]-pd.Timedelta(seconds=tail)]
            for window,g in subsets.items():windows.append(dict(**state,窗口=window,**metrics(g)))
            # 固定滞后全部保留，不报告最佳滞后为同步校正。
            for lag in range(-10,11):
                x=whole.temp.to_numpy();y=whole.v.to_numpy()
                a,b=(x[:-lag],y[lag:]) if lag>0 else (x[-lag:],y[:lag]) if lag<0 else (x,y)
                lags.append(dict(**state,滞后秒=lag,定义='corr(T(t),V(t+lag))，非因果/校正',相关=pearson(a,b)))


def run(cfg,out):
    oldcfg=yaml.safe_load((GUIDE/cfg['原阶段1规则']).read_text(encoding='utf-8-sig'))
    original=GUIDE/cfg['原结果'];meta=pd.read_csv(GUIDE/'数据处理/输出结果/表格/运行数据与极化测试对应表.csv')
    identity=pd.read_csv(original/'表格/极化源身份复核.csv');identity=identity.loc[identity.电堆=='RealHyFC']
    oldaudit=pd.read_csv(original/'表格/平台及分组筛选明细.csv');oldreps=pd.read_csv(original/'表格/合格代表点及条件.csv')
    profiles={};checks=[];repro=[];windows=[];platforms=[];frames=[];lags=[];source=[];resolution=[];rep_table=[];retained=[]
    for info in identity.to_dict('records'):
        path=GUIDE.parent/info['原始路径'];assert sha(path)==info['SHA256']
        frame=pd.read_csv(path,sep=';',encoding='utf-8-sig',low_memory=False)
        assert len(frame)==info['实际行数']
        raw,chosen=prepare(frame,'RealHyFC',oldcfg);name=info['文件名'];assert chosen==info['扫描分段']
        raw['direction']=scan_directions(raw.j.to_numpy(),.05)[0]
        for label,field in [('current','I.S.act [A]'),('tout','T.So.CL [°C]'),('tgas_a','T.Si.A [°C]'),('tgas_c','T.Si.C [°C]'),
                            ('pout_a','p.So.A [kPag]'),('pout_c','p.So.C [kPag]')]:raw[label]=col(frame,field)
        source.append(dict(**info,重新SHA一致=True,实际字段=';'.join(frame.columns),单片数量字段=';'.join(map(str,frame['Number of cells'].unique())) if 'Number of cells' in frame else 'NA'))
        dt=raw.datetime.diff().dt.total_seconds().dropna()
        for field in ['temp','tout','tgas_a','tgas_c','current','j','v']:
            vals=raw[field].to_numpy();unique=np.unique(vals[np.isfinite(vals)]);d=np.diff(unique);adj=np.diff(vals)
            resolution.append(dict(文件名=name,字段=field,有效值数=int(np.isfinite(vals).sum()),唯一值数=len(unique),
                最小正值间隔=float(d[d>1e-12].min()) if np.any(d>1e-12) else np.nan,
                相邻未变化比例=float(np.mean(adj==0)),日历间隔中位秒=float(dt.median()),
                正常一秒间隔比例=float(np.mean(abs(dt-1)<.01)),仪器精度='UNKNOWN；观测值间隔不等于测量准确度'))
        raw_diagnostics(raw,name,chosen,out,frames,windows,platforms,lags)
        for case in cfg['诊断组']:
            conf=copy.deepcopy(oldcfg);conf['RealHyFC']['末尾窗口秒']=case['末尾秒']
            if not case['温度限制']:conf['RealHyFC']['冷却液温度峰峰上限_C']=np.inf
            reps,audit,members=representatives(raw,chosen,'RealHyFC','主规则',conf)
            profiles[(name,case['名称'])]=reps
            checks.extend(dict(文件名=name,方案=case['名称'],身份='DIAGNOSTIC_ONLY',**a) for a in audit)
            retained.extend(dict(文件名=name,方案=case['名称'],**a) for a in members)
            rep_table.extend(dict(文件名=name,方案=case['名称'],**{k:v for k,v in p.items() if k!='block'},
                     分块代表值=';'.join(format(float(v),'.17g') for v in p['block'])) for p in reps)
            if case['名称']=='A60':
                expected=oldaudit.loc[(oldaudit.文件名==name)&(oldaudit.方案=='主规则')].reset_index(drop=True)
                actual=pd.DataFrame(audit)
                assert len(expected)==len(actual)
                assert np.array_equal(actual.保留,expected.保留)
                assert np.array_equal(actual.拒绝原因.fillna(''),expected.拒绝原因.fillna(''))
                assert np.allclose(actual.实测j,expected.实测j,equal_nan=True,atol=1e-14,rtol=0)
                rp=oldreps.loc[(oldreps.文件名==name)&(oldreps.方案=='主规则')].sort_values('j')
                assert len(rp)==len(reps)
                if reps:assert np.allclose([p['v'] for p in reps],rp.v,atol=1e-14,rtol=0)
                repro.append(dict(文件名=name,阶段1平台数=len(audit),保留数=len(reps),判据和拒绝原因严格复现=True))
        print('读取及A60复现:',name,flush=True)
    for filename,data in [('原始极化来源核对.csv',source),('原A60实现一致性.csv',repro),('全部诊断筛选明细.csv',checks),
                          ('合格代表点及条件.csv',rep_table),('保留原始行成员.csv',retained),('长平台时序绘图数据.csv',pd.concat(frames,ignore_index=True)),
                          ('全部长平台概况.csv',platforms),('长平台分窗口统计.csv',windows),('全部固定滞后相关.csv',lags),('观测分辨率与采样检查.csv',resolution)]:
        table(out/'表格'/filename,data)
    audit=pd.DataFrame(checks);pw=pd.DataFrame(windows);cohort=pd.read_csv(original/'表格/RealHyFC长平台拒绝独立复核.csv')
    cohort=cohort.loc[cohort.温度超冻结1C阈值]
    assert len(cohort)==69
    table(out/'表格/历史69平台固定队列.csv',cohort)
    rules=['DWELL_LT_180S_OR_TIME_MISSING','TAIL_INCOMPLETE','MISSING','CURRENT_UNSTABLE','VOLTAGE_SPREAD','VOLTAGE_DRIFT','CONDITION_UNSTABLE_temp','CONDITION_UNSTABLE_pa','CONDITION_UNSTABLE_pc']
    rule_rows=[]
    for case,group in audit.groupby('方案'):
        reasons=group.拒绝原因.fillna('');masks={r:reasons.str.contains(r,regex=False) for r in rules}
        for rule,m in masks.items():
            others=pd.concat([b for key,b in masks.items() if key!=rule],axis=1).any(axis=1)
            only=m&~others
            for scope,eligible in [('全部候选段',np.ones(len(group),bool)),('至少180秒且j在0.2至1.1',((group.平台秒>=180)&group.实测j.between(.2,1.1)).to_numpy())]:
                selected=m.to_numpy()&eligible;sole=only.to_numpy()&eligible
                rule_rows.append(dict(方案=case,范围=scope,规则=rule,边际拒绝平台数=int(selected.sum()),仅该规则拒绝平台数=int(sole.sum()),
                  边际受影响原始平台行数=int(group.loc[selected,'平台原始点数'].sum()),边际受影响末窗口行数=int(group.loc[selected,'代表窗口点数'].sum()),
                  边际影响文件数=group.loc[selected,'文件名'].nunique(),仅单项影响文件数=group.loc[sole,'文件名'].nunique(),
                  注意='多标签边际不可相加；不同方案重叠行不可视为独立样本'))
    table(out/'表格/逐规则独立排除贡献.csv',rule_rows)
    # 数值支持及诊断性损失；所有原始时间资格逐条沿用，不把标称小时当真实时钟。
    g=grid(oldcfg);bol='Polcurve_Nominal_RHFC_BoL.csv';features=[];gridrows=[];supportrows=[];perturb=[]
    for name in identity.文件名:
        t=meta.loc[meta.文件名==name].iloc[0]
        for case in cfg['诊断组']:
            label=case['名称'];pr=profiles[(name,label)];bp=profiles[(bol,label)]
            v,su=interpolate(pr,g,.20);b,bs=interpolate(bp,g,.20);loss=b-v;mask=np.isfinite(loss);x=g[mask]
            for kind,s in [('本次',su),('BoL',bs)]:supportrows.extend(dict(文件名=name,方案=label,来源=kind,**r) for r in s)
            ref=interpolate(pr,np.array([.9]),.2)[0][0];bref=interpolate(bp,np.array([.9]),.2)[0][0]
            r=dict(文件名=name,方案=label,身份='DIAGNOSTIC_ONLY',标称小时=t.标称老化小时,启停后=t.启停后,
              实际测试开始=t.实际测试开始,实际测试结束=t.实际测试结束,时钟资格=t.时钟资格,累计小时下界=t.累计小时下界,累计小时上界=t.累计小时上界,
              前300h资格=t.前300h物理结束资格,前600h资格=t.前600h物理结束资格,发布时间='UNKNOWN',原始路径=t.原始路径,
              合格平台数=len(pr),BoL合格平台数=len(bp),共同网格数=int(mask.sum()),最小j=float(x.min()) if len(x) else np.nan,
              最大j=float(x.max()) if len(x) else np.nan,完整0_3至1覆盖=bool(mask.all()),参考0_9支持=bool(np.isfinite(ref) and np.isfinite(bref)),
              缺失网格=';'.join(map(str,g[~mask])),足够数学支持=bool(len(x)>=10 and np.ptp(x)>=.5-1e-12),l_1=np.nan,l_2=np.nan)
            fit=None
            if r['足够数学支持']:
                fit=solve(x,loss[mask],.9);r.update({k:v for k,v in fit.items() if k not in ['pred','residual','weights']})
                conf=copy.deepcopy(oldcfg);conf['稳定性']['条件扰动次数']=100
                seed=20261025
                db=block_draws(bp,g,conf,np.random.default_rng(seed))
                dv=db if name==bol else block_draws(pr,g,conf,np.random.default_rng(seed+int(t.标称老化小时)+int(t.启停后)+1))
                X=np.column_stack([np.ones(len(x)),x/.9]);w=fit['weights']
                proj=np.linalg.inv(X.T@(w[:,None]*X))@(X.T*w);draws=(db[:,mask]-dv[:,mask])@proj.T
                bounds=np.quantile(draws,[.025,.975],axis=0)
                r.update(l1条件下界=bounds[0,0],l1条件上界=bounds[1,0],l2条件下界=bounds[0,1],l2条件上界=bounds[1,1])
                perturb.extend(dict(文件名=name,方案=label,扰动编号=i,l1=float(a),l2=float(b)) for i,(a,b) in enumerate(draws))
                for field,cap in [('temp',2.),('pa',5.),('pc',5.)]:
                    a=interpolate(pr,g,.2,[p[field] for p in pr])[0];c=interpolate(bp,g,.2,[p[field] for p in bp])[0]
                    r[field+'相对BoL均差']=float(np.nanmean((a-c)[mask]));r[field+'相对BoL最大差']=float(np.nanmax(abs((a-c)[mask])))
            k=0
            for i,j in enumerate(g):
                rec=dict(文件名=name,方案=label,j=j,BoL_V=b[i],测量_V=v[i],损失_V=loss[i],支持=bool(mask[i]),二维重构=np.nan,一维重构=np.nan,残差=np.nan,权重=np.nan)
                if fit is not None and mask[i]:
                    rec.update(二维重构=fit['pred'][k],一维重构=fit['一维c'],残差=fit['residual'][k],权重=fit['weights'][k]);k+=1
                gridrows.append(rec)
            features.append(r)
    feature=pd.DataFrame(features);points=pd.DataFrame(gridrows)
    table(out/'表格/诊断二维参数及支持资格.csv',feature);table(out/'表格/诊断网格损失及重构.csv',points)
    table(out/'表格/局部插值左右支持.csv',supportrows);table(out/'表格/条件分块扰动.csv',perturb)
    common=[]
    for case in cfg['诊断组']:
        name=case['名称'];main=feature.loc[(feature.方案==name)&~feature.启停后]
        masks=[points.loc[(points.文件名==f)&(points.方案==name),'支持'].to_numpy() for f in main.文件名]
        mask=np.logical_and.reduce(masks)
        common.append(dict(方案=name,主曲线含BoL=len(main),共同网格数=int(mask.sum()),
          最小j=float(g[mask].min()) if mask.any() else np.nan,最大j=float(g[mask].max()) if mask.any() else np.nan,
          全部主曲线同支持=';'.join(map(str,g[mask])),说明='不因各曲线支持不同而直接比较原拟合参数'))
    table(out/'表格/全部主曲线共同支持.csv',common)
    sensitivity=[]
    for name in identity.文件名:
        base=points.loc[(points.文件名==name)&(points.方案=='B60')]
        for case in cfg['诊断组']:
            label=case['名称']
            if label=='B60':continue
            other=points.loc[(points.文件名==name)&(points.方案==label)]
            mask=base.支持.to_numpy()&other.支持.to_numpy();x=g[mask]
            if len(x)<10 or np.ptp(x)<.5-1e-12:continue
            a=solve(x,base.损失_V.to_numpy()[mask],.9);b=solve(x,other.损失_V.to_numpy()[mask],.9)
            d=max(abs(a['l_1']-b['l_1']),abs(a['l_2']-b['l_2']));scale=max(abs(a['l_1']),abs(a['l_2']),.01)
            sensitivity.append(dict(文件名=name,对照=label,参照='B60',同网格数=len(x),最小j=x.min(),最大j=x.max(),
              参照l1=a['l_1'],参照l2=a['l_2'],对照l1=b['l_1'],对照l2=b['l_2'],最大参数差_V=d,相对差=d/scale,
              诊断稳定性标记=bool(d<=.005 and d/scale<=.25),身份='DIAGNOSTIC_ONLY'))
    table(out/'表格/同支持网格参数敏感性.csv',sensitivity)
    return feature,points,pd.DataFrame(sensitivity),pd.DataFrame(platforms),pd.DataFrame(windows),audit
