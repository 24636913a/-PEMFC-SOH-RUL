"""阶段1纯CPU特征算法；不进行时间预测或模型训练。"""
import numpy as np
import pandas as pd
from 核验基础 import col, mean_cells, scan_directions, CELLS_FC1, CELLS_REAL


def grid(cfg):
    c = cfg['共同网格']
    return np.round(np.arange(c['起点'], c['终点']+c['步长']/2, c['步长']), 8)


def variants(stack):
    return ['主规则', 'FC1窄分组', 'FC1宽分组'] if stack == 'FC1' else ['主规则', '末30秒', '末90秒', '严格稳态']


def prepare(frame, stack, cfg):
    voltage, _ = mean_cells(frame, CELLS_FC1 if stack == 'FC1' else CELLS_REAL)
    jname = next(n for n in frame if n.startswith('J (A/cm')) if stack == 'FC1' else 'j.S.act [A/cm2]'
    j = col(frame, jname)
    direction, segments = scan_directions(j, cfg['扫描']['确认幅度'])
    lo, hi = grid(cfg)[[0,-1]]
    chosen = None
    for sid in np.unique(segments):
        ix = np.flatnonzero(segments == sid); good = ix[np.isfinite(j[ix]) & np.isfinite(voltage[ix])]
        if len(good) and direction[ix][-1] == -1 and j[good].min() <= lo and j[good].max() >= hi:
            chosen = int(sid); break
    result = pd.DataFrame({'row':np.arange(len(frame)), 'j':j, 'v':voltage, 'segment':segments})
    for name, raw in [('set','I.S.set [A]'),('temp','T.Si.CL [°C]'),('pa','p.Si.A [kPag]'),
                      ('pc','p.Si.C [kPag]'),('hydrogen','F.Si.H.A [nl/min]'),('air','F.Si.Air.C [nl/min]')]:
        result[name] = col(frame, raw)
    result['datetime'] = pd.to_datetime(frame['DateTime'], format='mixed', errors='coerce').to_numpy() if 'DateTime' in frame else pd.NaT
    return result, chosen


def representatives(raw, chosen, stack, variant, cfg):
    """返回合格代表点、每组审计、原始行成员；无选择最佳拟合的逻辑。"""
    if chosen is None:
        return [], [dict(拒绝原因='NO_COMPLETE_DESCENDING_SCAN',保留=False)], []
    part = raw.loc[raw.segment == chosen].copy()
    audit=[]; reps=[]; members=[]
    if stack == 'FC1':
        width = {'主规则': cfg['FC1']['电流密度分组宽度'], 'FC1窄分组':cfg['稳定性']['FC1宽度对照'][0],
                 'FC1宽分组':cfg['稳定性']['FC1宽度对照'][1]}[variant]
        part = part.loc[part.j.between(0.2,1.1) & np.isfinite(part.v)]
        groupids = np.floor(part.j.to_numpy()/width+0.5).astype(int)
        groups = part.groupby(groupids, sort=True)
    else:
        # 缺失设定值或时间间断也断开平台，绝不拼接跨断点/多次扫描。
        delta = part.datetime.diff().dt.total_seconds()
        breaks = (part['set'].diff().abs() > cfg['RealHyFC']['设定电流变化阈值_A']) | part['set'].isna() | (delta > cfg['RealHyFC']['最大相邻间隔秒']) | (delta <= 0)
        groups = part.groupby(breaks.cumsum(),sort=True)
    for gid, whole in groups:
        reasons=[]; group=whole; duration=np.nan
        if stack=='FC1':
            if len(group) < cfg['FC1']['最少组内点数']: reasons.append('TOO_FEW_BIN_POINTS')
        else:
            c=cfg['RealHyFC']; duration=(whole.datetime.iloc[-1]-whole.datetime.iloc[0]).total_seconds()
            if not np.isfinite(duration) or duration < c['最短平台秒']: reasons.append('DWELL_LT_180S_OR_TIME_MISSING')
            tail={'末30秒':30, '末90秒':90}.get(variant,c['末尾窗口秒'])
            group=whole.loc[whole.datetime > whole.datetime.iloc[-1]-pd.Timedelta(seconds=tail)]
            if len(group)<max(10,int(tail*.9)):reasons.append('TAIL_INCOMPLETE')
            if len(group):
                for field in ['j','v','temp','pa','pc']:
                    if not np.isfinite(group[field]).all():reasons.append('MISSING_'+field)
                if group.j.max()-group.j.min()>c['电流密度峰峰上限']:reasons.append('CURRENT_UNSTABLE')
                vmax=cfg['稳定性']['严格电压波动上限_V'] if variant=='严格稳态' else c['电压P95减P5上限_V']
                if np.nanquantile(group.v,.95)-np.nanquantile(group.v,.05)>vmax:reasons.append('VOLTAGE_SPREAD')
                seconds=(group.datetime-group.datetime.iloc[0]).dt.total_seconds().to_numpy()
                slope=np.polyfit(seconds,group.v,1)[0] if len(group)>2 and np.ptp(seconds)>0 else np.nan
                if not np.isfinite(slope) or abs(slope)>c['电压斜率绝对值上限_V_s']:reasons.append('VOLTAGE_DRIFT')
                for field,cap in [('temp',c['冷却液温度峰峰上限_C']),('pa',c['入口压力峰峰上限_kPa']),('pc',c['入口压力峰峰上限_kPa'])]:
                    if group[field].max()-group[field].min()>cap:reasons.append('CONDITION_UNSTABLE_'+field)
        j=float(group.j.median()) if len(group) else np.nan
        stat=dict(组号=int(gid),扫描分段=chosen,平台原始点数=len(whole),代表窗口点数=len(group),平台秒=duration,
                  实测j=j,原始首行=int(whole.row.iloc[0]),原始末行=int(whole.row.iloc[-1]),保留=not reasons,
                  拒绝原因=';'.join(reasons),稳态资格='NO_CLOCK_SCAN_DESCRIPTOR' if stack=='FC1' else 'OBSERVED_STABILITY_SCREEN')
        audit.append(stat)
        if reasons or not np.isfinite(j):continue
        aggregate=np.median if stack=='FC1' else np.mean
        blocks=np.array_split(group.v.to_numpy(),5)
        point=dict(j=j,v=float(aggregate(group.v)),block=np.array([aggregate(b) for b in blocks if len(b)]),
                   row_start=int(group.row.iloc[0]),row_end=int(group.row.iloc[-1]),group=int(gid),
                   **{f:float(group[f].mean()) if group[f].notna().any() else np.nan for f in ['temp','pa','pc','hydrogen','air']})
        reps.append(point)
        members.extend(dict(原始行号=int(r.row),组号=int(gid),j=float(r.j),平均单片电压_V=float(r.v)) for r in group.itertuples())
    # 重复电流代表点只在同文件同方向内合并，避免重复平台支配权重。
    merged=[]
    for j in sorted(set(p['j'] for p in reps)):
        same=[p for p in reps if p['j']==j]; p=dict(same[0])
        if len(same)>1:
            p['v']=float(np.median([q['v'] for q in same]));p['block']=np.concatenate([q['block'] for q in same])
        merged.append(p)
    return merged,audit,members


def interpolate(reps, targets, max_gap, values=None):
    """不外推，不跨超过max_gap的缺测；返回真实左右支持身份。"""
    y=np.full(len(targets),np.nan); support=[]
    x=np.array([p['j'] for p in reps]);v=np.array([p['v'] for p in reps]) if values is None else np.asarray(values)
    for k,q in enumerate(targets):
        pos=np.searchsorted(x,q); left=pos-1; right=pos
        if pos<len(x) and abs(x[pos]-q)<1e-12:left=right=pos
        ok=left>=0 and right<len(x)
        if ok:ok=(x[right]-x[left]<=max_gap+1e-12)
        if ok:
            y[k]=v[left] if left==right else v[left]+(q-x[left])/(x[right]-x[left])*(v[right]-v[left])
        support.append(dict(j=float(q),支持=bool(ok),左j=float(x[left]) if left>=0 and left<len(x) else np.nan,
                            右j=float(x[right]) if right<len(x) else np.nan,
                            左原始行=reps[left]['row_start'] if left>=0 and left<len(x) else None,
                            右原始行=reps[right]['row_end'] if right<len(x) else None,
                            拒绝原因='' if ok else 'NO_BRACKET_OR_GAP_GT_LIMIT'))
    return y,support


def quadrature(x):
    d=np.diff(x)
    w=np.r_[d[0]/2,(d[:-1]+d[1:])/2,d[-1]/2]
    return w/w.sum()


def solve(x,y,jref):
    w=quadrature(x);X=np.column_stack([np.ones(len(x)),x/jref]); sw=np.sqrt(w)
    beta=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
    pred=X@beta;c=float(w@y); err=y-pred
    gram=X.T@(w[:,None]*X);inv=np.linalg.inv(gram)
    corr=float(inv[0,1]/np.sqrt(inv[0,0]*inv[1,1]))
    rmse=float(np.sqrt(w@(err**2)));r1=float(np.sqrt(w@((y-c)**2)))
    variance=float(w@((y-c)**2))
    cv1=np.zeros(len(x));cv2=np.zeros(len(x))
    for ids in np.array_split(np.arange(len(x)),5):
        keep=np.ones(len(x),bool);keep[ids]=False
        cw=quadrature(x[keep]);b=np.linalg.lstsq(X[keep]*np.sqrt(cw)[:,None],y[keep]*np.sqrt(cw),rcond=None)[0]
        cv2[ids]=X[ids]@b;cv1[ids]=cw@y[keep]
    return dict(l_1=float(beta[0]),l_2=float(beta[1]),一维c=c,二维RMSE_V=rmse,一维RMSE_V=r1,
                二维R2=1-rmse**2/variance if variance>1e-24 else np.nan,
                负载留组二维RMSE_V=float(np.sqrt(w@((y-cv2)**2))),负载留组一维RMSE_V=float(np.sqrt(w@((y-cv1)**2))),
                条件数=float(np.linalg.cond(X*sw[:,None])),参数相关系数=corr,
                最大残差_V=float(abs(err).max()),网格数=len(x),最小拟合j=float(x.min()),最大拟合j=float(x.max()),
                pred=pred,residual=err,weights=w)


def block_draws(reps, targets, cfg, rng):
    draws=[]
    for _ in range(cfg['稳定性']['条件扰动次数']):
        v=[]
        for p in reps:
            b=p['block']; sample=rng.choice(b,size=len(b),replace=True)
            v.append(float(sample.mean()))
        draws.append(interpolate(reps,targets,cfg['插值']['最大括距'],v)[0])
    return np.asarray(draws)


def unit_tests():
    p=[dict(j=.3,v=.7,row_start=0,row_end=1),dict(j=.5,v=.6,row_start=2,row_end=3)]
    v,_=interpolate(p,np.array([.2,.3,.4,.5,.6]),.2)
    assert np.isnan(v[[0,4]]).all() and np.allclose(v[1:4],[.7,.65,.6])
    assert np.isnan(interpolate(p,np.array([.4]),.1)[0][0])
    x=np.linspace(.3,1,15);f=solve(x,-.03+.02*x/.9,.9)
    assert np.allclose([f['l_1'],f['l_2']],[-.03,.02],atol=1e-12)
    assert f['二维RMSE_V']<1e-12 and f['负载留组二维RMSE_V']<1e-12
    b=solve(x,np.zeros(15),.9);assert b['l_1']==0 and b['l_2']==0
    return ['禁止外推','插值最大括距','保留负损失','二维解析解','BoL恒等零','负载留组重构']
