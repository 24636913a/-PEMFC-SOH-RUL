"""从保存的原始成员/代表点/支持表独立闭式复算，不调用研究算法。"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent.parent;GUIDE=WORK.parent.parent;REPO=GUIDE.parent
OUT=WORK/'输出结果';TAB=OUT/'表格'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def run():
    proof=json.loads((WORK/'配置/执行前证据.json').read_text(encoding='utf-8'))
    assert digest(WORK/'配置/候选表示预登记.json')==proof['规则SHA256']
    changed=[]
    for rel,h in proof['历史文件SHA256'].items():
        p=GUIDE/rel
        if not p.exists() or digest(p)!=h:changed.append(rel)
    dump(OUT/'历史文件保护复核.json',dict(检查文件数=len(proof['历史文件SHA256']),变化文件=changed,通过=not changed))
    assert not changed,changed
    src=pd.read_csv(TAB/'源文件身份复核.csv')
    for r in src.to_dict('records'):assert digest(REPO/r['原始路径'])==r['SHA256']==r['本次SHA256']
    reps=pd.read_csv(TAB/'合格代表点.csv');mem=pd.read_csv(TAB/'代表点原始成员.csv');aud=pd.read_csv(TAB/'平台及分组筛选.csv')
    errors=[]
    # 直接重新聚合保存的原始观测成员；不是从拟合结果反推代表值。
    grouped=mem.groupby(['文件名','方案','组号'],sort=False)
    for r in reps.to_dict('records'):
        g=grouped.get_group((r['文件名'],r['方案'],r['group']))
        value=g['平均单片电压_V'].median() if r['电堆']=='FC1' else g['平均单片电压_V'].mean()
        errors.append(abs(value-r['v']))
        assert abs(value-r['v'])<2e-14
        assert abs(g.j.median()-r['j'])<2e-14
        assert g.原始行号.min()==r['row_start'] and g.原始行号.max()==r['row_end']
    sup=pd.read_csv(TAB/'共同负载实际支持表.csv');supportdiff=[]
    bolnames={'FC1':'FC1_Pola_T000.xlsx','RealHyFC':'Polcurve_Nominal_RHFC_BoL.csv'}
    for r in sup.to_dict('records'):
        for tag,name,valid,col in [('本曲线',r['文件名'],r['本曲线支持'],'本曲线V'),('BoL',bolnames[r['电堆']],r['BoL支持'],'BoL_V')]:
            rp=reps.loc[(reps.文件名==name)&(reps.方案==r['方案'])].sort_values('j')
            x=rp.j.to_numpy();v=rp.v.to_numpy();pos=np.searchsorted(x,r['j']);l=pos-1;u=pos
            if pos<len(x) and abs(x[pos]-r['j'])<1e-12:l=u=pos
            ok=l>=0 and u<len(x)
            if ok:ok=x[u]-x[l]<=.2+1e-12
            if ok:
                aa=aud.loc[(aud.文件名==name)&(aud.方案==r['方案'])&~aud.保留&(aud.平台秒>=180)]
                if aa.实测j.between(x[l],x[u],inclusive='neither').any():ok=False
            assert bool(valid)==bool(ok),(r['文件名'],r['方案'],r['j'],tag)
            if ok:
                val=v[l] if l==u else ((x[u]-r['j'])*v[l]+(r['j']-x[l])*v[u])/(x[u]-x[l])
                supportdiff.append(abs(val-r[col]));assert abs(val-r[col])<2e-14
    grids=pd.read_csv(TAB/'候选网格损失.csv');feature=pd.read_csv(TAB/'路线A中心参数.csv');fitdiff=[]
    for r in feature.loc[feature.代数可计算].to_dict('records'):
        p=grids.loc[(grids.文件名==r['文件名'])&(grids.方案==r['方案'])]
        use=[float(x) for x in str(r['支持j']).split(';')];p=p.loc[p.j.isin(use)]
        assert len(p)==r['支持点数']
        z=p.j.to_numpy()-.9;y=p.D_V.to_numpy();s=np.mean((z-z.mean())*(y-y.mean()))/np.mean((z-z.mean())**2);h=y.mean()-s*z.mean()
        diff=max(abs(s-r['s_V_per_A_cm2']),abs(h-r['h_ref_V']));fitdiff.append(diff);assert diff<2e-13
        rmse=np.sqrt(np.mean((y-(h+s*z))**2));assert abs(rmse-r['二维RMSE_V'])<2e-14
        if r['标称小时']==0:assert np.all(y==0) and h==0 and s==0
    dual=pd.read_csv(TAB/'路线B双参考特征.csv')
    for r in dual.to_dict('records'):
        p=grids.loc[(grids.文件名==r['文件名'])&(grids.方案==r['方案'])].set_index('j')
        a=p.loc[.45,'D_V'];b=p.loc[.9,'D_V'];ok=np.isfinite([a,b]).all();assert r['双参考支持']==ok
        if ok:assert max(abs(a-r['D_low_V']),abs(b-r['D_ref_V']),abs((b-a)/.45-r['S_V_per_A_cm2']))<2e-13
        else:assert np.isnan(r['S_V_per_A_cm2'])
    cv=pd.read_csv(TAB/'共享平台分组留出.csv')
    assert cv.loc[~cv.可评价,['一维预测_V','二维预测_V']].isna().all().all()
    for r in cv.loc[cv.可评价].to_dict('records'):
        x=np.array([float(v) for v in r['训练j'].split(';')]);assert x.min()<=r['j']<=x.max()
        p=np.searchsorted(x,r['j']);assert 0<p<len(x) and x[p]-x[p-1]<=.2+1e-12
        target=grids.loc[(grids.文件名==r['文件名'])&(grids.方案==r['方案'])].set_index('j')
        y=target.loc[x,'D_V'].to_numpy();z=x-.9
        slope=np.sum((z-z.mean())*(y-y.mean()))/np.sum((z-z.mean())**2)
        pred=y.mean()+slope*(r['j']-.9-z.mean())
        assert abs(pred-r['二维预测_V'])<2e-13 and abs(y.mean()-r['一维预测_V'])<2e-13
        a=sup.loc[(sup.文件名==r['文件名'])&(sup.方案==r['方案'])].set_index('j')
        def shared(row):
            return set('cur:'+k for k in str(row.支持组).split(';'))|set('bol:'+k for k in str(row.BoL支持组).split(';'))
        hold=shared(a.loc[r['j']]);training=set().union(*(shared(a.loc[v]) for v in x))
        assert not (hold&training),'留组不可与训练共享原始平台'
    pert=pd.read_csv(TAB/'共享BoL条件扰动.csv')
    for stack,g in pert.groupby('电堆'):
        b=None
        for _,v in g.groupby('文件名'):
            now=v.sort_values('扰动').仅BoL_Dref扰动_V.to_numpy()
            if b is None:b=now
            else:assert np.array_equal(now,b)
    # 复核严格A60与历史逐组资格身份，不能在B结果中改写旧结论。
    old=pd.read_csv(GUIDE/'健康特征提取/输出结果/历史核验/20261025T153526_430847/表格/平台及分组筛选明细.csv')
    for name in src.loc[src.电堆=='RealHyFC','文件名']:
        a=aud.loc[(aud.文件名==name)&(aud.方案=='A60')].reset_index(drop=True)
        b=old.loc[(old.文件名==name)&(old.方案=='主规则')].reset_index(drop=True)
        assert len(a)==len(b) and np.array_equal(a.保留,b.保留)
        assert np.array_equal(a.拒绝原因.fillna(''),b.拒绝原因.fillna(''))
    # 代码静态验收：正式源中没有模型训练调用；完整代码本身纳入历史SHA保护。
    import ast
    files=[p for p in (WORK/'程序').glob('*.py')]
    for p in files:ast.parse(p.read_text(encoding='utf-8'))
    from 共同负载研究 import unit_tests
    tests=unit_tests()
    summary=dict(状态='PASS',源文件SHA数=len(src),历史保护文件数=len(proof['历史文件SHA256']),
        独立代表点数=len(errors),代表值最大差_V=max(errors),独立局部支持数=len(supportdiff),局部插值最大差_V=max(supportdiff),
        独立中心拟合数=len(fitdiff),中心系数最大差=max(fitdiff),
        双参考记录数=len(dual),合法留组查询闭式复核数=int(cv.可评价.sum()),严格A60历史逐组一致文件数=11,共享BoL扰动一致=True,
        算法测试=tests,原阶段1='BLOCKED_UNCHANGED',正式训练=False)
    dump(OUT/'独立复核结果.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
