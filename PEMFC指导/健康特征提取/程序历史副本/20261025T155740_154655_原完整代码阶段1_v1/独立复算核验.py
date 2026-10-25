"""不调用提取/拟合函数，以保存CSV独立闭式复算；仅CPU，不覆盖原结果。"""
from pathlib import Path
import argparse,json,hashlib,datetime
import numpy as np
import pandas as pd


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def closed(x,y,w):
    z=x/.9; mz=np.sum(w*z);my=np.sum(w*y)
    slope=np.sum(w*(z-mz)*(y-my))/np.sum(w*(z-mz)**2)
    intercept=my-slope*mz
    return np.array([intercept,slope])


def integral(x):
    w=np.zeros(len(x));d=x[1:]-x[:-1];w[:-1]+=d/2;w[1:]+=d/2
    return w/w.sum()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    guide=next(p for p in Path(__file__).resolve().parents if (p/'数据处理/数据核验报告.md').exists())
    out=args.output or guide/'健康特征提取/输出结果'
    f=pd.read_csv(out/'表格/全部方案二维特征.csv');pts=pd.read_csv(out/'表格/共同网格损失及一二维重构.csv')
    reps=pd.read_csv(out/'表格/合格代表点及条件.csv');mem=pd.read_csv(out/'表格/代表点原始行成员.csv')
    raw=pd.read_csv(guide/'数据处理/输出结果/表格/极化原始测量点.csv',low_memory=False)
    rawindex=raw.set_index(['电堆','文件名','原始行号'])
    errors=[];n=0
    for key,m in mem.groupby(['电堆','文件名','方案','组号'],sort=False):
        stack,name,variant,group=key
        exact=rawindex.loc[[(stack,name,int(i)) for i in m.原始行号]]
        assert np.allclose(m.平均单片电压_V,exact.平均单片电压_V,atol=1e-14,rtol=0)
        assert np.allclose(m.j,exact.电流密度_A_cm2,atol=1e-14,rtol=0)
        rp=reps.loc[(reps.电堆==stack)&(reps.文件名==name)&(reps.方案==variant)&(reps.group==group)]
        assert len(rp)==1,'重复平台需专项核对，不静默跳过'
        expected=m.平均单片电压_V.median() if stack=='FC1' else m.平均单片电压_V.mean()
        assert abs(expected-rp.iloc[0].v)<1e-12
        n+=1
    for _,r in f.loc[f.计算状态=='COMPUTED'].iterrows():
        p=pts.loc[(pts.电堆==r.电堆)&(pts.文件名==r.文件名)&(pts.方案==r.方案)&pts.纳入拟合]
        x=p.j.to_numpy();y=p.损失_V.to_numpy();w=integral(x);beta=closed(x,y,w)
        assert np.allclose(w,p.权重,atol=1e-13,rtol=0)
        assert np.allclose(beta,[r.l_1,r.l_2],atol=1e-12,rtol=0)
        assert np.allclose(y,p.BoL电压_V-p.本次电压_V,atol=1e-13,rtol=0)
        reconstructed=beta[0]+beta[1]*x/.9
        rmse=np.sqrt(np.sum(w*(y-reconstructed)**2));one=np.sqrt(np.sum(w*(y-np.sum(w*y))**2))
        assert abs(rmse-r.二维RMSE_V)<1e-12 and abs(one-r.一维RMSE_V)<1e-12
        cv1=np.empty(len(x));cv2=np.empty(len(x))
        for test in np.array_split(np.arange(len(x)),5):
            train=np.setdiff1d(np.arange(len(x)),test);tw=integral(x[train]);b=closed(x[train],y[train],tw)
            cv2[test]=b[0]+b[1]*x[test]/.9;cv1[test]=np.sum(tw*y[train])
        assert abs(np.sqrt(w@((cv1-y)**2))-r.负载留组一维RMSE_V)<1e-12
        assert abs(np.sqrt(w@((cv2-y)**2))-r.负载留组二维RMSE_V)<1e-12
        for which in ['本次','BoL']:
            name=r.文件名 if which=='本次' else r.基准文件
            v=reps.loc[(reps.电堆==r.电堆)&(reps.文件名==name)&(reps.方案==r.方案)].sort_values('j')
            assert x.min()>=v.j.min()-1e-12 and x.max()<=v.j.max()+1e-12
            direct=np.interp(x,v.j,v.v)
            assert np.allclose(direct,p.本次电压_V if which=='本次' else p.BoL电压_V,atol=1e-12,rtol=0)
        if r.文件名==r.基准文件:
            assert np.all(y==0) and r.l_1==0 and r.l_2==0
        errors.append(dict(电堆=r.电堆,文件名=r.文件名,方案=r.方案,系数最大差=float(np.max(abs(beta-[r.l_1,r.l_2]))),RMSE差=float(abs(rmse-r.二维RMSE_V))))
    source=pd.read_csv(out/'表格/极化源身份复核.csv')
    for _,r in source.iterrows():assert digest(guide.parent/r.原始路径)==r.SHA256
    frozen=json.loads((guide/'健康特征提取/配置/计算前冻结记录.json').read_text(encoding='utf-8'))
    # 正式代码同步后允许完整代码变更，但阶段0数据处理目录必须保持逐文件相同。
    preserved=[r for r in frozen['protected_files'] if r['path'].startswith('数据处理/')]
    for r in preserved:assert digest(guide/r['path'])==r['sha256']
    residual_rows=[]
    mask=(pts.电堆=='FC1')&(pts.方案=='主规则')&(pts.文件名!='FC1_Pola_T000.xlsx')&pts.纳入拟合
    for name,group in pts.loc[mask].groupby('文件名'):
        for label,lo,hi in [('低负载',.3,.45),('中负载',.5,.75),('高负载',.8,.95)]:
            s=group.loc[group.j.between(lo-1e-10,hi+1e-10)];res=s.损失_V-s.二维重构_V
            residual_rows.append(dict(文件名=name,负载区间=label,残差均值_mV=res.mean()*1000,最大绝对残差_mV=res.abs().max()*1000))
    residual=pd.DataFrame(residual_rows);respath=out/'表格/FC1分负载残差检查.csv'
    if respath.exists():
        old=pd.read_csv(respath);assert np.allclose(old.残差均值_mV,residual.残差均值_mV,rtol=0,atol=1e-12)
    else:residual.to_csv(respath,index=False,encoding='utf-8-sig')
    checks=dict(status='PASS',checked_at=datetime.datetime.now().astimezone().isoformat(),source_files=len(source),
                fitted_records=len(errors),raw_member_groups=n,protected_stage0_files=len(preserved),maximum_coefficient_difference=max(e['系数最大差'] for e in errors),
                tests=['原始行与阶段0点值一致','代表点均值/中位数','独立闭式系数','一二维重构RMSE','5组留负载重构','无外推与局部插值','BoL严格零','原始19源SHA','阶段0结果未变'],
                training=False,scope='同曲线描述性核验，不是未来预测评价')
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    (out/f'独立复算核验_{stamp}.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(errors).to_csv(out/f'表格/独立复算差异_{stamp}.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
