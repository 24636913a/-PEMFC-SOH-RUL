"""独立闭式复算、规则归因与频谱/同步补充；不改变六组诊断结果或历史文件。"""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent.parent;GUIDE=WORK.parent.parent;OUT=WORK/'输出结果'
sys.path[:0]=[str(GUIDE/'数据处理/程序')]
from 核验基础 import table,save_json,sha


def fit(x,y):
    w=np.zeros(len(x));d=np.diff(x);w[:-1]+=d/2;w[1:]+=d/2;w/=w.sum()
    z=x/.9;mz=w@z;my=w@y;b=(w@((z-mz)*(y-my)))/(w@((z-mz)**2));a=my-b*mz
    return a,b,w


def run():
    feature=pd.read_csv(OUT/'表格/诊断二维参数及支持资格.csv');points=pd.read_csv(OUT/'表格/诊断网格损失及重构.csv')
    raw=pd.read_csv(OUT/'表格/长平台时序绘图数据.csv',low_memory=False);co=pd.read_csv(OUT/'表格/历史69平台固定队列.csv')
    check=[]
    for r in feature.loc[feature.足够数学支持].itertuples():
        f=points.loc[(points.文件名==r.文件名)&(points.方案==r.方案)&points.支持]
        x=f.j.to_numpy();y=f.损失_V.to_numpy();a,b,w=fit(x,y)
        assert np.allclose([a,b],[r.l_1,r.l_2],atol=1e-12,rtol=0)
        assert np.allclose(y,f.BoL_V-f.测量_V,atol=1e-13,rtol=0)
        rmse=np.sqrt(w@((y-a-b*x/.9)**2));assert abs(rmse-r.二维RMSE_V)<1e-12
        assert len(x)>=10 and np.ptp(x)>=.5-1e-12
        check.append(dict(文件名=r.文件名,方案=r.方案,系数差=max(abs(a-r.l_1),abs(b-r.l_2)),RMSE差=abs(rmse-r.二维RMSE_V)))
    identity=pd.read_csv(OUT/'表格/原始极化来源核对.csv')
    for r in identity.itertuples():assert sha(GUIDE.parent/r.原始路径)==r.SHA256
    original=GUIDE/'健康特征提取/输出结果/历史核验/20261025T153526_430847'
    old=pd.read_csv(original/'表格/RealHyFC长平台拒绝独立复核.csv');assert int(old.温度超冻结1C阈值.sum())==69
    a=pd.read_csv(OUT/'表格/全部诊断筛选明细.csv');aa=a.loc[(a.方案=='A60')&(a.平台秒>=180)&a.实测j.between(.2,1.1)]
    bb=a.loc[(a.方案=='B60')&(a.平台秒>=180)&a.实测j.between(.2,1.1)]
    assert len(aa)==77 and int(aa.拒绝原因.fillna('').eq('CONDITION_UNSTABLE_temp').sum())==66
    assert int(aa.保留.sum())==8 and int(bb.保留.sum())==74
    table(OUT/'表格/独立二维复算差异.csv',check)
    windows=pd.read_csv(OUT/'表格/长平台分窗口统计.csv')
    remaining=bb.loc[~bb.保留].merge(windows.loc[(windows.窗口=='末60秒')&windows.主扫描],on=['文件名','组号'])
    table(OUT/'表格/取消温度后独立压力障碍.csv',remaining[['文件名','组号','实测j','pa_range','pa_p95_p5','pc_range','temp_range','v_p95_p5']])
    cohort=windows.merge(co[['文件名','组号']],on=['文件名','组号']);cohort=cohort.loc[cohort.主扫描]
    summary=[]
    for (name,win),g in cohort.groupby(['文件名','窗口']):
        summary.append(dict(文件名=name,窗口=win,平台数=len(g),温度峰峰中位=g.temp_range.median(),温度均值中位=g.temp_mean.median(),
            电压P95减P5中位_mV=g.v_p95_p5.median()*1000,温度电压相关中位=g.temp_v_r.median(),温度电压差分相关中位=g.temp_v_diff_r.median()))
    table(OUT/'表格/按日期和窗口的固定69平台汇总.csv',summary)
    # 全Nyquist区间复查最强频率，原计算中4–300秒频段结果原样保留，不将边界峰当真。
    spectra=[];align=[]
    for key,g in raw.groupby('平台键',sort=False):
        t=g.平台内秒.to_numpy();temp=g.temp.to_numpy();dt=np.median(np.diff(t));n=len(t)
        res=temp-np.polyval(np.polyfit(t,temp,1),t);pow=abs(np.fft.rfft(res))**2;freq=np.fft.rfftfreq(n,dt)
        peak=1+np.argmax(pow[1:]);ratio=pow[peak]/pow[1:].sum() if pow[1:].sum()>0 else np.nan
        spectra.append(dict(文件名=g.文件名.iloc[0],平台键=key,主扫描=bool(g.主扫描.iloc[0]),扫描分段=int(g.segment.iloc[0]),
          组号=int(key.split('_p')[-1]),最强全带频率_Hz=freq[peak],最强全带周期_s=1/freq[peak],全带谱峰占比=ratio,
          采样间隔中位_s=dt,频率上限_Hz=.5/dt,解释='全Nyquist探索性频谱；不可识别控制器或传感器硬件原因'))
        if bool(g.主扫描.iloc[0]):
            for lag in [-2,-1,0,1,2]:
                x=g.current.to_numpy();y=g.j.to_numpy()*280
                x,y=(x[:-lag],y[lag:]) if lag>0 else (x[-lag:],y[:lag]) if lag<0 else (x,y)
                align.append(dict(文件名=g.文件名.iloc[0],平台键=key,滞后秒=lag,定义='I(t)与280*j(t+lag)；不用于修正',RMS_A=float(np.sqrt(np.mean((x-y)**2)))))
    table(OUT/'表格/全Nyquist温度频谱复核.csv',spectra);table(OUT/'表格/固定电流密度同步敏感性.csv',align)
    lag=pd.read_csv(OUT/'表格/全部固定滞后相关.csv').merge(co[['文件名','组号']],on=['文件名','组号']);lag=lag.loc[lag.主扫描]
    lt=lag.groupby('滞后秒').相关.agg(中位='median',最小='min',最大='max').reset_index()
    table(OUT/'表格/69平台全部滞后汇总.csv',lt)
    sf=feature.loc[(feature.方案=='B60')&feature.足够数学支持&~feature.启停后&(feature.标称小时>0)]
    associations=[]
    for parameter in ['l_1','l_2']:
        for field in ['temp相对BoL均差','pa相对BoL均差','pc相对BoL均差']:
            associations.append(dict(参数=parameter,观测量=field,曲线数=len(sf),Pearson=float(sf[[parameter,field]].corr().iloc[0,1]),
                身份='同一条寿命轨迹的少数诊断，不独立；年龄/恢复/负载混杂，不能归因或做温度校正'))
    table(OUT/'表格/参数与温压差的描述关联.csv',associations)
    record=dict(status='PASS',time=datetime.datetime.now().astimezone().isoformat(),source_files=len(identity),fitted_cases=len(check),
                max_coefficient_difference=max(x['系数差'] for x in check),A60_long_77_pass=8,B60_long_77_pass=74,temp_only_reject=66,
                full_nyquist_platforms=len(spectra),training=False,independent_temporal_experiments=False)
    save_json(OUT/'独立诊断复算记录.json',record)
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':run()
