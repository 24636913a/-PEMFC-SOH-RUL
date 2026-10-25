"""只读复核平台拒绝原因与合成边界测试；不更改提取政策或结果。"""
from pathlib import Path
import sys,json,datetime,argparse
import numpy as np
import pandas as pd
import yaml

guide=next(p for p in Path(__file__).resolve().parents if (p/'数据处理/数据核验报告.md').exists())
if not (Path(__file__).parent/'核验基础.py').exists():sys.path.insert(0,str(guide/'数据处理/程序'))
from 极化特征计算 import representatives,unit_tests


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output
    cfg=yaml.safe_load((guide/'健康特征提取/配置/健康特征冻结规则.yaml').read_text(encoding='utf-8-sig'))
    passed=unit_tests()
    data=pd.DataFrame(dict(row=np.arange(181),j=np.full(181,.9),v=np.full(181,.7),segment=0,
          set=np.full(181,252.),temp=np.full(181,70.),pa=np.full(181,165.),pc=np.full(181,157.),
          hydrogen=16.,air=44.,datetime=pd.date_range('2025-01-01',periods=181,freq='s')))
    assert len(representatives(data,0,'RealHyFC','主规则',cfg)[0])==1
    assert not representatives(data.iloc[:-1],0,'RealHyFC','主规则',cfg)[0]
    hot=data.copy();hot['temp']=70+2*(np.arange(181)%2)
    assert not representatives(hot,0,'RealHyFC','主规则',cfg)[0]
    missing=data.copy();missing.loc[160,'v']=np.nan
    assert not representatives(missing,0,'RealHyFC','主规则',cfg)[0]
    passed+=['180秒平台边界','179秒拒绝','温度波动拒绝','缺失电压拒绝']
    audit=pd.read_csv(out/'表格/平台及分组筛选明细.csv')
    raw=pd.read_csv(guide/'数据处理/输出结果/表格/极化原始测量点.csv',low_memory=False)
    raw=raw.loc[raw.电堆=='RealHyFC'].copy();raw['clock']=pd.to_datetime(raw.实际测量时间,format='mixed')
    rows=[]
    for name,g in audit.loc[(audit.电堆=='RealHyFC')&(audit.方案=='主规则')&(audit.平台秒>=180)&audit.实测j.between(.2,1.1)].groupby('文件名'):
        r=raw.loc[raw.文件名==name]
        for row in g.itertuples():
            whole=r.loc[r.原始行号.between(row.原始首行,row.原始末行)]
            tail=whole.loc[whole.clock>whole.clock.iloc[-1]-pd.Timedelta(seconds=60)]
            temp=float(np.ptp(tail.冷却液入口_C));v=tail.平均单片电压_V.to_numpy()
            rows.append(dict(文件名=name,组号=int(row.组号),实测j=row.实测j,平台秒=row.平台秒,
                末60秒温度峰峰_C=temp,末60秒电压P95减P5_V=float(np.quantile(v,.95)-np.quantile(v,.05)),
                温度超冻结1C阈值=bool(temp>cfg['RealHyFC']['冷却液温度峰峰上限_C']),保留=row.保留,拒绝原因=row.拒绝原因))
            if temp>cfg['RealHyFC']['冷却液温度峰峰上限_C']:assert 'CONDITION_UNSTABLE_temp' in str(row.拒绝原因)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'表格/RealHyFC长平台拒绝独立复核.csv',index=False,encoding='utf-8-sig')
    record=dict(status='PASS',tests=passed,raw_long_plateaus_checked=len(rows),
                temperature_rejections=int(summary.温度超冻结1C阈值.sum()),
                temperature_peak_to_peak_min=float(summary.末60秒温度峰峰_C.min()),
                temperature_peak_to_peak_max=float(summary.末60秒温度峰峰_C.max()),
                qualification_scope='1C是本阶段预注册的保守筛选值，不是已证实的仪器精度或跨堆等价阈值；拒绝只说明本协议未取得资格',
                time=datetime.datetime.now().astimezone().isoformat(),training=False)
    (out/'平台可靠性复核.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
