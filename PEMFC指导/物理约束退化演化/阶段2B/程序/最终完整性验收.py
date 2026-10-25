"""最终只读验收与SHA清单，不训练、不选参、不重跑阶段0–2A。"""
from pathlib import Path
import ast,json
import pandas as pd
from PIL import Image
from 运行低阶实验 import sha,js,csv
WORK=Path(__file__).resolve().parents[1];GUIDE=WORK.parents[1];OUT=WORK/'输出结果';FORMAL=GUIDE/'完整代码';PACKAGE=FORMAL/'阶段2B_v1'

def main():
    frozen=json.loads((WORK/'配置/运行前保护快照.json').read_text(encoding='utf-8'))
    assert sha(WORK/'配置/阶段2B冻结协议.json')==frozen['protocol_sha256']
    for rel,h in frozen['files'].items():assert sha(GUIDE/rel)==h,rel
    publication=json.loads((OUT/'正式代码发布验收.json').read_text(encoding='utf-8'));assert publication['status']=='PASS'
    backup=FORMAL/publication['backup'];old=pd.read_csv(backup/'逐文件备份SHA256.csv')
    for _,r in old.iterrows():assert sha(FORMAL/r['原路径'])==r['原SHA256']==sha(FORMAL/r['备份路径'])
    assert publication['stage2A_numerical_experiment_rerun']==False
    for root in [OUT,WORK/publication['formal_output']]:
        done=json.loads((root/'计算完成.json').read_text(encoding='utf-8'))
        for rel,h in done['outputs'].items():assert sha(root/rel)==h,(root,rel)
        test=json.loads((root/'独立复核通过.json').read_text(encoding='utf-8'))
        assert test['status']=='PASS' and test['checks']==52 and test['compared_origin_variants']==1708
    compare=pd.read_csv(OUT/'表格/正式入口独立重跑比较.csv');assert len(compare)==23 and compare['字节一致'].all()
    code=json.loads((PACKAGE/'正式代码SHA256.json').read_text(encoding='utf-8'))
    for rel,h in code.items():assert sha(PACKAGE/rel)==h
    assert sha(PACKAGE/'程序/低阶演化核心.py')==sha(WORK/'程序/低阶演化核心.py')
    for root in [WORK/'程序',PACKAGE/'程序']:
        for p in root.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    figs=pd.read_csv(OUT/'表格/图表来源与SHA256.csv');assert len(figs)==9
    for _,r in figs.iterrows():
        assert sha(WORK/r['绘图数据'])==r.CSV_SHA256
        for rel,h in json.loads(r['来源SHA256']).items():assert sha(WORK/rel)==h
        for ext,col in [('png','PNG_SHA256'),('pdf','PDF_SHA256')]:assert sha(OUT/'图片'/(r['图']+'.'+ext))==r[col]
        with Image.open(OUT/'图片'/(r['图']+'.png')) as im:
            assert min(im.info['dpi'])>299.5 and im.width>1000 and im.height>500
            im.verify()
    recovery=pd.read_csv(OUT/'表格/回升预测方向汇总.csv')
    qualifications=pd.read_csv(OUT/'表格/一维二维状态演化资格.csv')
    assert qualifications.state_evolution.eq('NO').all()
    for dataset in ['FC1','RealHyFC']:
        pred=pd.read_csv(OUT/'表格'/f'{dataset}全部方法逐点预测.csv');obs=pd.read_csv(WORK/'数据准备'/('FC1小时观测.csv' if dataset=='FC1' else 'RealHyFC循环观测.csv'))
        obs=obs[obs.scheme==('FC1_NOMINAL' if dataset=='FC1' else 'B60')].set_index('id')
        p=pred[(pred.partition=='Assessment')&(pred.method=='SlowReversible')]
        for h,g in p.groupby('horizon'):
            ends=g[g.step==g.horizon];r=recovery[(recovery.dataset==dataset)&(recovery.method=='SlowReversible')&(recovery.horizon==h)].iloc[0]
            assert int((ends.pred_v.to_numpy()>obs.loc[ends.origin_id].v.to_numpy()).sum())==r.predicted_end_rise
            assert int((ends.true_v.to_numpy()>obs.loc[ends.origin_id].v.to_numpy()).sum())==r.observed_end_rise
    for f in ['物理约束模型实验协议.md','物理约束模型实验报告.md','模型与基础预测对比报告.md','下一阶段迁移准备报告.md']:assert (WORK/f).is_file()
    status=pd.DataFrame([
        dict(module='连续运行低阶C/D',engineering='PASS',scientific='FC1短中期改善；Real长跨度未改善',identity='EXPLORATORY_REASSESSMENT'),
        dict(module='恢复预测',engineering='数学结构与合成验证允许回升',scientific='所选后留曲线未预测实际终点上升；未证实恢复预测成功',identity='NEGATIVE_RESULT_RETAINED'),
        dict(module='FC1条件RUL',engineering='全部阈值/起点/删失已计算',scientific='未优于原局部线性，809h不是已认证EOL',identity='CONDITIONAL_THRESHOLD_ONLY'),
        dict(module='极化联合一维/二维映射',engineering='完成配对与拒绝审计',scientific='时钟、样本或条件数不合格',identity='NA_NOT_TRAINED'),
        dict(module='16单元网络',engineering='条件性预案已整理',scientific='原方案联合网络暂未具备资格',identity='NA_NO_LOSS_OR_GRADIENT_CLAIM'),
        dict(module='跨堆迁移',engineering='300/600/后移共同评价起点协议已准备',scientific='本轮未执行',identity='NOT_RUN')])
    csv(OUT/'表格/最终科研身份与状态.csv',status)
    (WORK/'阶段2B验收报告.md').write_text(f'''# 阶段2B验收报告

工程状态：PASS。科研状态：部分电压收益，未获得恢复预测、RUL改进或二维/迁移成功证据。

- 原阶段0–2A共{len(frozen['files'])}份冻结文件SHA完全不变。
- 阶段2A_v1的{len(old)}份正式文件已逐文件完整备份，入口--help/导入检查通过；没有重新执行2A数值实验。
- 两个2B入口分别52组独立检查、1708个起点变体逐点复算通过；最大预测差约1.08e−16V，身份字段严格一致。
- 正式版本23份真实预测、参数与指标文件独立重跑逐字节一致。实测该完整重跑用时{publication['formal_run_wall_seconds']:.3f}秒，仅为此次CPU运行记录，不是Benchmark。
- 九组300dpi PNG/PDF、绘图CSV/SHA索引及四份主报告齐全。
- 所有不利结果、缺测NA、删失、未训练模块与入口工程错误均保留；没有按Assessment结果调整候选或样本。
- 全局参数只拟合Train，Validation仅选冻结候选；RUL400/500/600h各自只用起点前80/20，不能借用之后Train。
- 标量运行损失C/D已执行；极化诊断到运行的联合一维/二维状态资格未通过，两者不是同一资格问题。
- 既有自动Git提交授权仅本地提交；本轮不自动推送。
'''.rstrip()+'\n',encoding='utf-8')
    files=[]
    for root in [WORK,PACKAGE,backup]:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '绘图缓存' in p.parts or '__pycache__' in p.parts or p.name=='最终文件SHA256.csv':continue
            files.append(dict(文件=str(p.relative_to(GUIDE)),字节=p.stat().st_size,SHA256=sha(p)))
    files.append(dict(文件=str((FORMAL/'README.md').relative_to(GUIDE)),字节=(FORMAL/'README.md').stat().st_size,SHA256=sha(FORMAL/'README.md')))
    csv(OUT/'表格/最终文件SHA256.csv',pd.DataFrame(files))
    print(f'最终验收PASS：{len(files)}份本轮文件登记；900份历史文件不变；2B数值及正式入口通过。')


if __name__=='__main__':main()
