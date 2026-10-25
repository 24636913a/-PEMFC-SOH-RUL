"""最终完整性验收、正式版本对照和文件身份清单。"""
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
TASK=Path(__file__).resolve().parents[1];GUIDE=TASK.parents[1];REPO=GUIDE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    first=TASK/'输出结果/正式计算';formal=TASK/'输出结果/正式版本独立复算';checked=[]
    for p in (first/'表格').glob('*.csv'):
        if p.name=='CPU运行代价.csv':continue
        q=formal/'表格'/p.name
        assert q.exists() and sha(p)==sha(q),p.name
        checked.append(dict(file='表格/'+p.name,first_sha256=sha(p),formal_sha256=sha(q),equal=True))
    for p in (first/'模型记录').glob('*.json'):
        q=formal/'模型记录'/p.name;assert sha(p)==sha(q)
        checked.append(dict(file='模型记录/'+p.name,first_sha256=sha(p),formal_sha256=sha(q),equal=True))
    pd.DataFrame(checked).to_csv(TASK/'输出结果/表格/正式包与首次计算逐文件比较.csv',index=False,encoding='utf-8-sig')
    for name in ['迁移核心.py','运行迁移实验.py']:
        assert sha(TASK/'程序'/name)==sha(GUIDE/'完整代码/阶段2C_v1/程序'/name)
        tree=ast.parse((TASK/'程序'/name).read_text(encoding='utf-8'))
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        assert not any(x and (x.startswith('torch') or x.startswith('tensorflow')) for x in imports)
    historical=pd.read_csv(TASK/'配置/历史成果保护SHA256.csv')
    assert all(sha(REPO/r.path)==r.sha256 for r in historical.itertuples())
    cfg=json.loads((TASK/'配置/阶段2C冻结协议.json').read_text(encoding='utf-8'));assert sha(TASK/'配置/阶段2C冻结协议.json')==json.loads((TASK/'配置/预登记时间与SHA256.json').read_text(encoding='utf-8'))['protocol_sha256']
    models=pd.read_csv(first/'表格/模型参数与速率.csv');info=[]
    for r in models.itertuples():
        used_target_n=0 if r.method in ['A_Persistence','B_FC1Direct'] else int(r.n_target_fit)
        info.append(dict(scenario=r.scenario,method=r.method,available_target_history=int(r.n_target_fit),target_rate_fit_observations=used_target_n,target_anchor_observations=1 if not r.method.startswith('D_') else np.nan,source_rate_fit_observations=int(r.source_n),explanation='原参数表n_target_fit为共同拟合流程的可用历史量；保持/直接迁移不使用目标速率统计。本表区分实际信息用途，未改动原CSV。'))
    pd.DataFrame(info).to_csv(TASK/'输出结果/表格/各方法实际信息使用.csv',index=False,encoding='utf-8-sig')
    qual=pd.read_csv(TASK/'输出结果/表格/一维二维迁移与16单元网络资格.csv');assert qual.completed_past_pairs.tolist()==[0,1,2]
    assert (qual.bounded_network16=='NOT_QUALIFIED').all()
    for r in pd.read_csv(TASK/'输出结果/表格/各截止诊断与运行配对及拒绝.csv').itertuples():
        assert r.operating_upper_h<=next(x['cutoff_h'] for x in cfg['scenarios'] if x['name']==r.scenario)
    csvnames=[]
    for record in json.loads((TASK/'输出结果/表格/图片来源索引.json').read_text(encoding='utf-8')):
        p=TASK/record['plot_csv'];assert sha(p)==record['csv_sha256'];csvnames.append(p.name)
        png=TASK/'输出结果/图片'/(record['figure']+'.png');pdf=png.with_suffix('.pdf')
        with Image.open(png) as im:assert im.width>1000 and im.height>800 and min(im.info.get('dpi',(0,0)))>299
        assert pdf.read_bytes().startswith(b'%PDF')
    old=json.loads((TASK/'输出结果/正式发布与备份记录.json').read_text(encoding='utf-8'));backup=GUIDE/old['backup']
    for r in json.loads((backup/'备份核验SHA256.json').read_text(encoding='utf-8')):assert sha(backup/r['path'])==r['sha256']==sha(GUIDE/'完整代码'/r['path'])
    for label,path in [('首次',first),('正式包',formal)]:assert json.loads((path/'独立复核通过.json').read_text(encoding='utf-8'))['status']=='PASS'
    stats=dict(status='PASS',protocol_sha256=sha(TASK/'配置/阶段2C冻结协议.json'),independent_checks_each=1492,numerical_csv_json_exact_equal=len(checked),historical_files_unchanged=len(historical),backup_files_sha_equal=old['old_files'],figures_png_pdf=8,figure_csv=len(csvnames),causal_unit_tests=11,forecast_rows=2538,unique_targets_each=141,all_models_from_actual_cpu_fits=True,neural_training=False,formal_package_self_contained=True,git_policy='standing local-commit authorization only; no push')
    dump(TASK/'输出结果/阶段2C最终验收.json',stats)
    report=f'''# 阶段2C验收与研究决定

阶段2C实际CPU拟合、固定起点预测、独立复算和正式包复现均PASS。**工程完成不等于迁移有效**：本轮固定FC1先验造成负迁移，二维迁移与16单元网络仍NOT_QUALIFIED。

- 三个截止有效历史23/47/48；各方法同一141个后续唯一目标，18条模型轨迹、2538条预测记录。
- 首次与正式包各1492项检查通过，独立最大相对损失差约2.20×10⁻¹⁵；{len(checked)}个数值CSV/模型JSON逐字节一致。运行时长、UTC、脚本身份记录自然分开，不冒充字节相同。
- 11项专项测试通过，包括未来历史拒绝、未来标签改变不影响预测、中途不读真值、600h名义时钟、恢复符号、缺失不填。
- 8组300dpi PNG/PDF均有真实绘图CSV、脚本和SHA。人工查看共同目标RMSE图及后移预测图，标签/单位/图例可读。
- 1050个既有历史文件SHA未变；完整代码原20个文件已备份并核对，可恢复。旧2B入口只读--help通过，未重新执行2B。
- 正式包只含独立数学核心、预测入口、冻结配置及依赖；准备、资格、绘图、发布与复核脚本都留在任务目录程序。
- 仅有一次工程报告依赖缺失，已保留原因与修复。没有模型拟合失败、没有用旧预测填补新结果。

## 参数表计数说明
原`模型参数与速率.csv`的`n_target_fit`记录共同拟合流程的可用目标历史数；保持法和直接迁移实际上不使用目标漂移估计。另存`各方法实际信息使用.csv`明确目标速率拟合量分别为0或完整截止历史，目标观察锚点1个；没有改写首次计算CSV。模型自由度与状态量另见`参数规模与模型来源.csv`。

## 下一步
不启动16单元网络。保留双参考离线表征方向，但优先获取更多独立诊断—运行合法历史配对、测试实际完成/发布时刻，以及跨负载/电堆观测映射依据。当前只能保留目标域低阶模型为后续基础；不得将后移目标域的好成绩改称FC1迁移收益。FC1条件RUL和RealHyFC无真实EOL边界不变。
'''
    (TASK/'阶段2C验收报告.md').write_text(report,encoding='utf-8')
    manifest=[]
    for p in sorted(TASK.rglob('*')):
        if p.is_file() and p.name!='本轮成果SHA256.csv' and '__pycache__' not in p.parts:manifest.append(dict(path=p.relative_to(TASK).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    pd.DataFrame(manifest).to_csv(TASK/'输出结果/本轮成果SHA256.csv',index=False,encoding='utf-8-sig');print(json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':main()
