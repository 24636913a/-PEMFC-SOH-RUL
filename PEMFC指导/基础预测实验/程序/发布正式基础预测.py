"""验收通过后首次发布预测/评价代码；仅备份并更新原占位README。"""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,subprocess,sys
from 运行基础预测 import sha,js,csv
import pandas as pd

WORK=Path(__file__).resolve().parents[1];FORMAL=WORK.parent/'完整代码'

def main():
    assert (WORK/'输出结果/独立复核通过.json').exists()
    receipt=WORK/'输出结果/正式版本发布记录.json'
    if receipt.exists():raise RuntimeError('已发布，请勿覆盖正式版本')
    assert not (FORMAL/'程序').exists(),'此入口仅用于首次发布，禁止覆盖后续正式版本'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=FORMAL/'完整代码历史副本'/(stamp+'_正式版本建立前')
    backup.mkdir(parents=True,exist_ok=False)
    originals=[p for p in FORMAL.rglob('*') if p.is_file() and '完整代码历史副本' not in p.parts]
    ledger=[]
    for p in originals:
        dest=backup/p.relative_to(FORMAL);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert sha(p)==sha(dest);ledger.append(dict(文件=str(p.relative_to(FORMAL)),原SHA256=sha(p),备份SHA256=sha(dest)))
    csv(backup/'备份SHA256.csv',pd.DataFrame(ledger))
    for folder in ['程序','配置']:(FORMAL/folder).mkdir(exist_ok=False)
    (FORMAL/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    shutil.copy2(WORK/'程序/基础预测核心.py',FORMAL/'程序/基础预测核心.py')
    code=(WORK/'程序/运行基础预测.py').read_text(encoding='utf-8')
    code=code.replace("default=Path(__file__).resolve().parents[1]", "default=Path(__file__).resolve().parents[2]/'基础预测实验'")
    code=code.replace("args.output or work/'输出结果'", "args.output or work/'输出结果/正式版本复算'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')")
    code=code.replace("cfgpath=work/'配置/阶段2A冻结协议.json'", "cfgpath=Path(__file__).resolve().parents[1]/'配置/阶段2A冻结协议.json'")
    code=code.replace("    prep=json.loads((work/'输出结果/数据准备完成.json').read_text(encoding='utf-8'))\n", '')
    code=code.replace("(work/'配置/执行前冻结.json')", "(Path(__file__).resolve().parents[1]/'配置/执行前冻结.json')")
    code=code.replace("contract=work/'配置/预测输入合同.json'", "contract=Path(__file__).resolve().parents[1]/'配置/预测输入合同.json'")
    (FORMAL/'程序/运行基础预测.py').write_text(code,encoding='utf-8')
    for name in ['阶段2A冻结协议.json','预测输入合同.json']:shutil.copy2(WORK/'配置'/name,FORMAL/'配置'/name)
    origin=json.loads((WORK/'配置/执行前冻结.json').read_text(encoding='utf-8'))
    js(FORMAL/'配置/执行前冻结.json',dict(协议SHA256=origin['协议SHA256'],HEAD=origin['HEAD'],UTC=origin['UTC'],
       全部原始保护清单='基础预测实验/配置/执行前冻结.json；正式运行只需协议身份与观测合同'))
    # Record actual installed versions rather than invent dependencies.
    import numpy as np
    (FORMAL/'requirements.txt').write_text(f'numpy=={np.__version__}\npandas=={pd.__version__}\n',encoding='utf-8')
    readme='''# 正式实验完整代码：阶段2A_v1

这是首个完成真实CPU预测、独立复算的正式代码版本。“正式”指代码和协议可复现，不表示已经通过严格在线迁移资格或证明二维收益。未进行神经训练。

仅含连续运行持续值/局部趋势、下一次诊断固定岭回归、条件阈值RUL及评价。所有自研数值模块已包含，无需导入数据处理、健康特征准备或旧研究Python程序。依赖见requirements.txt；已在项目Python3.11实测。

从任意工作目录执行：

`python -X utf8 -B "PEMFC指导/完整代码/程序/运行基础预测.py"`

路径应按实际仓库位置填写。默认按自身位置定位同级基础预测实验/数据准备中的三份冻结观测CSV（数据而非预测缓存），以本目录自己的配置/输入SHA合同验证，输出到基础预测实验/输出结果/正式版本复算/新UTC目录。也可用--work指定完整数据包所在阶段目录、--output指定全新输出目录。拒绝覆盖已有完成目录。

模型拟合与预测从观测重新计算，不读历史预测；输入CSV的原始来源、温压/时间资格由数据合同和基础预测实验记录追溯。原始数据准备、平台筛选、温压解释、原始曲线绘图不属于本完整代码，仍在各任务目录内。

输出包含全部逐点预测、指标、NA排除原因、局部/岭回归参数、RUL内部选择与删失记录、代码与输出SHA。结果仍为已看过全轨迹的探索性分析；FC1名义诊断时钟与Real B60身份不得升级。RUL是条件阈值寿命，不是已确认物理EOL。

使用说明更新前已逐文件备份并校验SHA，见完整代码历史副本。以后更新正式代码前必须先完整备份旧版本。不得将准备代码批量复制到本目录。
'''
    (FORMAL/'使用说明.md').write_text(readme,encoding='utf-8')
    js(FORMAL/'版本记录.json',dict(版本='阶段2A_v1',UTC=stamp,源HEAD=origin['HEAD'],
       身份='已验收的基础预测代码；探索性离线结论，不含迁移或物理网络',备份=str(backup.relative_to(FORMAL)),
       输入='同级基础预测实验/数据准备三个冻结CSV',来源模块={n:sha(WORK/'程序'/n) for n in ['基础预测核心.py','运行基础预测.py']},
       入口差异='仅默认输入/输出/配置路径适配；数值核心字节一致'))
    hashes={str(p.relative_to(FORMAL)):sha(p) for p in FORMAL.rglob('*') if p.is_file() and '完整代码历史副本' not in p.parts}
    js(FORMAL/'正式代码SHA256.json',hashes)
    dest=WORK/'输出结果/正式版本独立运行'/stamp
    cmd=[sys.executable,'-X','utf8','-B',str(FORMAL/'程序/运行基础预测.py'),'--output',str(dest)]
    result=subprocess.run(cmd,cwd=FORMAL,text=True,capture_output=True,encoding='utf-8')
    (WORK/'输出结果/正式入口执行日志.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('正式入口执行失败，保留现场：'+result.stderr)
    initial=json.loads((WORK/'输出结果/基础预测完成.json').read_text(encoding='utf-8'))
    current=json.loads((dest/'基础预测完成.json').read_text(encoding='utf-8'))
    comparisons=[]
    for rel,h in current['output_sha256'].items():
        check=sha(WORK/'输出结果'/rel)==h
        comparisons.append(dict(输出文件=rel,阶段SHA256=sha(WORK/'输出结果'/rel),正式独立运行SHA256=h,字节完全一致=check))
        assert check,rel
    csv(WORK/'输出结果/表格/正式入口独立重跑比较.csv',pd.DataFrame(comparisons))
    js(receipt,dict(status='PASS',formal_output=str(dest.relative_to(WORK)),compared_files=len(comparisons),
                   backup_SHA=ledger,formal_core_equals_stage=sha(FORMAL/'程序/基础预测核心.py')==sha(WORK/'程序/基础预测核心.py'),
                   formal_files_sha256=hashes))
    (WORK/'输出结果/正式版本验收说明.md').write_text(f'# 正式版本验收\n\n独立入口从冻结观测重新拟合并预测，{len(comparisons)}份预测、指标和参数文件与阶段执行字节完全一致。自研预测核心SHA完全一致，入口仅路径适配。历史占位说明备份SHA核对通过。没有准备代码进入完整代码。\n',encoding='utf-8')
    print(f'正式入口独立重跑通过，{len(comparisons)}文件字节一致。')


if __name__=='__main__':main()
