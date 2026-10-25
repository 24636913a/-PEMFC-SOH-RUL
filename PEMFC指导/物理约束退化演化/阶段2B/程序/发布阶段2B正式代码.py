"""先逐文件备份2A，再发布2B独立数值核心；不复制准备/绘图程序。"""
from pathlib import Path
from datetime import datetime,timezone
import shutil,json,sys,subprocess,time
import numpy as np
import pandas as pd
from 运行低阶实验 import sha,js,csv
WORK=Path(__file__).resolve().parents[1];GUIDE=WORK.parents[1];FORMAL=GUIDE/'完整代码'

def main():
    out=WORK/'输出结果';receipt=out/'正式代码发布验收.json'
    assert (out/'独立复核通过.json').exists()
    if receipt.exists():raise RuntimeError('本版本已发布，不覆盖。')
    package=FORMAL/'阶段2B_v1';assert not package.exists(),'已有版本必须先复核，不覆盖'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    previous=[p for p in FORMAL.rglob('*') if p.is_file() and '完整代码历史副本' not in p.parts and '__pycache__' not in p.parts]
    backup=FORMAL/'完整代码历史副本'/('阶段2A_v1_'+stamp);backup.mkdir(parents=True,exist_ok=False)
    records=[]
    for p in previous:
        dest=backup/p.relative_to(FORMAL);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert sha(p)==sha(dest);records.append(dict(原路径=str(p.relative_to(FORMAL)),备份路径=str(dest.relative_to(FORMAL)),原SHA256=sha(p),备份SHA256=sha(dest)))
    csv(backup/'逐文件备份SHA256.csv',pd.DataFrame(records))
    # --help only verifies imports/entry; do not rerun stage 2A calculations.
    result=subprocess.run([sys.executable,'-X','utf8','-B',str(backup/'程序/运行基础预测.py'),'--help'],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    (backup/'入口静态检查.txt').write_text(result.stdout.rstrip()+'\n',encoding='utf-8')
    (backup/'恢复方法.md').write_text('# 阶段2A_v1恢复方法\n\n逐文件备份SHA一致。依据逐文件备份SHA256.csv，将各相对原路径从本备份复制回完整代码即可恢复原正式版本；恢复前同样先保存当前文件。原2A文件本轮实际均未覆盖，所以无需现在恢复。历史副本不递归嵌套复制，之前的历史目录仍原位保留。\n\n备份入口--help与自有模块导入检查通过；不重复执行2A数值实验。恢复后使用原使用说明中的命令与数据合同即可运行。阶段2B独立子目录无需删除，不能以恢复为名删历史成果。\n',encoding='utf-8')
    for name in ['程序','配置']:(package/name).mkdir(parents=True,exist_ok=False)
    shutil.copy2(WORK/'程序/低阶演化核心.py',package/'程序/低阶演化核心.py')
    code=(WORK/'程序/运行低阶实验.py').read_text(encoding='utf-8')
    code=code.replace("default=Path(__file__).resolve().parents[1]", "default=Path(__file__).resolve().parents[3]/'物理约束退化演化/阶段2B'")
    code=code.replace("args.output or work/'输出结果'", "args.output or work/'输出结果/正式版本复算'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')")
    for name in ['阶段2B冻结协议.json','运行前保护快照.json','冻结输入身份.json']:
        code=code.replace(f"work/'配置/{name}'",f"Path(__file__).resolve().parents[1]/'配置/{name}'")
    (package/'程序/运行低阶实验.py').write_text(code,encoding='utf-8')
    for name in ['阶段2B冻结协议.json','冻结输入身份.json']:shutil.copy2(WORK/'配置'/name,package/'配置'/name)
    frozen=json.loads((WORK/'配置/运行前保护快照.json').read_text(encoding='utf-8'))
    js(package/'配置/运行前保护快照.json',dict(protocol_sha256=frozen['protocol_sha256'],HEAD=frozen['HEAD'],UTC=frozen['UTC'],
        原全量保护清单='物理约束退化演化/阶段2B/配置/运行前保护快照.json；数值入口只需要协议身份'))
    (package/'requirements.txt').write_text(f'numpy=={np.__version__}\npandas=={pd.__version__}\n',encoding='utf-8')
    (package/'使用说明.md').write_text('''# 阶段2B_v1：已运行的低阶约束预测完整代码

本版本含C有界速率、D慢变与可逆分量及去投影对照、Train统计拟合、Validation冻结选择、同查询评价、FC1独立内层RUL与删失。它是可复现数值实验，不是已证明的电化学机理或完成的跨堆迁移。

从任意目录运行本目录程序/运行低阶实验.py，默认按自身位置定位PEMFC指导/物理约束退化演化/阶段2B的数据准备（三份冻结观测CSV），只读2A冻结A/B预测作为对照；本目录自己的配置与SHA合同核验输入。全部自研数值模块均在本包内，无需导入任何准备任务或旧项目Python代码。CPU依赖见requirements.txt。

例如：`python -X utf8 -B "PEMFC指导/完整代码/阶段2B_v1/程序/运行低阶实验.py"`

默认输出为阶段2B/输出结果/正式版本复算/新UTC目录；也可--work指定冻结数据所在阶段目录、--output指定新输出目录。完成目录拒绝覆盖。重跑会真实拟合C/D，不使用2B旧预测缓存。2A不重新执行、不训练网络或迁移、不解析原始百万行数据。

输出有全部逐点预测、全候选Validation评价、同结构消融、参数、起点状态、SOH误差、RUL内层选择/删失与SHA。Real B60仍探索性；Assessment为已看过的再次验证。2D映射与神经网络资格未通过，没有虚构其成绩。

原阶段2A_v1已完整逐文件备份并核对SHA，其现有文件保持不变；版本索引见上级README.md。
'''.rstrip()+'\n',encoding='utf-8')
    js(package/'版本身份.json',dict(version='阶段2B_v1',UTC=stamp,source_HEAD=frozen['HEAD'],
        source_core_SHA256=sha(WORK/'程序/低阶演化核心.py'),scope='低阶标量观测损失演化；不含准备/绘图/二维映射训练/迁移',
        entry_difference='仅默认数据、输出、配置路径适配；数值核心字节一致'))
    js(package/'正式代码SHA256.json',{str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()})
    (FORMAL/'README.md').write_text('''# 正式完整实验代码版本索引

当前最新正式数值实验为[阶段2B_v1](阶段2B_v1/使用说明.md)：低阶有界速率、慢变与可逆状态、同信息消融及条件RUL。正式代码可运行不代表二维观测映射、物理网络或跨堆迁移已通过科研资格。

阶段2A_v1原入口与原[使用说明](使用说明.md)保留，不覆盖。阶段2B发布前已将其全部正式文件逐一备份到完整代码历史副本/阶段2A_v1_时间戳，附SHA、入口静态检查和恢复方法。没有重新执行2A实验。

数据准备、温压筛选、原始绘图和辅助审计仍在各自任务目录；不批量复制到此处。以后更新前仍须先完整备份旧正式版本。
'''.rstrip()+'\n',encoding='utf-8')
    dest=out/'正式版本独立运行'/stamp
    start=time.perf_counter();result=subprocess.run([sys.executable,'-X','utf8','-B',str(package/'程序/运行低阶实验.py'),'--output',str(dest)],
        cwd=package,capture_output=True,text=True,encoding='utf-8');elapsed=time.perf_counter()-start
    (out/'正式入口执行日志.txt').write_text((result.stdout+'\n'+result.stderr).rstrip()+'\n',encoding='utf-8')
    assert result.returncode==0,result.stderr
    fresh=json.loads((dest/'计算完成.json').read_text(encoding='utf-8'))
    compared=[]
    for rel,h in fresh['outputs'].items():
        oldsha=sha(out/rel);compared.append(dict(文件=rel,阶段SHA256=oldsha,正式独立运行SHA256=h,字节一致=oldsha==h));assert oldsha==h,rel
    csv(out/'表格/正式入口独立重跑比较.csv',pd.DataFrame(compared))
    assert sha(package/'程序/低阶演化核心.py')==sha(WORK/'程序/低阶演化核心.py')
    js(receipt,dict(status='PASS',backup=str(backup.relative_to(FORMAL)),backup_files=len(records),
        backup_entry_help=True,stage2A_numerical_experiment_rerun=False,formal_output=str(dest.relative_to(WORK)),
        independent_files=len(compared),formal_run_wall_seconds=elapsed,core_SHA=sha(package/'程序/低阶演化核心.py')))
    print(f'2B正式入口重跑通过：{len(compared)}份文件字节一致；2A备份{len(records)}份文件未改原件。')


if __name__=='__main__':main()
