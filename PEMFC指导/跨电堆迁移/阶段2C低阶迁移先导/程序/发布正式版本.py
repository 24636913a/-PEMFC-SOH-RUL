"""验收后备份2B完整代码并新增2C正式数值包，不修改旧索引/入口。"""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
TASK=Path(__file__).resolve().parents[1];GUIDE=TASK.parents[1];ROOT=GUIDE/'完整代码'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    report=json.loads((TASK/'输出结果/正式计算/独立复核通过.json').read_text(encoding='utf-8'))
    assert report['status']=='PASS'
    dest=ROOT/'阶段2C_v1'
    if dest.exists():raise RuntimeError('ALREADY_PUBLISHED_NO_OVERWRITE')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');backup=ROOT/'完整代码历史副本'/(stamp+'_阶段2B_v1完整备份')
    old=[p for p in ROOT.rglob('*') if p.is_file() and '完整代码历史副本' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts]
    records=[]
    for p in old:
        rel=p.relative_to(ROOT);out=backup/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
        assert sha(p)==sha(out);records.append(dict(path=rel.as_posix(),sha256=sha(p),backup_sha256=sha(out)))
    dump(backup/'备份核验SHA256.json',records)
    (backup/'恢复说明.md').write_text('该备份在新增阶段2C_v1前建立。保留原阶段2B_v1的程序、配置、依赖和入口，以及2A旧代码和当时全部索引。按备份核验SHA256.json逐文件校验；如需恢复，将列明文件复制到完整代码下同一相对位置。不要删除任何历史目录。本次没有改写被备份的原文件。\n',encoding='utf-8')
    # 入口help只解析参数，不重跑2B实验。
    checked=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'阶段2B_v1/程序/运行低阶实验.py'),'--help'],capture_output=True,text=True,encoding='utf-8')
    (backup/'旧2B入口只读help验收.txt').write_text(checked.stdout+checked.stderr,encoding='utf-8');assert checked.returncode==0
    for d in ['程序','配置']:(dest/d).mkdir(parents=True,exist_ok=True)
    for name in ['迁移核心.py','运行迁移实验.py']:shutil.copy2(TASK/'程序'/name,dest/'程序'/name)
    for name in ['阶段2C冻结协议.json','预登记时间与SHA256.json']:shutil.copy2(TASK/'配置'/name,dest/'配置'/name)
    identity=json.loads((TASK/'输出结果/正式计算/执行身份.json').read_text(encoding='utf-8'))
    (dest/'requirements.txt').write_text('numpy=='+identity['numpy']+'\npandas=='+identity['pandas']+'\n',encoding='utf-8')
    (dest/'使用说明.md').write_text('''# 阶段2C_v1：已验收的低阶跨堆迁移数值实验

本版本含完整数学核心、Train拟合、截止前内部验证、固定起点自主传播、同目标指标及迁移收益计算，不导入准备任务Python模块。只有numpy和pandas；无需神经网络或GPU。

运行：`python -X utf8 -B "PEMFC指导/完整代码/阶段2C_v1/程序/运行迁移实验.py"`。从任意工作目录可启动，默认只读阶段2C/数据准备三个冻结CSV及SHA合同，采用本包冻结配置；默认写入阶段2C/输出结果/正式版本复算/新UTC目录。可用--data、--config、--output明确设置，已存在输出目录拒绝覆盖。

结果是真实重新拟合和预测，不读取旧预测缓存。运行后三场景各6方法、相同141个未来目标，保留全部不利结果及25循环缺失NA。独立复核入口仍位于任务程序目录，不属于正式模型依赖。准备/时间核对/资格分析/绘图程序没有放入本包。

当前结论是探索性标量迁移负面证据，不是二维健康状态迁移成功或真实RUL。未知发布时刻、名义未来每循环12h、不同负载及原Assessment已查看均需保留。后移起点目标慢变模型较好不计为FC1贡献。

本次发布前完整代码旧文件已完整备份并逐文件核对SHA；源2A和2B原文件均未改写。恢复方法见相应历史备份/恢复说明.md。
''',encoding='utf-8')
    manifest={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}
    dump(dest/'正式代码SHA256.json',manifest)
    record=dict(version='阶段2C_v1',identity='已验收的低阶标量迁移先导；负迁移结果保留',source_head=identity['head'],backup=backup.relative_to(GUIDE).as_posix(),old_files=len(old),backup_all_sha_equal=True,formal_numerical_core_exact_copy=True)
    dump(dest/'版本记录.json',record)
    (ROOT/'当前版本_阶段2C.md').write_text('# 当前新增正式版本：阶段2C_v1\n\n正式入口：阶段2C_v1/程序/运行迁移实验.py。详见阶段2C_v1/使用说明.md。旧README、使用说明和2A/2B入口全部保留原样；本索引通过新增文件声明本次版本，不覆盖历史记录。\n\n本版本是已执行且可复现的低阶迁移实验，结论为固定FC1先验未改善目标共同未来预测；二维迁移和16单元网络没有启动。\n\n备份位置：'+record['backup']+'\n',encoding='utf-8')
    dump(TASK/'输出结果/正式发布与备份记录.json',record);print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
