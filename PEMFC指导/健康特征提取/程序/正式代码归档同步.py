"""阶段1一次性正式发布：先完整归档阶段0并验证，失败恢复旧代码；不执行Git。"""
from pathlib import Path
import datetime,hashlib,json,shutil,sys


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    raise RuntimeError('阶段1.1起准备工作不得同步到完整代码。旧实现已归档，仅保留历史身份；正式实验尚未建立。')
    guide=next(p for p in Path(__file__).resolve().parents if (p/'数据处理/数据核验报告.md').exists())
    stage=guide/'健康特征提取';formal=guide/'完整代码'
    previous=json.loads((formal/'版本与核验记录.json').read_text(encoding='utf-8'))
    if previous['version']!='阶段0_v1':raise RuntimeError('仅适用于阶段0_v1首次同步，禁止重复发布或覆盖新版本')
    final=json.loads((stage/'输出结果/最终交付索引.json').read_text(encoding='utf-8'))
    out=guide/final['result_relative_to_guide']
    checks=sorted(out.glob('独立复算核验_*.json'))
    if not checks or json.loads(checks[-1].read_text(encoding='utf-8'))['status']!='PASS':raise RuntimeError('独立复算未通过')
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    archive=formal/'完整代码历史副本'/f'{stamp}_阶段0_v1'
    archive.mkdir(parents=True,exist_ok=False)
    oldfiles=[p for p in formal.rglob('*') if p.is_file() and '完整代码历史副本' not in p.relative_to(formal).parts]
    inventory=[]
    for p in oldfiles:
        rel=p.relative_to(formal);dest=archive/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        digest=sha(p)
        if sha(dest)!=digest:raise RuntimeError('完整历史备份SHA失败')
        inventory.append(dict(file=rel.as_posix(),sha256=digest))
    write(archive/'归档文件SHA256.json',dict(version='阶段0_v1',files=inventory,verified=True))
    # 所有旧文件备份核验成功后才写最新正式目录。
    added=[]
    try:
        for source in sorted((stage/'程序').glob('*.py')):
            dest=formal/source.name
            if not dest.exists():added.append(dest)
            shutil.copy2(source,dest)
        for name in ['健康特征冻结规则.yaml','计算前冻结记录.json','指导方案只读文本.txt']:
            dest=formal/'配置'/name
            if not dest.exists():added.append(dest)
            shutil.copy2(stage/'配置'/name,dest)
        files=[p for p in formal.rglob('*') if p.is_file() and '完整代码历史副本' not in p.relative_to(formal).parts]
        for p in files:
            if p.suffix=='.py':compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
        info=dict(version='阶段1_v1',updated=datetime.datetime.now().astimezone().isoformat(),
             implemented=['阶段0原始数据核验','阶段1二维特征、质量拒绝、敏感性、独立复算与绘图'],
             not_implemented=['退化预测','神经网络','迁移','SOH/RUL'],scientific_status=final['status'],
             backup=str(archive),backup_files=len(inventory),backup_verified=True,source_result=str(out),
             files=[dict(file=p.relative_to(formal).as_posix(),sha256=sha(p)) for p in files if p.name not in ['版本与核验记录.json','使用说明.md']])
        write(formal/'版本与核验记录.json',info)
        readme=f'''# 最新完整代码 阶段1_v1

更新时间：{info['updated']}。包含阶段0和阶段1全部自研模块，不依赖历史训练入口，不加载模型或预测缓存。

正式入口：`健康特征主程序.py`（阶段1）；`数据核验主程序.py`（阶段0）。

已实现原始数据审计、降载平台筛选、局部插值、同堆BoL损失、一维与二维最小二乘、参数敏感性和独立复算。没有实现或启动退化网络、跨堆迁移、SOH/RUL预测。阶段1科学结果为{final['status']}：程序通过工程检查不等于两堆特征均取得资格。

运行环境：原项目Python 3.11.9；依赖见依赖环境.txt。原始数据仍在现有项目数据目录，路径及SHA在阶段0登记表；不复制大型原始文件。阶段1读取原始极化文件；继承阶段0表中的时间资格，不把它当临时缓存。源码使用本目录配置副本，SHA必须与冻结记录一致。

在本目录运行：
```powershell
python -X utf8 -B 健康特征主程序.py --check-only
python -X utf8 -B 健康特征主程序.py
python -X utf8 -B 独立复算核验.py --output "阶段1实际结果目录绝对路径"
python -X utf8 -B 平台可靠性复核.py --output "阶段1实际结果目录绝对路径"
python -X utf8 -B 数据核验主程序.py --check-only
```
默认新结果统一写入PEMFC指导/健康特征提取/输出结果；存在正式运行记录则新建历史核验目录，不覆盖旧结果。`--redraw-from`只从完整性核对后的真实CSV重绘，不重新提取数值；输出独立目录。

阶段0完整备份：{archive.name}，共{len(inventory)}个旧文件，逐文件SHA核对通过，归档不得改写。恢复时先归档当前版本，再按该归档文件清单恢复，不能混合配置。当前版本同样须先完整备份后才能更新。

验证：开发入口已真实读取19个极化源并完成数值计算；独立闭式复算、平台边界测试通过；本目录同步后另做静态、入口和原始读取能力检查。不重复扫描860万行运行数据。
'''
        (formal/'使用说明.md').write_text(readme,encoding='utf-8')
        write(out/'正式代码同步记录.json',dict(status='PASS',version=info['version'],archive=str(archive),backup_files=inventory,
            new_code=[dict(file=p.name,sha256=sha(p)) for p in sorted((stage/'程序').glob('*.py'))],training=False,git=False))
    except Exception:
        failure=formal/'完整代码历史副本'/f'{stamp}_阶段1更新失败现场';failure.mkdir(exist_ok=False)
        for p in list(added)+[formal/'使用说明.md',formal/'版本与核验记录.json']:
            if p.is_file():
                rel=p.relative_to(formal);d=failure/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
        for p in added:
            if p.is_file():p.unlink()  # 仅移除本次新建且已留档的文件；旧文件由归档恢复。
        for item in inventory:
            dest=formal/item['file'];shutil.copy2(archive/item['file'],dest)
            assert sha(dest)==item['sha256']
        raise
    print('PASS 阶段0完整备份并核对:',len(inventory),'文件；正式代码阶段1_v1同步完成')


if __name__=='__main__':main()
