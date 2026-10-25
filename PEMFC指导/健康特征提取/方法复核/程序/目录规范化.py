"""只处理PEMFC指导内已盘点的准备代码。先逐文件归档核验，再按显式清单清理重复文件。"""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,csv

GUIDE=next(p for p in Path(__file__).resolve().parents if p.name=='PEMFC指导')
WORK=GUIDE/'健康特征提取/方法复核'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')


def archive():
    manifest=WORK/'配置/目录归档清单.json'
    if manifest.exists():raise RuntimeError('已有归档清单，停止避免重复归档')
    formal=(GUIDE/'完整代码').resolve()
    if formal.parent!=GUIDE.resolve() or formal.is_symlink():raise RuntimeError('完整代码路径异常')
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    current=GUIDE/'健康特征提取/程序历史副本'/f'{stamp}_原完整代码阶段1_v1'
    zero=GUIDE/'数据处理/程序历史副本'/f'{stamp}_阶段0原程序及归档'
    current.mkdir(parents=True,exist_ok=False);zero.mkdir(parents=True,exist_ok=False)
    rows=[];matches=[]
    for original in sorted(formal.rglob('*')):
        if not original.is_file():continue
        rel=original.relative_to(formal);historic='完整代码历史副本' in rel.parts
        dest=(zero/'原历史目录'/rel.relative_to('完整代码历史副本')) if historic else current/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,dest)
        digest=sha(original)
        if digest!=sha(dest):raise RuntimeError('归档SHA失败，禁止清理')
        row=dict(原路径=original.relative_to(GUIDE).as_posix(),归档路径=dest.relative_to(GUIDE).as_posix(),SHA256=digest,类型='旧阶段0归档' if historic else '原混合准备代码版本',归档验证=True)
        rows.append(row)
        if original.suffix=='.py' and not historic:
            stage='数据处理' if (GUIDE/'数据处理/程序'/original.name).exists() else '健康特征提取'
            source=GUIDE/stage/'程序'/original.name
            if not source.is_file() or sha(source)!=digest:raise RuntimeError('正式目录与所属准备源不一致：'+str(original))
            matches.append(dict(原路径=row['原路径'],任务源路径=source.relative_to(GUIDE).as_posix(),SHA256=digest,一致=True))
    # 阶段0现有入口的管理逻辑将修改；将完整源版本单独留存，不仅备份改动片段。
    for folder in ['程序','配置']:
        for source in sorted((GUIDE/'数据处理'/folder).rglob('*')):
            if source.is_file():
                dest=zero/'整理前任务源'/folder/source.relative_to(GUIDE/'数据处理'/folder)
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
                assert sha(source)==sha(dest)
                rows.append(dict(原路径=source.relative_to(GUIDE).as_posix(),归档路径=dest.relative_to(GUIDE).as_posix(),SHA256=sha(source),类型='整理前任务源',归档验证=True))
    # 仅原完整代码内存在的依赖说明，分别归入两个任务；不变更原配置。
    dep=formal/'依赖环境.txt'
    for stage in ['数据处理','健康特征提取']:
        dest=GUIDE/stage/'配置/依赖环境.txt'
        if dest.exists() and sha(dest)!=sha(dep):raise RuntimeError('已有依赖说明不同，停止覆盖')
        if not dest.exists():shutil.copy2(dep,dest)
        assert sha(dep)==sha(dest)
    # 配置全部逐文件核对，避免只保留py遗漏依赖。
    for source in (formal/'配置').iterdir():
        task='数据处理' if source.name=='数据核验配置.yaml' else '健康特征提取'
        target=GUIDE/task/'配置'/source.name
        if not target.exists() or sha(target)!=sha(source):raise RuntimeError('配置来源不完整：'+source.name)
        matches.append(dict(原路径=source.relative_to(GUIDE).as_posix(),任务源路径=target.relative_to(GUIDE).as_posix(),SHA256=sha(source),一致=True))
    data=dict(version='阶段1.1_目录归档_v1',created=datetime.datetime.now().astimezone().isoformat(),rows=rows,source_matches=matches,cleanup_done=False)
    write(manifest,data)
    write(WORK/'配置/准备代码路径索引.json',rows)
    for d in [current,zero]:
        write(d/'归档来源说明.json',dict(source='原完整代码中的准备程序；按阶段1.1新规则归回任务历史副本，保留原文件字节',files=[r for r in rows if (GUIDE/r['归档路径']).is_relative_to(d)],immutable=True))
    print('ARCHIVE_VERIFIED',len(rows),'entries; source matches',len(matches))


def cleanup():
    data=json.loads((WORK/'配置/目录归档清单.json').read_text(encoding='utf-8'))
    if data['cleanup_done']:raise RuntimeError('已整理，不重复执行')
    proof=json.loads((WORK/'输出结果/目录运行验证.json').read_text(encoding='utf-8'))
    if proof['status']!='PASS':raise RuntimeError('先完成入口运行验证再清理')
    formal=(GUIDE/'完整代码').resolve();rows=[r for r in data['rows'] if r['原路径'].startswith('完整代码/')]
    expected={GUIDE/r['原路径'] for r in rows}
    actual={p for p in formal.rglob('*') if p.is_file()}
    if actual!=expected:raise RuntimeError('正式目录有盘点外变化，停止清理')
    for r in rows:
        old=(GUIDE/r['原路径']).resolve();backup=(GUIDE/r['归档路径']).resolve()
        if not old.is_relative_to(formal) or not backup.is_relative_to(GUIDE.resolve()) or old.is_symlink():raise RuntimeError('清理路径越界')
        if sha(old)!=r['SHA256'] or sha(backup)!=r['SHA256']:raise RuntimeError('清理前完整性不符')
    # 所有文件均已完整独立归档；只删除显式清单中的重复文件，无递归删除命令。
    for r in rows:(GUIDE/r['原路径']).unlink()
    for d in sorted([p for p in formal.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        if not any(d.iterdir()):d.rmdir()
    (formal/'完整代码历史副本').mkdir(exist_ok=True)
    (formal/'使用说明.md').write_text('# 正式实验完整代码\n\n正式实验代码尚未建立，等待后续建模阶段。\n\n数据处理、平台筛选、特征可行性分析属于准备工作，分别保存在数据处理/程序、健康特征提取/程序及健康特征提取/方法复核/程序。准备代码与历史版本不再同步到本目录。目录迁移及SHA核验见同级代码目录整理记录.md。\n\n未来仅保存已建立的正式退化、迁移、训练验证和预测评价完整代码；更新前完整备份、核对SHA256。\n',encoding='utf-8')
    data['cleanup_done']=True;data['cleanup_time']=datetime.datetime.now().astimezone().isoformat()
    write(WORK/'输出结果/目录清理完成记录.json',data)
    with (WORK/'输出结果/表格/代码目录迁移清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data['rows'][0]));w.writeheader();w.writerows(data['rows'])
    text=['# 代码目录整理记录','', '新规则：准备代码仅在所属任务，完整代码只留正式模型实验；本次没有建立正式模型。原准备文件先完整归档并逐文件SHA核对，验证入口后清理重复文件。未改写Git历史、原始数据或阶段0/1报告结果。',
          '', '原源程序与原完整目录中的10个Python文件及4个配置逐一SHA一致。阶段0/1准备入口的数值算法未改变，只处理禁止自动发布及历史路径解析。两个旧任务的使用说明保留历史原文，现行运行方式见任务内新增的准备代码运行说明.md。',
          '', '|原位置（相对PEMFC指导）|归档位置|SHA256|验证|','|---|---|---|---|']
    text += [f"|{r['原路径']}|{r['归档路径']}|{r['SHA256']}|PASS|" for r in data['rows']]
    text+=['','清理路径均解析为当前PEMFC指导/完整代码之内；未递归删除未知目录。完整代码现在仅有中文说明和空的正式历史副本目录。所有旧版本字节均可按清单恢复；归档内旧绝对路径仅作历史证据，现行入口使用新索引解析。',
            '', '未执行Git commit或push。']
    (GUIDE/'代码目录整理记录.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('DIRECTORY_NORMALIZATION_COMPLETED',len(rows),'duplicated files safely archived')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cleanup',action='store_true');a=p.parse_args()
    cleanup() if a.cleanup else archive()
