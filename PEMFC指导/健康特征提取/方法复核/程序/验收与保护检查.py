"""本轮最终交付检查：原证据、归档、入口和所有PNG。无Git修改、训练或全量运行数据读取。"""
from pathlib import Path
import sys,json,subprocess,datetime,importlib.util,ast
from PIL import Image

WORK=Path(__file__).resolve().parent.parent;GUIDE=WORK.parent.parent
sys.path[:0]=[str(WORK/'程序'),str(GUIDE/'数据处理/程序')]
from 核验基础 import sha,save_json
from 路径与证据 import resolve_evidence


def node(path,name):
    return ast.dump(next(x for x in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)


def run():
    out=WORK/'输出结果';snapshot=json.loads((WORK/'配置/任务前身份快照.json').read_text(encoding='utf-8'))
    moves=json.loads((WORK/'配置/目录归档清单.json').read_text(encoding='utf-8'))
    sourcefix={'数据处理/程序/数据核验主程序.py','健康特征提取/程序/健康特征主程序.py',
               '健康特征提取/程序/独立复算核验.py','健康特征提取/程序/正式代码归档同步.py'}
    check=[]
    for r in snapshot['protected_before_changes']:
        if r['path'] in sourcefix:
            archived=resolve_evidence(r['path'],r['sha256']);assert sha(archived)==r['sha256']
            check.append(dict(**r,status='OLD_BYTES_ARCHIVED_MANAGEMENT_FIX_ONLY',archived=str(archived)))
        else:
            assert sha(GUIDE/r['path'])==r['sha256'],r['path']
            check.append(dict(**r,status='UNCHANGED'))
    for r in moves['rows']:assert sha(GUIDE/r['归档路径'])==r['SHA256']
    old0=next(GUIDE/r['归档路径'] for r in moves['rows'] if r['原路径']=='数据处理/程序/数据核验主程序.py')
    old1=next(GUIDE/r['归档路径'] for r in moves['rows'] if r['原路径']=='完整代码/健康特征主程序.py')
    assert node(old0,'main')==node(GUIDE/'数据处理/程序/数据核验主程序.py','main')
    assert node(old1,'compute')==node(GUIDE/'健康特征提取/程序/健康特征主程序.py','compute')
    formal=GUIDE/'完整代码';assert not list(formal.rglob('*.py'))
    assert [p.name for p in formal.rglob('*') if p.is_file()]==['使用说明.md']
    assert '正式实验代码尚未建立，等待后续建模阶段。' in (formal/'使用说明.md').read_text(encoding='utf-8')
    before={str(p):sha(p) for p in formal.rglob('*') if p.is_file()}
    import 数据核验主程序 as zero
    assert zero.sync_complete(None,None,None,None,None,None)['status']=='PREPARATORY_CODE_TASK_LOCAL_ONLY'
    spec=importlib.util.spec_from_file_location('disabled_publisher',GUIDE/'健康特征提取/程序/正式代码归档同步.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:module.main()
    except RuntimeError as exc:assert '不得同步' in str(exc)
    else:raise AssertionError('旧同步器未禁用')
    assert before=={str(p):sha(p) for p in formal.rglob('*') if p.is_file()}
    tests=[]
    for relative in ['数据处理/程序/数据核验主程序.py','健康特征提取/程序/健康特征主程序.py','健康特征提取/方法复核/程序/方法复核主程序.py']:
        r=subprocess.run([sys.executable,'-X','utf8','-B',str(GUIDE/relative),'--check-only'],capture_output=True,text=True,encoding='utf-8')
        assert r.returncode==0,r.stderr;tests.append(dict(entry=relative,exit_code=r.returncode,stdout=r.stdout))
    copy=out/'目录验证副本'
    r=subprocess.run([sys.executable,'-X','utf8','-B',str(GUIDE/'健康特征提取/程序/独立复算核验.py'),'--output',str(copy)],capture_output=True,text=True,encoding='utf-8')
    assert r.returncode==0,r.stderr;tests.append(dict(entry='清理后阶段1独立复算，输入复制到本轮副本',exit_code=0,stdout=r.stdout))
    images=[]
    for p in sorted((out/'图片').glob('*.png')):
        with Image.open(p) as im:dim=im.size;dpi=im.info.get('dpi');im.verify()
        assert dpi and abs(dpi[0]-300)<.1 and min(dim)>500
        images.append(dict(file=p.name,sha256=sha(p),dimensions=dim,dpi=dpi))
    assert len(images)==260
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=GUIDE.parent,text=True).strip()
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=GUIDE.parent,text=True).splitlines()
    changed=subprocess.check_output(['git','-c','core.quotepath=false','diff','--name-only'],cwd=GUIDE.parent,text=True,encoding='utf-8').splitlines()
    assert head==snapshot['git_head'] and not staged
    assert all(x.startswith('PEMFC指导/') for x in changed)
    record=dict(status='PASS',time=datetime.datetime.now().astimezone().isoformat(),identity='DIAGNOSTIC_ONLY',
      unchanged_original_files=sum(x['status']=='UNCHANGED' for x in check),archived_management_only_fixes=4,
      all_old_evidence_bytes_preserved=True,archived_entries=len(moves['rows']),removed_formal_duplicates=26,
      formal_experiment_program_count=0,numeric_compute_ast_unchanged=True,entry_tests=tests,images=images,
      image_review_scope='全部260张文件结构与300dpi检查；人工查看汇总图、原论文关键页及部分平台原始图，未宣称人工逐张审阅252个平台图',
      git_head=head,staged_changes=staged,git_commit_or_push=False,training=False,full_operation_scan=False)
    save_json(out/'旧证据保护核验.json',check);save_json(out/'最终验收记录.json',record)
    exclude={'最终交付SHA256.json'}
    files=[p for p in sorted(out.rglob('*')) if p.is_file() and p.name not in exclude and '绘图配置缓存' not in p.parts]
    save_json(out/'最终交付SHA256.json',dict(time=record['time'],files=[dict(file=p.relative_to(out).as_posix(),sha256=sha(p)) for p in files],
        code=[dict(file=p.name,sha256=sha(p)) for p in sorted(WORK.joinpath('程序').glob('*.py'))],
        reports=[dict(file=p.relative_to(GUIDE).as_posix(),sha256=sha(p)) for p in [WORK/'平台失败原因分析报告.md',WORK/'健康特征方法复核报告.md',WORK/'使用说明.md',GUIDE/'代码目录整理记录.md',GUIDE/'研究说明.md']]))
    print(json.dumps({k:record[k] for k in ['status','unchanged_original_files','archived_management_only_fixes','archived_entries','formal_experiment_program_count','git_head','training']},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
