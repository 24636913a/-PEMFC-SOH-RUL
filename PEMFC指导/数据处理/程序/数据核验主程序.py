# %%
"""阶段0完整CPU核验入口；不导入旧训练代码，不训练、不拟合健康特征。"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import json
import shutil
import sys
import time
import traceback
from importlib import metadata
import yaml
import pandas as pd
from 核验基础 import sha, table, save_json, Registry, self_test
from 运行与极化 import polarization, fc1_operation, real_operation, align_polar
from 报告与绘图 import create_figures, code_review, make_report


def locate():
    return next(p for p in Path(__file__).resolve().parents if p.name=='PEMFC指导')


def config_path():
    here=Path(__file__).resolve().parent
    for path in (here/'配置/数据核验配置.yaml',here.parent/'配置/数据核验配置.yaml'):
        if path.is_file():return path
    raise FileNotFoundError('必须随完整程序保存配置/数据核验配置.yaml')


def snapshot_old(root):
    names=['README.md','New-PEMFC-Pytorch/New-PEMFC-Pytorch - 副本22,FC1版,最终版Ⅱ.py',
           'RENew-PEMFC-2300h-Pytorch/Before/REAL_Paper_File/F2主程序/raw1s/data.py',
           'RENew-PEMFC-2300h-Pytorch/Before/REAL_Paper_File/F2主程序/raw1s/features.py',
           'RENew-PEMFC-2300h-Pytorch/RealHyFC_Raw1s_Rolling_300to120/reports/LEGACY_RAW_SOURCE_AUTHORITY.json',
           'RENew-PEMFC-2300h-Pytorch/Before/Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv']
    return [dict(file=p,sha256=sha(root/p)) for p in names if (root/p).exists()]


def sync_complete(guide,cfg,source,config,out,verified):
    """阶段1.1新规则：准备程序不再发布到正式实验完整代码。"""
    return dict(status='PREPARATORY_CODE_TASK_LOCAL_ONLY',backup=None,
                policy='阶段1.1目录规则；旧同步实现已完整归档，当前不创建或写入完整代码')


def main(argv=None):
    parser=argparse.ArgumentParser(description='FC1与RealHyFC原始数据阶段0核验，不训练')
    parser.add_argument('--config',type=Path,default=config_path())
    parser.add_argument('--check-only',action='store_true',help='只验证完整程序语法、配置路径和内存合成语义测试')
    args=parser.parse_args(argv)
    guide=locate();cfg=yaml.safe_load(args.config.read_text(encoding='utf-8-sig'))
    root=(guide/cfg['项目根目录']).resolve()
    if not (root/'数据').is_dir():raise RuntimeError('配置未指向现有数据目录，不创建另一个项目')
    for p in (root/cfg['FC1目录'],root/cfg['RealHyFC目录'],root/cfg['历史运行清单']):
        if not p.exists():raise FileNotFoundError(str(p))
    source=Path(__file__).resolve().parent
    syntax=[]
    for p in source.glob('*.py'):
        compile(p.read_text(encoding='utf-8'),str(p),'exec');syntax.append(p.name)
    tested=self_test()
    if args.check_only:
        print(json.dumps(dict(status='PASS',stage=0,syntax=syntax,self_test=tested,full_raw_scan=False),ensure_ascii=False,indent=2));return
    base=(guide/cfg['输出目录']).resolve()
    if not base.is_relative_to(guide):raise ValueError('输出必须位于PEMFC指导内部')
    existing=base.exists() and any(p.is_file() for p in base.rglob('*'))
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    out=base/'历史核验'/stamp if existing else base
    for folder in ('图片','表格'): (out/folder).mkdir(parents=True,exist_ok=True)
    report_dir=out if existing else guide/'数据处理'
    def log(message):
        line=datetime.now().astimezone().isoformat()+' '+message
        print(line,flush=True)
        with (out/'执行日志.txt').open('a',encoding='utf-8') as stream:stream.write(line+'\n')
    manifest=dict(status='RUNNING',version=cfg['版本'],started=datetime.now().astimezone().isoformat(),
        project_root=str(root),guidance=str(guide),output=str(out),training=False,prediction_accuracy=False,
        phase=0,all_domain_use='DESCRIPTIVE_ONLY_NOT_GLOBAL_MODEL_FITTING',
        self_test=tested,syntax=syntax,code_sha256={p.name:sha(p) for p in source.glob('*.py')})
    save_json(out/'运行记录.json',manifest)
    save_json(out/'实际配置.json',cfg)
    old=snapshot_old(root);registry=Registry(root,log);started=time.perf_counter()
    try:
        log('开始阶段0：重新读取原始文件；不导入任何旧训练入口')
        code_review(root,report_dir/'数据读取代码审查报告.md',old)
        coverage,points=polarization(root,cfg,registry,out)
        fc_integrity,fc_plot,fc_gap,fc_summary,fc_all=fc1_operation(root,cfg,registry,out)
        real_integrity,real_plot,real_gap,real_summary,neighbors=real_operation(root,cfg,registry,out,coverage)
        integrity=fc_integrity+real_integrity;gaps=fc_gap+real_gap
        table(out/'表格/数据源文件登记表.csv',registry.public())
        table(out/'表格/老化运行数据完整性检查.csv',integrity)
        table(out/'表格/运行观测间断明细.csv',gaps)
        alignment=align_polar(root,cfg,coverage,neighbors,fc_all)
        table(out/'表格/运行数据与极化测试对应表.csv',alignment)
        log('全量统计完成；绘图仅使用可追溯原始点，无平滑和插值')
        figures=create_figures(cfg,out,fc_plot,real_plot,points,gaps,fc_summary,real_summary)
        status,blocks=make_report(report_dir/'数据核验报告.md',cfg,fc_summary,real_summary,integrity,coverage,alignment,figures,out,cfg['版本'])
        after=snapshot_old(root)
        if after!=old:raise RuntimeError('历史只读来源发生变化；不覆盖或替换它们')
        save_json(out/'历史文件保护检查.json',dict(all_unchanged=True,files=old))
        # 只复核本次派生表和文件身份，不读取缓存代替原始扫描。
        registered=pd.read_csv(out/'表格/数据源文件登记表.csv')
        assert registered.读取成功.eq(True).all() and not registered.原始路径.duplicated().any()
        assert int(registered.loc[registered.数据类型=='Loadcycle运行','实际行数'].sum())==real_summary['rows']
        for figure in figures:
            assert sha(out/'图片'/figure['图片'])==figure['图片SHA256']
        verification=dict(status='PASS',source_files=len(registered),raw_loadcycle_rows=real_summary['rows'],
            figures=len(figures),all_old_sources_unchanged=True,syntax_files=len(syntax),self_test=tested)
        save_json(out/'运行验证记录.json',verification)
        sync=sync_complete(guide,cfg,source,args.config,out,verification) if status!='BLOCKED' else dict(status='NOT_PROMOTED_BLOCKED')
        save_json(out/'完整代码同步记录.json',sync)
        manifest.update(status=status,completed=datetime.now().astimezone().isoformat(),elapsed_seconds=time.perf_counter()-started,
            fc1=fc_summary,realhyfc=real_summary,polar_curves=len(coverage),images=len(figures),blocks=blocks,
            source_files=len(registered),report=str(report_dir/'数据核验报告.md'))
        save_json(out/'运行记录.json',manifest)
        hashes=[dict(file=p.relative_to(out).as_posix(),sha256=sha(p)) for p in out.rglob('*')
                if p.is_file() and p.name not in ('执行日志.txt','输出文件SHA256.json') and '绘图缓存' not in p.parts]
        save_json(out/'输出文件SHA256.json',hashes)
        log('阶段0完成：'+status+'；结果：'+str(out))
    except BaseException:
        manifest.update(status='FAILED',error=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        save_json(out/'运行记录.json',manifest)
        (out/'失败记录.txt').write_text(traceback.format_exc(),encoding='utf-8')
        table(out/'表格/数据源文件登记表_部分.csv',registry.public())
        log('失败现场已保留，不改写历史文件或既有正式版本')
        raise


# %% CLI入口
if __name__=='__main__':
    main()
