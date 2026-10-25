"""阶段1.1独立入口。六组探索性诊断一次运行，不调用神经训练或旧评价入口。"""
from pathlib import Path
import argparse,json,sys,datetime,traceback
import yaml

WORK=Path(__file__).resolve().parent.parent
GUIDE=WORK.parent.parent
sys.path[:0]=[str(GUIDE/'数据处理/程序'),str(GUIDE/'健康特征提取/程序')]
from 核验基础 import sha,save_json
from 极化特征计算 import unit_tests
from 平台诊断计算 import run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');parser.add_argument('--no-platform-plots',action='store_true',help='保留用于工程检查，正式运行不使用');args=parser.parse_args()
    conf=WORK/'配置/诊断冻结规则.yaml';frozen=json.loads((WORK/'配置/诊断计算前冻结记录.json').read_text(encoding='utf-8'))
    assert sha(conf)==frozen['sha256']
    cfg=yaml.safe_load(conf.read_text(encoding='utf-8-sig'))
    assert sha(GUIDE/cfg['原阶段1规则'])==frozen['historical_rules_sha256']
    for p in WORK.joinpath('程序').glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
    tests=unit_tests()
    if args.check_only:print('PASS',tests);return
    if args.no_platform_plots:raise RuntimeError('正式本轮必须生成全部长平台图，不接受跳过')
    out=WORK/'输出结果'
    if (out/'运行记录.json').exists():raise RuntimeError('本轮已执行，停止重复运行；如需新诊断先另行登记')
    record=dict(started=datetime.datetime.now().astimezone().isoformat(),identity='DIAGNOSTIC_ONLY',config_sha256=sha(conf),tests=tests,training=False)
    save_json(out/'运行记录.json',dict(record,status='RUNNING'))
    try:
        values=run(cfg,out)
        from 诊断报告绘图 import deliver
        summary=deliver(WORK,out,cfg,*values)
        record.update(status='DIAGNOSTIC_ONLY_COMPLETED',finished=datetime.datetime.now().astimezone().isoformat(),summary=summary,
                      source_code=[dict(file=p.name,sha256=sha(p)) for p in WORK.joinpath('程序').glob('*.py')])
        save_json(out/'运行记录.json',record)
        print('DIAGNOSTIC_ONLY_COMPLETED',flush=True)
    except Exception:
        save_json(out/'故障记录.json',dict(record,error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
