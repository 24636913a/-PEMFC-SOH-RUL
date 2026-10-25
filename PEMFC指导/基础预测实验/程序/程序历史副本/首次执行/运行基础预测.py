"""从冻结观测表执行真实基础预测；不调用任何准备脚本或历史预测缓存。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, platform
import numpy as np
import pandas as pd
from 基础预测核心 import continuous, sparse, rul, metrics


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def js(p,x):Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(p,d):d.to_csv(p,index=False,encoding='utf-8-sig')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();work=args.work.resolve();out=(args.output or work/'输出结果').resolve()
    sentinel=out/'基础预测完成.json'
    if sentinel.exists():raise RuntimeError('已有完成结果，不覆盖；请指定新的 --output')
    tab=out/'表格';models=out/'模型记录'
    for p in [tab,models]:p.mkdir(parents=True,exist_ok=True)
    cfgpath=work/'配置/阶段2A冻结协议.json';cfg=json.loads(cfgpath.read_text(encoding='utf-8'))
    prep=json.loads((work/'输出结果/数据准备完成.json').read_text(encoding='utf-8'))
    # Require the exact pre-forecast frozen protocol and observation contract.
    freeze=json.loads((work/'配置/执行前冻结.json').read_text(encoding='utf-8'))
    assert sha(cfgpath)==freeze['协议SHA256']
    inputs={f:sha(work/'数据准备'/f) for f in ['FC1小时观测.csv','RealHyFC循环观测.csv','稀疏诊断观测.csv']}
    contract=work/'配置/预测输入合同.json'
    if contract.exists():assert json.loads(contract.read_text(encoding='utf-8'))==inputs
    else:js(contract,inputs)
    fc=pd.read_csv(work/'数据准备/FC1小时观测.csv');real=pd.read_csv(work/'数据准备/RealHyFC循环观测.csv')
    diag=pd.read_csv(work/'数据准备/稀疏诊断观测.csv')
    print('正在执行冻结的持续值和局部线性预测',flush=True)
    allpred=[];allmet=[]
    for dataset,obs in [('FC1',fc),('RealHyFC',real)]:
        c=cfg[dataset]
        pred,ex,logs=continuous(obs,dataset,c['历史长度'],c['历史有效点至少'],c['跨度'])
        csv(tab/f'{dataset}基础预测逐起点结果.csv',pred)
        csv(tab/f'{dataset}连续预测排除记录.csv',ex);csv(models/f'{dataset}局部模型参数.csv',logs)
        met=metrics(pred,['dataset','scheme','partition','horizon','method'])
        # Explicit NA rows for every missing prespecified evaluation cell.
        for scheme in obs.scheme.unique():
            for part in ['Train','Validation','Assessment']:
                for h in c['跨度']:
                    for method in ['Persistence','LocalLinear']:
                        if met.empty or not ((met.scheme==scheme)&(met.partition==part)&(met.horizon==h)&(met.method==method)).any():
                            met=pd.concat([met,pd.DataFrame([dict(dataset=dataset,scheme=scheme,partition=part,horizon=h,
                                method=method,n=0,reason='NO_COMPLETE_VALID_TARGET_BLOCK_OR_BASE')])],ignore_index=True)
        csv(tab/f'{dataset}连续预测指标.csv',met);allmet.append(met);allpred.append(pred)
    print('正在执行下一次真实诊断预测及秩检查',flush=True)
    sp,fair,spmodels=sparse(diag,cfg['稀疏预测']['alpha'])
    csv(tab/'稀疏诊断全部预测与NA.csv',sp);csv(tab/'一维二维健康特征公平对比.csv',fair)
    js(models/'稀疏回归逐起点参数.json',spmodels)
    sm=metrics(sp[sp.pred_D.notna()],['dataset','scheme','method'])
    fm=metrics(fair,['dataset','scheme','method'])
    csv(tab/'稀疏诊断全部起点指标.csv',sm);csv(tab/'稀疏诊断公平交集指标.csv',fm)
    print('正在执行FC1条件阈值RUL及删失核验',flush=True)
    rc=cfg['RUL'];r,c,sel=rul(fc,rc['起点_h'],rc['敏感性下降比例'],rc['线性历史候选_h'],rc['外推上限_h'])
    csv(tab/'FC1_RUL阈值与删失结果.csv',r);csv(tab/'FC1_RUL外推曲线.csv',c);js(models/'RUL历史内部选择记录.json',sel)
    soh=pd.concat([pd.concat(allmet).assign(task='CONTINUOUS_OPERATION'),sm.assign(task='SPARSE_DIAGNOSTIC_ALL'),
                   fm.assign(task='SPARSE_DIAGNOSTIC_FAIR')],ignore_index=True)
    csv(tab/'SOH评价结果.csv',soh)
    hashes={str(p.relative_to(out)):sha(p) for folder in [tab,models] for p in folder.rglob('*') if p.is_file()}
    js(sentinel,dict(UTC=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(cfgpath),input_sha256=inputs,
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['基础预测核心.py','运行基础预测.py']},
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,output_sha256=hashes,
        identity='EXPLORATORY_OFFLINE_BASELINES_NO_NEURAL_TRAINING'))
    print('基础预测完成；已保留所有NA及排除原因',flush=True)


if __name__=='__main__':main()
