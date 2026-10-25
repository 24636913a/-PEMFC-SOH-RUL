"""阶段2C一次性预登记、既有观测复制和历史保护；不拟合模型。"""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

TASK=Path(__file__).resolve().parents[1]
GUIDE=TASK.parents[1]
REPO=GUIDE.parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    frozen=TASK/'配置/阶段2C冻结协议.json'
    if frozen.exists():raise RuntimeError('ALREADY_FROZEN: 不覆盖已有登记')
    for d in ['配置','数据准备','输出结果/图片','输出结果/表格','输出结果/模型记录','输出结果/绘图数据']:(TASK/d).mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    protected=[]
    for p in sorted(GUIDE.rglob('*')):
        if not p.is_file() or TASK in p.parents or any(x in p.parts for x in ['__pycache__','绘图缓存']) or p.name.startswith('~$'):continue
        protected.append(dict(path=p.relative_to(REPO).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    pd.DataFrame(protected).to_csv(TASK/'配置/历史成果保护SHA256.csv',index=False,encoding='utf-8-sig')
    protocol=dict(version='阶段2C_v1',head=head,utc=datetime.now(timezone.utc).isoformat(),
      identity='标量运行性能探索性低阶迁移；既有Assessment已查看；非盲测、非二维迁移、非真实RUL',
      source=dict(dataset='FC1',partition='Train',scheme='FC1_NOMINAL',j_A_cm2=.70,coordinate='actual hour',lookback=48,min_history=24,dt_ref_h=1),
      target=dict(dataset='RealHyFC',scheme='B60',j_A_cm2=.558161974,coordinate='completed available_h',lookback_raw_cycles=12,min_history=6,min_fit=12,dt_ref_h=12),
      scenarios=[dict(name='A_严格300h',cutoff_h=300.,diagnostic_nominal_h=[0]),dict(name='B_严格600h',cutoff_h=600.,diagnostic_nominal_h=[0,300]),dict(name='C_后移629h',cutoff_h=629.2038,diagnostic_nominal_h=[0,300,600])],
      state='仅最后12个原始循环内合法观测初始化2B型状态；全截止前历史拟合统计。A/B/C从最后实测d锚定；到截止点用已知历史时间传播，不接触未来观测。',
      methods=['A_Persistence','A_TargetBounded','B_FC1Direct','C_FC1PriorAdapted','D_Target2B_Bounded','D_Target2B_SlowReversible'],
      prior_weight_source=.25,
      rate='d=1-SOH dimensionless; global OLS(d,actual past h) clipped [0,Q95(abs(local rates))]; source FC1 Train only; C=.25*q_source+.75*q_target',
      local='FC1过去48h至少24点；Real过去12个原始cycle ID至少6点；均对真实过去运行小时OLS，缺失不填',
      floors=dict(rate_V=1e-6,acceleration_V=1e-8,noise_V=1e-5),
      statistics='rate cap=max(Q95(abs local d/h),floor_V/(base_V*dt_ref)); acceleration=max(.25*Q95(abs diff local rates / diff actual h),floor_V/(base_V*dt_ref^2)); noise=MAD(diff d)/(.6744897501960817*sqrt(2)) with V/base floor; reversible=max(std(d-causalEWMA),noise); EWMA tc=lookback*dt_ref/2',
      bounded_k=[.25,1.],reversible_tau_h=[12.,36.,72.],
      inner='各截止前合法历史按顺序前80%拟合，最后20%为单一起点固定预测选择；未来名义12h/cycle，不读实际未来lead；按RMSE、MAE、候选次序；随后全部截止前历史重拟合。',
      D_model='x=[L,q,R]; F=[[1,dt,0],[0,1,0],[0,0,exp(-dt/tau)]]; H=[1,0,1]; Joseph update; q projected to [0,cap] and |dq|<=accel*dt; P0=diag(sR^2,(.25cap)^2,sR^2); Q=diag((.05noise)^2*dt/dt_ref,(accel*dt_ref)^2*dt/dt_ref,sR^2*(1-exp(-2dt/tau))); no future observation update',
      units='2B型算法重新按d/h参数化；不是原2B以V/cycle计算结果的复用。Q的参考步长显式12h，保留量纲。',
      nominal_future='从实际截止点重新锚定名义循环排程：target_nominal_h=cutoff_h+12*(target_id-last_completed_cycle_id)。截止时未完成循环的相位未知，不推测剩余时长。真实未来时长仅用于误差和时间审计。',
      evaluation=dict(common_after_raw_cycle=51,common_after_h=629.2038,primary='所有真实合格共同未来目标；每个场景一个固定预测起点；同一目标仅一个查询',strata='保留旧partition标签，分别报告；这些标签不再决定本轮拟合样本',auxiliary_complete_blocks=[1,4,12,25],block_rule='从原始cycle52起完整前缀，任何缺失则该块NA；不补齐。主共同未来逐点评价仍保留全部合法观测。'),
      pairing=dict(max_calendar_lag_h=48,min_pairs_1d=6,min_pairs_2d=9,max_standardized_condition=100,min_2d_transitions=6,FC1_clock='NOMINAL_ONLY',release='UNKNOWN；诊断完成后可用仅离线回放假设'),
      selection='不使用共同未来、旧Assessment或后期诊断选择迁移权重、参数或方法；所有模型和场景均报告；无后续搜索',
      scope='不训练网络，不更改0-2B，不重解析原始860万CSV；只复用冻结观测和缓存时间邻点',
      git='延续用户自动Git授权，仅本地提交，不推送；不改旧文件')
    dump(frozen,protocol)
    dump(TASK/'配置/预登记时间与SHA256.json',dict(utc=protocol['utc'],head=head,protocol_sha256=sha(frozen),before_fitting=True))
    sources=[]
    for name in ['FC1小时观测.csv','RealHyFC循环观测.csv','稀疏诊断观测.csv']:
        src=GUIDE/'基础预测实验/数据准备'/name;dst=TASK/'数据准备'/name
        if dst.exists():raise RuntimeError('COPY_TARGET_EXISTS')
        shutil.copy2(src,dst);assert sha(src)==sha(dst)
        sources.append(dict(source=src.relative_to(REPO).as_posix(),local=dst.relative_to(TASK).as_posix(),sha256=sha(src)))
    # 缓存只随机读取两个截止点的邻点；不重跑原始数据审计。
    run=REPO/'RENew-PEMFC-2300h-Pytorch/RealHyFC_Raw1s_Rolling_300to120/audits/Startup_Precursor_Audit/runs/20261022T071957_960946Z_b1590612'
    manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8-sig'))
    n=manifest['rows'];cols=manifest['columns']
    values=np.memmap(run/'cache/values.bin',dtype=np.float64,mode='r',shape=(n,len(cols)))
    time=np.memmap(run/'cache/time_ns.bin',dtype=np.int64,mode='r',shape=n)
    files=np.memmap(run/'cache/file_id.bin',dtype=np.int16,mode='r',shape=n)
    rows=np.memmap(run/'cache/source_row.bin',dtype=np.int64,mode='r',shape=n)
    age=values[:,cols.index('OpHrs2 [h]')];clocks=[]
    for cutoff in [300.,600.]:
        i=int(np.searchsorted(age,cutoff,side='right'))
        for pos,role in [(i-1,'LAST_LE_CUTOFF'),(i,'FIRST_GT_CUTOFF')]:
            clocks.append(dict(cutoff_h=cutoff,role=role,cache_row=pos,actual_h=float(age[pos]),calendar=pd.Timestamp(int(time[pos])).isoformat(),file_id=int(files[pos]),source_row=int(rows[pos]),source_path=manifest['source_files'][int(files[pos])]['path'],meaning='运行小时截止的日历包络；不插值成精确测量时间'))
    pd.DataFrame(clocks).to_csv(TASK/'数据准备/严格截止日历包络.csv',index=False,encoding='utf-8-sig')
    sources.append(dict(source=(run/'source_manifest.json').relative_to(REPO).as_posix(),sha256=sha(run/'source_manifest.json'),use='仅搜索时钟邻点；cache只读；未扫描原始CSV'))
    for rel in ['step1/燃料电池三档研究方案.docx','物理约束退化演化/阶段2B/配置/阶段2B冻结协议.json','物理约束退化演化/阶段2B/程序/低阶演化核心.py','物理约束退化演化/阶段2B/输出结果/表格/后续300_600及后移起点协议.csv','物理约束退化演化/阶段2B/输出结果/表格/诊断与运行历史配对及排除.csv']:
        p=GUIDE/rel;sources.append(dict(source=p.relative_to(REPO).as_posix(),sha256=sha(p)))
    dump(TASK/'数据准备/来源合同与SHA256.json',sources)
    (TASK/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    (TASK/'.gitignore').write_text('__pycache__/\n输出结果/绘图缓存/\n',encoding='utf-8')
    print('PRE_REGISTERED',sha(frozen),'protected',len(protected))
if __name__=='__main__':main()
