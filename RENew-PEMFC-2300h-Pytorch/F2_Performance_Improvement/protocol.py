"""Freeze finite experiments, legal Train/Validation identities and history protection."""
from datetime import datetime,timezone
from pathlib import Path
import importlib.metadata
import os
import subprocess
import numpy as np
import pandas as pd
import torch
import frozen_f2 as f
import settings as s
from io_utils import js,csv,text

MATH_FILES=['frozen_f2.py','settings.py','diagnostics.py','train.py','io_utils.py','selection.py','stage_a.py']


def history_snapshot(run):
    repo=f.ROOT.parent
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode('utf-8').rstrip('\0').split('\0')
    selected=[p for p in paths if p.startswith('RENew-PEMFC-2300h-Pytorch/') and (repo/p).is_file()]
    snapshot={}
    for index,name in enumerate(selected):
        snapshot[name]=f.sha(repo/name)
        if index%10000==0:print('Historical SHA protection',index,'/',len(selected),flush=True)
    js(run/'HISTORY_BEFORE.json',snapshot)
    return len(snapshot)


def verify_history(run):
    before=f.read_json(run/'HISTORY_BEFORE.json');repo=f.ROOT.parent
    for index,(name,digest) in enumerate(before.items(),start=1):
        f.require(f.sha(repo/name)==digest,'HISTORY_CHANGED:'+name)
        if index%10000==0:print('Historical SHA unchanged',index,'/',len(before),flush=True)
    js(run/'HISTORY_PROTECTION.json',dict(status='PASS',unchanged_SHA256_files=len(before)))


def prepare():
    f.model_identity();environment=f.configure_environment()
    frame,windows,scaler,prepared,excluded=f.load_selection_data()
    run=f.HERE/'outputs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_F2_improvement')
    run.mkdir(parents=True,exist_ok=False)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=f.ROOT.parent).decode().strip()
    age=frame.bin_end_ophrs2_h;v=frame.cell_voltage_mean
    valid=frame.bin_valid_for_windows.eq(1)&frame.bin_valid_for_windows.shift().eq(1)
    same=frame.temporal_segment_id.eq(frame.temporal_segment_id.shift())&age.diff().sub(.5).abs().lt(1e-6)
    jumps=v.diff().abs().loc[:2689];eligible=(valid&same).loc[:2689]
    threshold=float(jumps.loc[eligible].quantile(.95))
    protocol=dict(base_git_sha=sha,data_sha256=f.sha(f.SOURCE),official_F2_sha256=f.sha(f.OFFICIAL),
        reference_run=f.REFERENCE_RUN.relative_to(f.ROOT).as_posix(),raw_rows=3843,
        boundaries=dict(TRAIN=[0,2690],VALIDATION=[2690,3074],EXPLORATORY_HOLDOUT=[3074,3843]),
        history=48,horizon=24,resolution_h=.5,Top10=f.MASKS['top10'],EWMA_alpha=.3,
        target='raw DeltaV',R2_target='raw DeltaV',scaler='Train legal raw histories only',
        initial_seeds=list(s.INITIAL_SEEDS),final_seeds=list(s.FINAL_SEEDS),A_candidates=s.CONFIGS_A,
        optimizer='AdamW',weight_decay=s.WEIGHT_DECAY,batch_size=32,max_epochs=450,patience=40,
        gradient_clip_norm=1.,early_stop_min_delta=1e-12,qualification=s.QUALIFICATION,
        B_gate=s.STAGE_B_GATE,C_gate=s.STAGE_C_GATE,selection=s.SELECTION_RULE,
        max_fresh_trainings=s.MAX_FRESH_TRAININGS,
        event_definition=dict(Train_jump_P95_V=threshold,formula='startup OR shutdown OR 0<zero_fraction<1 OR same-segment abs adjacent voltage jump >= Train P95',
            weighted_MSE='mean((1+event)/mean_Train_queries(1+event) * squared_error)',
            training_weight=2.,future_conditions_are_never_predictor_inputs=True),
        model_selection_partitions=['TRAIN','VALIDATION'],holdout_access='No windows, transformed inputs, inference or new metrics before final configuration and six new weights are frozen',
        ordinary_CUDA=dict(deterministic_algorithms=False,cudnn_deterministic=True,cudnn_benchmark=False,
            matmul_tf32=False,cudnn_tf32=True,CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
        environment=environment,partitions={})
    for part,w in windows.items():
        protocol['partitions'][part]=dict(origins=len(w.origins),queries=int(w.targets.size),unique_targets=int(np.unique(w.targets).size),
            input_sha256=f.array_sha(prepared[part]),labels_sha256=f.array_sha(w.y),origins_sha256=f.array_sha(w.origins),targets_sha256=f.array_sha(w.targets))
        csv(run/f'windows/{part}.csv',dict(origin_row=w.origins,history_first_row=w.origins-47,first_target_row=w.targets[:,0],
            last_target_row=w.targets[:,-1],origin_age_h=w.origin_age_h))
    js(run/'PROTOCOL.json',protocol);csv(run/'EXCLUDED_SELECTION_WINDOWS.csv',excluded)
    js(run/'PREPROCESSOR.json',dict(mean=scaler.mean.tolist(),scale=scaler.scale.tolist(),target_scale_V=scaler.target_scale,
        Top10=f.MASKS['top10'],EWMA_alpha=.3))
    packages={name:importlib.metadata.version(name) for name in ['torch','pytorch-forecasting','lightning','numpy','pandas','matplotlib']}
    js(run/'ENVIRONMENT.json',dict(packages=packages,**environment))
    js(run/'CODE_BEFORE_A.json',{name:f.sha(f.HERE/name) for name in MATH_FILES})
    history_snapshot(run)
    js(run/'P0_ACCEPTANCE.json',dict(status='PASS',Train_origins=1750,Validation_origins=218,
        historical_inputs_labels_and_time_order_identical=True,no_holdout_windows_constructed=True))
    js(f.HERE/'ACTIVE_RUN.json',dict(run=run.name))
    text(run/'00_冻结协议.md','''# F2性能提升：预先冻结的有限预算

3843行，Train[0,2690)、Validation[2690,3074)、探索性后留[3074,3843)。固定Top10、48→24、0.5h、因果EWMA alpha=.3、Train-only scaler、原始ΔV及其R²。研究阶段只构造1750个Train和218个Validation窗口；不构造或推理后留窗口。

阶段A：A0原始配置、A1仅学习率改.0006、A2仅Dropout改.05，分别Seed42/202/212，共9次新训练。450轮、patience40、AdamW/MSE、batch32、clip1不变。相同种子共同Epoch的采样顺序必须一致。固定预算，不重跑差种子、不搜索其他参数。

候选初步升级要求：三个种子Validation RMSE全部优于同阶段从头训练对照，平均降幅≥5%；平均MAE不增加；平均P99≤对照1.05倍，任一种子的MAE或P99不超过对照1.10倍；所有梯度有限。该门槛是探索性筛选规则，不是统计显著性。

阶段B只有A完成后登记。保留TFT/LSTM/CNN/历史门控，仅比较一个有界残差融合与同配置的原始融合，各3次全新训练。若A0两路径确有非零梯度、门控尚未接近全LSTM，且融合RMS相对LSTM RMS平均低于.98，才有继续检验衰减假设的资格。参数尺度/相关性不构成因果证明。B升级还要求关闭CNN补充后Validation RMSE在≥2/3种子恶化，不能以退化成LSTM路径替代双路径贡献。

阶段C最多一个Train事件加权MSE候选，只在A、B均完成且已有合格前置候选、Validation事件目标≥5个时考虑。阈值来自Train，事件权重固定2并按Train查询均值归一化；与普通MSE各3次新训练。同样按无权重全Validation RMSE早停。升级还要求事件RMSE至少下降5%、非事件RMSE恶化≤2%、P99不升高。

阶段D只冻结最多一个满足规则、且相对本轮A0也具一致收益的方案；以三种子Validation平均RMSE最低选择，同差≤1e-6mV时选较早/简单方案。只有合格才六种子从头训练。全部六个最佳权重冻结后，才进行一次探索性后留评价；若不合格，后留结果记NA，不强行升级。

上限27次新训练：A9+B6+C6+D6。未触发的阶段如实记录，不补造六种子成绩。普通CUDA存在数值重复性限制；增加的只读梯度探针不改变原生前向公式，但不承诺数值轨迹与无探针旧运行逐位一致。六种子曾经按旧F2表现筛选，当前结果也不是无偏随机初始化总体。

最后20%已受历史研究反馈影响，所有后留比较都只能标为探索性。绝不根据新后留成绩回到A/B/C改方案。
''')
    return run,frame,windows,scaler,prepared
