"""Synthetic-only real TFT/F2 engineering acceptance; never a training runner."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import traceback
import unittest

import p1_safety as safety

PHASE=safety.PHASE
REPO=safety.REPO
PROTOCOL=PHASE/'protocol/P1_EXECUTION_PROTOCOL_DRAFT.json'
OUTPUT=PHASE/'outputs'

def digest(payload):
    return hashlib.sha256(payload).hexdigest()

def write_json(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(PHASE):
        raise PermissionError('P1_OUTPUT_PATH_ONLY')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO)

def source_identity():
    import yaml
    paths={p:(REPO/(safety.ROOT+p)) for p in safety.SOURCE_NAMES}
    sources={}
    for name,path in paths.items():
        observed=path.read_bytes()
        expected=git('show',safety.BASE+':'+safety.ROOT+name)
        if observed!=expected:
            raise RuntimeError('FROZEN_BASE_SOURCE_CHANGED:'+name)
        sources[safety.ROOT+name]={'sha256':digest(observed),'source_commit':safety.BASE,'kind':'STATIC_SOURCE_CONFIG_OR_METADATA_ONLY'}
    hpath=paths[safety.SOURCE_NAMES[0]]
    h_sha=digest(hpath.read_bytes())
    if h_sha!=paths[safety.SOURCE_NAMES[1]].read_text(encoding='utf-8').split()[0]:
        raise RuntimeError('H_PROTOCOL_HASH_MISMATCH')
    cfg=yaml.safe_load(hpath.read_text(encoding='utf-8'))
    ppath=paths['Phase_P_Final_Innovation_and_Experiment_Planning/protocol/P_PROPOSED_GATE_PROTOCOL.json']
    p_sha=digest(ppath.read_bytes())
    if p_sha!=ppath.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]:
        raise RuntimeError('P_PROTOCOL_HASH_MISMATCH')
    prior=json.loads(ppath.read_text(encoding='utf-8'))
    for path,expected in cfg['code_identity']['files'].items():
        if path.startswith('C:'):
            observed=digest(Path(path).read_bytes())
            if observed!=expected:
                raise RuntimeError('UPSTREAM_TFT_IDENTITY_CHANGED:'+path)
            sources[path]={'sha256':observed,'kind':'INSTALLED_FROZEN_TFT_SOURCE'}
        elif path in sources and sources[path]['sha256']!=expected:
            raise RuntimeError('H_CODE_SHA_MISMATCH:'+path)
    local={str(p.relative_to(PHASE)).replace('\\','/'):digest(p.read_bytes())
           for folder in ('scripts','tests') for p in sorted((PHASE/folder).glob('*.py'))}
    models=json.loads(paths[safety.SOURCE_NAMES[-1]].read_text(encoding='utf-8'))['models']
    if len(models)!=9:
        raise RuntimeError('HISTORICAL_METADATA_MODEL_COUNT_MISMATCH')
    return cfg,prior,h_sha,p_sha,sources,local,models

def protocol_candidate(cfg,prior,h_sha,p_sha,sources,local):
    return {'phase':PHASE.name,'base_commit':safety.BASE,'mode':'ENGINEERING_PREFLIGHT_ONLY',
        'training_authorized':False,'optimizer_steps_allowed':0,'real_data_load_allowed':False,
        'assessment_access_allowed':False,'test_access_allowed':False,'historical_checkpoint_load_allowed':False,
        'paper_body_authorized':False,'H_protocol_sha256':h_sha,'P_protocol_sha256':p_sha,
        'groups':prior['groups'],'task':cfg['task'],'data':cfg['data'],
        'preprocessing':cfg['preprocessing'],'hyperparameters':cfg['hyperparameters'],
        'normal_runtime':cfg['normal_runtime'],'dataloader':cfg['dataloader'],'seeds':[42,52,62],
        'early_stopping':{'monitor':'validation_standardized_RMSE','improves':'validation_standardized_RMSE < best - 1e-12',
                          'patience':40,'max_epochs':450,'scheduler':None,'each_group_own_best_checkpoint':True,
                          'P_vague_monitor_clarified_without_editing_P':True},
        'prediction':{'entry':'AST-extracted original E.predict -> original C.predict; H batch32',
                      'eval':True,'grad_mode':'torch.no_grad','batch_size':32,'dtype':'float32',
                      'inference_mode':False,'AMP':False,'requires_grad_mutation':False,
                      'initial_group_atol_standardized':1e-7,'initial_group_rtol':0.,'synthetic_reload':'np.array_equal EXACT'},
        'initialization':{'fresh_core_template':'G0 constructed in isolated RNG seeded run seed',
                          'fresh_gate_template':'original G2 gate constructed in isolated RNG seeded run seed + 100000',
                          'template_source':'FRESH_SYNTHETIC_INITIALIZATION_NOT_HISTORICAL_WEIGHTS',
                          'full_CPU_CUDA_Python_NumPy_RNG_restored':True,'shared_core_within_seed':True,
                          'shared_G1_G2_gate':True,'initial_gate_output':0.5,'no_change_to_requires_grad':True},
        'synthetic_checks':{'input_shape':[65,48,14],'output_shape':[65,24],'input_generator_seed':991,
                            'mask_zero_indices':[1,2,5,12],'index_sample_count':1380,'index_sampling_epochs':2,
                            'optimizer_created':False,'backward_only':True,'synthetic_checkpoint_directory':'runtime/',
                            'manual_gate_weight_fixture_only':'synthetic context response diagnostic; never optimizer update'},
        'future_execution_requirements':{'separate_explicit_authorization_required':True,'runs':9,
                            'per_epoch_actual_origin_order_log_required':True,'common_epoch_orders_must_match':True,
                            'same_seed_initial_CPU_CUDA_Python_NumPy_loader_and_dropout_required':True,
                            'N1_contract':'read_csv(nrows=3007), then bin_end_ophrs2_h < 1819.997738 yields 3005',
                            'N1_loaded_here':False},
        'success_criteria_inherited_unchanged':prior['success_proposal'],
        'evaluation_inherited_unchanged':prior['evaluation'],
        'scientific_boundaries':['validation reused: exploratory, not independent confirmation',
                                '57 parameters do not imply equal effective functional degrees',
                                'engineering thresholds are not statistical significance or electrochemical causality'],
        'frozen_source_identity':sources,'P1_executed_code_identity':local,
        'stop':'technical PASS permits asking for authorization only; no automatic real training or paper writing'}

class EvidenceResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.records=[]
    def addSuccess(self,test):
        super().addSuccess(test);self.records.append({'test':test.id(),'status':'PASS'})
    def addFailure(self,test,err):
        super().addFailure(test,err);self.records.append({'test':test.id(),'status':'FAIL','error':self._exc_info_to_string(err,test)})
    def addError(self,test,err):
        super().addError(test,err);self.records.append({'test':test.id(),'status':'ERROR','error':self._exc_info_to_string(err,test)})

def reports(audit,evidence):
    ready=audit['final_status']=='GATE_TRAINING_TECHNICALLY_READY_FOR_AUTHORIZATION'
    rows=[]
    for seed,r in evidence.items():
        rows.append(f"| {seed} | 332379 / 332436 / 332436 | {r['initial_prediction_max_abs_difference']['G1']:.12g} | {r['initial_prediction_max_abs_difference']['G2']:.12g} | {r['initial_prediction_exact']} |")
    table='| Seed | G0 / G1 / G2 参数 | G1−G0 最大绝对差 | G2−G0 最大绝对差 | 逐点严格相等 |\n|---|---|---|---|---|\n'+'\n'.join(rows)
    texts={
      'P1_MODEL_IMPLEMENTATION_AUDIT.md':f'''# 模型实现验收

仅工程合成检查，没有真实性能结果。状态：{audit['checks']['model_structure_and_parameters']}。

G0 直接复用冻结 F1；G2 直接复用冻结 F2；G1 仅继承 G2 并覆盖 `regime_context` 为固定 `[1,1,1,1,1]`。没有复制三套 TFT，也没有改变 encoder、decoder 或 24 步 ΔV 输出。

{table}

核心 state_dict 键、形状、逐张量初始值与可训练标志均核验；G1/G2 门控结构和全部 57 个参数相同。G0 没有无效门控参数。初始输出层零权重/偏置使 g=0.5。非优化 backward 验证核心和门控输出有有限、非零梯度；所有 state_dict 在 backward 后保持原值。

另外只在合成 fixture 中手动设置非零门控输出，检验 G1 始终全局共享，G2 随历史上下文改变。该 fixture 不用于初始化、保存前预测或科研结果。57 参数相同不代表两组有效函数自由度相同。
''',
      'P1_RNG_INITIALIZATION_AUDIT.md':f'''# 初始化、RNG 与采样验收

状态：{audit['checks']['initialization_and_RNG']}。Seed42/52/62，全部是 fresh random initialization，无历史 checkpoint 热启动。

每个 Seed 的 G0 fresh 核心模板、隔离 RNG 下 Seed+100000 的原始 G2 门控模板在三组中共享。所有构造调用保存并恢复 CPU/CUDA/Python/NumPy RNG；正序、逆序构造逐点一致。这个 gate RNG offset 只隔离初始化流，不构成模型调参。

创建和复制完成后，各组恢复同一 run-seed RNG；独立 DataLoader generator 以 run seed 初始化。真实 PyTorch DataLoader 对合成索引 0..1379 的前两个完整 epoch 逐项核对，未读取真实窗口。dropout probe 使用相同 CUDA RNG 且保存/恢复状态，三组 mask 一致。

完整初始核心/门控、CPU/CUDA/Python/NumPy/loader 状态 SHA256，以及两个 epoch 全序列和 SHA256 均保存于 `outputs/P1_SYNTHETIC_MODEL_EVIDENCE.json`。未来授权训练必须逐 epoch 记录实际 origin 顺序并核对共同 epoch；本次索引模拟不能替代未来真实运行审计。H 不宣称 CUDA 严格位级训练确定性。
''',
      'P1_EARLY_STOPPING_AND_RUNTIME_AUDIT.md':f'''# 原 H 协议与运行语义

状态：{audit['checks']['H_training_protocol_consistency']}。

直接 AST 核验原 H：`validation_standardized_RMSE < best - 1e-12`。监控标准化 RMSE，不是 mV RMSE；每组自身最佳 checkpoint；max450、patience40、clip1.0，无 scheduler。AdamW lr=0.0009605449687795866、weight_decay=6.95942232671908e-6、batch32 保持不变。本次没有构造优化器或执行 epoch。

Raw F2 48×14→24 ΔV、冻结 Top10、标准化空间零掩码、逐窗口因果 EWMA alpha0.3、Train-only scaler、Train [0,2103)、Validation [2103,2404)、1380/205 窗口与 H 原协议相同。这些仅从配置/源码核对，本次不运行真实数据构建器。

P 的早停描述没有明确空间，P1 只在新 draft 明确 H 的真实 standardized 语义，不修改 P 历史文件。将来 N1 必须 nrows3007 后年龄过滤为3005，禁止直接 nrows3005 或完整文件哈希；此处读取次数为0。

实际调用 AST 提取的 H.configure_normal/environment，环境与 H 全部字段一致；记录于执行审计。原 E.predict→C.predict 使用 eval + no_grad + batch32；没有 inference_mode、AMP、requires_grad 改写或 TF32 改动。
''',
      'P1_CHECKPOINT_AND_PREDICTION_PARITY.md':f'''# 合成保存、重载与预测一致性

状态：{audit['checks']['synthetic_reload_and_prediction']}。

65 条合成输入按 32/32/1 批预测，输出65×24。9 个新合成 state_dict 仅写入 P1/runtime/{audit['attempt_id']}/，严格 group/seed/protocol SHA 身份校验后，以同结构重载。每组重载前后必须 `np.array_equal`，不能放宽 H 的严格重载要求。临时权重和预测只在本地 runtime，Git忽略；哈希与路径保留在模型证据 JSON。

跨组三组初始 eval 比较预先冻结 atol=1e-7 standardized、rtol=0，并单独报告是否严格相同：

{table}

历史九个正式 checkpoint 未打开、未加载、未哈希、未覆盖、未重保存；只读 L 的小型模型 manifest 用作身份元数据。合成结果只证明实现一致性，不是预测性能或科研结论。
''',
      'P1_FINAL_TRAINING_READINESS_DECISION.md':f'''# 最终工程准入决定

**{audit['final_status']}**

| 验收轴 | 状态 |
|---|---|
'''+''.join(f'| {k} | {v} |\n' for k,v in audit['checks'].items())+f'''
实际执行 {audit.get('tests_run',0)} 项 unittest；证据为测试日志、逐测试状态、初始化/RNG/sampler 哈希和真实合成模型重载记录。成功不等于完成正式消融。

是否可以申请九次正式训练授权：{'YES' if ready else 'NO'}。
训练授权仍为 false，正式训练0次、optimizer更新0次、真实数据推理0次、Test访问0次。入口不能切换到训练模式。

继承 P 的 G2−G1 主比较及 G2−G0 对照、max(0.5mV,5%control mean)改进、3/3 Seed配对优势及endpoint/target-time/MAE/P99限制；不因合成结果调整。真实验证已经复用，未来结论只能是探索性工程证据。技术失败、平均改善不足、指标依赖及全部工程门槛通过应区分；不证明电化学因果或统计显著性。若未来未证明门控价值，不增加第四组或修改门控。

剩余技术阻断：{audit.get('exception','NONE')}。
完成后停止；等待研究负责人另行授权，不启动训练，不写论文正文。
'''}
    for name,text in texts.items():
        path=PHASE/'reports'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding='utf-8',newline='\n')

def main():
    parser=argparse.ArgumentParser(description='Engineering-only; no real training mode exists.')
    parser.add_argument('--freeze-engineering-draft',action='store_true',help='Freeze/revise technical draft; never authorizes training.')
    args=parser.parse_args()
    safety.install_repo_firewall()
    attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    audit={'phase':PHASE.name,'base_commit':safety.BASE,'attempt_id':attempt,
           'mode':'ENGINEERING_PREFLIGHT_ONLY','training_authorized':False,'optimizer_steps':0,
           'real_training_runs':0,'real_data_inference':0,'N1_read_or_full_hash':False,
           'Assessment_prediction_read':False,'test_accessed':False,'raw_data_accessed':False,
           'historical_checkpoint_loaded_or_hashed':False,'historical_phase_modified':False,
           'synthetic_backward_only':True,'paper_written':False,
           'checks':{k:'FAIL' for k in ('model_structure_and_parameters','initialization_and_RNG',
                       'H_training_protocol_consistency','synthetic_reload_and_prediction','data_access_firewall')},
           'final_status':'GATE_TRAINING_PREFLIGHT_BLOCKED'}
    evidence={};test_record={'tests':[],'passed':False};log=io.StringIO()
    try:
        head=git('rev-parse','HEAD').decode().strip()
        git('merge-base','--is-ancestor',safety.BASE,'HEAD')
        changed=git('diff','--name-only','HEAD').decode().splitlines()
        if any(not p.startswith(safety.ROOT+PHASE.name+'/') for p in changed):
            raise RuntimeError('UNRELATED_TRACKED_WORKTREE_CHANGE')
        audit['execution_HEAD']=head
        cfg,prior,h_sha,p_sha,sources,local,models=source_identity()
        candidate=protocol_candidate(cfg,prior,h_sha,p_sha,sources,local)
        if PROTOCOL.exists() and json.loads(PROTOCOL.read_text(encoding='utf-8'))!=candidate:
            if not args.freeze_engineering_draft:
                raise RuntimeError('P1_DRAFT_OR_CODE_IDENTITY_CHANGED_EXPLICIT_REFREEZE_REQUIRED')
            write_json(PHASE/'runtime'/attempt/'superseded_engineering_draft.json',json.loads(PROTOCOL.read_text(encoding='utf-8')))
        if args.freeze_engineering_draft or not PROTOCOL.exists():
            if not args.freeze_engineering_draft:
                raise RuntimeError('ENGINEERING_DRAFT_MUST_BE_FROZEN_BEFORE_CHECKS')
            write_json(PROTOCOL,candidate)
            PROTOCOL.with_suffix('.sha256').write_text(digest(PROTOCOL.read_bytes())+'  '+PROTOCOL.name+'\n',encoding='utf-8',newline='\n')
        protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'));protocol_sha=digest(PROTOCOL.read_bytes())
        if protocol_sha!=PROTOCOL.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]:
            raise RuntimeError('P1_PROTOCOL_HASH_MISMATCH')
        safety.require_preflight_only(protocol)
        audit.update(protocol_sha256=protocol_sha,frozen_source_identity=sources,P1_executed_code_identity=local,
                     historical_checkpoint_metadata_count=len(models),historical_metadata_only=True)
        audit['early_stop_source_check']=safety.check_h_early_stop((REPO/(safety.ROOT+safety.SOURCE_NAMES[2])).read_text(encoding='utf-8'))
        import gate_ablation_models as a
        audit['runtime'],audit['pure_function_AST_identity']=a.configure_runtime(cfg)
        safety.forbid_optimizer_steps()
        sys.path.insert(0,str(PHASE/'tests'))
        import p1_test_context as ctx
        ctx.CONFIG=cfg;ctx.PROTOCOL=protocol;ctx.PROTOCOL_SHA=protocol_sha;ctx.DESTINATION=PHASE/'runtime'/attempt
        ctx.DESTINATION.mkdir(parents=True,exist_ok=True)
        suite=unittest.defaultTestLoader.discover(str(PHASE/'tests'),pattern='test_*.py')
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=EvidenceResult).run(suite)
        evidence=ctx.RESULTS
        test_record={'tests':result.records,'tests_run':result.testsRun,'passed':result.wasSuccessful(),
                     'failure_count':len(result.failures),'error_count':len(result.errors),
                     'test_data':'SYNTHETIC_ONLY','real_optimizer_steps':0}
        audit['tests_run']=result.testsRun
        if not result.wasSuccessful() or len(evidence)!=3:
            raise RuntimeError('P1_ACTUAL_SYNTHETIC_TEST_FAILURE')
        _,_,_,_,after_sources,after_local,_=source_identity()
        if sources!=after_sources or local!=after_local:
            raise RuntimeError('EXECUTED_SOURCE_CHANGED_DURING_PREFLIGHT')
        forbidden_allowed=[e for e in safety.EVENTS if e['action']=='ALLOWED' and
                           any(s in e['path'] for s in ('causal_features_30min.csv','Loadcycle_','Polcurve_','.pt','.npz')) and
                           not Path(e['path']).resolve().is_relative_to(PHASE)]
        if forbidden_allowed:
            raise RuntimeError('FORBIDDEN_DATA_ACCESS_DETECTED')
        audit['checks']={k:'PASS' for k in audit['checks']}
        audit['final_status']='GATE_TRAINING_TECHNICALLY_READY_FOR_AUTHORIZATION'
    except Exception:
        audit['exception']=traceback.format_exc()
        log.write('\n'+audit['exception'])
    finally:
        audit['blocked_optimizer_test_attempts']=len(safety.OPTIMIZER_ATTEMPTS)
        audit['repo_open_events']=safety.EVENTS
        audit['completed_UTC']=datetime.now(timezone.utc).isoformat()
        OUTPUT.mkdir(parents=True,exist_ok=True)
        logfile=OUTPUT/'attempt_logs'/f'{attempt}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
        logfile.write_text(log.getvalue(),encoding='utf-8',newline='\n')
        audit['actual_test_log']=str(logfile.relative_to(PHASE)).replace('\\','/')
        write_json(OUTPUT/'P1_PREFLIGHT_TEST_RESULTS.json',test_record)
        write_json(OUTPUT/'P1_EXECUTION_AUDIT.json',audit)
        write_json(OUTPUT/'P1_SYNTHETIC_MODEL_EVIDENCE.json',evidence)
        write_json(OUTPUT/'attempt_logs'/f'{attempt}_audit.json',audit)
        reports(audit,evidence)
        print(log.getvalue(),flush=True)
        print(json.dumps({'status':audit['final_status'],'tests_run':audit.get('tests_run',0),
                          'training_authorized':False,'optimizer_updates':0,'test_accessed':False,
                          'protocol_sha256':audit.get('protocol_sha256'),'attempt':attempt},ensure_ascii=False),flush=True)
    return 0 if audit['final_status']=='GATE_TRAINING_TECHNICALLY_READY_FOR_AUTHORIZATION' else 1

if __name__=='__main__':
    raise SystemExit(main())
