"""Freeze science, arrays, causal preprocessing and training code after engineering."""
from common import *
import yaml
def main():
 assert not (P/'configs/FROZEN_PROTOCOL.json').exists(),'No refreeze'
 assert json.loads((P/'outputs/ENGINEERING_EVIDENCE.json').read_text())['status']=='PASS'
 tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=P,capture_output=True,text=True,encoding='utf-8')
 js('outputs/PRETRAINING_TESTS.json',dict(returncode=tests.returncode,stdout=tests.stdout,stderr=tests.stderr));assert tests.returncode==0,tests.stderr
 old=yaml.safe_load((ROOT/'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))
 h=old['hyperparameters'];h={k:v for k,v in h.items() if k!='seed'}
 critical=['scripts/common.py','scripts/prepare_data.py','scripts/train.py','scripts/engineering.py','models/event_tft.py']
 code={f:sha(P/f) for f in critical}
 data={f.relative_to(P).as_posix():sha(f) for f in sorted((P/'data_contracts').glob('*')) if f.is_file()}
 cfg=dict(status='FROZEN_BEFORE_REAL_TRAINING',base_commit=BASE,
  authorization='User explicit27 event-model experiments and GPU authorization',
  event='Target raw_cycle=origin+1, frozen141 middle50 representative V; new5audit cycles excluded',
  target='Absolute real voltage standardized by training-target mean/std; never SOH labels',
  completion='Origin asof successor15||14,15||0 anchor fully observed; target initial anchor may be known but every target fixed-state voltage measurement is future',
  horizon='Irregular H1 event, now8.76–9.13 actual operating hours after delayed confirmed origin; actual future lead/time never features',
  event_readout='One constant zero decoder event token through full inherited native forward: VSN->LSTM(+CNN/gate)->static enrichment->attention->GRN/FF->pre_output_gate_norm->linear64to1',
  head='Existing native scalar output_layer independently initialized, one event query; no24position relabeling, interpolation, MLP replacement or old checkpoint loads',
  input_shape=[48,14],output_shape=[1],raw_feature_order=old['task']['raw_f2_feature_order'],
  preprocessing='Train-only raw-history population mean/std, EWMAalpha.3 on historymeanV, Top10 standardized zero mask; innerFit-only during epoch selection, fullTrain-only during finalrefit',
  Top10='Recompute fixed absSpearman-at-origin rule on correspondingTrain scalarH1V; mandatorymeanV. Historical indices derived later dynamic data forbidden at early folds',
  y_scaling_reason='Original target_scale_dynamic_V replaced by Train targetV std and mean because new absolute voltage task has different units/center; not a hyperparameter search',
  folds='OriginalRidge expanding boundaries, restricted to prior37/13,50/6,56/22 candidate intersection; no boundary moves',
  inner='Lastceil15percentTrain chronological selection; innerfit labels confirmed at least1h before first inner origin',
  fit_recipe='Inner training selects epoch via inner RMSE; independent same-seed reinitialization refits fullTrain for selected epoch count, no outerValidation early stopping',
  registered_experiments=27,optimizer_fit_passes=54,fit_count_definition='27 inner selection fits+27 final fullTrain refits; synthetic engineering excluded',
  seeds=[42,52,62],models=['F0','F1','F2'],hyperparameters=h,
  runtime='GPU, float32, CPUthreads1,cudnn deterministicTrue benchmarkFalse, torch deterministic algorithmsFalse, TF32matmulFalse cudnnTrue; no strict CUDA bitwise claim',
  preorigin_readiness='Fixed1h label-to-origin reserve; additionally measured full fit runtime placed after lastTrainconfirmedlabel must end before firstValidationorigin. Contemporary2026run is historicalasof replay, not actual2025deployment',
  primary='Threefold equally weighted MAE_mV, average over all3seeds within eachfold; no twofold substitute',
  supported_rule=dict(Persistence_MAE_improvement_at_least=.05,F2_better_F0_and_F1_equal_fold_MAE=True,
   minimum_fold_wins_vs_each_comparator=2,minimum_seed_equal_fold_MAE_wins_vs_Persistence=2,
   tail='Eachfold seed-mean RMSE/P95/max no more than110percent of Persistence; report actual extremes',all3folds_required=True),
  decision='Notqualified if data/time/technical gates fail; noGain if F2 equalfoldMAE not belowPersistence; supported only all criteria; otherwiseexploratory',
  baselines='Origin latest frozenfixedV persistence; independent sameTrain Ridge96 fivefeatures at confirmedorigin, alpha[.0001,.01,1,100], train-innerMAE choice and Train-onlyscaler, residualDeltaR; retain oldRidge unchanged',
  proxy='100V/.71383023635 only after frozenfirst10reference available; no clipping/monotonicity; not standardj1polarizationSOH',
  reference='Recheck exact H1 event links and frozen past M2 metadata only; no new calibration, no standardSOH errors without qualified clock/condition/release gates',
  stops='Nonfinite/identity/time/firewall/reload failure stops sweep with evidence; no outcome-driven retuning; completed runIDs never restart',
  training_code_sha256=code,data_sha256=data,counts=json.loads((P/'outputs/ENGINEERING_EVIDENCE.json').read_text())['counts'])
 js('configs/FROZEN_PROTOCOL.json',cfg);md('configs/FROZEN_PROTOCOL.sha256',sha(P/'configs/FROZEN_PROTOCOL.json'))
 md('README.md','''# H1健康导向事件级双路径TFT

新任务：完整周期确认后预测下一raw物理周期的冻结middle50固定状态电压。141原标签，新增5周期不入训练。

阶段一：prepare_data.py、engineering.py、freeze_protocol.py，先提交协议；阶段二train.py执行27登记实验，每项内选epoch再独立全Train重拟合，共54个真实GPU拟合pass。原模型/权重只读，无Test数值或原始高频重读。

一次性运行，runID状态存在时不自动重训；控制器跳过COMPLETE，遇FAILED停止并保存错误。恢复状态位于本阶段ignored outputs/runtime，正式inner/finalcheckpoint均保存。

模型输入48×14，保留原骨干及native注意力；仅一个全零事件查询，输出标准化绝对V。Top10和X/Yscaler都在对应Train拟合，不能复用较晚动态任务掩码。外层验证从不选择epoch。

真实训练发生在2026年，训练耗时放入历史标签可用时间之后检查asof可实现性；不声称2025年实际在线部署。等待完成后读08通俗总结和逐点CSV。
''')
 assert protected_pass();print('FROZEN',sha(P/'configs/FROZEN_PROTOCOL.json'),'real training fits54')
if __name__=='__main__':main()
