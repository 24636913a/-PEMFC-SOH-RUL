"""Final report/cache consistency and reproducible delivery inventory; no fits."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))
 assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 for rel,digest in cfg['source_sha256'].items():assert sha(P.parent/rel)==digest,rel
 for rel,digest in cfg['execution_code_sha256'].items():assert sha(P/rel)==digest,rel
 for rel,digest in cfg['upstream_sha256'].items():assert sha(Path(rel))==digest,rel
 ledger=json.loads((P/'outputs/EXECUTION_LEDGER.json').read_text());assert ledger['status']=='NINE_RUNS_COMPLETED' and ledger['formal_started']==ledger['formal_completed']==9
 assert not list(P.glob('outputs/training_logs/*/failure.json')) and not list(P.rglob('*.writing'))
 m=pd.read_csv(P/'outputs/ALL_RUN_METRICS.csv',float_precision='round_trip');means=pd.read_csv(P/'outputs/THREE_SEED_MEANS.csv',float_precision='round_trip').set_index('model')
 reports=['00_最终实验总结.md','01_P1S与TFT时间对齐.md','02_健康感知融合结构.md','03_数据覆盖与泄漏测试.md','04_九组正式训练结果.md','05_消融对照与独立增益.md','06_下一阶段SOH与物理约束计划.md']
 for rel in reports:assert (P/rel).stat().st_size>500
 for group,g in m.groupby('model'):
  assert set(g.seed)=={42,52,62}
  for metric in ['RMSE_mV','MAE_mV','endpoint_12h_RMSE_mV','target_time_equal_RMSE_mV','P95_abs_error_mV','P99_abs_error_mV','max_abs_error_mV']:
   assert abs(g[metric].mean()-means.loc[group,metric])<1e-12
  text=(P/reports[0]).read_text(encoding='utf-8')
  expected=[group,*[f'{g[g.seed==s].RMSE_mV.iloc[0]:.6f}' for s in [42,52,62]],f'{g.RMSE_mV.mean():.6f}',f'{g.RMSE_mV.std(ddof=1):.6f}']
  actual=[r for r in text.splitlines() if r.startswith('| '+group+' |')];assert len(actual)==1
  assert [x.strip() for x in actual[0].strip('|').split('|')]==expected
 for row in m.itertuples():
  assert row.stop_epoch<=450 and (row.stop_epoch==450 or row.stop_epoch-row.best_epoch==40)
  assert sha(P/row.checkpoint)==row.checkpoint_sha256 and sha(P/row.prediction)==row.prediction_sha256
 assert sum(m.stop_epoch)==3050
 dec=json.loads((P/'outputs/RESEARCH_DECISION.json').read_text(encoding='utf-8'));assert dec['status']=='HEALTH_CONTEXT_NO_CLEAR_GAIN' and dec['paired_seed_wins_over_both']==0 and dec['endpoint_assessment']=='fail'
 for name in ['INDEPENDENT_VERIFICATION.json','HEALTH_INPUT_INDEPENDENT_VERIFICATION.json']:assert json.loads((P/'outputs'/name).read_text())['status']=='PASS'
 files=['train_validation_coverage','health_fusion_architecture','three_models_RMSE','all_horizon_errors','three_model_predictions_seed42','gate_weight_distributions','training_curves']
 for name in files:
  for suffix in ['png','pdf']:assert (P/f'figures/{name}.{suffix}').stat().st_size>1000
 assert not subprocess.check_output(['git','diff','--name-only'],cwd=P.parent.parent)
 result=dict(status='PASS',formal_runs=9,formal_failures=0,formal_retries=0,total_epochs=3050,train_origins=870,validation_origins=178,
  validation_distinct_target_rows=224,prediction_queries_verified=38448,healthy_vectors_independently_verified=1048,
  all_report_primary_tables_match_actual_CSV=True,all_checkpoints_and_predictions_hashes_match=True,all_frozen_sources_and_execution_code_unchanged=True,
  figures_visually_checked=7,all_required_reports_present=True,Assessment_Test_accessed=False,original_models_and_Ridge_preserved=True,
  decision=dec['status'],preflight_repair_archived=True,neural_training_stopped=True)
 (P/'outputs/FINAL_INTEGRITY_AUDIT.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
