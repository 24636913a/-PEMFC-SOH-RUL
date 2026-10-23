"""Register cache-only calculations AFTER literature comparison and revised outline."""
from common import *
def main():
    p=P/'outputs/audits/FROZEN_ANALYSIS.json';assert not p.exists(),'ALREADY_REGISTERED_DO_NOT_REFREEZE'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    first=read(P/'outputs/audits/LITERATURE_PHASE_COMPLETE.json');assert first['papers']==9
    outline=P/'outline/04_正式SCI论文大纲_文献对标修订版.md';assert outline.is_file()
    paths=[H/'configs/H_FROZEN_PROTOCOL.yaml',H/'outputs/metrics/H_VALIDATION_METRICS.csv',I2/'outputs/metrics/I2_ASSESSMENT_METRICS.csv',
        J/'outputs/gate_diagnostics/J_TRAIN_ONLY_CONDITION_THRESHOLDS.csv',O1/'outputs/O1_NINE_MODEL_VALIDATION_METRICS.csv',
        P2/'outputs/metrics/P2_NINE_RUN_METRICS.csv',P2/'reports/P2_FINAL_RESEARCH_DECISION.md',
        R1/'outputs/FINAL_SUMMARY.json',R1/'outputs/F2_SEED_RANKING.csv',R1/'configs/frozen_seed_screening.json',
        R2/'outputs/F2_SIX_CANDIDATE_SEEDS.csv',R2/'outputs/FINAL_SUMMARY.json',R2/'configs/frozen_seed_screening.json',
        SIX/'outputs/METRICS.csv',SIX/'outputs/GATE_STATISTICS.csv',SIX/'outputs/GATE_CORRELATIONS.csv',SIX/'outputs/SUMMARY.json',
        SIX/'outputs/ALL_VALIDATION_GATES.csv',SIX/'outputs/PERMUTATION_SUMMARY.csv',SIX/'outputs/audits/HISTORY_BEFORE.csv']
    paths += [R2/'outputs/audits'/f for f in ['FROZEN_H_DATA.npz','VALIDATION_KEY_TRUTH.csv','SCALER.json','DATA_FINGERPRINTS.json']]
    for row in rd(H/'outputs/metrics/H_VALIDATION_METRICS.csv').query("role=='MULTISEED_PRIMARY'").itertuples():
        path=H/row.validation_prediction;assert sha(path)==row.prediction_file_sha256
        paths.append(path)
    paths += [SIX/f'outputs/predictions/seed_{s}_B0.csv' for s in SELECTED]
    F=ROOT/'Phase_F_TFT_DualPath_RegimeFusion'
    paths += sorted((F/'models').glob('*.py'))+[ROOT/'Phase_TFT_Original_Baseline/models/canonical_tft.py']
    paths += [P/'literature'/f for f in ['01_已发表PEMFC论文横向比较矩阵.md','02_参考文献核心实验与图表清单.md','03_本项目创新点与重叠风险.md','REFERENCES.md','core_P1_P5_evidence.json','P6_aux_evidence.json','verified_metadata.json']]+[outline]
    unique=sorted(set(paths));sources=[dict(path=str(f.relative_to(REPO)).replace('\\','/'),sha256=sha(f),bytes=f.stat().st_size) for f in unique]
    # Recheck historical checkpoint/screening protection, including local partial outputs.
    history=rd(SIX/'outputs/audits/HISTORY_BEFORE.csv');before={r.path:r.sha256 for r in history.itertuples()}
    for phase in [R1,R2,P2]:
        for f in phase.rglob('*'):
            if not f.is_file() or '__pycache__' in f.parts or 'runtime' in f.parts or f.suffix=='.pyc':continue
            before[str(f.relative_to(REPO)).replace('\\','/')]=sha(f)
    for f,digest in before.items():assert sha(REPO/f)==digest,('HISTORICAL_SOURCE_CHANGED',f)
    cs('outputs/audits/HISTORY_BEFORE.csv',[dict(path=f,sha256=v) for f,v in before.items()])
    c=dict(base_commit=BASE,literature_first_complete=True,literature_phase_sha256=sha(P/'outputs/audits/LITERATURE_PHASE_COMPLETE.json'),
        sources=sources,formal_seeds=SEEDS,selected_seeds=SELECTED,new_training=False,model_forward=False,Test=False,
        Assessment='existing I2 metric summary only; no target data or inference',
        condition_rules='Reuse J Train-only 1/3 and2/3 quantiles exactly; three history means (indices9,11,8); no Validation-error threshold tuning',
        baseline='DeltaV Persistence=0; same4920 query identities and true targets',
        case_rules={'typical_origin':2205,'difficult':'maximum equal-seed mean per-origin RMSE of formal F2 seeds42/52/62, ties lowest origin; diagnostic only'},
        unit='existing physical DeltaV V; metrics*1000 to mV once; absolute voltage=raw origin V+DeltaV',
        no_p_values='overlapping histories, shared targets and selected instances are not independent battery samples',
        horizons='24 lead-specific diagnostics of one fixed48to24 model; not24 separately trained horizon tasks')
    js('outputs/audits/FROZEN_ANALYSIS.json',c);text('outputs/audits/FROZEN_ANALYSIS.sha256',sha(p)+'\n')
    cs('outputs/evidence_index/COMPUTATIONAL_SOURCE_INDEX.csv',sources);print('FROZEN_AFTER_LITERATURE_AND_OUTLINE',len(sources),'protected',len(before))
if __name__=='__main__':main()
