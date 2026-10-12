"""Document/plan integrity only. Never import a model, loader or trainer."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PHASE=Path(__file__).resolve().parents[1]
REPO=PHASE.parent.parent
ROOT='RENew-PEMFC-2300h-Pytorch/'
BASE='ffcd9d836b1dfe12feeee220423f50be8f8ca241'
SOURCES=(
 'Phase_F_TFT_DualPath_RegimeFusion/models/regime_aware_dualpath_tft.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/dualpath_strict_history_tft.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/causal_multiscale_branch.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/frozen_feature_schema.py',
 'Phase_TFT_Original_Baseline/models/canonical_tft.py',
 'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml',
 'Phase_H_Determinism_MultiSeed_Stability/outputs/metrics/H_MULTISEED_STATISTICS.csv',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_FIXED_GATE_DIAGNOSTIC.md',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_FINAL_ARCHITECTURE_DECISION.md',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/gate_diagnostics/J_GATE_STATISTICS.csv',
 'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/manifests/L_FROZEN_MODEL_MANIFEST.json',
 'Phase_M_F2_PINN_Physics_Feasibility/reports/M_FINAL_PHYSICS_RESEARCH_DECISION.md',
 'Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/reports/M2_FINAL_P3_RESEARCH_DECISION.md',
 'Phase_N_Physics_Informed_F2_Method_Convergence/N_FINAL_TRAINING_READINESS_DECISION.md',
 'Phase_O_F2_Paper_FullText_Novelty_Audit/reports/O_03_PHYSICS_INFORMED_PRIOR_ART_COMPARISON.md',
 'Phase_O_F2_Paper_FullText_Novelty_Audit/reports/O_10_FINAL_PUBLICATION_DECISION.md',
 'Phase_O1_Nine_Model_Validation_Robustness_Audit/reports/O1_FINAL_RESEARCH_DECISION.md',
 'Phase_O1_Nine_Model_Validation_Robustness_Audit/reports/O1_SEED62_EXTREME_ERROR_ANALYSIS.md',
 'Phase_O1_Nine_Model_Validation_Robustness_Audit/outputs/O1_EQUAL_SEED_SUMMARY.csv',
)
ALLOW=frozenset(ROOT+s for s in SOURCES)
DOCS=('P_CURRENT_RESEARCH_STATUS','P_GATE_INDEPENDENT_ABLATION_PLAN',
      'P_PHYSICS_LITE_CANDIDATE_REVIEW','P_PHYSICS_LITE_EQUATION_AND_INFORMATION_AUDIT',
      'P_MATCHED_EXPERIMENT_DESIGN','P_FINAL_GO_NO_GO_DECISION',
      'P_SUPERVISOR_PRESENTATION_PREPARATION')


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def approved_bytes(relative):
    demand(relative in ALLOW,'P_STATIC_SOURCE_NOT_ALLOWED:'+relative)
    path=(REPO/relative).resolve()
    demand(path.is_relative_to(REPO.resolve()),'P_SOURCE_OUTSIDE_REPO')
    return path.read_bytes()


def git(*args,binary=False):
    value=subprocess.run(['git',*args],cwd=REPO,check=True,capture_output=True).stdout
    return value if binary else value.decode('utf-8').strip()


def plan_check(plan):
    demand(plan['training_authorized'] is False,'P_TRAINING_NOT_AUTHORIZED')
    demand(plan['paper_body_authorized'] is False,'P_PAPER_BODY_NOT_AUTHORIZED')
    demand(plan['test_authorized'] is False,'P_TEST_PROHIBITED')
    demand(plan['physics_training_runs_authorized']==0,'P_NO_PHYSICS_RUNS')
    demand(set(plan['groups'])=={'G0','G1','G2'},'P_ONLY_THREE_GROUPS')
    demand(plan['training']['seeds']==[42,52,62] and plan['training']['runs']==9,'P_NINE_RUNS')
    demand(plan['groups']['G1']['trainable_gate_parameters']==plan['groups']['G2']['trainable_gate_parameters']==57,'P_CAPACITY_MATCHED_GLOBAL_CONTROL')
    demand(plan['groups']['G1']['gate_context']=='fixed_dimensionless_ones_5_vector_for_all_windows','P_GLOBAL_NO_CONDITIONING')
    demand(plan['initialization']['historical_checkpoint_warm_start'] is False,'P_NO_WARM_START')
    demand(plan['evaluation']['outlier_exclusion'] is False and plan['evaluation']['seed_selection'] is False,'P_NO_SELECTION')
    demand(plan['evaluation']['primary']=='FORMAL_QUERY_WEIGHTED_RMSE_mV','P_PRIMARY_METRIC_CHANGED')
    demand(plan['training']['max_epochs']==450 and plan['training']['patience']==40,'P_FROZEN_BUDGET_CHANGED')
    demand(plan['gate_decision']=='GATE_ABLATION_READY_FOR_AUTHORIZATION','P_GATE_STATUS')
    demand(plan['physics_decision']=='NO_CREDIBLE_PHYSICS_LITE_ROUTE','P_PHYSICS_STATUS')
    # Algebraic design accounting only, not neural-model instantiation.
    demand(5*8+8+8*1+1==57,'P_GATE_PARAMETER_ARITHMETIC')
    demand(plan['cost']['maximum_epochs_all_runs']==9*450,'P_EPOCH_BUDGET_ARITHMETIC')
    demand(plan['cost']['maximum_optimizer_steps_all_runs']==9*450*((1380+31)//32),'P_STEP_BUDGET_ARITHMETIC')
    return True


def save(path,value):
    demand(path.resolve().is_relative_to(PHASE.resolve()),'P_WRITE_OUTSIDE_PHASE')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run():
    git('merge-base','--is-ancestor',BASE,'HEAD')
    demand(git('branch','--show-current')=='main','P_MAIN_REQUIRED')
    prefix=PHASE.relative_to(REPO).as_posix()+'/'
    for args in (('diff','--name-only',BASE,'HEAD'),('diff','--name-only'),('diff','--cached','--name-only')):
        demand(all(x.startswith(prefix) for x in git(*args).splitlines()),'P_HISTORICAL_CHANGE')
    rows=[]
    for relative in sorted(ALLOW):
        data=approved_bytes(relative)
        original=git('show',BASE+':'+relative,binary=True)
        demand(data.replace(b'\r\n',b'\n')==original.replace(b'\r\n',b'\n'),'P_SOURCE_DIFFERS_FROM_BASE:'+relative)
        rows.append({'path':relative,'file_sha256':digest(data),'base_git_blob_sha256':digest(original),
                     'source_commit':BASE,'use':'static code/report/config/saved aggregate only'})
    # Only metadata identity; NEVER open manifest weight paths.
    frozen=json.loads(approved_bytes(ROOT+'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/manifests/L_FROZEN_MODEL_MANIFEST.json'))
    demand(len(frozen['models'])==9,'P_NINE_HISTORICAL_IDENTITIES')
    protocol=PHASE/'protocol/P_PROPOSED_GATE_PROTOCOL.json'
    plan=json.loads(protocol.read_text(encoding='utf-8'));plan_check(plan)
    checksum=digest(protocol.read_bytes());sha_path=protocol.with_suffix('.sha256')
    if sha_path.exists():
        demand(sha_path.read_text().split()[0]==checksum,'P_PROPOSED_PLAN_CHANGED_AFTER_CHECKSUM')
    else:
        sha_path.write_text(checksum+'  '+protocol.name+'\n',encoding='utf-8')
    docs=[]
    for name in DOCS:
        path=PHASE/(name+'.md')
        demand(path.is_file() and len(path.read_text(encoding='utf-8'))>500,'P_MISSING_REQUIRED_REPORT:'+name)
        docs.append({'path':path.name,'sha256':digest(path.read_bytes())})
    pdf=REPO/'参考论文/1-s2.0-S0378775326006919-main.pdf'
    paper_sha=digest(pdf.read_bytes())
    demand(paper_sha=='02aa3f525909f1590cf51be3a4fa7468143dfbed6756a0a291cbe2e0791ca18f','P_REFERENCE_PDF_IDENTITY_CHANGED')
    save(PHASE/'evidence/P_STATIC_SOURCE_MANIFEST.json',{'base_commit':BASE,'sources':rows,
        'literature':[{'path':pdf.relative_to(REPO).as_posix(),'sha256':paper_sha,'verified_pages':[6],
                       'doi':'10.1016/j.jpowsour.2026.239941'}],
        'bounded_web_verification':[
            {'url':'https://doi.org/10.1038/s42254-021-00314-5','identity':'Karniadakis et al., 2021, Nature Reviews Physics 3,422-440',
             'scope':'primary publisher indexed metadata/abstract; direct full-page redirect inaccessible; not a full-text review'},
            {'url':'https://netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf','identity':'DOE/NETL Fuel Cell Handbook, Seventh Edition, November 2004',
             'scope':'official download and title/contents verified; subsequent method-page requests failed; not used as a newly verified equation source'},
            {'url':'https://doi.org/10.1016/j.jpowsour.2026.239941','scope':'primary publisher identity plus local user-supplied original PDF p6 equations/visual check'}],
        'weight_identity_scope':'historical metadata only; no current weight byte hash/load'})
    test_file=PHASE/'outputs/P_DOCUMENT_GUARD_TEST_RESULTS.json'
    demand(test_file.exists(),'P_DOCUMENT_TESTS_NOT_RUN')
    test=json.loads(test_file.read_text(encoding='utf-8'));demand(test['pass'],'P_DOCUMENT_TESTS_FAILED')
    audit={'base_commit':BASE,'execution_head':git('rev-parse','HEAD'),'utc':datetime.now(timezone.utc).isoformat(),
       'technical_status':'STATIC_PLANNING_AUDIT_PASS','gate_decision':plan['gate_decision'],
       'physics_decision':plan['physics_decision'],'proposed_training_runs':9,'training_runs_executed':0,
       'training_authorized':False,'source_count':len(rows),'proposed_protocol_sha256':checksum,
       'reports':docs,'historical_model_identities':9,'checkpoint_bytes_read':False,
       'model_forward_calls':0,'prediction_cache_read':False,'raw_N1_read_or_hash':False,
       'new_performance_metrics':False,'physics_parameters_fitted':False,'test_numeric_accessed':False,
       'assessment_used_for_new_selection':False,'old_protocol_modified':False,'paper_body_written':False,
       'PPT_created':False,'GPU_probed':False,'document_guard_tests':test,
       'code_sha256':digest(Path(__file__).read_bytes()),
       'IO_scope':'closed-list script plus agent tool trace; not global OS monitoring',
       'stop':'await separate human authorization; no training or paper drafting'}
    save(PHASE/'outputs/P_STATIC_PLANNING_AUDIT.json',audit)
    print(json.dumps({k:audit[k] for k in ('technical_status','gate_decision','physics_decision','proposed_training_runs','training_runs_executed')},ensure_ascii=False))


if __name__=='__main__':
    run()
