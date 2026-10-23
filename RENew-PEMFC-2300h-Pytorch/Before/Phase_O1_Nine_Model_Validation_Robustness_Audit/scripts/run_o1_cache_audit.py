"""Frozen Validation CSV audit only. No torch/model/raw-data imports."""
import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parent.parent
ROOT = 'RENew-PEMFC-2300h-Pytorch/'
BASE = '7553fcf3ad5f2b10bcf1f590ce90d19d46468bb4'
H = ROOT+'Phase_H_Determinism_MultiSeed_Stability/'
J = ROOT+'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/'
L = ROOT+'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/'
O = ROOT+'Phase_O_F2_Paper_FullText_Novelty_Audit/'
SEEDS = (42,52,62)
MODELS = ('F0','F1','F2')
RUNS = tuple(f'H-{m}-Seed{s}' for m in MODELS for s in SEEDS)
H_SHA = '88aba09c7f9a08d43f658f9c66e482713bbb60607bc88bb8059f3638d5aad9dd'
META = (H+'outputs/metrics/H_VALIDATION_METRICS.csv',
        H+'configs/H_FROZEN_PROTOCOL.yaml', H+'configs/H_FROZEN_PROTOCOL.sha256',
        H+'scripts/audit_h_predictions.py',
        L+'manifests/L_FROZEN_MODEL_MANIFEST.json',
        J+'outputs/checkpoint_identity/J_NINE_CHECKPOINT_IDENTITY.json',
        J+'outputs/fixed_gate_diagnostic/J_VALIDATION_METRICS.csv',
        J+'outputs/integrity_audit/J_ORIGINAL_H_FULL_VALIDATION_CACHE_AUDIT.csv',
        O+'reports/O_07_F2_EXPERIMENT_EVIDENCE_AUDIT.md',
        O+'reports/O_10_FINAL_PUBLICATION_DECISION.md')
CACHES = tuple(H+f'outputs/validation_predictions/{run}.csv' for run in RUNS)
ALLOWED = frozenset(META+CACHES)
KEYS = ['origin_row','target_row','horizon_step']
TOL = 1e-10


class AuditError(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise AuditError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(relative):
    require(relative in ALLOWED, 'SOURCE_NOT_ALLOWED:'+relative)
    p = (REPO/relative).resolve()
    require(p.is_relative_to(REPO.resolve()), 'SOURCE_OUTSIDE_REPO')
    return p


def read_csv(relative):
    return pd.read_csv(source(relative), float_precision='round_trip')


def write_json(path, value):
    require(path.resolve().is_relative_to(PHASE.resolve()), 'WRITE_OUTSIDE_O1')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def git(*args):
    return subprocess.run(['git',*args], cwd=REPO, check=True, capture_output=True).stdout.decode('utf-8').strip()


def verify_hash(relative, expected):
    observed = sha(source(relative).read_bytes())
    require(observed == expected, 'SOURCE_HASH_MISMATCH:'+relative)
    return observed


def freeze():
    require(git('rev-parse','HEAD') == BASE, 'BASE_COMMIT_CHANGED')
    require(git('branch','--show-current') == 'main', 'MAIN_REQUIRED')
    manifest_path = PHASE/'protocol/O1_ALLOWED_CACHE_MANIFEST.json'
    require(not manifest_path.exists(), 'PROTOCOL_ALREADY_FROZEN')
    protocol = PHASE/'protocol/O1_FROZEN_METRIC_PROTOCOL.md'
    digest = sha(protocol.read_bytes())
    protocol.with_suffix('.sha256').write_text(digest+'  '+protocol.name+'\n', encoding='utf-8')
    authority = read_csv(META[0])
    primary = authority[authority.role == 'MULTISEED_PRIMARY']
    require(len(primary) == 9 and set(primary.run_id) == set(RUNS), 'H_PRIMARY_SET_MISMATCH')
    require((primary.records == 4920).all() and (primary.protocol_sha256 == H_SHA).all(), 'H_COUNT_PROTOCOL_MISMATCH')
    verify_hash(META[1], H_SHA)
    cfg = yaml.safe_load(source(META[1]).read_text(encoding='utf-8'))
    require(cfg['data']['validation_rows'] == [2103,2404] and cfg['data']['validation_windows'] == 205,
            'H_FROZEN_BOUNDARY_COUNT_MISMATCH')
    require(cfg['task']['lookback_steps'] == 48 and cfg['task']['horizon_steps'] == 24,
            'H_FROZEN_TASK_MISMATCH')
    refs = [{'path':p,'sha256':sha(source(p).read_bytes())} for p in META]
    cache_rows = []
    for run in RUNS:
        row = primary[primary.run_id == run].iloc[0]
        path = H+f'outputs/validation_predictions/{run}.csv'
        cache_rows.append({'run_id':run,'model':row.model,'seed':int(row.seed),
            'role':row.role,'path':path,'sha256':row.prediction_file_sha256,
            'checkpoint_relative_to_H':row.checkpoint,'protocol_sha256':row.protocol_sha256,
            'code_identity_sha256':row.code_identity_sha256,'expected_records':int(row.records)})
    write_json(manifest_path, {'base_commit':BASE,'protocol_sha256':digest,
        'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'stage':'FROZEN_BEFORE_FORMAL_METRIC_RECOMPUTATION','metadata':refs,'caches':cache_rows,
        'no_prediction_file_parsed_during_freeze':True, 'expected_hash_source':META[0]})
    print('O1_PROTOCOL_FROZEN '+digest)


def validate_frame(frame, run, model, seed, protocol_sha, n_origins=205, horizon=24):
    require(run in RUNS, 'REPEAT_OR_UNKNOWN_IDENTITY_PROHIBITED')
    require(run == f'H-{model}-Seed{seed}', 'RUN_MODEL_SEED_CONTRACT_MISMATCH')
    expected_columns = {'run_id','model','seed','protocol_sha256',*KEYS,'y_true_delta_V','y_pred_delta_V'}
    require(expected_columns.issubset(frame.columns), 'CACHE_SCHEMA_MISSING')
    require(len(frame) == n_origins*horizon, 'QUERY_COUNT_MISMATCH')
    require(not frame[KEYS].duplicated().any(), 'DUPLICATE_KEY')
    require((frame.run_id == run).all() and (frame.model == model).all() and (frame.seed == seed).all(), 'RUN_MODEL_SEED_MISMATCH')
    require((frame.protocol_sha256 == protocol_sha).all(), 'PROTOCOL_MISMATCH')
    a = frame[KEYS].to_numpy(dtype=float)
    require(np.isfinite(a).all() and np.array_equal(a, np.floor(a)), 'KEY_NOT_FINITE_INTEGER')
    require(np.isfinite(frame[['y_true_delta_V','y_pred_delta_V']].to_numpy(float)).all(), 'NONFINITE_VALUE')
    require((frame.target_row == frame.origin_row+frame.horizon_step).all(), 'TARGET_ORIGIN_HORIZON_MISMATCH')
    require(frame.target_row.between(2103,2403).all() and frame.origin_row.between(2103,2403).all(), 'VALIDATION_BOUNDARY_MISMATCH')
    require(frame.origin_row.nunique() == n_origins, 'ORIGIN_COUNT_MISMATCH')
    require(set(frame.horizon_step) == set(range(1,horizon+1)), 'HORIZON_SET_MISMATCH')
    require((frame.groupby('origin_row').horizon_step.nunique() == horizon).all(), 'MISSING_HORIZON')
    require((frame.groupby('horizon_step').size() == n_origins).all(), 'HORIZON_QUERY_COUNT_MISMATCH')


def align_truth(frame, reference):
    require(set(map(tuple,frame[KEYS].to_numpy())) == set(map(tuple,reference[KEYS].to_numpy())), 'COMMON_KEY_SET_MISMATCH')
    # Explicit keyed comparison, not position-based matching; original files untouched.
    a = frame.set_index(KEYS).sort_index()
    b = reference.set_index(KEYS).sort_index()
    require(np.array_equal(a.y_true_delta_V.to_numpy(), b.y_true_delta_V.to_numpy()), 'COMMON_TRUTH_MISMATCH')
    return a.reset_index()


def error_metrics(e):
    e = np.asarray(e, dtype=np.float64)
    require(len(e)>0 and np.isfinite(e).all(), 'INVALID_METRIC_INPUT')
    absolute = np.abs(e)
    return {'RMSE_mV':math.sqrt(math.fsum(float(v)*float(v) for v in e)/len(e)),
            'MAE_mV':math.fsum(float(v) for v in absolute)/len(e),
            'P95_mV':float(np.quantile(absolute,.95,method='linear')),
            'P99_mV':float(np.quantile(absolute,.99,method='linear')),
            'max_abs_error_mV':float(absolute.max()), 'n_queries':len(e)}


def target_metrics(frame):
    f = frame[['target_row','horizon_step']].copy()
    f['SE_mV2'] = np.square((frame.y_pred_delta_V.to_numpy()-frame.y_true_delta_V.to_numpy())*1000)
    rows = []
    for target, block in f.groupby('target_row', sort=True):
        sse = math.fsum(float(x) for x in block.SE_mV2)
        rows.append({'target_row':int(target),'multiplicity':len(block), 'SSE_mV2':sse,
            'mean_SE_mV2':sse/len(block),'horizon_steps':';'.join(map(str,sorted(block.horizon_step.tolist())))})
    targets = pd.DataFrame(rows)
    total = math.fsum(targets.SSE_mV2)
    targets['formal_SSE_share'] = targets.SSE_mV2/total if total else 0.
    equal_total = math.fsum(targets.mean_SE_mV2)
    targets['target_equal_MSE_share'] = targets.mean_SE_mV2/equal_total if equal_total else 0.
    ordered = targets.sort_values(['SSE_mV2','target_row'], ascending=[False,True])
    summary = {'target_time_equal_RMSE_mV':math.sqrt(math.fsum(targets.mean_SE_mV2)/len(targets)),
       'n_unique_target_times':len(targets), 'total_SSE_mV2':total,
       'top1_target_SSE_share':float(ordered.formal_SSE_share.iloc[0]),
       'top5_target_SSE_share':math.fsum(ordered.formal_SSE_share.iloc[:5]),
       'top10_target_SSE_share':math.fsum(ordered.formal_SSE_share.iloc[:10]),
       'max_SSE_target_row':int(ordered.target_row.iloc[0]),
       'max_SSE_target_multiplicity':int(ordered.multiplicity.iloc[0])}
    return summary, targets


def target_semantics(independence_claim=False):
    require(not independence_claim, 'UNIQUE_TARGETS_ARE_NOT_INDEPENDENT_SAMPLES')
    return 'unique target timestamps; statistical independence not assumed'


def check_identity_metadata(entry, hrow, lrow, jrow):
    require(hrow.role == 'MULTISEED_PRIMARY', 'NON_PRIMARY_ROLE')
    for field in ('model','seed','best_epoch','parameter_count'):
        require(hrow[field] == lrow[field] == jrow[field], 'MODEL_MANIFEST_IDENTITY_MISMATCH:'+field)
    checkpoint = H+str(hrow.checkpoint).replace('\\','/')
    require(checkpoint == lrow['path'] == jrow['path'], 'CHECKPOINT_PATH_MISMATCH')
    require(lrow['sha256'] == jrow['sha256'], 'HISTORICAL_CHECKPOINT_HASH_MISMATCH')
    require(hrow.prediction_file_sha256 == entry['sha256'], 'CACHE_REFERENCE_HASH_MISMATCH')
    require(hrow.protocol_sha256 == entry['protocol_sha256'] == H_SHA, 'H_PROTOCOL_IDENTITY_MISMATCH')
    require(hrow.code_identity_sha256 == entry['code_identity_sha256'], 'H_CODE_IDENTITY_MISMATCH')


def verify_saved_tables(expected):
    records=[]
    for name,table in expected.items():
        path=PHASE/'outputs'/(name+'.csv')
        require(path.is_file() and path.stat().st_size>0,'OUTPUT_MISSING:'+name)
        actual=pd.read_csv(path,float_precision='round_trip')
        require(list(actual.columns)==list(table.columns) and len(actual)==len(table),'OUTPUT_SCHEMA_COUNT:'+name)
        for column in table.columns:
            if pd.api.types.is_numeric_dtype(table[column]):
                require(np.allclose(actual[column].to_numpy(float),table[column].to_numpy(float),rtol=0,atol=TOL),'OUTPUT_NUMERIC_MISMATCH:'+column)
            else:
                require(actual[column].astype(str).tolist()==table[column].astype(str).tolist(),'OUTPUT_IDENTITY_MISMATCH:'+column)
        records.append({'path':path.relative_to(PHASE).as_posix(),'rows':len(actual),'sha256':sha(path.read_bytes()),'round_trip_pass':True})
    return records


def audit():
    protocol = PHASE/'protocol/O1_FROZEN_METRIC_PROTOCOL.md'
    manifest = json.loads((PHASE/'protocol/O1_ALLOWED_CACHE_MANIFEST.json').read_text(encoding='utf-8'))
    require(sha(protocol.read_bytes()) == manifest['protocol_sha256'] == protocol.with_suffix('.sha256').read_text().split()[0], 'O1_PROTOCOL_CHANGED')
    require(manifest['base_commit'] == BASE, 'O1_BASE_IDENTITY_CHANGED')
    git('merge-base','--is-ancestor',BASE,'HEAD')
    require(git('branch','--show-current') == 'main', 'MAIN_REQUIRED')
    own=PHASE.relative_to(REPO).as_posix()+'/'
    for command in (('diff','--name-only',BASE,'HEAD'),('diff','--name-only'),('diff','--cached','--name-only')):
        require(all(x.startswith(own) for x in git(*command).splitlines()),'HISTORICAL_FILE_CHANGED')
    test = json.loads((PHASE/'outputs/O1_SYNTHETIC_TEST_RESULTS.json').read_text(encoding='utf-8'))
    require(test['pass'], 'SYNTHETIC_TESTS_FAILED')
    for row in manifest['metadata']:
        verify_hash(row['path'], row['sha256'])
    require(len(manifest['caches']) == 9 and {r['run_id'] for r in manifest['caches']} == set(RUNS), 'NINE_CACHE_SET_MISMATCH')
    htable = read_csv(META[0]).set_index('run_id',drop=False)
    ltable = {x['identity']:x for x in json.loads(source(META[4]).read_text(encoding='utf-8'))['models']}
    jidentity = json.loads(source(META[5]).read_text(encoding='utf-8'))
    frames = {}; checks = []
    for entry in manifest['caches']:
        run = entry['run_id']
        check_identity_metadata(entry,htable.loc[run],ltable[run],jidentity[run])
        observed = verify_hash(entry['path'], entry['sha256'])
        frame = read_csv(entry['path'])
        validate_frame(frame,run,entry['model'],entry['seed'],entry['protocol_sha256'])
        frames[run] = frame
        checks.append({'run_id':run,'file_hash_pass':True,'sha256':observed,
            'records':len(frame),'n_origins':int(frame.origin_row.nunique()),
            'key_uniqueness':True,'identity_pass':True,'truth_max_abs_difference_V':0.})
    reference = frames[RUNS[0]]
    for run in RUNS:
        frames[run] = align_truth(frames[run],reference)
    # All nine identity and truth checks above finish BEFORE any metric comparison.
    metrics=[]; horizons=[]; targets=[]; diffs=[]
    for run in RUNS:
        f=frames[run]; model=f.model.iloc[0]; seed=int(f.seed.iloc[0]); identity={'run_id':run,'model':model,'seed':seed}
        e=(f.y_pred_delta_V.to_numpy()-f.y_true_delta_V.to_numpy())*1000
        m={**identity,**error_metrics(e),'n_origins':int(f.origin_row.nunique())}
        tm, t=target_metrics(f); m.update(tm)
        for h in range(1,25):
            hm=error_metrics(e[f.horizon_step.to_numpy()==h])
            horizons.append({**identity,'horizon_step':h,'forecast_hours':h*.5,**hm})
            if h==24:
                m.update({'endpoint_'+k:v for k,v in hm.items()})
        for field in ('RMSE_mV','MAE_mV'):
            difference=float(m[field]-htable.loc[run,field]); diffs.append({'run_id':run,'source':'H','metric':field,'difference_mV':difference})
            require(abs(difference)<=TOL,'H_METRIC_REPRODUCTION_MISMATCH:'+run+':'+field)
        metrics.append(m); t.insert(0,'seed',seed); t.insert(0,'model',model); t.insert(0,'run_id',run);targets.append(t)
    metric_df=pd.DataFrame(metrics);horizon_df=pd.DataFrame(horizons);target_df=pd.concat(targets,ignore_index=True)
    jt=read_csv(META[6]); ja=read_csv(META[7])
    for seed in SEEDS:
        require(ja.loc[ja.seed==seed,'all_4920_keys_truth_and_prediction_exact'].astype(str).str.lower().eq('true').all(), 'J_H_HISTORICAL_EXACT_AUDIT_NOT_PASS')
        jrow=jt[(jt.seed==seed)&(jt['mode']=='learned')]
        require(len(jrow)==1,'J_LEARNED_IDENTITY_MISMATCH')
        row=metric_df[(metric_df.model=='F2')&(metric_df.seed==seed)].iloc[0]
        for field in ('RMSE_mV','MAE_mV','P95_mV','P99_mV','endpoint_RMSE_mV','max_abs_error_mV','target_time_equal_RMSE_mV'):
            delta=float(row[field]-jrow.iloc[0][field]);diffs.append({'run_id':row.run_id,'source':'J learned','metric':field,'difference_mV':delta})
            require(abs(delta)<=TOL,'J_METRIC_REPRODUCTION_MISMATCH:'+field)
    fields=['RMSE_mV','MAE_mV','P95_mV','P99_mV','max_abs_error_mV','endpoint_RMSE_mV',
      'endpoint_MAE_mV','endpoint_P95_mV','endpoint_P99_mV','endpoint_max_abs_error_mV',
      'target_time_equal_RMSE_mV','top1_target_SSE_share','top5_target_SSE_share','top10_target_SSE_share']
    pairs=[];hpairs=[]
    for new,old in (('F2','F0'),('F2','F1'),('F1','F0')):
        for seed in SEEDS:
            n=metric_df[(metric_df.model==new)&(metric_df.seed==seed)].iloc[0]
            b=metric_df[(metric_df.model==old)&(metric_df.seed==seed)].iloc[0]
            pairs.append({'comparison':new+'-'+old,'seed':seed,**{k+'_difference':float(n[k]-b[k]) for k in fields}})
            for h in range(1,25):
                nh=horizon_df[(horizon_df.model==new)&(horizon_df.seed==seed)&(horizon_df.horizon_step==h)].iloc[0]
                bh=horizon_df[(horizon_df.model==old)&(horizon_df.seed==seed)&(horizon_df.horizon_step==h)].iloc[0]
                hpairs.append({'comparison':new+'-'+old,'seed':seed,'horizon_step':h,
                    **{k+'_difference':float(nh[k]-bh[k]) for k in ('RMSE_mV','MAE_mV','P95_mV','P99_mV','max_abs_error_mV')}})
    pair_df=pd.DataFrame(pairs);hpair_df=pd.DataFrame(hpairs)
    summaries=[];hsummaries=[]
    for model, group in metric_df.groupby('model'):
        for k in fields:
            summaries.append({'model':model,'metric':k,'equal_seed_mean':float(group[k].mean()),'sample_SD_ddof1':float(group[k].std(ddof=1)),'n_seeds':3})
    for (model,h),g in horizon_df.groupby(['model','horizon_step']):
        for k in ('RMSE_mV','MAE_mV','P95_mV','P99_mV','max_abs_error_mV'):
            hsummaries.append({'model':model,'horizon_step':int(h),'metric':k,'equal_seed_mean':float(g[k].mean()),'sample_SD_ddof1':float(g[k].std(ddof=1))})
    summary_df=pd.DataFrame(summaries)
    overall={m:float(metric_df.loc[metric_df.model==m,'RMSE_mV'].mean()) for m in MODELS}
    advantage=all(overall['F2']<overall[m] for m in ('F0','F1'))
    robust_pairs=pair_df[pair_df.comparison.isin(['F2-F0','F2-F1'])]
    all_positive=all((robust_pairs[k+'_difference']<=0).all() for k in fields)
    status='F2_ADVANTAGE_NOT_SUPPORTED' if not advantage else 'F2_ADVANTAGE_ROBUST_ACROSS_REPORTED_METRICS' if all_positive else 'F2_ADVANTAGE_METRIC_DEPENDENT'
    outputs=PHASE/'outputs';outputs.mkdir(exist_ok=True)
    tables={'O1_NINE_MODEL_VALIDATION_METRICS':metric_df,'O1_HORIZON_METRICS':horizon_df,
      'O1_TARGET_SSE_CONCENTRATION':target_df,'O1_PAIRED_MODEL_COMPARISON':pair_df,
      'O1_PAIRED_HORIZON_COMPARISON':hpair_df,'O1_EQUAL_SEED_SUMMARY':summary_df,
      'O1_HORIZON_SEED_SUMMARY':pd.DataFrame(hsummaries),'O1_HISTORICAL_METRIC_DIFFERENCES':pd.DataFrame(diffs)}
    for name,table in tables.items():
        table.to_csv(outputs/(name+'.csv'),index=False,float_format='%.17g',lineterminator='\n')
    # Exact key-level paired residual/SSE differences; no independence inference.
    q=[]
    for new,old in (('F2','F0'),('F2','F1'),('F1','F0')):
        for seed in SEEDS:
            nf=frames[f'H-{new}-Seed{seed}'];bf=frames[f'H-{old}-Seed{seed}']
            en=(nf.y_pred_delta_V.to_numpy()-nf.y_true_delta_V.to_numpy())*1000
            eb=(bf.y_pred_delta_V.to_numpy()-bf.y_true_delta_V.to_numpy())*1000
            item=nf[KEYS].copy();item.insert(0,'seed',seed);item.insert(0,'comparison',new+'-'+old)
            item['absolute_error_difference_mV']=np.abs(en)-np.abs(eb);item['SE_difference_mV2']=en*en-eb*eb;q.append(item)
    pd.concat(q).to_csv(outputs/'O1_PAIRED_QUERY_ERROR_DIFFERENCES.csv',index=False,float_format='%.17g',lineterminator='\n')
    extreme=[]
    for model in MODELS:
        f=frames[f'H-{model}-Seed62'].copy()
        f['signed_error_mV']=(f.y_pred_delta_V-f.y_true_delta_V)*1000
        f['absolute_error_mV']=f.signed_error_mV.abs()
        top=f.sort_values(['absolute_error_mV',*KEYS],ascending=[False,True,True,True]).head(10)
        extreme.append(top[['model','seed',*KEYS,'signed_error_mV','absolute_error_mV']])
    extreme_df=pd.concat(extreme,ignore_index=True)
    extreme_df.to_csv(outputs/'O1_SEED62_EXTREME_QUERIES.csv',index=False,float_format='%.17g',lineterminator='\n')
    from o1_figures_reports import deliver
    deliver(PHASE, metric_df,horizon_df,target_df,pair_df,summary_df,extreme_df,status)
    table_records=verify_saved_tables(tables)
    required_reports=['O1_VALIDATION_IDENTITY_REPORT','O1_NINE_MODEL_ROBUSTNESS_COMPARISON',
        'O1_SEED62_EXTREME_ERROR_ANALYSIS','O1_PHASE_O_TERMINOLOGY_ERRATA',
        'O1_PAPER_FIGURE_AND_TABLE_INDEX','O1_FINAL_RESEARCH_DECISION']
    for name in required_reports:
        path=PHASE/'reports'/(name+'.md')
        require(path.is_file() and path.stat().st_size>0,'REPORT_MISSING:'+name)
    figure_paths=sorted((PHASE/'figures').glob('*'))
    require(len(figure_paths)==12 and all(p.stat().st_size>0 for p in figure_paths),'FIGURE_SET_INCOMPLETE')
    audit_record={'base_commit':BASE,'identity_status':'VALIDATION_CACHE_IDENTITY_PASS',
        'robustness_status':status,'evidence_status':'VALIDATION_EVIDENCE_COMPLETED',
        'protocol_sha256':manifest['protocol_sha256'],'historical_protocol_sha256':H_SHA,
        'cache_checks':checks,'hist_metric_differences':diffs,'numeric_tolerance_mV':TOL,
        'exact_shared_keys_and_truth':True,'unique_targets_statistically_independent':False,
        'historical_J_exact_query_identity_report_verified':True,'J_prediction_cache_read':False,
        'all_queries_retained':True,'seed_selected':False,'training_performed':False,
        'model_forward_calls':0,'checkpoint_bytes_read':False,'assessment_point_predictions_read':False,
        'test_numeric_accessed':False,'raw_N1_read_or_hash':False,'new_model_predictions':False,
        'formal_metric_unchanged':'FORMAL_QUERY_WEIGHTED_RMSE','no_significance_testing':True,
        'synthetic_tests':test,'source_hashes_rechecked_after_run':True,
        'saved_metric_table_round_trip_audit':table_records,
        'figure_outputs':[{ 'path':p.relative_to(PHASE).as_posix(),'sha256':sha(p.read_bytes())} for p in figure_paths],
        'execution_code_sha256':{str(p.relative_to(PHASE)):sha(p.read_bytes()) for p in
           [Path(__file__).resolve(),PHASE/'scripts/o1_figures_reports.py',PHASE/'tests/test_o1_synthetic_metrics.py']},
        'source_access_scope':'closed allowlist and recorded tool trace; not OS-wide I/O instrumentation',
        'utc':datetime.now(timezone.utc).isoformat()}
    for row in manifest['metadata']+manifest['caches']:
        verify_hash(row['path'],row['sha256'])
    write_json(outputs/'O1_IDENTITY_AND_RECOMPUTATION_AUDIT.json',audit_record)
    print(json.dumps({'identity':audit_record['identity_status'],'robustness':status,
       'formal_equal_seed_RMSE_mV':overall,'test_accessed':False},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    try:
        freeze() if args.freeze else audit()
    except (AuditError,FileNotFoundError,ValueError,KeyError) as error:
        write_json(PHASE/'outputs/O1_BLOCKER.json',{'identity_status':'VALIDATION_CACHE_IDENTITY_BLOCKED',
            'robustness_status':'INCONCLUSIVE','evidence_status':'VALIDATION_EVIDENCE_BLOCKED',
            'reason':str(error),'comparison_published':False,'test_accessed':False})
        raise
