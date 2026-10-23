"""Stage-gated, resumable read-only evaluation. See README for commands."""
from common import *
import argparse,traceback
from load_full_realhyfc import load_full,save_structure
from build_chronological_partitions import save_partitions,partitions
from build_legal_windows import build,save_windows
from preprocess_frozen_inputs import preprocess,verify_h_data,scaler
from load_frozen_models import make_inventory
def prepare():
    assert not (P/'outputs/audits/WINDOW_FREEZE.json').exists(),'WINDOWS_ALREADY_FROZEN'
    base=git('rev-parse','HEAD');assert not git('diff','--name-only'),'PREEXISTING_TRACKED_MODIFICATIONS'
    event('START',base_commit=base);print('Protecting tracked history...',flush=True);protected=snapshot_history()
    f=load_full();save_structure(f);save_partitions(f);ds={};counts=[]
    for name,(a,b) in partitions(len(f)).items():
        d,legal,excluded=build(f,a,b);assert len(d['origins'])>0
        d['x']=preprocess(d['x_raw']);ds[name]=d;counts.append(save_windows(name,d,legal,excluded))
        print(name,counts[-1],flush=True)
    verify_h_data(ds)
    for i,a in enumerate(PARTS):
        for b in PARTS[i+1:]:assert not np.intersect1d(ds[a]['targets'],ds[b]['targets']).size
    js('outputs/audits/WINDOW_COUNTS.json',counts)
    cs('tables/WINDOW_COUNTS.csv',[{k:v for k,v in z.items() if k!='primary_reason_counts'} for z in counts])
    make_inventory();js('outputs/audits/SCALER.json',scaler())
    frozen={f.relative_to(REPO).as_posix():sha(f) for f in (P/'outputs/audits').glob('*') if f.is_file()}
    js('outputs/audits/WINDOW_FREEZE.json',dict(base_commit=base,protected_historical_files=protected,source_sha256=sha(SOURCE),frozen_sha256=frozen,origin_convention='origin in partition, all 24 future targets in partition; history may be from previous partition',input_only='48 rows <=origin; future raw14 checked ONLY for inherited eligibility',exploratory=True))
def freeze():
    assert not (P/'configs/FROZEN_CHRONOLOGICAL_PROTOCOL.yaml').exists(),'PROTOCOL_ALREADY_FROZEN'
    assert len(rd(P/'outputs/audits/VALIDATION_EXACT_REPLAY.csv'))==15
    assert read(P/'outputs/audits/PREFLIGHT_TESTS.json')['pass']
    frozen=read(P/'outputs/audits/WINDOW_FREEZE.json')['frozen_sha256']
    for key,v in frozen.items():assert sha(REPO/key)==v
    files=list((P/'scripts').glob('*.py'))+list((P/'tests').glob('*.py'))+[SOURCE,P/'outputs/audits/WINDOW_FREEZE.json',P/'outputs/audits/VALIDATION_EXACT_REPLAY.csv']
    files+=list((H/'scripts').glob('*.py'))+list((B/'models').glob('*.py'))+[R2/'outputs/audits/SCALER.json',H/'configs/H_FROZEN_PROTOCOL.yaml',B/'configs/FROZEN_BASELINE_PROTOCOL.yaml']
    # Cover all historical tracked code/config/checkpoints through the immutable protection manifest.
    for r in inventory().itertuples():files.extend([REPO/r.checkpoint,REPO/r.reference_prediction])
    frozen.update({f.relative_to(REPO).as_posix():sha(f) for f in files})
    c=dict(phase=P.name,base_commit=read(P/'outputs/audits/WINDOW_FREEZE.json')['base_commit'],status='FROZEN_AFTER_EXACT_VALIDATION_REPLAY_BEFORE_FULL_DOMAIN_INFERENCE',authorization='user scheme B full chronological exploratory evaluation',partitions=partitions(read(P/'outputs/audits/FULL_SOURCE.json')['rows']),models=MODELS,seeds=SEEDS,history_steps=48,horizon_steps=24,resolution_h=.5,batch_size=32,raw_features=FEATURES,Top10_indices=TOP10,EWMA_alpha=.3,scaler='historical Train-only unfiltered legal histories; no refit used',target='raw delta V, scale once, raw origin anchor',metrics=['RMSE_mV','MAE_mV','R2_on_deltaV','endpoint_RMSE_mV','target_time_equal_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV','24_horizon_RMSE'],target_time_equal='sqrt(mean_unique_target(mean_queries(squared_error)))',seed_summary='equal seed arithmetic mean and sample SD ddof=1',case_rules=dict(illustration='middle legal LATE_LIFE origin, Seed42, all five models',hard='largest three-seed F2 per-origin mean RMSE in LATE_LIFE; tie lowest origin',F2_life_bins='20 equal operating-hour bins within each partition, per-origin errors; no voltage window stitching'),constraints=['no training, model selection or hyperparameter changes','no AHC or ELTE entrypoints','Phase L confirmatory Test not eligible unchanged','Assessment and late-life exploratory, not independent Test'],frozen_sha256=frozen)
    text=yaml.safe_dump(c,allow_unicode=True,sort_keys=False)
    write('configs/FROZEN_CHRONOLOGICAL_PROTOCOL.yaml',text);write('configs/FROZEN_CHRONOLOGICAL_PROTOCOL.sha256',sha(P/'configs/FROZEN_CHRONOLOGICAL_PROTOCOL.yaml')+'\n');event('PROTOCOL_FROZEN',sha256=sha(P/'configs/FROZEN_CHRONOLOGICAL_PROTOCOL.yaml'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','validation','freeze','infer','audit','figures','reports','finalize']);args=ap.parse_args()
    try:
        if args.stage=='prepare':prepare()
        elif args.stage=='preflight':
            from audit_prediction_alignment import run_tests
            run_tests('PREFLIGHT_TESTS')
        elif args.stage=='validation':
            from run_full_chronological_inference import validation_replay
            validation_replay()
        elif args.stage=='freeze':freeze()
        elif args.stage=='infer':
            from run_full_chronological_inference import run_remaining
            run_remaining()
            from calculate_partition_metrics import calculate
            calculate()
        elif args.stage=='audit':
            from audit_prediction_alignment import audit,run_tests
            audit();run_tests('FINAL_TESTS')
        elif args.stage=='figures':
            from generate_full_life_figures import generate
            generate()
        elif args.stage=='reports':
            from write_reports import generate
            generate()
        elif args.stage=='finalize':
            from audit_prediction_alignment import finalize
            finalize()
    except Exception:
        error=traceback.format_exc();event('BLOCKED',stage=args.stage,error=error);write(f'outputs/execution_ledger/FAILURE_{args.stage}.txt',error);raise
if __name__=='__main__':main()
