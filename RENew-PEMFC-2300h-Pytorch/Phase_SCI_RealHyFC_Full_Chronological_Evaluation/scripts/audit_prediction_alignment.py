"""Independent scalar recomputation and strict identity checks, no model fitting."""
from common import *
import math,unittest,io
def quantile(values,p):
    a=sorted(values);position=(len(a)-1)*p;i=int(math.floor(position));j=int(math.ceil(position))
    return a[i]+(a[j]-a[i])*(position-i)
def independent(f):
    y=list(map(float,f.y_true_delta_V));p=list(map(float,f.y_pred_delta_V));e=[(b-a)*1000 for a,b in zip(y,p)];a=list(map(abs,e));n=len(e)
    se=math.fsum(z*z for z in e);mean=math.fsum(y)/n;sst=math.fsum((z-mean)**2 for z in y)
    endpoint=[z for z,h in zip(e,f.horizon_step) if h==24];by={}
    for t,z in zip(f.target_row,e):by.setdefault(int(t),[]).append(z*z)
    return dict(RMSE_mV=math.sqrt(se/n),MAE_mV=math.fsum(a)/n,R2=1-math.fsum((z/1000)**2 for z in e)/sst,endpoint_RMSE_mV=math.sqrt(math.fsum(z*z for z in endpoint)/len(endpoint)),target_time_equal_RMSE_mV=math.sqrt(math.fsum(math.fsum(z)/len(z) for z in by.values())/len(by)),P95_mV=quantile(a,.95),P99_mV=quantile(a,.99),max_abs_error_mV=max(a))
def run_tests(name):
    suite=unittest.defaultTestLoader.discover(str(P/'tests'),pattern='test_*.py');stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(f'outputs/audits/{name}.txt',stream.getvalue());js(f'outputs/audits/{name}.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),pass_=result.wasSuccessful(),**{'pass':result.wasSuccessful()}))
    print(stream.getvalue(),flush=True);assert result.wasSuccessful(),'ENGINEERING_TESTS_FAILED'
def audit():
    protocol();from calculate_partition_metrics import metrics
    master=rd(P/'outputs/metrics/PER_MODEL_SEED_PARTITION.csv');steps=rd(P/'outputs/metrics/HORIZON_METRICS.csv');rows=[];maxdiff=0.;checked=0
    src=rd(P/'outputs/audits/FULL_ROW_IDENTITY.csv');v=src.cell_voltage_mean.to_numpy(float)
    clocks=[];calendar_gap=(pd.to_datetime(src.first_datetime)-pd.to_datetime(src.last_datetime).shift()).dt.total_seconds()
    for part in PARTS:
        d=data(part);o=d['origins'];t=d['targets'];expect=dict(origin_row=np.repeat(o,24),target_row=t.ravel(),horizon_step=np.tile(np.arange(1,25),len(o)),y_true_delta_V=d['y'].ravel())
        pairs=np.unique((o[:,None]+np.arange(-46,25)).ravel())
        flags=int(src.loc[pairs,'gap_before_flag'].sum());maxgap=float(calendar_gap.iloc[pairs].max())
        assert flags==0 and maxgap<=1.01,('LEGAL_WINDOW_CALENDAR_BREAK',part,flags,maxgap)
        clocks.append(dict(partition=part,internal_row_pairs=len(pairs),gap_flags=flags,max_calendar_interrow_gap_s=maxgap,nominal_sample_interval_s=1,chronological_source_order_unchanged=True))
        assert np.array_equal(d['x_raw'][:,:,0],v[o[:,None]+np.arange(-47,1)].astype(np.float32))
        assert np.array_equal(d['y'],(v[t]-v[o,None]).astype(np.float32))
        for model in MODELS:
            for seed in SEEDS:
                f=rd(P/f'outputs/predictions/{part}_{model}_seed{seed}.csv')
                for k,a in expect.items():assert np.array_equal(f[k],a),('PREDICTION_KEY_TRUTH_MISMATCH',part,model,seed,k)
                assert np.array_equal(f.origin_raw_voltage_V,np.repeat(v[o],24))
                assert np.array_equal(f.predicted_voltage_V,f.origin_raw_voltage_V+f.y_pred_delta_V)
                assert np.array_equal(f.true_raw_target_voltage_V,v[t].ravel())
                old=master.loc[(master.partition==part)&(master.model==model)&(master.seed==seed)].iloc[0]
                independent_result=independent(f)
                for k,val in independent_result.items():
                    diff=abs(val-float(old[k]));maxdiff=max(maxdiff,diff);checked+=1;assert diff<1e-9,('METRIC_MISMATCH',part,model,seed,k,diff)
                for horizon,z in f.groupby('horizon_step'):
                    e=[(p-y)*1000 for y,p in zip(z.y_true_delta_V,z.y_pred_delta_V)];value=math.sqrt(math.fsum(a*a for a in e)/len(e))
                    oldstep=steps.loc[(steps.partition==part)&(steps.model==model)&(steps.seed==seed)&(steps.horizon_step==horizon)].iloc[0]
                    diff=abs(value-oldstep.RMSE_mV);maxdiff=max(maxdiff,diff);checked+=1;assert diff<1e-9
                rows.append(dict(partition=part,model=model,seed=seed,queries=len(f),keys_truth_raw_anchor_exact=True,independent_metrics_pass=True,prediction_sha256=sha(P/f'outputs/predictions/{part}_{model}_seed{seed}.csv')))
    cs('outputs/audits/PREDICTION_AND_METRIC_AUDIT.csv',rows);cs('outputs/audits/LEGAL_CLOCK_CONTINUITY.csv',clocks)
    # Sample SD and mean independently, rather than trusting groupby summaries.
    for r in rd(P/'tables/MODEL_THREE_SEED_SUMMARY.csv').itertuples():
        z=master.loc[(master.partition==r.partition)&(master.model==r.model)]
        for k in independent_result:
            vals=list(map(float,z[k]));mean=math.fsum(vals)/3;sd=math.sqrt(math.fsum((a-mean)**2 for a in vals)/2)
            for suffix,value in [('mean',mean),('sample_sd',sd)]:
                diff=abs(value-getattr(r,k+'_'+suffix));checked+=1;maxdiff=max(maxdiff,diff);assert diff<1e-9
    protected=protect()
    for r in inventory().itertuples():assert sha(REPO/r.checkpoint)==r.checkpoint_sha256
    js('outputs/audits/INDEPENDENT_RECALCULATION.json',dict(pass_=True,numeric_checks=checked,max_absolute_metric_difference=maxdiff,prediction_sets=60,historical_files_unchanged=protected,raw_anchor_exact=True,all_queries_retained=True,**{'pass':True}))
def finalize():
    protocol();qa=read(P/'outputs/audits/INDEPENDENT_RECALCULATION.json');protected=qa['historical_files_unchanged']
    # The full SHA sweep is already done by audit; no historical write follows.
    changed=git('diff','--name-only').splitlines();assert all((REPO/f).resolve().is_relative_to(P) for f in changed)
    for r in inventory().itertuples():assert sha(REPO/r.checkpoint)==r.checkpoint_sha256
    assert read(P/'outputs/audits/FINAL_TESTS.json')['pass'];assert qa['pass']
    assert len(list((P/'figures').glob('*.png')))==7 and len(list((P/'figures').glob('*.pdf')))==7
    rows=[]
    for f in sorted(P.rglob('*')):
        if f.is_file() and f.name not in ['ARTIFACT_MANIFEST.csv','FINAL_QA.json','GIT_RECEIPT.local.json'] and not f.name.endswith('.writing') and '__pycache__' not in f.parts:
            rows.append(dict(path=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
    cs('outputs/audits/ARTIFACT_MANIFEST.csv',rows)
    js('outputs/audits/FINAL_QA.json',dict(status='REALHYFC_FULL_CHRONOLOGICAL_EVALUATION_COMPLETED',protocol_sha256=sha(P/'configs/FROZEN_CHRONOLOGICAL_PROTOCOL.yaml'),checkpoint_count=15,prediction_sets=60,validation_replays_exact=15,historical_files_unchanged=protected,engineering_tests=read(P/'outputs/audits/FINAL_TESTS.json'),independent_recalculation=read(P/'outputs/audits/INDEPENDENT_RECALCULATION.json'),figures_png=7,figures_pdf=7,training_performed=False,phase_L_unchanged=True,independent_Test_claim=False,artifact_count=len(rows)))
