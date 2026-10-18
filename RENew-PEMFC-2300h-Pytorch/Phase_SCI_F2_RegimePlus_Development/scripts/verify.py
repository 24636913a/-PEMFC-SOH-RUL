"""Independent numerical recomputation using Python math and CSV records.

Does not call the metric function in analyze.py. Synthetic tests cover
feature information boundaries and censoring, not merely output shapes.
"""
from experiment import *
import inspect

def rd(s): return pd.read_csv(path(s),float_precision='round_trip')

def synthetic_tests():
    from analyze import metric
    toy=pd.DataFrame(dict(origin_row=[0,0,1,1],target_row=[1,24,2,25],horizon_step=[1,24,1,24],
        y_true_delta_V=[0,.001,0,.001],y_pred_delta_V=[.001,.002,-.001,0]))
    toy_result=metric(toy)
    for field in ['RMSE_mV','MAE_mV','endpoint_RMSE_mV','target_time_equal_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV']:
        req(abs(toy_result[field]-1)<1e-12,'KNOWN_VALUE_UNIT_METRIC:'+field)
    req(abs(toy_result['R2_deltaV']+3)<1e-12,'KNOWN_VALUE_R2')
    z,sc=load_data(); p=get_protocol(); x=torch.from_numpy(z['tx'][:10]).clone()
    a=derive(x,sc['mean'][13],sc['scale'][13])
    # Neither voltage nor masked channels influence added state features.
    b=x.clone(); b[:,:,[0,1,2,3,4,5,6,7,9,10,11,12]]=12345
    req(torch.equal(a,derive(b,sc['mean'][13],sc['scale'][13])),'FEATURE_UNEXPECTED_CHANNEL')
    none=x.clone(); none[:,:,13]=(0-sc['mean'][13])/sc['scale'][13]
    req(torch.equal(derive(none,sc['mean'][13],sc['scale'][13])[:,2],torch.ones(10)),'CENSOR_ABSENCE')
    last=none.clone(); last[:,-1,13]=(1-sc['mean'][13])/sc['scale'][13]
    req(torch.equal(derive(last,sc['mean'][13],sc['scale'][13])[:,2],torch.zeros(10)),'RECENCY_LATEST')
    first=none.clone(); first[:,0,13]=(1-sc['mean'][13])/sc['scale'][13]
    req(torch.equal(derive(first,sc['mean'][13],sc['scale'][13])[:,2],torch.full((10,),47/48)),'RECENCY_FIRST')
    req(len(inspect.signature(derive).parameters)==3,'FEATURE_API_ACCEPTS_FUTURE')
    # Independently recompute features in physical/raw units at every Train origin.
    f=derive(torch.from_numpy(z['tx']),sc['mean'][13],sc['scale'][13]).numpy()
    raw=z['train_x_raw'].astype(float); j=raw[:,:,8]
    expected=[]
    for i in range(len(raw)):
        stops=np.flatnonzero(raw[i,:,13]>.5)
        expected.append([(np.mean(j[i,-6:])-np.mean(j[i]))/sc['scale'][8],np.mean(raw[i,-12:,13])-np.mean(raw[i,-24:-12,13]),(47-stops[-1])/48 if len(stops) else 1])
    dif=np.max(abs(f-np.array(expected)),axis=0)
    # Numerical tolerance applies only to float32 arithmetic, never sample identity.
    req(bool((dif<5e-7).all()),'INDEPENDENT_FEATURE_ARITHMETIC')
    for seed in SEEDS:
        a=initialize(seed,'A1',p,torch.device('cpu')); b=initialize(seed,'B1',p,torch.device('cpu'))
        req(old.state_sha256(a)==old.state_sha256(b),'PAIR_INIT_CHANGED')
        a.eval(); b.eval()
        cx=a.regime_context(x); req(bool((cx[:,5:]==0).all()),'ABLATION_NOT_ZERO')
        req(bool((b.regime_context(x)[:,5:]!=0).any()),'CANDIDATE_INACTIVE_FEATURES')
        native=old.initialize_f2(seed,p['hyperparameters'],torch.device('cpu'))
        for name,t in native.state_dict().items():
            if not name.startswith('regime_gate.'):
                req(torch.equal(t,a.state_dict()[name]),'BACKBONE_INIT_CHANGED')
    return dict(passed=True,known_value_metric_tests=True,feature_independent_physical_recomputation_max_diff=dif.tolist(),censor_tests=True,voltage_not_used_for_state=True,
        no_future_arguments=True,paired_initialization=True,unchanged_backbone_initialization=True)

def main():
    p=get_protocol(); old.configure_environment('cuda'); tests=synthetic_tests(); z,sc=load_data()
    m=rd('outputs/METRICS.csv'); horizons=rd('outputs/HORIZON_METRICS.csv'); delta=rd('outputs/PAIRED_POINTWISE_DELTAS.csv')
    target_table=rd('outputs/PER_TARGET_SQUARED_ERRORS.csv')
    checks=[]; target_conservation=0
    for variant in ['B0','A1','B1']:
        for seed in SEEDS:
            f=rd(f'outputs/predictions/{variant}_seed{seed}.csv'); errors=[(float(r.y_pred_delta_V)-float(r.y_true_delta_V))*1000 for r in f.itertuples()]
            total=math.fsum(e*e for e in errors); n=len(errors); y=f.y_true_delta_V.to_list(); ym=math.fsum(y)/n
            targets=target_table.loc[(target_table.seed==seed)&(target_table.variant==variant)]
            req(len(targets)==251 and abs(math.fsum(targets.SSE_mV2)-total)<1e-7,'TARGET_SSE_CONSERVATION')
            req(abs(math.fsum(targets.SSE_share_pct)-100)<1e-9,'TARGET_SHARE_CONSERVATION')
            target_conservation+=1
            by={}
            for row,e in zip(f.itertuples(),errors): by.setdefault(row.target_row,[]).append(e*e)
            nums=dict(RMSE_mV=math.sqrt(total/n),MAE_mV=math.fsum(abs(e) for e in errors)/n,
                R2_deltaV=1-total/(math.fsum((v-ym)**2 for v in y)*1e6),
                endpoint_RMSE_mV=math.sqrt(math.fsum(e*e for r,e in zip(f.itertuples(),errors) if r.horizon_step==24)/205),
                target_time_equal_RMSE_mV=math.sqrt(math.fsum(math.fsum(v)/len(v) for v in by.values())/len(by)),
                max_abs_error_mV=max(abs(e) for e in errors),P95_mV=float(np.quantile(np.abs(errors),.95)),P99_mV=float(np.quantile(np.abs(errors),.99)))
            row=m.loc[(m.variant==variant)&(m.seed==seed)].iloc[0]
            for k,v in nums.items():
                diff=abs(v-row[k]); req(diff<1e-9,'INDEPENDENT_METRIC:'+variant+str(seed)+k)
                checks.append(dict(variant=variant,seed=seed,metric=k,independent_value=v,saved_value=row[k],difference=diff))
            req(len(f)==4920 and len(by)==251 and f.origin_row.nunique()==205,'NUMERIC_COUNT')
            req(not f.duplicated(['origin_row','target_row','horizon_step']).any(),'DUPLICATE_IDENTITY')
            for h in range(1,25):
                values=[e*e for r,e in zip(f.itertuples(),errors) if r.horizon_step==h]
                saved=horizons.loc[(horizons.variant==variant)&(horizons.seed==seed)&(horizons.horizon_step==h)].RMSE_mV.item()
                req(abs(math.sqrt(math.fsum(values)/len(values))-saved)<1e-9,'INDEPENDENT_HORIZON')
            if variant!='B0':
                ledger=readj(f'outputs/ledger/{variant}_seed{seed}.json'); req(ledger['status']=='COMPLETED','RUN_INCOMPLETE')
                req(sha(path(ledger['checkpoint']))==ledger['checkpoint_sha256'],'CHECKPOINT_SHA')
                log=rd(f'outputs/training_logs/{variant}_seed{seed}.csv'); best=log.loc[log.validation_standardized_RMSE.idxmin()]
                req(int(best.epoch)==ledger['best_epoch'],'BEST_EPOCH_SELECTION')
                req(len(log)==ledger['stop_epoch'] and len(log)<=450,'TRAINING_BUDGET')
                req(len(log)==450 or int(log.iloc[-1].stale_epochs)==40,'EARLY_STOP')
                ck=torch.load(path(ledger['checkpoint']),map_location='cpu',weights_only=True)
                req(ck['protocol_sha256']==sha(path('configs/FROZEN_PROTOCOL.json')),'CHECKPOINT_PROTOCOL')
    # All paired squared-error gains agree with predictions, with no deleted queries.
    req(len(delta)==2*3*4920,'PAIRED_COUNT')
    req(float(abs(delta.reference_error_mV**2-delta.B1_error_mV**2-delta.SSE_gain_mV2).max())<1e-8,'PAIRED_SSE')
    diagonal=rd('outputs/POSTHOC_ZERO_FEATURE_METRICS.csv')
    for seed in SEEDS:
        f=rd(f'outputs/predictions/DIAGNOSTIC_B1_new_features_zero_seed{seed}.csv')
        errors=[(float(r.y_pred_delta_V)-float(r.y_true_delta_V))*1000 for r in f.itertuples()]
        rmse=math.sqrt(math.fsum(e*e for e in errors)/len(errors))
        req(abs(rmse-diagonal.loc[diagonal.seed==seed].RMSE_mV.item())<1e-9,'POSTHOC_ZERO_RMSE')
    csv('outputs/INDEPENDENT_METRIC_RECOMPUTATION.csv',checks)
    forward=rd('outputs/FINAL_FORWARD_IDENTITY.csv'); req(len(forward)==9 and forward.prediction_exact.all() and forward.weights_unchanged.all(),'FINAL_FORWARD')
    js('outputs/INDEPENDENT_QA.json',dict(status='PASS',synthetic_tests=tests,independent_metrics=len(checks),horizon_checks=216,
        pointwise_rows=9*4920,paired_rows=len(delta),checkpoint_and_best_epoch=True,all_queries_retained=True,final_exact_forward_sets=9,
        target_SSE_conservation_sets=target_conservation,posthoc_zero_feature_RMSE_checks=3,
        features_and_protocol_frozen=True,new_Assessment_or_late_inference=False))
    print('INDEPENDENT_QA_PASS',len(checks))

if __name__=='__main__': main()
