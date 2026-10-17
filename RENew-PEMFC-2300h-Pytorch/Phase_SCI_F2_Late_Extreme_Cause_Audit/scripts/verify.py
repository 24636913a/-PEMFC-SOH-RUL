"""Independent conservation arithmetic, input identity and historical protection."""
from audit_common import *
import math,ast
def verify_official_metrics():
    observed=rd(P/'outputs/EVENT_SSE_CONTRIBUTIONS.csv');observed=observed.loc[observed.event=='COMBINED']
    frozen=rd(FULL/'outputs/metrics/PER_MODEL_SEED_PARTITION.csv');frozen=frozen.loc[frozen.partition=='LATE_LIFE'];rows=[]
    for row in observed.itertuples():
        old=frozen.loc[(frozen.model==row.model)&(frozen.seed==row.seed)].iloc[0]
        difference=abs(row.formal_RMSE_mV-old.RMSE_mV);assert difference<1e-10
        rows.append(dict(model=row.model,seed=row.seed,all_queries=row.all_queries,diagnostic_recomputed_RMSE_mV=row.formal_RMSE_mV,frozen_official_RMSE_mV=old.RMSE_mV,numeric_difference_mV=difference,formal_score_unchanged=True))
    for seed in SEEDS:
        current=observed.loc[observed.seed==seed].sort_values('formal_RMSE_mV').model.tolist()
        old=frozen.loc[frozen.seed==seed].sort_values('RMSE_mV').model.tolist();assert current==old
    cs('outputs/FORMAL_METRIC_AND_RANKING_PARITY.csv',rows)
def main():
    check_protocol();verify_official_metrics();errors=rd(P/'outputs/POINTWISE_ERRORS.csv');c=rd(P/'outputs/EVENT_SSE_CONTRIBUTIONS.csv');t=rd(P/'outputs/PER_TARGET_ERRORS.csv');h=rd(P/'outputs/PER_HORIZON_ERRORS.csv');o=rd(P/'outputs/PER_ORIGIN_ERRORS.csv');checks=0;maximum=0.
    assert len(errors)==15*13152 and not errors.duplicated(['model','seed','origin_row','target_row','horizon_step']).any()
    for (model,seed),z in errors.groupby(['model','seed']):
        total=math.fsum(float(v)**2 for v in z.error_mV);n=len(z)
        for table in [t,h,o]:
            a=math.fsum(table.loc[(table.model==model)&(table.seed==seed),'SSE_mV2']);diff=abs(a-total);assert diff<1e-7;maximum=max(maximum,diff);checks+=1
        for row in c.loc[(c.model==model)&(c.seed==seed)].itertuples():
            ids=list(EVENTS[row.event]) if row.event in EVENTS else [3112,3113,3627,3628]
            event=math.fsum(float(v)**2 for v in z.loc[z.target_row.isin(ids),'error_mV'])
            by={}
            for r in z.itertuples():by.setdefault(r.target_row,[]).append(float(r.error_mV)**2)
            means={k:math.fsum(v)/len(v) for k,v in by.items()}
            values={'formal_RMSE_mV':math.sqrt(total/n),'query_weighted_SSE_share_pct':event/total*100,'target_time_equal_MSE_share_pct':math.fsum(means[k] for k in ids)/math.fsum(means.values())*100}
            for key,value in values.items():
                diff=abs(value-getattr(row,key));assert diff<1e-10;maximum=max(maximum,diff);checks+=1
            assert row.event_queries==24*len(ids) and row.all_unique_targets==640
    rawsummary=rd(P/'outputs/RAW_BIN_MIXTURE_RECOMPUTATION.csv')
    for event in EVENTS:
        samples=rd(P/f'outputs/RAW_EVENT_SAMPLES_{event}.csv')
        cell=[c for c in samples if c.startswith('U') and c.endswith('[V]')]
        assert len(cell)==10 and 'U6 [V]' not in cell
        voltage=samples[cell].mean(axis=1).to_numpy();assert np.max(abs(voltage-samples.active_10cell_mean_V))<1e-15
        for r in rawsummary.loc[rawsummary.event==event].itertuples():
            z=samples.loc[samples.bin_start_h==r.age_h-.5];assert len(z)==r.raw_samples
            value=math.fsum(z.active_10cell_mean_V)/len(z);assert abs(value-r.frozen_bin_mean_V)<1e-14;checks+=1
            assert int((abs(z.current_A)<=.1).sum())==r.zero_current_samples
    forwards=rd(P/'outputs/FORWARD_IDENTITY_AUDIT.csv');assert len(forwards)==9
    assert forwards.prehook_replay_exact.all() and forwards.observing_hooks_prediction_exact.all() and forwards.weights_unchanged.all() and forwards.no_interventions.all()
    pair=rd(P/'outputs/PAIR_STATE_INVERSION.csv');assert (pair.pred_first_V<pair.pred_second_V).equals(pair.pair_inverted)
    # No calls to training, automatic differentiation or optimizers in new code.
    for file in (P/'scripts').glob('*.py'):
        tree=ast.parse(file.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                assert name not in {'backward','step','fit','train_new_f2','train_seed','AdamW','Adam','SGD'},(file.name,name)
    js('outputs/INDEPENDENT_NUMERIC_VERIFICATION.json',dict(status='PASS',checks=checks,max_numeric_difference=maximum,pointwise_rows=len(errors),original_queries_retained=True,raw_mixture_recomputed=True,original_forward_sets_exact=9,no_training_or_interventions=True))
    # Reuse already-sealed hashes rather than generating a new historical baseline.
    rows=rd(FULL/'outputs/audits/HISTORY_BEFORE.csv').to_dict('records')
    for base,manifest in [(FULL,FULL/'outputs/audits/ARTIFACT_MANIFEST.csv'),(ROOT,ROOT/'RealHyFC_F2_Main_Outputs/DELIVERY_MANIFEST.csv')]:
        for r in rd(manifest).to_dict('records'):rows.append(dict(path=(base/r['path']).relative_to(REPO).as_posix(),sha256=r['sha256']))
    for file in [FULL/'outputs/audits/ARTIFACT_MANIFEST.csv',FULL/'outputs/audits/FINAL_QA.json',ROOT/'RealHyFC_F2_Main_Outputs/DELIVERY_MANIFEST.csv']:
        name=file.relative_to(REPO).as_posix();blob=file.read_bytes();expected=subprocess.check_output(['git','rev-parse','HEAD:'+name],cwd=REPO).decode().strip()
        assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==expected
        rows.append(dict(path=name,sha256=sha(file)))
    assert len(rows)==len({r['path'] for r in rows})==65683
    for i,r in enumerate(rows):
        assert sha(REPO/r['path'])==r['sha256'],('HISTORY_CHANGED',r['path'])
        if (i+1)%15000==0:print('Historical SHA checked',i+1,flush=True)
    js('outputs/HISTORICAL_PROTECTION.json',dict(status='PASS',historical_files=65683,all_SHA256_unchanged=True,checkpoint_weights_unchanged=True,formal_predictions_unchanged=True,paper_conclusions_unmodified=True))
    print('AUDIT_VERIFICATION_PASS',checks,flush=True)
if __name__=='__main__':main()
