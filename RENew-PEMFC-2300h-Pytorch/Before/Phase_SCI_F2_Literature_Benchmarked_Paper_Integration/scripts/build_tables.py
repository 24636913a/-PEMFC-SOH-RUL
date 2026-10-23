"""Paper tables and figure inputs from verified cache-only results."""
from common import *
def main():
    c=verified_config();assert read(P/'outputs/audits/INDEPENDENT_RESULT_AUDIT.json')['pass_']
    frozen=read(R2/'configs/frozen_seed_screening.json');hp=frozen['hyperparameters'];fields=frozen['task']['raw_f2_feature_order'];top=frozen['preprocessing']['Top10_indices']
    features=[dict(slot=i,name=name,unit='V' if i<5 else ('A' if i<8 else ('A/cm2' if i<11 else ('count/bin' if i<13 else 'fraction'))),Top10_selected=i in top,role='past only',EWMA=i==0) for i,name in enumerate(fields)]
    cs('tables/T01_FEATURES.csv',features)
    split=[dict(region='Train',row_start=0,row_stop_exclusive=2103,windows=1380,forecast_queries=33120,use='historical fitting only'),dict(region='Validation',row_start=2103,row_stop_exclusive=2404,windows=205,forecast_queries=4920,use='reused Development; primary comparison and screening'),dict(region='Assessment',row_start=2404,row_stop_exclusive=3005,windows=378,forecast_queries=9072,use='reused Development; existing I2 summary only')]
    cs('tables/T02_SPLITS.csv',split)
    settings=[dict(setting=k,value=str(v)) for k,v in hp.items() if k!='seed']+[dict(setting='formal_seeds',value='42,52,62'),dict(setting='history_future_steps',value='48->24'),dict(setting='resolution_h',value='.5'),dict(setting='Top10_indices',value=str(top)),dict(setting='EWMA_alpha',value='.3'),dict(setting='loss',value='MSE on DeltaV / Train voltage SD'),dict(setting='attention_heads',value='4'),dict(setting='lstm_layers',value='1')]
    cs('tables/T03_FROZEN_SETTINGS.csv',settings)
    m=rd(P/'outputs/experiment_tables/FORMAL_VALIDATION_METRICS.csv');cs('tables/T04_FORMAL_MODEL_COMPARISON.csv',m)
    summary=rd(P/'outputs/experiment_tables/EQUAL_SEED_SUMMARY.csv');cs('tables/T05_EQUAL_SEED_SUMMARY.csv',summary)
    selected=rd(P/'outputs/experiment_tables/SIX_SELECTED_INSTANCES.csv')
    b0=rd(SIX/'outputs/METRICS.csv');b0=b0[b0['mode']=='B0'][['seed','target_time_equal_RMSE_mV']]
    selected=selected.merge(b0,on='seed',validate='one_to_one');cs('tables/T06_SELECTED_F2_INSTANCES.csv',selected)
    p2=rd(P2/'outputs/metrics/P2_NINE_RUN_METRICS.csv');cs('tables/T07_INDEPENDENT_TRAINING_ABLATION.csv',p2)
    cs('tables/T08_CONDITION_RESULTS.csv',rd(P/'outputs/condition_analysis/CONDITION_METRICS.csv'))
    cs('tables/T09_PAIRED_REGIME_RESULTS.csv',rd(P/'outputs/condition_analysis/PAIRED_REGIME_COMPARISONS.csv'))
    cs('tables/T10_REUSED_ASSESSMENT.csv',rd(P/'outputs/experiment_tables/REUSED_ASSESSMENT_METRICS.csv'))
    r1=rd(R1/'outputs/F2_SEED_RANKING.csv');r2=rd(R2/'outputs/ROUND2_SEED_RANKING.csv')
    pool=pd.concat([r1[['seed','RMSE_mV']].assign(round='R1'),r2[['seed','RMSE_mV']].assign(round='R2')],ignore_index=True);assert len(pool)==30 and pool.seed.nunique()==30
    cs('outputs/experiment_tables/ALL_COMPLETED_SCREENED_SEEDS.csv',pool)
    c1=read(R1/'configs/frozen_seed_screening.json');s1=read(R1/'outputs/FINAL_SUMMARY.json');s2=read(R2/'outputs/FINAL_SUMMARY.json')
    budget=dict(R1_registered=len(c1['seeds']),R1_pool=c1['seeds'],R1_attempted=len(c1['seeds'])-len(s1['not_started']),R1_completed=len(r1),R1_failed=s1['failed'],R1_unstarted=len(s1['not_started']),R1_wall_limit_h=c1['maximum_wall_s']/3600,
        R2_registered_maximum=len(frozen['seeds']),R2_pool=frozen['seeds'],R2_attempted=s2['attempted'],R2_completed=len(r2),R2_qualified=s2['qualified_new_seeds'],R2_unstarted=s2['not_started'],R2_wall_limit_h=frozen['maximum_wall_s']/3600,
        new_attempts=len(c1['seeds'])-len(s1['not_started'])+s2['attempted'],new_completed=len(r1)+len(r2),new_failed=len(s1['failed'])+len(s2['failed_or_timeout']),seed42_source='historical H, not retrained in two rounds',
        selected_six=SELECTED,selection='original F2 Validation pooled RMSE and R2 predeclared endpoint/tail conditions; selected instances are not unbiased seed population',
        R2_thresholds=frozen['thresholds'],threshold_timing='R2 thresholds frozen after R1 results were already known; historical42/103/193 retrospectively checked, not R1 preregistered success',R2_stop='three new A+B seeds found; remaining38 not run')
    js('outputs/experiment_tables/SCREENING_DISCLOSURE.json',budget)
    states=[]
    for seed in c1['seeds']:
        states.append(dict(round='R1',seed=seed,state='COMPLETED' if seed in set(r1.seed) else ('FAILED_PRESERVED_NOT_RETRAINED' if seed in s1['failed'] else 'NOT_STARTED')))
    for seed in frozen['seeds']:
        states.append(dict(round='R2',seed=seed,state='COMPLETED' if seed in set(r2.seed) else 'NOT_STARTED'))
    cs('outputs/experiment_tables/SCREENING_CANDIDATE_STATES.csv',states)
    extra=[R2/'outputs/ROUND2_SEED_RANKING.csv']
    before=rd(P/'outputs/audits/HISTORY_BEFORE.csv').set_index('path')
    for path in extra:assert sha(path)==before.loc[str(path.relative_to(REPO)).replace('\\','/'),'sha256']
    cs('outputs/evidence_index/ADDITIONAL_TABLE_SOURCES.csv',[dict(path=str(p.relative_to(REPO)).replace('\\','/'),sha256=sha(p)) for p in extra])
    # Structural figure coordinates are original diagram specifications, not published figures.
    spec=dict(task=dict(history_steps=48,horizon_steps=24,resolution_h=.5,raw_slots=14,Top10=top,target='raw DeltaV',decoder='relative horizon index only'),
        branch=dict(kernel_size=3,dilations=[1,4],causal_padding='left only',input='same encoder VSN sequence',mix='concat ->1x1 conv ->dropout+residual+LayerNorm'),
        fusion='g*post_LSTM_encoder+(1-g)*CNN_encoder',encoder_state_route='original encoder LSTM hidden/cell -> native decoder LSTM; fused features do not replace these states',
        downstream='concatenate fused48 encoder representations with24 decoder representations before native enrichment and attention',
        gate_inputs=[11,13,9,6,8],gate_layers=[5,8,1],gate_activation=['ELU','sigmoid'],
        parameters={'F0':323483,'F1':332379,'F2':332436},inherited=['VSN','LSTM encoder/decoder','GRN/GLU','attention','residual normalization'],
        added=['causal multiscale depthwise convolution','encoder path fusion','history-statistics scalar gate'],sources=[str((ROOT/'Phase_F_TFT_DualPath_RegimeFusion/models').relative_to(REPO)),str((ROOT/'Phase_TFT_Original_Baseline/models/canonical_tft.py').relative_to(REPO))])
    js('outputs/experiment_tables/DIAGRAM_SPEC.json',spec)
    for name,title,cols in [('T01_FEATURES','冻结输入槽位',None),('T02_SPLITS','数据划分',None),('T03_FROZEN_SETTINGS','训练设置',None),('T04_FORMAL_MODEL_COMPARISON','正式Validation比较',None),('T05_EQUAL_SEED_SUMMARY','等Seed汇总',None),('T06_SELECTED_F2_INSTANCES','筛选后六实例',['seed','RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','target_time_equal_RMSE_mV','max_abs_error_mV']),('T07_INDEPENDENT_TRAINING_ABLATION','独立训练门控消融',['model','seed','RMSE_mV','endpoint_12h_RMSE_mV','target_time_equal_RMSE_mV','max_abs_error_mV']),('T09_PAIRED_REGIME_RESULTS','同工况配对结果',None)]:
        f=rd(P/'tables'/f'{name}.csv');f=f[cols] if cols else f;text('tables/'+name+'.md','# '+title+'\n\n'+mdtable(f)+'\n')
    # A compact, unambiguous result bundle for manuscript writers/reviewers.
    avg=m[m.model!='Persistence'].groupby('model').RMSE_mV.mean();groups=rd(P/'outputs/condition_analysis/PAIRED_REGIME_COMPARISONS.csv')
    result=dict(formal_mean_RMSE_mV=avg.to_dict(),formal_SD_RMSE_mV=m[m.model!='Persistence'].groupby('model').RMSE_mV.std(ddof=1).to_dict(),
        relative_mean_RMSE_improvement_percent={k:float(100*(avg[k]-avg['F2'])/avg[k]) for k in ['F0','F1']},
        persistence=m[m.model=='Persistence'].iloc[0].to_dict(),condition_pairs=groups.to_dict('records'),
        condition_counts={k:rd(P/'outputs/condition_analysis/ORIGIN_CONDITIONS.csv')[k+'_group'].value_counts().to_dict() for k in CONDITIONS},
        selected_best_existing=dict(seed=202,RMSE_mV=float(selected[selected.seed==202].RMSE_mV.iloc[0])),
        gate=read(SIX/'outputs/SUMMARY.json'),scope='all numbers Development; selected6 must not be treated as unbiased; no training or new inference')
    js('outputs/experiment_tables/PAPER_NUMERIC_FACTS.json',result);print('TEN_PAPER_TABLES_AND_FIGURE_DATA_READY')
if __name__=='__main__':main()
