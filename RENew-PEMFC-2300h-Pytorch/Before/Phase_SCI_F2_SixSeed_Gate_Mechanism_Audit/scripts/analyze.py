"""All diagnostic statistics from persisted physical-unit predictions and gate CSVs."""
import itertools
from scipy.stats import spearmanr
from common import *
def main():
    c=cfg(); allm=[]; horizons=[]; win=[];stats=[];cor=[];strata=[];pairs=[];gates=[]
    thresholds=rd(P/'configs/train_condition_thresholds.csv')
    for s in SEEDS:
        assert read(P/f'outputs/audits/seed_{s}.json')['pass']
        f=rd(P/f'outputs/validation_gates_seed_{s}.csv');g=f.g.to_numpy();gates.append(f)
        q=np.percentile(g,[5,25,75,95]); sd=float(g.std(ddof=1)); rg=float(np.ptp(g))
        stats.append(dict(seed=s,n=len(g),mean=float(g.mean()),median=float(np.median(g)),SD=sd,min=float(g.min()),max=float(g.max()),
            P5=q[0],P25=q[1],P75=q[2],P95=q[3],range=rg,IQR=q[2]-q[1],dynamic=sd>.001 and rg>.01,
            low_lt_0p1_fraction=float(np.mean(g<.1)),middle_0p4_to_0p6_fraction=float(np.mean((g>=.4)&(g<=.6))),high_gt_0p9_fraction=float(np.mean(g>.9)),
            mean_absolute_deviation_from_0p5=float(np.mean(abs(g-.5)))))
        for name in FEATURES:
            x=f[name+'_raw_mean'].to_numpy();rho=float(spearmanr(x,g).statistic)
            cor.append(dict(seed=s,feature=name,Spearman_rho=rho))
            t=thresholds[thresholds.feature==name].iloc[0];labels=np.where(x<=t.low_upper,'LOW',np.where(x<=t.high_lower,'MID','HIGH'))
            for lab in ['LOW','MID','HIGH']:
                a=g[labels==lab];strata.append(dict(seed=s,feature=name,group=lab,n=len(a),g_mean=float(a.mean()) if len(a) else None,
                    g_SD=float(a.std(ddof=1)) if len(a)>1 else None,train_q1=t.low_upper,train_q2=t.high_lower))
        base=rd(P/f'outputs/predictions/seed_{s}_B0.csv')
        for path in sorted((P/'outputs/predictions').glob(f'seed_{s}_*.csv')):
            mode=path.stem.split(f'seed_{s}_')[1];a=rd(path);m=metrics(a)
            diff=(a.y_pred_delta_V-base.y_pred_delta_V).to_numpy()*1000
            m.update(seed=s,mode=mode,delta_RMSE_vs_B0_mV=m['RMSE_mV']-metrics(base)['RMSE_mV'],
                delta_RMSE_vs_B0_percent=100*(m['RMSE_mV']/metrics(base)['RMSE_mV']-1),
                prediction_change_RMS_mV=float(np.sqrt(np.mean(diff*diff))),prediction_change_MAE_mV=float(abs(diff).mean()),
                prediction_change_P95_mV=float(np.percentile(abs(diff),95)),prediction_change_max_mV=float(abs(diff).max()),
                different_prediction_fraction=float(np.mean(diff!=0)))
            allm.append(m)
            for hh,z in a.groupby('horizon_step'):
                e=(z.y_pred_delta_V-z.y_true_delta_V).to_numpy()*1000
                horizons.append(dict(seed=s,mode=mode,horizon_step=hh,lead_h=hh*.5,n=len(z),RMSE_mV=float(np.sqrt(np.mean(e*e)))))
            for origin,z in a.groupby('origin_row'):
                e=(z.y_pred_delta_V-z.y_true_delta_V).to_numpy()*1000
                win.append(dict(seed=s,mode=mode,origin_row=origin,n=len(z),RMSE_mV=float(np.sqrt(np.mean(e*e)))))
    allg=pd.concat(gates,ignore_index=True)
    for a,b in itertools.combinations(SEEDS,2):
        x=allg[allg.seed==a].g.to_numpy();y=allg[allg.seed==b].g.to_numpy()
        pairs.append(dict(seed_a=a,seed_b=b,Spearman_rho=float(spearmanr(x,y).statistic),mean_absolute_gate_difference=float(abs(x-y).mean()),gate_difference_RMS=float(np.sqrt(np.mean((x-y)**2)))))
    m=pd.DataFrame(allm);w=pd.DataFrame(win);perms=[];cases=[]
    for s in SEEDS:
        f=m[(m.seed==s)&m['mode'].str.startswith('C_')];base=m[(m.seed==s)&(m['mode']=='B0')].iloc[0]
        perms.append(dict(seed=s,repeats=len(f),RMSE_mean_mV=f.RMSE_mV.mean(),RMSE_SD_mV=f.RMSE_mV.std(ddof=1),
            RMSE_min_mV=f.RMSE_mV.min(),RMSE_max_mV=f.RMSE_mV.max(),delta_mean_mV=f.RMSE_mV.mean()-base.RMSE_mV,
            worse_than_B0=int((f.RMSE_mV>base.RMSE_mV).sum())))
        z=w[w.seed==s].pivot(index='origin_row',columns='mode',values='RMSE_mV');delta=z.B0-z.B2
        origin=int(delta.sort_values(ascending=False,kind='stable').index[0])
        cases.extend([dict(seed=s,case='fixed_typical',origin_row=c['fixed_window_origin_row'],rule='predeclared origin 2205'),
            dict(seed=s,case='most_unfavorable_to_B0_vs_B2',origin_row=origin,rule=c['unfavorable_rule'])])
    for name,f in [('METRICS',m),('HORIZON_METRICS',horizons),('WINDOW_METRICS',w),('GATE_STATISTICS',stats),('GATE_CORRELATIONS',cor),
                   ('GATE_STRATA',strata),('CROSS_SEED_GATES',pairs),('ALL_VALIDATION_GATES',allg),('PERMUTATION_SUMMARY',perms),('CASE_SELECTION',cases)]:cs('outputs/'+name+'.csv',f)
    pp=pd.DataFrame(perms);b1=m[m['mode']=='B1'];b2=m[m['mode']=='B2']
    summary=dict(seeds=SEEDS,B1_worse_count=int((b1.delta_RMSE_vs_B0_mV>0).sum()),B2_worse_count=int((b2.delta_RMSE_vs_B0_mV>0).sum()),
        permutation_mean_worse_seed_count=int((pp.delta_mean_mV>0).sum()),permutation_worse_runs=int(pp.worse_than_B0.sum()),permutation_runs=len(pp)*10,
        B1_mean_delta_mV=float(b1.delta_RMSE_vs_B0_mV.mean()),B2_mean_delta_mV=float(b2.delta_RMSE_vs_B0_mV.mean()),
        C_mean_delta_mV=float(pp.delta_mean_mV.mean()),dynamic_seed_count=int(sum(r['dynamic'] for r in stats)),
        baseline_prediction_points=6*4920,all_prediction_points=len(m)*4920,minimum_cross_seed_gate_rho=float(min(r['Spearman_rho'] for r in pairs)),
        maximum_cross_seed_gate_rho=float(max(r['Spearman_rho'] for r in pairs)),status='SELECTED_INSTANCE_GATE_DEPENDENCE_DIAGNOSTIC',independent_accuracy_gain_proven=False)
    js('outputs/SUMMARY.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
