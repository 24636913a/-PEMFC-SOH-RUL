"""Independent scalar recalculation of all new cache-derived results, no model import."""
from common import *
import math
from collections import defaultdict
def scalar(f):
    y=list(f.y_true_delta_V);pred=list(f.y_pred_delta_V);e=[(b-a)*1000 for a,b in zip(y,pred)];n=len(e);aa=sorted(map(abs,e));mu=math.fsum(y)/n
    def quant(q):
        k=(n-1)*q;i=int(k);return aa[i]+(aa[min(i+1,n-1)]-aa[i])*(k-i)
    tgt=defaultdict(list);end=[]
    for r,v in zip(f.itertuples(),e):
        tgt[r.target_row].append(v*v)
        if r.horizon_step==24:end.append(v*v)
    return dict(RMSE_mV=math.sqrt(math.fsum(v*v for v in e)/n),MAE_mV=math.fsum(map(abs,e))/n,
        R2=1-math.fsum(v*v for v in e)/math.fsum(((v-mu)*1000)**2 for v in y),
        endpoint_RMSE_mV=math.sqrt(math.fsum(end)/len(end)) if end else None,
        target_time_equal_RMSE_mV=math.sqrt(math.fsum(math.fsum(v)/len(v) for v in tgt.values())/len(tgt)),
        P95_mV=quant(.95),P99_mV=quant(.99),max_abs_error_mV=max(aa))
def main():
    verified_config();f=rd(P/'outputs/experiment_tables/ALIGNED_FORMAL_PREDICTIONS.csv');m=rd(P/'outputs/experiment_tables/FORMAL_VALIDATION_METRICS.csv')
    conditions=rd(P/'outputs/condition_analysis/ORIGIN_CONDITIONS.csv');gm=rd(P/'outputs/condition_analysis/CONDITION_METRICS.csv');h=rd(P/'outputs/experiment_tables/HORIZON_METRICS.csv');checks=[]
    keys=rd(R2/'outputs/audits/VALIDATION_KEY_TRUTH.csv')
    for row in m.itertuples():
        z=f[(f.model==row.model)&(f.seed==row.seed)];assert len(z)==4920
        for key in ['origin_row','target_row','horizon_step','y_true_delta_V']:assert np.array_equal(z[key],keys[key]),(row.model,row.seed,key)
        for key,value in scalar(z).items():
            diff=abs(value-getattr(row,key));assert diff<1e-9;checks.append(dict(model=row.model,seed=row.seed,scope='overall',metric=key,difference=diff))
        for step,zz in z.groupby('horizon_step'):
            e=[(b-a)*1000 for a,b in zip(zz.y_true_delta_V,zz.y_pred_delta_V)];v=math.sqrt(math.fsum(t*t for t in e)/len(e))
            old=h[(h.model==row.model)&(h.seed==row.seed)&(h.horizon_step==step)].RMSE_mV.iloc[0];assert abs(v-old)<1e-9
        e=z.merge(conditions,on='origin_row',suffixes=('','_group'),validate='many_to_one')
        for cond in CONDITIONS:
            for group in ['LOW','MID','HIGH']:
                zz=e[e[cond+'_group']==group];old=gm[(gm.model==row.model)&(gm.seed==row.seed)&(gm.condition==cond)&(gm.group==group)].iloc[0]
                assert len(zz)==old.n
                for key,value in scalar(zz).items():
                    diff=abs(value-old[key]);assert diff<1e-9;checks.append(dict(model=row.model,seed=row.seed,scope=cond+'/'+group,metric=key,difference=diff))
    assert (f[f.model=='Persistence'].y_pred_delta_V==0).all()
    # Every historical prediction/checkpoint/partial training file protected at entry stays byte-identical.
    history=rd(P/'outputs/audits/HISTORY_BEFORE.csv')
    for r in history.itertuples():assert sha(REPO/r.path)==r.sha256,('HISTORICAL_MODIFICATION',r.path)
    dirty=subprocess.check_output(['git','diff','--name-only',BASE],cwd=REPO,text=True,encoding='utf-8').splitlines();assert all(P.name in v for v in dirty)
    cs('outputs/audits/INDEPENDENT_METRIC_DIFFERENCES.csv',checks)
    js('outputs/audits/INDEPENDENT_RESULT_AUDIT.json',dict(pass_=True,metric_checks=len(checks),max_difference=max(r['difference'] for r in checks),
        overall_settings=len(m),regime_settings=len(gm),prediction_points=len(f),history_files_unchanged=len(history),
        training=False,model_forward=False,Test=False,new_Assessment=False))
    print('INDEPENDENT_CACHE_RESULT_VERIFICATION_PASS',len(checks))
if __name__=='__main__':main()
