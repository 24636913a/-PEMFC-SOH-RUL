"""不导入审计生成程序；独立检查真实诊断配对、传播代数和历史保护。"""
import ast,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
TASK=Path(__file__).resolve().parents[1];GUIDE=TASK.parents[1];REPO=GUIDE.parent;T=TASK/'输出结果/表格'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks=[]
    def check(label,result,detail=''):
        checks.append(dict(check=label,passed=bool(result),detail=str(detail)))
        if not result:raise AssertionError(label+': '+str(detail))
    a=GUIDE/'基础预测实验/数据准备';c=GUIDE/'跨电堆迁移/阶段2C低阶迁移先导';b=GUIDE/'物理约束退化演化/阶段2B'
    fc=pd.read_csv(a/'FC1小时观测.csv');r=pd.read_csv(a/'RealHyFC循环观测.csv');r=r[(r.scheme=='B60')&r.valid].sort_values('available_ns');f=fc[fc.valid].sort_values('time')
    d=pd.read_csv(a/'稀疏诊断观测.csv');d=d[(d.scheme=='FC1_10')|(d.scheme=='B60')]
    pairs=pd.read_csv(T/'全部真实诊断身份与运行配对.csv').set_index('diagnostic');check('19个独立文件身份，无方案重复计数',len(d)==len(pairs)==19 and pairs.index.is_unique)
    frozen_support=pd.read_csv(GUIDE/'健康特征提取/共同负载特征研究/输出结果/表格/共同负载实际支持表.csv')
    rebuilt=[]
    for z in d.to_dict('records'):
        ds=z['电堆'];obs=f if ds=='FC1' else r;clock=obs.time.to_numpy() if ds=='FC1' else obs.available_ns.to_numpy();end=z['标称小时'] if ds=='FC1' else pd.Timestamp(z['实际测试结束']).value
        i=int(np.searchsorted(clock,end,side='right'))-1;got=pairs.loc[z['文件名']]
        o=None if i<0 else obs.iloc[i];lag=np.nan if o is None else (end-clock[i])/(1 if ds=='FC1' else 3.6e12)
        sp=frozen_support[(frozen_support['文件名']==z['文件名'])&(frozen_support['方案']==z['方案'])];low=bool(sp[np.isclose(sp.j,.45)].iloc[0]['共同支持']);ref=bool(sp[np.isclose(sp.j,.90)].iloc[0]['共同支持'])
        pair_ref=o is not None and lag<=48 and ref;pair_dual=pair_ref and low
        check(z['文件名']+'锚点身份',pd.isna(got.operating_id) if o is None else got.operating_id==int(o.id))
        check(z['文件名']+'双参考及单参考资格',got.pair_dual==pair_dual and got.pair_ref==pair_ref)
        check(z['文件名']+'实测值未改',pd.isna(got.D_low) if pd.isna(z['D_low_V']) else abs(got.D_low-z['D_low_V'])<1e-14)
        check(z['文件名']+'无在线发布时间伪造',got.release=='UNKNOWN' and not got.strict_online_qualified)
        rebuilt.append(dict(ds=ds,name=z['文件名'],primary=z['primary'],nominal=z['标称小时'],upper=z['累计小时上界'],pair=pair_dual,anchor=np.nan if o is None else int(o.id)))
    rb=pd.DataFrame(rebuilt);summary=pd.read_csv(T/'全寿命与各截止独立配对上限.csv')
    for row in summary.itertuples():
        cut={'严格300h':300.,'严格600h':600.,'后移629.2038h':629.2038,'全寿命_仅描述':np.inf}[row.scope];sub=rb[rb.ds==row.dataset];sub=sub[(sub.nominal<=cut) if row.dataset=='FC1' else (sub.upper<=cut)];main=sub[sub.primary];usable=main[main.pair]
        check(row.dataset+row.scope+'配对计数',len(usable)==row.main_dual_pairs and usable.anchor.nunique()==row.main_unique_operating_anchors)
        check(row.dataset+row.scope+'配对转移',max(0,len(usable)-1)==row.paired_transitions)
    full=summary[summary.scope=='全寿命_仅描述'].set_index('dataset');check('全寿命二维仍未达9对',full.loc['FC1','main_dual_pairs']<9 and full.loc['RealHyFC','main_dual_pairs']<9)
    check('启停后不是新增唯一运行锚点',full.loc['RealHyFC','all_assays_dual_pairs']>full.loc['RealHyFC','all_assays_unique_operating_anchors'])
    q=pd.read_csv(T/'一维二维融合分层资格.csv');bp=json.loads((b/'配置/阶段2B冻结协议.json').read_text(encoding='utf-8'));check('冻结门槛未放宽',(q.minimum_pairs==q.dimension.map({1:6,2:9})).all() and bp['pairing']['min_distinct_pairs_per_parameter']==3 and (q.condition_limit==100).all())
    check('二维未冒称训练合格',(~q[q.dimension==2].mathematical_sample_gate).all() and (q.joint_training=='NO').all())
    for row in q.itertuples():
        cut={'严格300h':300.,'严格600h':600.,'后移629.2038h':629.2038,'全寿命_仅描述':np.inf}[row.scope]
        u=pairs[(pairs.dataset==row.dataset)&pairs.primary&pairs.pair_dual];u=u[(u.nominal_h<=cut) if row.dataset=='FC1' else (u.operating_upper_h<=cut)]
        x=u[['D_ref','D_low']].to_numpy()[:,:row.dimension]
        if len(x):
            sdev=np.std(x,axis=0);z=np.c_[np.ones(len(x)),(x-x.mean(0))/np.where(sdev>1e-14,sdev,1)];sing=np.linalg.svd(z,compute_uv=False);rank=int(np.linalg.matrix_rank(z));cond=float(sing[0]/sing[-1]) if rank==row.dimension+1 else np.nan
        else:rank=0;cond=np.nan
        check(row.dataset+row.scope+str(row.dimension)+'秩及条件数',rank==row.rank and (pd.isna(row.standardized_condition) if pd.isna(cond) else abs(cond-row.standardized_condition)<1e-9))
    pred=pd.read_csv(c/'输出结果/正式计算/表格/各方法逐起点预测.csv');time=pd.read_csv(T/'逐目标时间场景传播差额.csv');rates=pd.read_csv(T/'源目标速率差异核对.csv')
    for sc in rates.itertuples():
        target=r[r.available_h<=dict(A_严格300h=300,B_严格600h=600,C_后移629h=629.2038)[sc.scenario]];src=f[f.partition=='Train']
        def independent_q(x,y):
            x=np.asarray(x,float);return np.linalg.lstsq(np.c_[np.ones(len(x)),x-x[0]],np.asarray(y,float),rcond=None)[0][1]
        check(sc.scenario+'实际小时速率',abs(independent_q(src.time,1-src.v/src.base_V)-sc.source_rate_per_h)<1e-14 and abs(independent_q(target.available_h,1-target.v/target.base_V)-sc.target_rate_per_h)<1e-14)
    for (scenario,method),g in pred.groupby(['scenario','method']):
        g=g.sort_values('target_id');record=json.loads((c/f'输出结果/正式计算/模型记录/{scenario}_参数与状态.json').read_text(encoding='utf-8'));s=record['states'][method]['at_cutoff'];lead=g.nominal_lead_h.to_numpy();rev=0 if s['tau_h'] is None else s['R']*np.exp(-lead/s['tau_h']);p=s['L']+s['q']*lead+rev
        check(scenario+method+'冻结值代数',np.max(abs(p-g.pred_d))<1e-13)
        truth=r.set_index('id').loc[g.target_id];check(scenario+method+'标签与时间身份',np.max(abs(truth.v.to_numpy()-g.true_v))<1e-14 and np.max(abs(truth.available_h.to_numpy()-g.origin_h-g.actual_lead_h))<1e-10)
        dt=lead-g.actual_lead_h;extra=s['q']*dt
        if s['tau_h'] is not None:extra+=s['R']*(np.exp(-lead/s['tau_h'])-np.exp(-g.actual_lead_h/s['tau_h']))
        t=time[(time.scenario==scenario)&(time.method==method)].sort_values('target_id')
        check(scenario+method+'事后时间差额',np.max(abs(-g.base_V.to_numpy()*1000*extra-t.clock_propagation_difference_mV))<1e-9)
    ex=pd.read_csv(T/'源信息增量与误差平方分解.csv');check('增量及MSE分解',max(ex.max_increment_formula_diff_mV)<1e-9 and max(abs(ex.decomposition_residual_mV2))<1e-8)
    # 只读2A/2B逐起点缓存，确认沿用基线并未因后续单位/公式改变。
    legacy=[];extra_sources=[]
    for ds in ['FC1','RealHyFC']:
        pa=GUIDE/f'基础预测实验/输出结果/表格/{ds}基础预测逐起点结果.csv';pb=b/f'输出结果/表格/{ds}全部方法逐点预测.csv';pm=b/f'输出结果/表格/{ds}各跨度指标.csv'
        old=pd.read_csv(pa);new=pd.read_csv(pb);met=pd.read_csv(pm);keys=['dataset','scheme','partition','origin_id','target_id','horizon','step','method']
        baseline=new[new.method.isin(['Persistence','LocalLinear'])];merged=baseline.merge(old,on=keys,how='left',suffixes=('_2B','_2A'),validate='one_to_one')
        check(ds+'2A_2B基线身份与值',len(merged)==len(baseline) and merged.pred_v_2A.notna().all() and max(abs(merged.pred_v_2B-merged.pred_v_2A))<1e-14 and max(abs(merged.true_v_2B-merged.true_v_2A))<1e-14)
        for z in met.itertuples():
            g=new[(new.partition==z.partition)&(new.horizon==z.horizon)&(new.method==z.method)]
            if len(g) and np.isfinite(z.RMSE_mV):check(ds+str(z.Index)+'既有指标单位',abs(1000*np.sqrt(np.mean((g.pred_v-g.true_v)**2))-z.RMSE_mV)<1e-9)
        legacy.append(dict(dataset=ds,stage2A_rows=len(old),stage2B_rows=len(new),baseline_rows=len(baseline),baseline_max_prediction_difference_V=float(max(abs(merged.pred_v_2B-merged.pred_v_2A))),true_label_difference_V=float(max(abs(merged.true_v_2B-merged.true_v_2A))),coordinate='FC1 actual hour; Real raw cycle step; future actual lead is audit only',status='PASS_READONLY_CACHE_CHECK'))
        for pth in [pa,pb,pm]:extra_sources.append(dict(path=pth.relative_to(REPO).as_posix(),sha256=sha(pth)))
    scale=b/'输出结果/表格/已选配置与训练尺度.csv';scale_rows=pd.read_csv(scale);source_q=scale_rows[(scale_rows.dataset=='FC1')&(scale_rows.model=='BoundedRate')].q_prior.iloc[0]
    check('2B源V_h到2C无量纲_h换算',abs(source_q/float(f.base_V.iloc[0])-rates.source_rate_per_h.iloc[0])<1e-14)
    extra_sources.append(dict(path=scale.relative_to(REPO).as_posix(),sha256=sha(scale)))
    pd.DataFrame(legacy).to_csv(T/'2A_2B冻结逐起点缓存核对.csv',index=False,encoding='utf-8-sig');pd.DataFrame(extra_sources).to_csv(TASK/'配置/补充只读核验来源SHA256.csv',index=False,encoding='utf-8-sig')
    for ds in ['FC1','RealHyFC']:
        u=pairs[(pairs.dataset==ds)&pairs.primary&pairs.pair_dual];rho=float(np.corrcoef(u.D_ref,u.D_low)[0,1]);reported=pd.read_csv(T/'特征相关性与第二维剩余变化.csv').set_index('dataset').loc[ds]
        check(ds+'特征相关性独立复算',abs(rho-reported.corr_Dref_Dlow)<1e-12)
    for rec in pd.read_csv(TASK/'配置/历史文件SHA256.csv').itertuples():check('历史SHA_'+str(rec.Index),sha(REPO/rec.path)==rec.sha256)
    for rec in pd.read_csv(TASK/'配置/来源证据SHA256.csv').itertuples():check('来源SHA_'+str(rec.Index),sha(REPO/rec.path)==rec.sha256)
    literature=TASK/'配置/原方案与报告来源SHA256.csv'
    if literature.exists():
        for rec in pd.read_csv(literature).itertuples():check('已有文献与报告SHA_'+str(rec.Index),sha(REPO/rec.path)==rec.sha256)
    roles=T/'逐诊断训练_校正_验证用途.csv'
    if roles.exists():
        role=pd.read_csv(roles);check('逐诊断用途不越权',len(role)==76 and (role.two_dim_joint_training=='NO').all() and (role.state_correction=='NO_UNQUALIFIED_MAPPING').all())
    for p in (TASK/'程序').glob('*.py'):
        tree=ast.parse(p.read_text(encoding='utf-8'));imports=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]+[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)];check(p.name+'无模型框架或历史模型导入',not any(x and ('torch' in x or 'tensorflow' in x or '迁移核心' in x or '低阶演化核心' in x) for x in imports))
    if (T/'图片来源索引.json').exists():
        from PIL import Image
        for fig in json.loads((T/'图片来源索引.json').read_text(encoding='utf-8')):
            check(fig['figure']+'绘图CSV',sha(TASK/fig['csv'])==fig['csv_sha256'])
            for source in fig['source_files']:check(fig['figure']+'原始来源SHA',sha(REPO/source['path'])==source['sha256'])
            with Image.open(TASK/fig['png']) as im:check(fig['figure']+'PNG分辨率',min(im.info.get('dpi',(0,0)))>299 and im.width>1000)
            check(fig['figure']+'PDF', (TASK/fig['pdf']).read_bytes().startswith(b'%PDF'))
    pd.DataFrame(checks).to_csv(T/'独立检查明细.csv',index=False,encoding='utf-8-sig')
    result=dict(status='PASS',checks=len(checks),historical_files_unchanged=len(pd.read_csv(TASK/'配置/历史文件SHA256.csv')),all_diagnostics_checked=19,retrained_models=0,formal_code_modified=False,method='独立二分时钟配对、SVD/lstsq速率、冻结传播代数、标签和SHA；未调用历史模型')
    (TASK/'输出结果/独立检查通过.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
