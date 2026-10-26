"""从冻结CSV/JSON做代数核验、描述统计和配对上限审计；无预测模型训练。"""
import ast,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
TASK=Path(__file__).resolve().parents[1];GUIDE=TASK.parents[1];REPO=GUIDE.parent;TAB=TASK/'输出结果/表格'
C=GUIDE/'跨电堆迁移/阶段2C低阶迁移先导';B=GUIDE/'物理约束退化演化/阶段2B';A=GUIDE/'基础预测实验'
SOURCES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);SOURCES[p.relative_to(REPO).as_posix()]=sha(p)
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.suffix=='.json' else pd.read_csv(p)
def save(df,name):df.to_csv(TAB/name,index=False,encoding='utf-8-sig',float_format='%.17g')
def slope(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);z=x-x.mean()
    return float(z@(y-y.mean())/(z@z)) if len(x)>1 and z@z>0 else np.nan
def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else np.nan
def design(d,dims):
    if d.empty:return 0,np.nan
    x=d[['D_ref','D_low']].to_numpy()[:,:dims];sd=x.std(0);z=(x-x.mean(0))/np.where(sd>1e-14,sd,1);mat=np.c_[np.ones(len(x)),z]
    rank=int(np.linalg.matrix_rank(mat));return rank,float(np.linalg.cond(mat)) if rank==dims+1 else np.nan

def main():
    if (TASK/'输出结果/审计计算完成.json').exists():raise FileExistsError('审计已完成，禁止覆盖；独立检查请运行独立检查.py')
    cfg=read(TASK/'配置/阶段2D审计协议.json');assert sha(TASK/'配置/阶段2D审计协议.json')==read(TASK/'配置/预登记SHA256.json')['protocol_sha256']
    fc=read(A/'数据准备/FC1小时观测.csv');real=read(A/'数据准备/RealHyFC循环观测.csv');real=real[real.scheme=='B60'].copy()
    fc['h']=fc.time;real['h']=real.available_h
    for d in [fc,real]:d['d']=1-d.SOH
    diag=read(A/'数据准备/稀疏诊断观测.csv');support=read(GUIDE/'健康特征提取/共同负载特征研究/输出结果/表格/共同负载实际支持表.csv');scans=read(GUIDE/'健康特征提取/共同负载特征研究/输出结果/表格/全部扫描身份.csv');conditions=read(GUIDE/'健康特征提取/共同负载特征研究/输出结果/表格/实际候选支持平台条件.csv')
    pred=read(C/'输出结果/正式计算/表格/各方法逐起点预测.csv');metrics=read(C/'输出结果/正式计算/表格/电压_SOH及损失误差.csv');cp=read(C/'配置/阶段2C冻结协议.json');bp=read(B/'配置/阶段2B冻结协议.json');ap=read(A/'配置/阶段2A冻结协议.json')
    source=read(C/'输出结果/正式计算/模型记录/FC1源端参数.json');oldpairs=read(B/'输出结果/表格/诊断与运行历史配对及排除.csv')
    for stage,path in [('2A',A/'输出结果/独立复核通过.json'),('2B',B/'输出结果/独立复核通过.json'),('2C',C/'输出结果/正式计算/独立复核通过.json')]:assert read(path)['status']=='PASS'
    for name in ['FC1小时观测.csv','RealHyFC循环观测.csv','稀疏诊断观测.csv']:
        assert sha(A/'数据准备'/name)==sha(B/'数据准备'/name)==sha(C/'数据准备'/name)
    # 既有代码只做文本/AST检查，不import，不调用历史运行或拟合入口。
    codechecks=[]
    for rel in ['基础预测实验/程序/基础预测核心.py','物理约束退化演化/阶段2B/程序/低阶演化核心.py','跨电堆迁移/阶段2C低阶迁移先导/程序/迁移核心.py','跨电堆迁移/阶段2C低阶迁移先导/程序/运行迁移实验.py']:
        path=GUIDE/rel;SOURCES[path.relative_to(REPO).as_posix()]=sha(path);ast.parse(path.read_text(encoding='utf-8-sig'));codechecks.append(dict(source=rel,sha256=sha(path),syntax='PASS',execution='NOT_IMPORTED_NOT_EXECUTED'))
    save(pd.DataFrame(codechecks),'源码只读身份核对.csv')
    # 运行阶段与逐相邻观测变动：不改变历史窗，不根据未来误差选取段。
    rate_rows=[];changes=[]
    groups=[('FC1','源原Train',fc[fc.valid&(fc.partition=='Train')])]+[('FC1','原'+str(k),g) for k,g in fc[fc.valid].groupby('partition') if k!='Train']
    groups += [('RealHyFC',s['name'],real[real.valid&(real.h<=s['cutoff_h'])]) for s in cp['scenarios']]
    groups += [('RealHyFC',f'固定区间_{lo}_{hi}',real[real.valid&(real.h>lo)&(real.h<=hi)]) for lo,hi in [(300,600),(600,629.2038),(629.2038,float(real.h.max()))]]
    groups += [('RealHyFC','原'+str(k),g) for k,g in real[real.valid].groupby('partition')]
    for dataset,label,d in groups:
        allobs=fc[fc.valid] if dataset=='FC1' else real[real.valid];d=d.sort_values('h')
        rate_rows.append(dict(dataset=dataset,group=label,n=len(d),first_h=float(d.h.min()),last_h=float(d.h.max()),observed_full_first_h=float(allobs.h.min()),observed_full_last_h=float(allobs.h.max()),observed_record_span_fraction=float((d.h.max()-d.h.min())/(allobs.h.max()-allobs.h.min())),mean_loss=float(d.d.mean()),min_loss=float(d.d.min()),max_loss=float(d.d.max()),OLS_loss_per_h=slope(d.h,d.d),identity='描述性斜率；未用于选择新参数；记录范围比例不是物理寿命百分比'))
    for dataset,d in [('FC1',fc[fc.valid]),('RealHyFC',real[real.valid])]:
        d=d.sort_values('id');records=d.to_dict('records')
        for p,r in zip(records,records[1:]):changes.append(dict(dataset=dataset,from_id=int(p['id']),to_id=int(r['id']),from_h=p['h'],to_h=r['h'],gap_ids=int(r['id']-p['id']),adjacent_raw_id=(r['id']-p['id']==1),delta_v_mV=(r['v']-p['v'])*1000,delta_d=r['d']-p['d'],sign='VOLTAGE_RISE' if r['v']>p['v'] else 'VOLTAGE_FALL' if r['v']<p['v'] else 'EQUAL',identity='符号统计，无恢复幅度阈值；缺口不当连续时刻'))
    save(pd.DataFrame(rate_rows),'源目标阶段范围与描述性速率.csv');save(pd.DataFrame(changes),'相邻观测非单调变化.csv')
    # 保存状态的代数复核，不重新估计状态或执行预测模型。
    algebra=[];timeparts=[];timemetrics=[];excess=[];states=[];rates=[]
    for sc in cp['scenarios']:
        name=sc['name'];record=read(C/f"输出结果/正式计算/模型记录/{name}_参数与状态.json");hist=real[real.valid&(real.h<=sc['cutoff_h'])]
        qtarget=record['target']['q_bounded'];qsource=source['q_bounded'];base=float(hist.base_V.iloc[0]);ratio=qsource/qtarget
        rates.append(dict(scenario=name,source_n=source['n_fit'],target_n=len(hist),source_rate_per_h=qsource,target_rate_per_h=qtarget,source_target_ratio=ratio,source_recomputed_per_h=slope(fc[fc.valid&(fc.partition=='Train')].h,fc[fc.valid&(fc.partition=='Train')].d),target_recomputed_per_h=slope(hist.h,hist.d),source_native_V_per_h=qsource*source['base_V'],target_native_V_per_h=qtarget*base))
        for method,ss in record['states'].items():
            st=ss['at_cutoff'];g=pred[(pred.scenario==name)&(pred.method==method)].sort_values('target_id');n=g.nominal_lead_h.to_numpy();actual=g.actual_lead_h.to_numpy();tau=st['tau_h']
            rev=np.zeros(len(g)) if tau is None else st['R']*np.exp(-n/tau);expected_d=st['L']+st['q']*n+rev
            algebra.append(dict(scenario=name,method=method,n=len(g),max_d_difference=float(max(abs(expected_d-g.pred_d))),max_v_difference=float(max(abs(base*(1-expected_d)-g.pred_v))),max_SOH_difference=float(max(abs(1-expected_d-g.pred_soh))),actual_lead_min_h=float(min(actual)),actual_lead_max_h=float(max(actual)),nominal_lead_min_h=float(min(n)),nominal_lead_max_h=float(max(n))))
            clockd=st['q']*(n-actual)+(0 if tau is None else st['R']*(np.exp(-n/tau)-np.exp(-actual/tau)))
            delta=-base*1000*clockd
            for i,r in enumerate(g.itertuples()):timeparts.append(dict(scenario=name,method=method,target_id=r.target_id,target_actual_h=r.target_actual_h,nominal_lead_h=n[i],actual_lead_h=actual[i],time_difference_h=n[i]-actual[i],clock_propagation_difference_mV=delta[i],identity='仅冻结参数事后传播差额；不是可用预测或新的误差排名'))
            timemetrics.append(dict(scenario=name,method=method,n=len(g),time_difference_mean_h=float(np.mean(n-actual)),time_difference_min_h=float(min(n-actual)),time_difference_max_h=float(max(n-actual)),clock_difference_RMS_mV=float(np.sqrt(np.mean(delta**2))),clock_difference_mean_mV=float(np.mean(delta)),clock_difference_max_abs_mV=float(max(abs(delta))),frozen_RMSE_mV=float(np.sqrt(np.mean((1000*(g.pred_v-g.true_v))**2)))))
            states.append(dict(scenario=name,method=method,L=st['L'],q_per_h=st['q'],R=st['R'],tau_h=tau,predicted_initial_loss=st['L']+st['R'],initial_rate_per_h=st['q'] if tau is None else st['q']-st['R']/tau,last_measured_d=float(hist.d.iloc[-1]),future_voltage_rise_pairs=int(np.sum(np.diff(g.pred_v)>0)),n_history_state=st['n_history']))
        target=pred[(pred.scenario==name)&(pred.method=='A_TargetBounded')].sort_values('target_id');e=(target.pred_v-target.true_v).to_numpy()*1000;total=(sc['cutoff_h']-hist.h.max())+target.nominal_lead_h.to_numpy()
        for method,w in [('B_FC1Direct',1.),('C_FC1PriorAdapted',.25)]:
            actualpred=pred[(pred.scenario==name)&(pred.method==method)].sort_values('target_id');dv=-base*(qsource-qtarget)*total*w*1000
            actualdelta=1000*(actualpred.pred_v.to_numpy()-target.pred_v.to_numpy());cross=float(2*np.mean(e*dv));square=float(np.mean(dv**2));actual_mse=float(np.mean((1000*(actualpred.pred_v-actualpred.true_v))**2)-np.mean(e**2))
            excess.append(dict(scenario=name,method=method,source_weight=w,max_increment_formula_diff_mV=float(max(abs(dv-actualdelta))),rate_increment_RMS_mV=float(np.sqrt(np.mean(dv**2))),cross_term_mV2=cross,squared_increment_mV2=square,actual_MSE_increase_mV2=actual_mse,decomposition_residual_mV2=actual_mse-cross-square,identity='精确代数分解；不是温压或电化学因果贡献比例'))
    save(pd.DataFrame(rates),'源目标速率差异核对.csv');save(pd.DataFrame(algebra),'冻结预测代数核对.csv');save(pd.DataFrame(timeparts),'逐目标时间场景传播差额.csv');save(pd.DataFrame(timemetrics),'时间场景影响汇总.csv');save(pd.DataFrame(states),'冻结状态与后移优势解释.csv');save(pd.DataFrame(excess),'源信息增量与误差平方分解.csv')
    # 两参考点原始支持、扫描和日历配对。拒绝记录同样保存。
    selected=diag[((diag['电堆']=='FC1')&(diag.scheme=='FC1_10'))|((diag['电堆']=='RealHyFC')&(diag.scheme=='B60'))].copy();allpairs=[];supports=[];platformrows=[]
    for d in selected.to_dict('records'):
        ds=d['电堆'];filename=d['文件名'];sp=support[(support['文件名']==filename)&(support['方案']==d['方案'])];ps=[]
        for j in cfg['points']:
            s=sp[np.isclose(sp.j,j,rtol=0,atol=1e-10)];assert len(s)==1,(filename,j,len(s));row=s.iloc[0].to_dict();supports.append(row);ps.append(bool(row['共同支持']))
            if row['共同支持']:
                expected=row['BoL_V']-row['本曲线V'];frozen=d['D_low_V'] if j==.45 else d['D_ref_V'];assert abs(expected-frozen)<1e-12,(filename,j)
            for token in str(row['支持组']).split(';'):
                try:gid=int(float(token))
                except ValueError:continue
                for p in conditions[(conditions['文件名']==filename)&(conditions['组号']==gid)].to_dict('records'):platformrows.append(dict(reference_j=j,**p))
        obs=fc[fc.valid].sort_values('h') if ds=='FC1' else real[real.valid].sort_values('available_ns')
        end=None if ds=='FC1' else pd.Timestamp(d['实际测试结束']);pre=obs[obs.h<=d['标称小时']] if ds=='FC1' else obs[obs.available_ns<=end.value]
        post=obs[obs.h>d['标称小时']] if ds=='FC1' else obs[obs.available_ns>end.value]
        last=None if pre.empty else pre.iloc[-1];nxt=None if post.empty else post.iloc[0]
        lag=np.nan if last is None else float(d['标称小时']-last.h) if ds=='FC1' else float((end.value-int(last.available_ns))/3.6e12)
        postlag=np.nan if nxt is None else float(nxt.h-d['标称小时']) if ds=='FC1' else float((int(nxt.available_ns)-end.value)/3.6e12)
        pair_ref=last is not None and lag<=48 and ps[1];pair_dual=last is not None and lag<=48 and all(ps)
        sc=scans[(scans['文件名']==filename)&(scans['扫描方向']=='降载')&scans['主扫描']]
        row=dict(dataset=ds,diagnostic=filename,scheme=d['scheme'],nominal_h=d['标称小时'],primary=bool(d['primary']),after_start=bool(d['启停后']),actual_completed=d['实际测试结束'],operating_lower_h=d['累计小时下界'],operating_upper_h=d['累计小时上界'],clock=d['时钟资格'],release='UNKNOWN',strict_online_qualified=False,
          selected_scan='降载',selected_segments=';'.join(map(str,sc['扫描分段'].tolist())),both_supported=all(ps),ref_supported=ps[1],low_supported=ps[0],D_ref=d['D_ref_V'],D_low=d['D_low_V'],diagnostic_base_ref_V=d['base_V'],
          operating_id=np.nan if last is None else int(last.id),operating_h=np.nan if last is None else last.h,operating_calendar='' if last is None or ds=='FC1' else pd.Timestamp(int(last.available_ns)).isoformat(),lag_h=lag,operating_loss_V=np.nan if last is None else last.D,operating_loss_relative=np.nan if last is None else last.d,
          pair_ref=bool(pair_ref),pair_dual=bool(pair_dual),completed_clock_pair=bool(pair_dual and ds=='RealHyFC'),nominal_only_pair=bool(pair_dual and ds=='FC1'),
          post_operating_id=np.nan if nxt is None else int(nxt.id),post_h=np.nan if nxt is None else nxt.h,post_lag_h=postlag,pre_post_available=bool(last is not None and nxt is not None and lag<=48 and postlag<=48),
          post_minus_pre_operating_mV=np.nan if last is None or nxt is None or lag>48 or postlag>48 else float((nxt.v-last.v)*1000),
          operating_condition='0.70 A/cm2, five-cell mean, operating nominal clock' if ds=='FC1' else '0.558161974 A/cm2, DOWN, LATER_OR_UNKNOWN, exploratory B60',measurement_comparability='FC1诊断条件/真实时钟不足' if ds=='FC1' else 'B60仅数学可算，未认证温度等价；运行和极化协议不同',reason='NO_PRIOR_OPERATING_POINT' if last is None else 'REFERENCE_UNSUPPORTED' if not all(ps) else 'LAG_GT_48H' if lag>48 else 'NOMINAL_ONLY' if ds=='FC1' else 'COMPLETION_ASSUMPTION_ONLY')
        allpairs.append(row)
    pairs=pd.DataFrame(allpairs);save(pairs,'全部真实诊断身份与运行配对.csv');save(pd.DataFrame(supports),'两参考点实测支持与原始行号.csv');save(pd.DataFrame(platformrows).drop_duplicates(),'两参考点支持平台测量条件.csv');save(diag,'全部A60_B60及FC1冻结资格.csv')
    # 保留每个截止的全部19个诊断（包括未在截止前可得的排除），禁止用后期值回灌。
    cutoffrows=[];summary=[];qualifications=[]
    scopes=[('严格300h',300.),('严格600h',600.),('后移629.2038h',629.2038),('全寿命_仅描述',np.inf)]
    for scope,cut in scopes:
        for r in pairs.to_dict('records'):
            available=bool(r['nominal_h']<=cut) if r['dataset']=='FC1' else bool(np.isfinite(r['operating_upper_h']) and r['operating_upper_h']<=cut)
            cutoffrows.append(dict(scope=scope,cutoff_h=cut if np.isfinite(cut) else np.nan,diagnostic=r['diagnostic'],dataset=r['dataset'],primary=r['primary'],after_start=r['after_start'],available_completion_or_nominal_assumption=available,operating_id=r['operating_id'],pair_dual_if_available=available and r['pair_dual'],pair_ref_if_available=available and r['pair_ref'],scope_use='FC1早期计数仅描述；实际迁移源模型仍使用原Train' if r['dataset']=='FC1' else '目标早期适配' if np.isfinite(cut) else '全寿命上限，禁止回灌早期',release='UNKNOWN'))
        for ds in ['FC1','RealHyFC']:
            d=pairs[pairs.dataset==ds];avail=d[d.nominal_h<=cut] if ds=='FC1' else d[d.operating_upper_h<=cut]
            primary=avail[avail.primary];usable=primary[primary.pair_dual];refonly=primary[primary.pair_ref];all_dual=avail[avail.pair_dual];all_ref=avail[avail.pair_ref]
            summary.append(dict(dataset=ds,scope=scope,physical_diagnostics=len(avail),main_diagnostics=len(primary),main_dual_features=int(primary.both_supported.sum()),main_ref_pairs=len(refonly),main_dual_pairs=len(usable),main_unique_operating_anchors=usable.operating_id.nunique(),all_assays_ref_pairs=len(all_ref),all_assays_dual_pairs=len(all_dual),all_assays_unique_operating_anchors=all_dual.operating_id.nunique(),main_diagnostic_transitions=max(0,int(primary.both_supported.sum())-1),paired_transitions=max(0,len(usable)-1),actual_completed_pairs=len(usable) if ds=='RealHyFC' else 0,published_online_pairs=0,independent_stacks=1))
            for dims in [1,2]:
                u=refonly if dims==1 else usable;n=u.operating_id.nunique();rank,cond=design(u,dims);minimum=6 if dims==1 else 9;math_ok=n>=minimum and rank==dims+1 and np.isfinite(cond) and cond<=100
                qualifications.append(dict(dataset=ds,scope=scope,dimension=dims,distinct_pairs=n,minimum_pairs=minimum,rank=rank,required_rank=dims+1,standardized_condition=cond,condition_limit=100,paired_transitions=max(0,n-1),required_transitions=6,mathematical_sample_gate=bool(math_ok),measurement_gate='NOT_CERTIFIED',clock_gate='NOMINAL_ONLY' if ds=='FC1' else 'COMPLETION_ASSUMPTION_ONLY',release_gate='UNKNOWN',allowed_use='有条件可探索：全寿命离线一维观测关系，不能回灌早期' if math_ok else '目前不具备资格',joint_training='NO',strict_early_transfer='NO'))
    save(pd.DataFrame(cutoffrows),'各截止时点诊断可用及排除明细.csv');save(pd.DataFrame(summary),'全寿命与各截止独立配对上限.csv');save(pd.DataFrame(qualifications),'一维二维融合分层资格.csv')
    # 相关性只描述真实已配对主序列，不拟合运行-诊断映射。
    relations=[];directions=[];shape=[]
    for ds in ['FC1','RealHyFC']:
        p=pairs[(pairs.dataset==ds)&pairs.primary&pairs.pair_dual].sort_values('nominal_h');a=p.D_ref.to_numpy();b=p.D_low.to_numpy()
        rho=corr(a,b);res=b-b.mean()-((a-a.mean())@(b-b.mean())/((a-a.mean())@(a-a.mean())))*(a-a.mean())
        relations.append(dict(dataset=ds,n_main_pairs=len(p),corr_op_Dref=corr(p.operating_loss_V,a),corr_op_Dlow=corr(p.operating_loss_V,b),spearman_op_Dref=corr(p.operating_loss_V.rank(),p.D_ref.rank()),corr_Dref_Dlow=rho,second_feature_residual_std_V=float(np.std(res)),second_feature_unexplained_fraction=float(1-rho**2),identity='全寿命描述，存在共同时间趋势和共享BoL误差；不是验证精度或物理因果'))
        records=p.to_dict('records')
        for a,b in zip(records,records[1:]):
            dr=b['D_ref']-a['D_ref'];dl=b['D_low']-a['D_low'];do=b['operating_loss_V']-a['operating_loss_V'];directions.append(dict(dataset=ds,from_diagnostic=a['diagnostic'],to_diagnostic=b['diagnostic'],delta_op_loss_V=do,delta_Dref_V=dr,delta_Dlow_V=dl,op_ref_same_sign=bool(np.sign(do)==np.sign(dr)),ref_low_same_sign=bool(np.sign(dl)==np.sign(dr))))
        j=.70 if ds=='FC1' else .558161974;w=(j-.45)/(.90-.45)
        shape.append(dict(dataset=ds,j_operating=j,j_low=.45,j_ref=.90,H_ref=w,H_low=1-w,instantaneous_observation_rank=1,unknown_state_dimensions=2,unknown_operating_offset=True,formal_2D_MLP_inputs=8,formal_2D_MLP_outputs=2,formal_2D_MLP_weights=(8+1)*16+(16+1)*2,FC1_scalar_MLP_weights=(5+1)*16+(16+1) if ds=='FC1' else np.nan,identity='仅线性表征的代数观测方向；不认证负载等价，不计算新SOH，不把两个电堆堆叠为同一状态'))
    save(pd.DataFrame(relations),'特征相关性与第二维剩余变化.csv');save(pd.DataFrame(directions),'运行与诊断相邻变化方向.csv');save(pd.DataFrame(shape),'单负载观测结构与网络规模.csv')
    interventions=[]
    for nominal in [1500,2300]:
        before=pairs[(pairs.dataset=='RealHyFC')&pairs.primary&(pairs.nominal_h==nominal)].iloc[0];after=pairs[(pairs.dataset=='RealHyFC')&pairs.after_start&(pairs.nominal_h==nominal)].iloc[0]
        interventions.append(dict(nominal_h=nominal,before=before.diagnostic,after=after.diagnostic,same_operating_anchor=before.operating_id==after.operating_id,operating_id=before.operating_id,after_minus_before_Dref_mV=(after.D_ref-before.D_ref)*1000,after_minus_before_Dlow_mV=(after.D_low-before.D_low)*1000,both_ref_supported=before.both_supported and after.both_supported,identity='启停干预诊断对照，不算新增独立老化配对，不混入主扫描序列'))
    save(pd.DataFrame(interventions),'启停诊断与共享运行锚点.csv')
    # 核对旧记录标签，原记录保持不动。
    reviews=[dict(item='2B与2C时间单位',finding='2B Real为V/循环，2C按相对损失/真实过去小时重估，未来按名义12h；代码与各自冻结协议一致，不能当作同一参数重用',effect='未发现V/h直接误当V/cycle'),dict(item='2C n_target_fit字段',finding='共同流程可用历史数；保持/直接迁移不使用目标漂移统计，2C已有实际信息使用表说明',effect='不改变预测，不能把字段当各模型自由拟合样本量'),dict(item='诊断转移计数口径',finding='2C表记全诊断链0/1/2；2B可用配对链按n_pairs-1。本轮同时列出两个口径，早期可用配对链0/0/1',effect='所有早期场景仍不合格，不改历史结论'),dict(item='恒定漂移',finding='2C未来q固定，q_change_after_origin及future_observation_updates为0',effect='速率偏差随长名义时长累积；不是恢复机理已辨识'),dict(item='数学共同表征',finding='两参考损失可算；固定负载瞬时观测只有一个线性组合，还有运行偏差；未建立合法观测标定',effect='数学公式不等于状态可识别或迁移合格')]
    save(pd.DataFrame(reviews),'公式_代码_输入合同核查.csv')
    pd.DataFrame([dict(path=p,sha256=h) for p,h in sorted(SOURCES.items())]).to_csv(TASK/'配置/来源证据SHA256.csv',index=False,encoding='utf-8-sig')
    identity=dict(status='READ_ONLY_AUDIT_COMPUTED',utc=datetime.now(timezone.utc).isoformat(),head=cfg['head'],protocol_sha256=sha(TASK/'配置/阶段2D审计协议.json'),source_files=len(SOURCES),historical_prediction_rows=len(pred),unique_physical_diagnostics=len(pairs),retrained_models=0,formal_code_modified=False)
    (TASK/'输出结果/审计计算完成.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(relations).to_string(index=False));print(pd.DataFrame(excess).to_string(index=False))
if __name__=='__main__':main()
