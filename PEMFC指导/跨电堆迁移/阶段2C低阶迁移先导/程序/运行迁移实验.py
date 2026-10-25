"""阶段2C正式CPU低阶预测入口；不读取旧预测或未来工况拟合。"""
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from 迁移核心 import prepare,fit,build_models,predict,metrics

HERE=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def save(df,p):df.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.17g')
def run(data,config,out):
    if out.exists():raise FileExistsError('新运行必须使用全新输出目录')
    (out/'表格').mkdir(parents=True);(out/'模型记录').mkdir()
    start=time.perf_counter();protocol=json.loads(config.read_text(encoding='utf-8-sig'))
    registration=json.loads((config.parent/'预登记时间与SHA256.json').read_text(encoding='utf-8-sig'))
    assert registration['protocol_sha256']==sha(config)
    fc0=pd.read_csv(data/'FC1小时观测.csv');real0=pd.read_csv(data/'RealHyFC循环观测.csv')
    contract=json.loads((data/'来源合同与SHA256.json').read_text(encoding='utf-8-sig'))
    for c in contract:
        if 'local' in c:assert sha(data/Path(c['local']).name)==c['sha256']
    fc,real=prepare(fc0,real0);source_data=fc[(fc.partition=='Train')&fc.valid];source=fit(source_data,'FC1')
    dump(out/'模型记录/FC1源端参数.json',source)
    common=real[(real.id>protocol['evaluation']['common_after_raw_cycle'])&(real.start_time.notna())&(real.operating_start_h>=protocol['evaluation']['common_after_h'])]
    # 保存缺失身份；评价只用既有B60合格实测，不以结果筛选。
    missing=common[['id','available_h','available_ns','partition','valid','reason']].copy()
    missing['role']=np.where(missing.valid,'EVALUABLE','EXCLUDED_FROZEN_OBSERVATION_INVALID')
    save(missing,out/'表格/全部共同目标及缺失记录.csv');truth=common[common.valid]
    rows=[];parameters=[];inners=[];scenarios=[];costs=[]
    for sc in protocol['scenarios']:
        begin=time.perf_counter();cut=sc['cutoff_h'];name=sc['name'];hist=real[real.valid&(real.available_h<=cut)].sort_values('h');clock=real[real.available_h<=cut].sort_values('id').iloc[-1]
        p,selected,inner,states=build_models(hist,source,cut,protocol)
        record=dict(scenario=name,cutoff_h=cut,clock_anchor_id=int(clock.id),last_observation_id=int(hist.id.iloc[-1]),last_observation_h=float(hist.h.iloc[-1]),last_observation_calendar=pd.Timestamp(int(hist.available_ns.iloc[-1])).isoformat(),n_history=len(hist),history_ids=hist.id.astype(int).tolist(),source=source,target=p,selected=selected,states=states,diagnostic_nominal_h=sc['diagnostic_nominal_h'],availability='完成后可用的探索性离线假设；实际发布时间未知')
        dump(out/f'模型记录/{name}_参数与状态.json',record)
        for x in inner:inners.append(dict(scenario=name,**x))
        scenarios.append({k:v for k,v in record.items() if k not in ['source','target','selected','states','history_ids']})
        for method,states_m in states.items():
            state=states_m['at_cutoff'];last_state=states_m['at_last_observation']
            nlead=12*(truth.id.to_numpy()-int(clock.id));yp,slow,rev=predict(state,nlead)
            for i,r in enumerate(truth.itertuples()):
                rows.append(dict(scenario=name,method=method,origin_h=cut,origin_clock_id=int(clock.id),history_last_id=int(hist.id.iloc[-1]),history_last_h=float(hist.h.iloc[-1]),n_history=len(hist),
                  target_id=int(r.id),target_calendar=pd.Timestamp(int(r.available_ns)).isoformat(),target_actual_h=float(r.available_h),target_legacy_partition=r.partition,nominal_target_h=float(cut+nlead[i]),nominal_lead_h=float(nlead[i]),actual_lead_h=float(r.available_h-cut),raw_cycle_lead=int(r.id-clock.id),
                  base_V=float(r.base_V),true_v=float(r.v),true_soh=float(r.SOH),true_d=float(r.d),pred_d=float(yp[i]),pred_soh=float(1-yp[i]),pred_v=float(r.base_V*(1-yp[i])),slow_d=float(slow[i]),reversible_d=float(rev[i]),rate_per_h=float(state['q']),q_change_after_origin=0,future_observation_updates=0))
            parameters.append(dict(scenario=name,method=method,n_target_fit=len(hist),source_n=len(source_data) if method.startswith(('B_','C_')) else 0,source_rate_per_h=source['q_bounded'],target_global_rate_per_h=p['q_bounded'],model_rate_per_h=state['q'],rate_V_per_h=state['q']*p['base_V'],rate_cap_per_h=p['rate_cap'],inner_setting=selected['Bounded'] if method=='D_Target2B_Bounded' else selected['SlowReversible'] if method=='D_Target2B_SlowReversible' else np.nan,
              state_L=state['L'],state_R=state['R'],state_q=state['q'],last_measured_d=float(hist.d.iloc[-1]),cutoff_residual_d=state['L']+state['R']-hist.d.iloc[-1],known_past_propagation_h=cut-last_state['last_h'],state_history=len(last_state['history_ids']),state_updates=last_state['measurement_updates'],tau_h=state['tau_h']))
        costs.append(dict(scenario=name,seconds=time.perf_counter()-begin,scope='target stats, five fixed inner candidates, six states and forecasts',hardware=platform.processor(),python=platform.python_version()))
    pred=pd.DataFrame(rows);save(pred,out/'表格/各方法逐起点预测.csv');save(pd.DataFrame(parameters),out/'表格/模型参数与速率.csv');save(pd.DataFrame(inners),out/'表格/训练内部验证候选.csv');save(pd.DataFrame(scenarios),out/'表格/300_600及后移起点资格.csv');save(pd.DataFrame(costs),out/'表格/CPU运行代价.csv')
    results=[]
    for (sc,method),g in pred.groupby(['scenario','method'],sort=False):
        results.append(dict(scenario=sc,method=method,scope='COMMON_ALL',status='EVALUABLE',**metrics(g)))
        for partition,sub in g.groupby('target_legacy_partition',sort=False):results.append(dict(scenario=sc,method=method,scope='LEGACY_'+partition,status='EVALUABLE',**metrics(sub)))
        for h in protocol['evaluation']['auxiliary_complete_blocks']:
            ids=set(range(52,52+h));sub=g[g.target_id.isin(ids)]
            if set(sub.target_id)==ids:results.append(dict(scenario=sc,method=method,scope=f'COMMON_COMPLETE_{h}_CYCLES',status='EVALUABLE',**metrics(sub)))
            else:results.append(dict(scenario=sc,method=method,scope=f'COMMON_COMPLETE_{h}_CYCLES',status='NA_MISSING_FROZEN_CYCLE',n=0,missing_ids=';'.join(map(str,sorted(ids-set(sub.target_id))))))
    table=pd.DataFrame(results);save(table,out/'表格/电压_SOH及损失误差.csv')
    gains=[]
    for (sc,scope),g in table[table.status=='EVALUABLE'].groupby(['scenario','scope'],sort=False):
        for transfer in ['B_FC1Direct','C_FC1PriorAdapted']:
            t=g[g.method==transfer].iloc[0]
            for reference in ['A_Persistence','A_TargetBounded','D_Target2B_Bounded','D_Target2B_SlowReversible']:
                b=g[g.method==reference].iloc[0]
                gains.append(dict(scenario=sc,scope=scope,transfer=transfer,reference=reference,RMSE_gain_pct=100*(1-t.RMSE_mV/b.RMSE_mV) if b.RMSE_mV>0 else np.nan,MAE_gain_pct=100*(1-t.MAE_mV/b.MAE_mV) if b.MAE_mV>0 else np.nan,transfer_RMSE_mV=t.RMSE_mV,reference_RMSE_mV=b.RMSE_mV,negative_transfer=bool(t.RMSE_mV>b.RMSE_mV)))
    save(pd.DataFrame(gains),out/'表格/迁移收益与负迁移.csv')
    # 分布只描述固定源Train、目标适配历史和既定评价；不据此调参。
    distributions=[]
    for label,g in [('FC1_Train',source_data)]+[(s['name'],real[real.valid&(real.h<=s['cutoff_h'])]) for s in protocol['scenarios']]+[('Real_共同未来_仅事后',truth)]:
        distributions.append(dict(group=label,n=len(g),first_h=float(g.h.min()),last_h=float(g.h.max()),loss_min=float(g.d.min()),loss_mean=float(g.d.mean()),loss_max=float(g.d.max()),loss_std=float(g.d.std()),base_V=float(g.base_V.iloc[0])))
    save(pd.DataFrame(distributions),out/'表格/源目标归一化损失分布.csv')
    dump(out/'执行身份.json',dict(protocol_sha256=sha(config),head=protocol['head'],utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,total_seconds=time.perf_counter()-start,n_predictions=len(pred),n_unique_targets=len(truth),n_origins_per_scenario=1,source_fit_ids=source['fit_ids'],source_fit_last_h=source['last_h'],data_contract_sha256=sha(data/'来源合同与SHA256.json'),script_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},status='COMPUTED_NOT_YET_INDEPENDENTLY_VERIFIED'))
    print(table[table.scope=='COMMON_ALL'][['scenario','method','n','RMSE_mV','MAE_mV','RMSE_SOH_pp']].to_string(index=False))

def main():
    work=HERE if (HERE/'数据准备').exists() else HERE.parents[1]/'跨电堆迁移/阶段2C低阶迁移先导'
    fresh=work/'输出结果/正式版本复算'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,default=work/'数据准备');parser.add_argument('--config',type=Path,default=HERE/'配置/阶段2C冻结协议.json');parser.add_argument('--output',type=Path,default=fresh);args=parser.parse_args();run(args.data,args.config,args.output)
if __name__=='__main__':main()
