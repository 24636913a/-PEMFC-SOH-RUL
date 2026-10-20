"""Frozen selection: Validation only, no holdout table is accepted here."""
import math
import pandas as pd
import frozen_f2 as f
import settings as s


def qualify(candidate,control):
    f.require(candidate.partition.eq('VALIDATION').all() and control.partition.eq('VALIDATION').all(),'VALIDATION_ONLY_SELECTION')
    f.require(set(candidate.seed)==set(control.seed)==set(s.INITIAL_SEEDS),'THREE_PAIRED_SEEDS')
    a=candidate.set_index('seed').loc[list(s.INITIAL_SEEDS)]
    b=control.set_index('seed').loc[list(s.INITIAL_SEEDS)]
    def mean(frame,key):return math.fsum(frame[key])/3
    checks=dict(all_three_RMSE_lower=bool((a.RMSE_mV<b.RMSE_mV).all()),
        mean_RMSE_at_least_5pct_lower=mean(a,'RMSE_mV')<=.95*mean(b,'RMSE_mV'),
        mean_MAE_not_higher=mean(a,'MAE_mV')<=mean(b,'MAE_mV'),
        mean_P99_guard=mean(a,'P99_mV')<=1.05*mean(b,'P99_mV'),
        worst_seed_MAE_guard=bool((a.MAE_mV<=1.10*b.MAE_mV).all()),
        worst_seed_P99_guard=bool((a.P99_mV<=1.10*b.P99_mV).all()))
    return dict(qualified=all(checks.values()),checks=checks,
        mean_RMSE_ratio=mean(a,'RMSE_mV')/mean(b,'RMSE_mV'),
        mean_MAE_ratio=mean(a,'MAE_mV')/mean(b,'MAE_mV'),paired_wins=int((a.RMSE_mV<b.RMSE_mV).sum()))


def collect(run,stage):
    rows=[]
    for folder in sorted((run/'models'/stage).iterdir()):
        ledger=f.read_json(folder/'TRAINING_LEDGER.json')
        f.require(ledger['status']=='TRAINED','INCOMPLETE_STAGE')
        f.require(f.read_json(folder/'EVALUATION_AUDIT.json')['status']=='PASS','INSTANCE_AUDIT_REQUIRED')
        score=f.read_csv(folder/'METRICS.csv');score=score.loc[score.partition=='VALIDATION'].iloc[0].to_dict()
        rows.append(dict(candidate=ledger['candidate'],seed=ledger['seed'],best_epoch=ledger['best_epoch'],
            stop_epoch=ledger['stop_epoch'],training_seconds=ledger['training_seconds'],parameters=ledger['parameters'],**score))
    return pd.DataFrame(rows)


def choose(table,qualified):
    options=[]
    for name in qualified:
        values=table.loc[table.candidate==name,'RMSE_mV']
        f.require(len(values)==3,'SELECT_THREE_SEED_MEAN')
        options.append((math.fsum(values)/3,name))
    if not options:return None
    minimum=min(score for score,name in options)
    return sorted(name for score,name in options if score<=minimum+1e-6)[0]
