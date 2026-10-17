"""Independent complete budget, minibatch journal and same-seed sampler audit."""
from baseline_common import *

def main():
    c=cfg();h.configure_normal(c);expected={(m,s) for m,s in RUNS};events=[json.loads(line) for line in (P/'outputs/execution_ledger/events.jsonl').read_text(encoding='utf-8').splitlines()]
    starts=[r for r in events if r['event']=='RUN_STARTED'];ends=[r for r in events if r['event']=='RUN_COMPLETED']
    assert len(starts)==len(ends)==6;assert {(r['model'],r['seed']) for r in starts}==expected;assert {(r['model'],r['seed']) for r in ends}==expected
    assert not any(r['event']=='RUN_FAILED' for r in events)
    states=list((P/'outputs/training_logs').glob('*/run_state.json'));assert len(states)==6
    rows=[];sources=[]
    for name,seed in RUNS:
        run=f'{name}_seed{seed}';log=rd(P/f'outputs/training_logs/{run}/epochs.csv');history=H/f'outputs/training_logs/H-F2-Seed{seed}/epochs.csv';old=rd(history)
        metric=read(P/f'outputs/metrics/{run}.json');ck=torch.load(P/metric['checkpoint'],map_location='cpu',weights_only=True)
        h.e.set_seed(seed);m=model(name,'cpu');assert h.e.state_digest(m)==metric['initialization_sha256'],'FROM_SCRATCH_INITIALIZATION_MISMATCH'
        changed=sum(not torch.equal(value.detach(),ck['state_dict'][key]) for key,value in m.named_parameters());assert changed>0,'UNCHANGED_TRAINABLE_WEIGHTS'
        m.load_state_dict(ck['state_dict'],strict=True)
        if name=='TCN':
            for block in m.tcn:assert int(block.norm.num_batches_tracked)==44*metric['best_epoch'],'BATCHNORM_TRAIN_ONLY_UPDATE_COUNT'
        assert 'sampler_origin_order_sha256' in old
        n=min(len(log),len(old));assert np.array_equal(log.sampler_origin_order_sha256.iloc[:n],old.sampler_origin_order_sha256.iloc[:n]),('H_SAMPLER_ORDER_CHANGED',run)
        for other in ['LSTM','TCN']:
            if other==name:continue
            right=rd(P/f'outputs/training_logs/{other}_seed{seed}/epochs.csv');common=min(len(log),len(right));assert np.array_equal(log.sampler_origin_order_sha256.iloc[:common],right.sampler_origin_order_sha256.iloc[:common])
        journal=P/f'outputs/training_logs/{run}/batch_events.jsonl';counts={}
        with journal.open(encoding='utf-8') as stream:
            for line in stream:
                batch=json.loads(line);epoch=int(batch['epoch']);i=int(batch['batch']);counts.setdefault(epoch,[]).append(i)
                assert math.isfinite(batch['train_standardized_MSE']) and math.isfinite(batch['gradient_norm_before_clip'])
        assert set(counts)==set(range(1,len(log)+1))
        for batches in counts.values():assert batches==list(range(44))
        rows.append(dict(model=name,seed=seed,epochs=len(log),minibatches=44*len(log),changed_parameter_tensors=changed,from_scratch_initialization_exact=True,sampler_equal_to_H_epochs=n,other_baseline_common_epochs=common,status='PASS'))
        sources.append(dict(path=history.relative_to(REPO).as_posix(),sha256=sha(history)))
    cs('outputs/audits/EXECUTION_AND_SAMPLER_AUDIT.csv',rows);cs('outputs/audits/H_SAMPLER_SOURCES.csv',sources)
    js('outputs/audits/EXECUTION_AND_SAMPLER_AUDIT.json',dict(status='PASS',formal_starts=len(starts),formal_completions=len(ends),additional_runs=0,restarts=0,all_epochs_exact_44_batches=True,same_seed_sampler_matches_H_and_other_baseline=True,all_best_weights_differ_from_initialization=True,TCN_batchnorm_updates_equal_Train_batches_only=True,comparison_does_not_require_same_architecture_initial_parameters=True))
    print('EXECUTION_AND_SAMPLER_AUDIT_PASS',flush=True)

if __name__=='__main__':main()
