"""One-shot protocol registration BEFORE any new model forward or outcome inspection."""
from common import *
def main():
    assert not (P/'configs/frozen_protocol.json').exists(),'PROTOCOL_ALREADY_FROZEN'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    for p in ['configs','outputs/predictions','outputs/audits','figures','reports','tests']:dest(p+'/placeholder').parent.mkdir(parents=True,exist_ok=True)
    c=read(R2/'configs/frozen_seed_screening.json'); sources={}
    # Read only explicitly frozen Development tensor authorities and model source.
    for k,v in c['source_sha256'].items():
        path=Path(k) if Path(k).is_absolute() else REPO/k
        assert sha(path)==v,('UPSTREAM_IDENTITY',k)
        sources[str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)]=v
    for k,v in c['data_sha256'].items():
        path=R2/k; assert sha(path)==v; sources[str(path.relative_to(REPO))]=v
    candidates=rd(R2/'outputs/F2_SIX_CANDIDATE_SEEDS.csv'); assert list(candidates.seed)==SEEDS
    models=[]
    for r in candidates.itertuples():
        ck=Path(r.checkpoint); ck=ck if ck.is_absolute() else R2/ck
        pred=R2/r.predictions
        assert sha(ck)==r.checkpoint_sha256 and sha(pred)==r.prediction_sha256
        sources[str(ck.relative_to(REPO))]=sha(ck); sources[str(pred.relative_to(REPO))]=sha(pred)
        models.append(dict(seed=r.seed,checkpoint=str(ck.relative_to(REPO)),predictions=str(pred.relative_to(REPO)),protocol_sha256=r.protocol_sha256))
    for f in [R2/'configs/frozen_seed_screening.json',J/'outputs/gate_diagnostics/J_TRAIN_ONLY_CONDITION_THRESHOLDS.csv',J/'outputs/gate_values/J_ALL_F2_GATES.csv',J/'outputs/fixed_gate_diagnostic/J_FIXED_GATE_COMPARISON.csv']:
        sources[str(f.relative_to(REPO))]=sha(f)
    d=np.load(R2/'outputs/audits/FROZEN_H_DATA.npz'); assert d['tx'].shape==(1380,48,14) and d['vx'].shape==(205,48,14)
    thresholds=rd(J/'outputs/gate_diagnostics/J_TRAIN_ONLY_CONDITION_THRESHOLDS.csv')
    precision=[]
    for r in thresholds.itertuples():
        raw=d['train_x_raw'][:,:,r.raw_f2_index]
        q=np.quantile(raw.astype('float64').mean(axis=1),[1/3,2/3]);original=np.array([r.low_upper,r.high_lower])
        # Snapshot raw features were serialized as float32; J used original float64.
        # Quantiles are 1-Lipschitz in max norm. Bound only that known serialization error.
        bound=float((np.abs(np.spacing(raw)).astype('float64')*.5).mean(axis=1).max())+1e-13
        assert np.max(abs(q-original))<=bound,('TRAIN_THRESHOLD_BEYOND_SERIALIZATION_PRECISION',r.feature)
        precision.append(dict(feature=r.feature,J_q1=original[0],J_q2=original[1],snapshot_q1=q[0],snapshot_q2=q[1],max_absolute_difference=float(max(abs(q-original))),float32_serialization_bound=bound))
    cs('outputs/audits/TRAIN_THRESHOLD_PRECISION.csv',precision)
    cs('configs/train_condition_thresholds.csv',thresholds)
    perms=list(range(2026101601,2026101611)); cs('configs/permutations.csv',derangements(205,perms))
    cnew=dict(base_commit=BASE,seeds=SEEDS,models=models,input_sha256=sources,
        hyperparameters=c['hyperparameters'],normal_runtime=c['normal_runtime'],data_protocol=c['data'],preprocessing=c['preprocessing'],
        gate_features=FEATURES,gate_raw_indices=INDICES,gate_definition='g*LSTM+(1-g)*CNN; one scalar per 48-step history, broadcast across encoder',
        B0='unmodified dynamic gate',B1='temporary g=0.5; NOT independently trained F1',
        B2='float64 arithmetic mean of 1380 Train forward gate outputs, cast once to float32 in hook',
        C='10 fixed label-free random derangements of all 205 Validation gates; no best-repeat selection',permutation_seeds=perms,
        gate_statistics=dict(SD_ddof=1,dynamic_SD_gt=.001,dynamic_range_gt=.01,saturation_low_lt=.1,saturation_high_gt=.9,middle=[.4,.6]),
        grouping='reuse Phase J original float64 raw-history Train terciles EXACTLY; float32 snapshot precision separately bounded, no Validation tuning',
        fixed_window_origin_row=2205,unfavorable_rule='per seed origin with largest RMSE(B0)-RMSE(B2), ties smaller origin; report even if negative',
        metric='pooled 4920 queries; target-time-equal mean of per-target MSE; R2 on physical DeltaV',
        selection_bias='Six instances selected using original B0 Validation performance; diagnostic only; overlapping histories not independent samples',
        training=False,Assessment_access=False,Test_access=False,raw_csv_read=False)
    for f in [P/'configs/permutations.csv',P/'configs/train_condition_thresholds.csv']+list((P/'scripts').glob('*.py'))+list((P/'tests').glob('*.py')):
        cnew['input_sha256'][str(f.relative_to(REPO))]=sha(f)
    cs('outputs/audits/HISTORY_BEFORE.csv',historical_inventory())
    js('configs/frozen_protocol.json',cnew); txt('configs/frozen_protocol.sha256',sha(P/'configs/frozen_protocol.json')+'\n')
    print('PROTOCOL_FROZEN_BEFORE_FORWARD',sha(P/'configs/frozen_protocol.json'),flush=True)
if __name__=='__main__':main()
