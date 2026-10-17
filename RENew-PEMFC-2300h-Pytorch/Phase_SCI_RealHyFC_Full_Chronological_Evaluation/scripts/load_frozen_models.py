from common import *
sys.path.insert(0,str(H/'scripts'));import h_common as h
sys.path.insert(0,str(B/'models'));from baseline_models import LSTMForecaster,TCNForecaster
ARCH=dict(LSTM=dict(input_dim=14,horizon=24,hidden_dim=64,num_layers=2,dropout=.1),TCN=dict(input_dim=14,horizon=24,hidden_dim=48,num_layers=3,kernel_size=3,dropout=.1))
COUNTS=dict(F0=323483,F1=332379,F2=332436,LSTM=55448,TCN=39048)
def make_inventory():
    metrics=rd(H/'outputs/metrics/H_VALIDATION_METRICS.csv');metrics=metrics.loc[metrics.role=='MULTISEED_PRIMARY']
    hc={r['run_id']:r for r in read(H/'outputs/audits/H_FINAL_ARTIFACT_INTEGRITY_AUDIT.json')['formal_checkpoints']};rows=[]
    for model in MODELS:
        for seed in SEEDS:
            if model in ['F0','F1','F2']:
                z=metrics.loc[(metrics.model==model)&(metrics.seed==seed)].iloc[0];ck=H/z.checkpoint;pr=H/z.validation_prediction
                cksha=hc[z.run_id]['checkpoint_sha256'];prsha=z.prediction_file_sha256;proto=z.protocol_sha256
            else:
                z=read(B/f'outputs/metrics/{model}_seed{seed}.json');ck=B/z['checkpoint'];pr=B/z['predictions'];cksha=z['checkpoint_sha256'];prsha=z['prediction_sha256'];proto=z['protocol_sha256']
            assert sha(ck)==cksha and sha(pr)==prsha,('HISTORICAL_CHECKPOINT_OR_PREDICTION_CHANGED',model,seed)
            rows.append(dict(model=model,seed=seed,checkpoint=ck.relative_to(REPO).as_posix(),checkpoint_sha256=cksha,checkpoint_bytes=ck.stat().st_size,reference_prediction=pr.relative_to(REPO).as_posix(),prediction_sha256=prsha,historical_protocol_sha256=proto,parameter_count=COUNTS[model]))
    cs('outputs/audits/CHECKPOINT_INVENTORY.csv',rows);return rows
def load(row,device='cuda'):
    assert sha(REPO/row.checkpoint)==row.checkpoint_sha256
    ck=torch.load(REPO/row.checkpoint,map_location='cpu',weights_only=True)
    assert ck['protocol_sha256']==row.historical_protocol_sha256
    c=h.load_protocol()
    if row.model in ['F0','F1','F2']:m=h.model(row.model,c['hyperparameters'],torch.device(device))
    else:m={'LSTM':LSTMForecaster,'TCN':TCNForecaster}[row.model](**ARCH[row.model]).to(device)
    assert sum(p.numel() for p in m.parameters())==row.parameter_count
    m.load_state_dict(ck['state_dict'],strict=True);m.eval()
    return m,ck
def state_hash(m):return h.e.state_digest(m)
def predict(m,x,model):
    m.eval();out=[]
    with torch.no_grad():
        for b in range(0,len(x),32):
            z=torch.from_numpy(x[b:b+32]).to('cuda')
            out.append((m.predict_delta(z) if model in ['F0','F1','F2'] else m(z)).cpu().numpy())
    return np.concatenate(out)
