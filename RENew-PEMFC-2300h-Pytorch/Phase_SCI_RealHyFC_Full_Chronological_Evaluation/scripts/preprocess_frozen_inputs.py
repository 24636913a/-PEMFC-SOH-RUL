from common import *
def scaler():return read(R2/'outputs/audits/SCALER.json')
def preprocess(x,s=None):
    s=scaler() if s is None else s
    z=x.copy()
    for t in range(1,48):z[:,t,0]=.3*x[:,t,0]+.7*z[:,t-1,0]
    z=((z-np.asarray(s['mean']))/np.asarray(s['scale'])).astype(np.float32)
    z[:,:,[i for i in range(14) if i not in TOP10]]=0.
    return z
def verify_h_data(ds):
    old=np.load(R2/'outputs/audits/FROZEN_H_DATA.npz',allow_pickle=False);checks=[]
    maps={'TRAIN':dict(x_raw='train_x_raw',x='tx',y='train_y',origins='train_origins',targets='train_targets'),'VALIDATION':dict(x_raw='val_x_raw',x='vx',y='val_y',origins='val_origins',targets='val_targets',origin_age_h='val_origin_age_h',target_age_h='val_target_age_h',origin_raw_voltage_V='val_origin_raw_voltage_V',target_raw_voltage_V='val_target_raw_voltage_V')}
    for name,cols in maps.items():
        for new,k in cols.items():
            a,b=ds[name][new],old[k];ok=a.dtype==b.dtype and np.array_equal(a,b)
            checks.append(dict(partition=name,field=new,shape=str(a.shape),new_sha256=ah(a),historical_sha256=ah(b),exact=ok))
            assert ok,('H_DATA_MISMATCH',name,new)
    raw=ds['TRAIN']['x_raw'].astype('float64').reshape(-1,14);scale=raw.std(axis=0);scale=np.where(scale>1e-8,scale,1.)
    assert np.array_equal(raw.mean(axis=0),np.asarray(scaler()['mean']))
    assert np.array_equal(scale,np.asarray(scaler()['scale']))
    cs('outputs/audits/H_INPUT_EXACT_PARITY.csv',checks)
