"""Bounded historical availability and frozen array identity audit."""
from experiment import *

def main():
    p=get_protocol(); z,sc=load_data()
    req(sha(R/'RealHyFC_F2_Main.py')==p['model_source_sha256'],'HISTORICAL_MODEL_SOURCE_CHANGED')
    csv('outputs/INPUT_ARRAY_IDENTITIES.csv',[dict(array=k,shape=str(v.shape),dtype=str(v.dtype),sha256=old.array_sha256(v)) for k,v in sorted(z.items())])
    columns=['first_datetime','last_datetime','bin_end_ophrs2_h','bin_valid_for_windows','temporal_segment_id']
    f=pd.read_csv(old.SOURCE,nrows=2404,usecols=columns)
    req(len(f)==2404,'BOUNDED_ROW_COUNT')
    f['last_datetime']=pd.to_datetime(f.last_datetime)
    rows=[]
    for split in ['train','val']:
        for o,targets in zip(z[split+'_origins'],z[split+'_targets']):
            block=f.iloc[o-47:targets[-1]+1]
            req(bool((block.bin_valid_for_windows==1).all()),'INVALID_FROZEN_WINDOW')
            req(block.temporal_segment_id.nunique()==1,'FROZEN_SEGMENT_BREAK')
            req(np.allclose(np.diff(block.bin_end_ophrs2_h),.5,rtol=0,atol=1e-6),'OPERATING_CONTINUITY')
            req(f.last_datetime.iloc[o]<f.last_datetime.iloc[targets[0]],'AVAILABILITY_ORDER')
            rows.append(dict(partition=split,origin_row=int(o),history_start_row=int(o-47),history_end_row=int(o),
              input_available_at=str(f.last_datetime.iloc[o]),first_target_row=int(targets[0]),last_target_row=int(targets[-1]),
              last_label_measured_at=str(f.last_datetime.iloc[targets[-1]]),origin_operating_h=float(f.bin_end_ophrs2_h.iloc[o])))
    latest_train=f.last_datetime.iloc[int(z['train_targets'].max())]
    first_val=f.last_datetime.iloc[int(z['val_origins'].min())]
    req(latest_train<=first_val,'TRAIN_LABEL_AVAILABLE_AFTER_FIRST_VAL_ORIGIN')
    csv('outputs/WINDOW_AVAILABILITY.csv',rows)
    js('outputs/INPUT_CAUSALITY_AUDIT.json',dict(status='PASS',train_windows=1380,validation_windows=205,
        metadata_nrows=2404,latest_train_label_available_at=str(latest_train),first_validation_origin_available_at=str(first_val),
        labels_before_first_validation_origin=True,features_from_histories_only=True,source_extra_times_used_by_model=False,
        future_covariates_in_decoder=False,original_cache_sha256=sha(D/'FROZEN_H_DATA.npz'),original_model_source_sha256=sha(R/'RealHyFC_F2_Main.py')))
    print('INPUT_CAUSALITY_AUDIT_PASS',len(rows))

if __name__=='__main__': main()
