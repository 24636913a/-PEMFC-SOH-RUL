"""The only unbounded feature reader; full-domain access explicitly authorized here."""
from common import *
def load_full():
    # Keep pandas' historical default numeric parser for bit-identical H inputs.
    f=pd.read_csv(SOURCE)
    assert f.index.equals(pd.RangeIndex(len(f)))
    assert f.columns.is_unique and set(FEATURES).issubset(f.columns)
    assert f.bin_end_ophrs2_h.is_monotonic_increasing and f.bin_end_ophrs2_h.is_unique
    assert set(f.bin_valid_for_windows.dropna().unique()).issubset({0,1})
    assert not f.bin_valid_for_windows.isna().any()
    dev=f.iloc[:3007].loc[lambda x:x.bin_end_ophrs2_h<1819.997738]
    assert len(dev)==3005 and np.array_equal(dev.index,np.arange(3005)), 'DEVELOPMENT_ROW_IDENTITY_CHANGED'
    return f
def save_structure(f):
    cs('outputs/audits/FULL_ROW_IDENTITY.csv',f.reset_index(names='row_id'))
    cs('outputs/audits/SCHEMA.csv',[dict(field=k,dtype=str(f[k].dtype),missing=int(f[k].isna().sum())) for k in f])
    js('outputs/audits/FULL_SOURCE.json',dict(path=str(SOURCE),sha256=sha(SOURCE),rows=len(f),columns=len(f.columns),parser='pandas read_csv defaults, historical H parser',row_order='unchanged physical CSV order',development_prefix_filtered_rows=3005,late_access='explicit user scheme B; exploratory only',phase_L_decision=read(ROOT/'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/outputs/L_FINAL_DECISION.json')))
