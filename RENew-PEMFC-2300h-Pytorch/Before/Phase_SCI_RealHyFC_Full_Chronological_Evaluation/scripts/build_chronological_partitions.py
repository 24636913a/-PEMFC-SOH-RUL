from common import *
def partitions(n):
    assert isinstance(n,int) and n>3005
    bounds=[0,2103,2404,3005,n]
    return {name:(a,b) for name,a,b in zip(PARTS,bounds[:-1],bounds[1:])}
def save_partitions(f):
    rows=[]
    for name,(a,b) in partitions(len(f)).items():
        z=f.iloc[a:b]
        rows.append(dict(partition=name,start_inclusive=a,end_exclusive=b,records=b-a,first_bin_end_h=float(z.bin_end_ophrs2_h.iloc[0]),last_bin_end_h=float(z.bin_end_ophrs2_h.iloc[-1]),first_datetime=str(z.first_datetime.iloc[0]),last_datetime=str(z.last_datetime.iloc[-1]),valid_records=int(z.bin_valid_for_windows.sum())))
    cs('tables/TIME_PARTITIONS.csv',rows);return rows
