"""Production byte-limited access. No boundary discovery from a raw value."""
import csv
import hashlib
import io
import re
from pathlib import Path
import numpy as np
import pandas as pd

class AccessFailure(RuntimeError):
    pass

def check(condition,message):
    if not condition:
        raise AccessFailure(message)

def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()

class LimitedBytes(io.RawIOBase):
    """Never asks the underlying unbuffered descriptor for a forbidden byte."""
    def __init__(self, stream, end):
        check(isinstance(end,int) and end>0,'INVALID_BYTE_BOUNDARY')
        self.stream=stream; self.end=end; self.count=0; self.max_request_end=0
    def readable(self):
        return True
    def read(self,n=-1):
        remaining=self.end-self.count
        if remaining==0:
            return b'' # no underlying EOF/next-byte probe
        wanted=remaining if n is None or n<0 else min(n,remaining)
        if wanted==0:
            return b''
        self.max_request_end=max(self.max_request_end,self.count+wanted)
        data=self.stream.read(wanted)
        check(len(data)==wanted,'SOURCE_TRUNCATED_BEFORE_REGISTERED_BOUNDARY')
        self.count+=len(data)
        return data
    def readinto(self, buffer):
        data=self.read(len(buffer)); buffer[:len(data)]=data
        return len(data)

def validate_permit(row):
    check(row.get('preauthorized_domain')=='ENTIRE_SOURCE_PROVEN_TRAIN','RAW_SCOPE_NOT_AUTHORIZED')
    check(row.get('byte_start')==0 and isinstance(row.get('byte_end'),int) and row['byte_end']>0,'EXACT_BYTE_BOUNDARY_MISSING')
    check(row['historical_age_min_h']>=72.5 and row['historical_age_max_h']<1310.,'NONTRAIN_SOURCE_DENIED')
    check(bool(re.fullmatch('[0-9a-f]{64}',row.get('historical_source_sha256',''))),'SOURCE_HASH_REFERENCE_MISSING')
    check(row.get('historical_record_count',0)>0,'RECORD_IDENTITY_MISSING')
    check(row.get('n1_rcr_range_and_count_agree') is True,'INDEPENDENT_RANGE_AUDITS_MISMATCH')
    check(row.get('historical_manifest_committed') is True,'SOURCE_PROVENANCE_UNBOUND')
    return row

def lookup_permit(relative, allowlist):
    check(relative in allowlist,'RAW_SOURCE_NOT_ALLOWLISTED:'+relative)
    return validate_permit(allowlist[relative])

def permitted_path(root,relative,allowlist):
    row=lookup_permit(relative,allowlist) # denial precedes is_file/open
    path=(root/relative).resolve()
    check(path.is_relative_to(root.resolve()),'PATH_ESCAPE_DENIED')
    check(path.is_file(),'SOURCE_UNAVAILABLE')
    check(path.stat().st_size==row['byte_end'],'SOURCE_SIZE_CHANGED')
    return path,row

def verify_registered_bytes(root,relative,allowlist):
    path,row=permitted_path(root,relative,allowlist)
    # This authenticates an already frozen fully-Train extent. It never finds a
    # cutoff by scanning ages, and MUST NOT be used on a mixed-source file.
    h=hashlib.sha256()
    with path.open('rb',buffering=0) as f:
        bounded=LimitedBytes(f,row['byte_end'])
        while bounded.count<bounded.end:
            h.update(bounded.read(1048576))
        check(f.tell()==row['byte_end'],'UNEXPECTED_PHYSICAL_POSITION')
    observed=h.hexdigest()
    check(observed==row['historical_source_sha256'],'SOURCE_CONTENT_IDENTITY_MISMATCH')
    return dict(source_file=relative,identity='PASS',observed_sha256=observed,
        bytes_authenticated=row['byte_end'],numeric_rows_parsed=0,
        extent='ENTIRE_ALREADY_PROVEN_TRAIN_SOURCE',mixed_file_whole_hash=False)

def inspect_authorized_header(root,relative,allowlist):
    path,row=permitted_path(root,relative,allowlist)
    payload=bytearray(); quoted=False
    with path.open('rb',buffering=0) as f:
        stream=LimitedBytes(f,row['byte_end'])
        while True:
            b=stream.read(1); check(bool(b),'RAW_HEADER_TRUNCATED')
            payload.extend(b); check(len(payload)<1048576,'RAW_HEADER_TOO_LARGE')
            if b==b'"': quoted=not quoted
            if b==b'\n' and not quoted: break
    header=list(csv.reader(io.StringIO(payload.decode('utf-8-sig')),delimiter=';',strict=True))
    check(len(header)==1 and len(header[0])==len(set(header[0])),'RAW_HEADER_SCHEMA_INVALID')
    return dict(UTF8_BOM=payload.startswith(b'\xef\xbb\xbf'),header_bytes_read=len(payload),
      header_has_quotes=b'"' in payload,line_terminator='CRLF' if payload.endswith(b'\r\n') else 'LF',
      columns=header[0],header_numeric_records_read=0)

def numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series,errors='coerce')
    x=series.astype('string').str.replace('\ufeff','',regex=False).str.strip()
    comma=x.str.contains(',',regex=False,na=False); dot=x.str.contains('.',regex=False,na=False)
    x=x.mask(comma & dot,x.str.replace(',','',regex=False))
    x=x.mask(comma & ~dot,x.str.replace(',','.',regex=False))
    return pd.to_numeric(x,errors='coerce')

def validate_chronology(age,timestamps, previous_age=None,previous_time=None):
    a=np.asarray(age,float); t=pd.to_datetime(timestamps,errors='coerce')
    check(np.isfinite(a).all() and not pd.isna(t).any(),'TIME_IDENTITY_NONFINITE')
    check(np.all(a>=72.5) and np.all(a<1310.),'RAW_TRAIN_TIME_BREACH')
    check(np.all(np.diff(a)>=-1e-9),'RAW_AGE_REVERSAL')
    ns=np.asarray(t,dtype='datetime64[ns]').astype('int64')
    check(np.all(np.diff(ns)>0),'RAW_DATETIME_NOT_STRICTLY_INCREASING')
    if previous_age is not None:
        check(a[0]>=previous_age-1e-9,'RAW_AGE_REVERSAL_AT_CHUNK')
        check(pd.Timestamp(t.iloc[0] if isinstance(t,pd.Series) else t[0])>previous_time,'RAW_TIME_REVERSAL_AT_CHUNK')
    return int(np.count_nonzero(np.diff(a)==0))+int(previous_age is not None and a[0]==previous_age),float(a[-1]),pd.Timestamp(t.iloc[-1] if isinstance(t,pd.Series) else t[-1])

def read_header_only(repo,relative,allowed):
    check(relative in allowed,'SCHEMA_PATH_NOT_ALLOWLISTED')
    p=(repo/relative).resolve(); check(p.is_relative_to(repo.resolve()),'SCHEMA_PATH_ESCAPE')
    payload=bytearray(); quoted=False
    with p.open('rb',buffering=0) as f:
        while True:
            b=f.read(1); check(bool(b),'SCHEMA_HEADER_TRUNCATED')
            payload.extend(b); check(len(payload)<1048576,'OVERSIZED_SCHEMA_HEADER')
            if b==b'"': quoted=not quoted
            if b==b'\n' and not quoted: break
    parsed=list(csv.reader(io.StringIO(payload.decode('utf-8-sig')),strict=True))
    check(len(parsed)==1,'SCHEMA_HEADER_RECORD_MISMATCH')
    return dict(path=relative,columns=parsed[0],bytes_read=len(payload),data_rows_read=0)

def read_f2_identity_prefix(repo, relative, byte_end, runtime_identity):
    """M1's audited byte count, not a new boundary inferred from raw ages."""
    with (repo/relative).open('rb',buffering=0) as f:
        stream=LimitedBytes(f,byte_end)
        data=stream.read()
        check(f.tell()==byte_end,'F2_READ_AHEAD')
    records=list(csv.reader(io.StringIO(data.decode('utf-8-sig')),strict=True))
    table=pd.DataFrame(records[1:],columns=records[0])
    check(len(table)==2103,'F2_PREFIX_RECORD_MISMATCH')
    cols=runtime_identity['fingerprint_columns']
    scope=table[cols].copy()
    for col in scope:
        if col not in {'first_datetime','last_datetime','source_file_names'}:
            scope[col]=pd.to_numeric(scope[col],errors='coerce')
    payload=scope.to_csv(index=False,lineterminator='\n',float_format='%.17g',na_rep='<NA>').encode('utf-8')
    check(sha_bytes(payload)==runtime_identity['allowed_train_fingerprint_sha256'],'F2_PREFIX_FINGERPRINT_MISMATCH')
    check(scope.bin_start_ophrs2_h.iloc[0]==72.5 and scope.bin_end_ophrs2_h.iloc[-1]==1310.,'F2_BOUNDARY_IDENTITY_MISMATCH')
    return scope[['bin_start_ophrs2_h','bin_end_ophrs2_h','first_datetime','last_datetime',
        'temporal_segment_id','bin_valid_for_windows','source_file_names','raw_sample_count',
        'actual_anode_inlet_rh_pct_mean','actual_cathode_inlet_rh_pct_mean']].copy()
