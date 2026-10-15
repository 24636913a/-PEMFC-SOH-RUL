"""All new writes stay in this audit. No model or fitting dependencies."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];ROOT=P.parent;REPO=ROOT.parent
RCR=ROOT/'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit'
PAA=ROOT/'Phase_PAA_P1_StateToken_PreAnchor_Alignment'
RHA=ROOT/'Phase_RHA_Raw_Operation_Point_Health_Alignment'
PREV=ROOT/'Phase_TFT_VoltageBased_SOH_Feasibility'
RAW=REPO/'数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据'
BOUNDARY=1819.997738
ACTIVE=[f'U{i} [V]' for i in [1,2,3,4,5,7,8,9,10,11]]
CHANNELS={'j':'j.S.act [A/cm2]','I_actual_A':'I.S.act [A]','V':'ACTIVE10_MEAN',
 'T_anode_C':'T.Si.A [°C]','T_cathode_C':'T.Si.C [°C]','T_coolant_C':'T.Si.CL [°C]',
 'RH_anode_pct':'RH.Si.A  [%]','RH_cathode_pct':'RH.Si.C [%]',
 'P_anode_kPag':'p.Si.A [kPag]','P_cathode_kPag':'p.Si.C [kPag]',
 'flow_H2_nl_min':'F.Si.H.A [nl/min]','stoic_anode':'Stoic.S.A [-]','stoic_cathode':'Stoic.S.C [-]'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return pd.read_csv(p,float_precision='round_trip')
def cs(name,d):
 f=P/name;f.parent.mkdir(parents=True,exist_ok=True);d.to_csv(f,index=False,float_format='%.17g',lineterminator='\n')
def js(name,d):
 f=P/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def md(name,s):
 f=P/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((s.strip()+'\n').encode())
def number(s):return pd.to_numeric(s.astype(str).str.replace(',','',regex=False),errors='coerce')
def authority_development():
 f=ROOT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
 meta=pd.read_csv(f,nrows=3007,usecols=['bin_end_ophrs2_h','split','first_datetime','last_datetime'])
 ids=np.flatnonzero(meta.bin_end_ophrs2_h.to_numpy()<BOUNDARY)
 assert len(meta)==3007 and np.array_equal(ids,np.arange(3005))
 assert not meta.iloc[ids]['split'].str.contains('test',case=False).any()
 b=pd.read_csv(f,nrows=len(ids));assert len(b)==len(ids)
 return meta,ids,b
