"""One-shot authority read and condition-only inventory; no HI or matching fit."""
from common import *
import yaml
def main():
 assert not (P/'outputs/PREPARE_RECEIPT.json').exists(),'Already prepared; use saved snapshot'
 for f in ['configs','data_contracts','outputs','figures','tests']: (P/f).mkdir(exist_ok=True)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='017ced9d670c22216e01d72a490b83059162666e'
 meta,ids,b=historical_development()
 cs('data_contracts/HISTORICAL_3007_METADATA_MASK.csv',meta.assign(source_row=np.arange(len(meta)),retained=np.isin(np.arange(len(meta)),ids)))
 cs('data_contracts/DEVELOPMENT.csv',b)
 old=read(BRIDGE/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv')
 checks=[]
 for c in old:
  a=b[c]; z=old[c]
  if pd.api.types.is_numeric_dtype(a):
   x=a.to_numpy(float);y=z.to_numpy(float);same=(x.view('uint64')==y.view('uint64'))|(np.isnan(x)&np.isnan(y))
   maximum=float(np.nanmax(abs(x-y)))
  else: same=((a==z)|(a.isna()&z.isna())).to_numpy(); maximum=None
  checks.append(dict(field=c,unequal=int((~same).sum()),max_abs_difference=maximum))
 assert all(r['unequal']==0 for r in checks)
 cs('outputs/READ_RULE_EXACT_COMPARISON.csv',pd.DataFrame(checks))
 cfg=yaml.safe_load((H/'configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))
 raw=cfg['task']['raw_f2_feature_order']; extra=[c for c in old if c.startswith('actual_') and c not in raw]
 cs('data_contracts/INPUT_VERSUS_AUDIT_CONDITIONS.csv',pd.DataFrame([dict(field=c,raw14_input=c in raw,qualification_only=c not in raw,unit=('A/cm2' if 'density' in c else 'C' if 'temperature' in c else '%RH' if 'rh_pct' in c else 'kPa(g)' if 'pressure' in c else 'dimensionless_or_field_named')) for c in dict.fromkeys(raw+extra)]))
 inventory=[]
 for role,d in [('Train',b.iloc[:2103]),('Validation',b.iloc[2103:2404])]:
  for c in old:
   if pd.api.types.is_numeric_dtype(d[c]) and c!='cell_voltage_mean' and c!='cell_voltage_temporal_std':
    s=d[c];inventory.append(dict(role=role,field=c,n=len(s),missing=int(s.isna().sum()),min=s.min(),q25=s.quantile(.25),median=s.median(),q75=s.quantile(.75),max=s.max()))
 cs('outputs/CONDITION_ONLY_INVENTORY.csv',pd.DataFrame(inventory))
 sources=[BRIDGE/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv',BRIDGE/'outputs/ALL_REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv',BRIDGE/'outputs/DECISION.json',BRIDGE/'00_通俗研究结论.md',
 ROOT/'Phase_F_TFT_DualPath_RegimeFusion/README.md',H/'README.md',H/'configs/H_FROZEN_PROTOCOL.yaml',H/'scripts/run_h_training.py',
 ROOT/'Phase_P1S_96h_ThreeFold_Comparison/00_实验总结.md',ROOT/'Phase_Integrated_SOH_Physics_RUL_Research/README.md',
 ROOT/'Phase_Integrated_SOH_Physics_RUL_Research/outputs/TFT_PAST_INFORMATION_AND_PARAMETER_AVAILABILITY.csv',
 ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_PREDICTIONS.csv',ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_EQUAL_FOLD_METRICS.csv',
 ROOT/'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv',
 ROOT/'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv',
 ROOT/'Phase_Polarization_Physics_to_Health_Qualification/outputs/POLARIZATION_QUALIFICATION.csv']
 sources += list((H/'outputs/validation_predictions').glob('H-F2-Seed*.csv'))
 sources += list((H/'outputs/checkpoints').glob('H-F2-Seed*_best.pt'))
 # Hash-only protect all historical core model weights; never deserialize/infer.
 for phase in ['Phase_F_TFT_DualPath_RegimeFusion','Phase_P1S_96h_ThreeFold_Comparison','Phase_Integrated_SOH_Physics_RUL_Research']:
  sources += [p for d in ['models','outputs/checkpoints'] for p in (ROOT/phase/d).glob('*') if p.is_file()]
 rows=[]
 for f in sorted(set(sources)):
  assert f.exists(),f
  rel=f.relative_to(ROOT).as_posix(); blob=subprocess.check_output(['git','rev-parse','HEAD:'+ROOT.name+'/'+rel],cwd=REPO,text=True).strip()
  rows.append(dict(path=rel,sha256=sha(f),git_blob=blob,base_commit='017ced9d670c22216e01d72a490b83059162666e'))
 cs('data_contracts/SOURCE_SHA256.csv',pd.DataFrame(rows))
 with N1.open('rb') as stream: prefix=b''.join(stream.readline() for _ in range(len(ids)+1))
 js('outputs/PREPARE_RECEIPT.json',dict(metadata_prefix_rows=len(meta),retained=len(ids),source_first=int(ids[0]),source_last=int(ids[-1]),
  common_fields=len(checks),exact_common_cells=len(old)*len(checks),max_numeric_difference=0,excluded_numeric_features_parsed=False,
  source_prefix_sha256=hashlib.sha256(prefix).hexdigest(),full_mixed_file_hashed=False,training=False,GPU=False))
 print('PREPARED',len(b),'strict common cells',len(old)*len(checks))
if __name__=='__main__': main()
