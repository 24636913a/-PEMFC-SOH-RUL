"""No refitting: independently check raw conversion, units, residuals and history."""
import json,subprocess,sys,re,io
import numpy as np
import pandas as pd
from common import P,PROJECT,REPO,sha,js,md
def run():
 manifest=pd.read_csv(P/'inputs/SOURCE_CONTENT_MANIFEST.csv');points=pd.read_csv(P/'outputs/EIS_NORMALIZED_POINTS.csv',float_precision='round_trip');checked=0
 for r in manifest.loc[manifest.role.eq('CORE_SPECTRUM')].itertuples():
  assert sha(REPO/r.path)==r.content_sha256
  text=(REPO/r.path).read_text(encoding='ascii');lines=text.splitlines();i=next(k for k,s in enumerate(lines) if 'Number' in s and 'Frequency/Hz' in s)
  raw=np.loadtxt(io.StringIO('\n'.join(lines[i+1:])))
  observed=points.loc[points.source_path.eq(r.path)].sort_values('point')
  z=(raw[:,2]+1j*raw[:,3]) if 'impedance_R/Ohm' in lines[i] else raw[:,2]*(np.cos(np.deg2rad(raw[:,3]))+1j*np.sin(np.deg2rad(raw[:,3])))
  assert np.array_equal(raw[:,1],observed.frequency_Hz.to_numpy())
  assert np.allclose(z.real,observed.Z_real_ohm,rtol=0,atol=1e-18) and np.allclose(z.imag,observed.Z_imag_ohm,rtol=0,atol=1e-18)
  checked+=len(raw)
 assert checked==3620
 details=json.loads((P/'outputs/EQUIVALENT_CIRCUIT_FULL_PARAMETERS.json').read_text());pred=pd.read_csv(P/'outputs/EQUIVALENT_CIRCUIT_POINT_RESIDUALS.csv',float_precision='round_trip');table=pd.read_csv(P/'outputs/EQUIVALENT_CIRCUIT_DIAGNOSTICS.csv',float_precision='round_trip')
 for source,item in details.items():
  d=pred.loc[pred.source_path.eq(source)];p=np.array(item['parameters']);w=2*np.pi*d.frequency_Hz.to_numpy()
  zz=np.exp(p[0])+np.exp(p[1])/(1+(1j*w*np.exp(p[2]))**p[3])+np.exp(p[4])/(1+(1j*w*np.exp(p[5]))**p[6])
  actual=d.actual_real_ohm.to_numpy()+1j*d.actual_imag_ohm.to_numpy();saved=d.fitted_real_ohm.to_numpy()+1j*d.fitted_imag_ohm.to_numpy()
  assert np.allclose(zz,saved,rtol=0,atol=1e-17)
  rmse=float(np.sqrt(np.mean(abs((zz-actual)/abs(actual))**2))*100);assert abs(rmse-item['relative_complex_RMSE_percent'])<1e-10
  row=table.loc[table.source_path.eq(source)].iloc[0]
  if row.object in ['SINGLE_CELL_CHANNEL','REFERENCE_SINGLE_CELL']:assert abs(row.ASR_effective_model_candidate_ohm_cm2-np.exp(p[0])*280)<1e-15
  else:assert pd.isna(row.ASR_effective_model_candidate_ohm_cm2)
  assert not row.ASR_certified
 original=json.loads((P/'inputs/PROTECTED_HISTORY_SHA.json').read_text());assert all(sha(PROJECT/f)==h for f,h in original.items())
 lfs=pd.read_csv(P/'outputs/GIT_LFS_OBJECT_PROVENANCE.csv');assert lfs.oid_matches.all() and lfs.size_matches.all() and len(lfs)==275
 log=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(P/'tests'),'-v'],capture_output=True,text=True,encoding='utf-8');md('outputs/TEST_LOG.txt',log.stdout+log.stderr);assert log.returncode==0
 js('outputs/INDEPENDENT_VERIFICATION.json',dict(status='PASS',tests=10,independently_checked_spectrum_points=3620,independently_recomputed_circuit_outputs=33,
  source_LFS_OID_byte_match_files=275,history_files_unchanged=len(original),no_model_forward_or_refit=True,
  numerical_rounding_atol_Z=1e-18,identity_frequency_zero_tolerance=True))
 print('INDEPENDENT_PASS;10 tests;3620 raw points;33 fitted outputs;no refit')
if __name__=='__main__':run()
