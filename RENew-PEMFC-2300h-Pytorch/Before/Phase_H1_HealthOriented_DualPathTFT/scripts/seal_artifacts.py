"""Seal completed phase, or verify its recorded byte identities. No fitting."""
from common import *
from datetime import datetime,timezone
EXCLUDE={'outputs/ARTIFACT_SHA256.csv','outputs/SEAL_RECEIPT.json','outputs/GIT_RECEIPT.local.json'}
def included(f):
 r=f.relative_to(P).as_posix()
 return f.is_file() and r not in EXCLUDE and not r.startswith('outputs/runtime/') and '__pycache__' not in f.parts and f.suffix not in ['.pyc','.writing']
def verify():
 table=rd(P/'outputs/ARTIFACT_SHA256.csv');current={f.relative_to(P).as_posix() for f in P.rglob('*') if included(f)}
 assert set(table.path)==current,('Artifact set changed',current.symmetric_difference(set(table.path)))
 for r in table.itertuples():assert (P/r.path).stat().st_size==r.bytes and sha(P/r.path)==r.sha256,r.path
 receipt=json.loads((P/'outputs/SEAL_RECEIPT.json').read_text())
 assert sha(P/'outputs/ARTIFACT_SHA256.csv')==receipt['artifact_manifest_sha256']
 print(json.dumps(dict(status='PASS',artifacts=len(table),bytes=int(table.bytes.sum())),ensure_ascii=False))
def main():
 if '--verify' in sys.argv:verify();return
 assert not (P/'outputs/SEAL_RECEIPT.json').exists(),'Already sealed; use --verify'
 quality=json.loads((P/'outputs/FINAL_QUALITY_GATE.json').read_text());assert quality['status']=='PASS'
 cfg=load_cfg()
 for f,h in {**cfg['data_sha256'],**cfg['training_code_sha256']}.items():assert sha(P/f)==h,f
 assert protected_pass()
 for r in rd(P/'data_contracts/REFERENCE_SOURCE_SHA256.csv').itertuples():assert sha(ROOT/r.path)==r.sha256
 code=[dict(path=f.relative_to(P).as_posix(),sha256=sha(f)) for d in ['scripts','models','tests'] for f in sorted((P/d).rglob('*.py'))]
 cs('outputs/FINAL_CODE_SHA256.csv',pd.DataFrame(code))
 artifacts=[dict(path=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(P.rglob('*')) if included(f)]
 cs('outputs/ARTIFACT_SHA256.csv',pd.DataFrame(artifacts))
 js('outputs/SEAL_RECEIPT.json',dict(status='SEALED',UTC=datetime.now(timezone.utc).isoformat(),
  source_base=BASE,pretraining_commit=json.loads((P/'outputs/PRETRAINING_COMMIT.json').read_text()),
  frozen_protocol_sha256=sha(P/'configs/FROZEN_PROTOCOL.json'),artifact_manifest_sha256=sha(P/'outputs/ARTIFACT_SHA256.csv'),
  artifact_count=len(artifacts),total_bytes=sum(r['bytes'] for r in artifacts),
  scientific_status=json.loads((P/'outputs/FINAL_DECISION.json').read_text())['status'],
  quality_gate=quality,backend_precision_failure_preserved=True,raw_Test_values_read=False,
  ignored_runtime='Local recovery states retained, not scientific outputs; all54formal scientific weights included'))
 verify()
if __name__=='__main__':main()
