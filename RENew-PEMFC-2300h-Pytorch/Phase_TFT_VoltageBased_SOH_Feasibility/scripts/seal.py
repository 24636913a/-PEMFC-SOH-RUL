"""Seal verified artifacts after figure inspection. Excludes self-referential manifests."""
from common import *
import sys,platform
def main():
 gate=json.loads((P/'outputs/FINAL_QUALITY_GATE.json').read_text());assert gate['status']=='PASS'
 ledger=read(P/'data_contracts/SOURCE_SHA256.csv');assert all(sha(ROOT/r.path)==r.sha256 for r in ledger.itertuples())
 js('outputs/EXECUTION_NOTES.json',dict(
  numerical_analysis_once=True,prepare_once=True,freeze_once=True,
  observed_warnings=['pandas PerformanceWarning for fragmented 214-column snapshot on analyze.py insert; no numerical failure; retained in this note'],
  failed_scientific_gates=['Authenticated BoL','Physically certified equivalent dynamic measurements','Strict online model/plan availability','Reference polarization SOH comparison','Common-origin equivalent cycle/dynamic target'],
  scientific_gate_failure_is_not_missing_data_filled_with_zero=True,
  figure_review='Four PNGs visually inspected; labels, units, n, retrospective stage and non-SOH status visible',
  env=dict(Python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__),
  original_phases_unchanged=True))
 sources=[]
 for path in sorted((P/'scripts').glob('*.py'))+sorted((P/'tests').glob('*.py')):
  sources.append(dict(path=path.relative_to(P).as_posix(),sha256=sha(path)))
 cs('outputs/FINAL_CODE_SHA256.csv',pd.DataFrame(sources))
 rows=[]
 for q in sorted(P.rglob('*')):
  if not q.is_file() or '__pycache__' in q.parts or q.suffix=='.pyc':continue
  rel=q.relative_to(P).as_posix()
  if rel in ['outputs/ARTIFACT_SHA256.csv','outputs/SEAL_RECEIPT.json','outputs/GIT_RECEIPT.local.json']:continue
  rows.append(dict(path=rel,size_bytes=q.stat().st_size,sha256=sha(q)))
 cs('outputs/ARTIFACT_SHA256.csv',pd.DataFrame(rows))
 js('outputs/SEAL_RECEIPT.json',dict(status='SEALED',files=len(rows),manifest_sha256=sha(P/'outputs/ARTIFACT_SHA256.csv'),
  excluded_self_referential_files=['outputs/ARTIFACT_SHA256.csv','outputs/SEAL_RECEIPT.json'],
  protocol_sha256=sha(P/'configs/FROZEN_PROTOCOL.json'),original_source_files_unchanged=len(ledger),
  no_paper_directory_overwrite=True))
 print('SEALED',len(rows),'artifacts')
if __name__=='__main__':main()
