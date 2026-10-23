"""Seal only this phase after visual QA. Never modify or export old results."""
from common import *
import sys,platform
def main():
 assert json.loads((P/'outputs/FINAL_QUALITY_GATE.json').read_text())['status']=='PASS'
 for filename in ['SOURCE_SHA256.csv','ADDITIONAL_METADATA_SOURCE_SHA256.csv']:
  d=rd(P/'data_contracts'/filename);assert all(sha(ROOT/r.path)==r.sha256 for r in d.itertuples())
 js('outputs/EXECUTION_NOTES.json',dict(
  final_recommendation='C_RETAIN_DYNAMIC_TFT_AND_RIDGE96_AUXILIARY',
  preserved_failures=['outputs/FIRST_EXTRACTION_FAILURE.json','outputs/FIRST_FAILED_EXTRACTION_PROGRESS.csv','outputs/failures/WINDOW_BOUNDARY_FAILURE.json','outputs/failures/INCLUSIVE75_BOUNDARY_RAW_FIXED_STATE_DWELL_MEASUREMENTS.csv'],
  additional_verifier_repair='raw_cycle_ids may deserialize as integer when only one ID; explicit string conversion fixed the initial verifier AttributeError, no scientific results changed',
  raw_iterator='Independent raw check uses bounded zip(range(certified_count),csvReader); no conversion of numerical feature values outside prefix',
  warnings='Constant-current Spearman undefined yields NA and NumPy RuntimeWarning in condition audit; no zero substitute',
  reference_initial75boundary_failure_does_not_change_frozen_protocol=True,
  raw_retries='Only technical parsing/unit/boundary repairs. No threshold/model/state selection tuning. Failed output retained separately',
  figure_QA='Four PNG sets inspected: units, realN, Development scope, no invented labels; route has explicit auxiliary/unknown interface',
  environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),
  original_models_and_results_unchanged=True,trained=False,GPU=False,Test_numeric_values=False))
 code=[dict(path=f.relative_to(P).as_posix(),sha256=sha(f)) for folder in ['scripts','tests'] for f in sorted((P/folder).glob('*.py'))]
 cs('outputs/FINAL_CODE_SHA256.csv',pd.DataFrame(code))
 rows=[]
 for f in sorted(P.rglob('*')):
  if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.pyc':continue
  rel=f.relative_to(P).as_posix()
  if rel in ['outputs/ARTIFACT_SHA256.csv','outputs/SEAL_RECEIPT.json','outputs/GIT_RECEIPT.local.json']:continue
  rows.append(dict(path=rel,size_bytes=f.stat().st_size,sha256=sha(f),role='FAILED_EVIDENCE_NOT_USED' if '/failures/' in '/'+rel or 'FIRST_' in f.name else 'ACTIVE_AUDIT_ARTIFACT'))
 cs('outputs/ARTIFACT_SHA256.csv',pd.DataFrame(rows))
 js('outputs/SEAL_RECEIPT.json',dict(status='SEALED',files=len(rows),manifest_sha256=sha(P/'outputs/ARTIFACT_SHA256.csv'),
  protocol_sha256=sha(P/'configs/FROZEN_AUDIT.json'),historical_source_files_unchanged=35,
  source_prefix_sha256_ledger='outputs/RAW_SOURCE_PREFIX_PROGRESS.csv',no_paper_overwrite=True))
 print('SEALED',len(rows),'files')
if __name__=='__main__':main()
