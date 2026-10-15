"""Seal this additive documentation bundle only after independent and visual verification."""
import csv,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def seal():
 v=json.loads((P/'INDEPENDENT_VERIFICATION.json').read_text(encoding='utf-8'));assert v['status']=='PASS'
 visual=json.loads((P/'VISUAL_QA.json').read_text(encoding='utf-8'));assert visual['status']=='PASS';assert visual['PNG_SHA256']==sha(P/'integrated_route.png');assert visual['plot_code_SHA256']==sha(P/'plot_route.py')
 files=[q for q in sorted(P.glob('*')) if q.is_file() and q.name not in ['CLOSURE_MANIFEST.json','GIT_RECEIPT.local.json']]
 manifest={'baseline_commit':'2afcfced13a40b280e93d01d63e993be6413af58','scientific_identity':'SINGLE_REALHYFC_PARTIALLY_VALIDATED_FRAMEWORK_NO_NEW_TRAINING','source_and_existing_figure_count':v['source_files_checked'],'independent_verification':'PASS','visual_inspection':'PASS','files':[{'path':q.name,'bytes':q.stat().st_size,'SHA256':sha(q)} for q in files]}
 q=P/'CLOSURE_MANIFEST.json';data=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
 if q.exists():assert q.read_bytes()==data,'Sealed bundle changed; version explicitly, do not silently replace'
 else:q.write_bytes(data)
 print('SEALED',len(files),'files',sum(r['bytes'] for r in manifest['files']),'bytes')
if __name__=='__main__':seal()
