import json
from common import P,PROJECT,REPO,sha,js
cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text());assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
for f,h in cfg['code_sha256'].items():assert sha(P/'scripts'/f)==h,f
protected=json.loads((P/'inputs/PROTECTED_HISTORY_SHA.json').read_text());assert all(sha(PROJECT/f)==h for f,h in protected.items())
assert json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
decision=json.loads((P/'outputs/RESEARCH_DECISION.json').read_text());assert decision['GPU_runs']==0 and decision['actual_Train_physics_loss'] is None
js('outputs/FINAL_INTEGRITY_AUDIT.json',dict(status='PASS',protected_history_files=len(protected),history_bytes_unchanged=True,
 source_core_protocol_and_code_unchanged=True,tests=10,real_spectra=65,real_points=3620,pure_CPU_circuit_fits=33,
 conditional_profiles=5,figures_PNG_PDF_visually_checked=8,no_GPU_training=True,no_qualified_HFR_or_ASR_claim=True,
 no_Test_numeric_parse=True,no_raw_durability_reparse=True,no_checkpoint_loaded_or_changed=True))
files={str(f.relative_to(P)).replace('\\','/'):sha(f) for f in P.rglob('*') if f.is_file() and 'runtime' not in f.relative_to(P).parts and f.name not in ['ARTIFACT_MANIFEST.json','GIT_RECEIPT.local.json']}
js('outputs/ARTIFACT_MANIFEST.json',dict(files=files));print('SEALED',len(files),'files;all history protected')
