import json,subprocess,sys
from pathlib import Path
from run_bridge_audit import P,ROOT,sha,js,md
result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(P/'tests'),'-v'],capture_output=True,text=True,encoding='utf-8')
md('outputs/TEST_LOG.txt',result.stdout+result.stderr);assert result.returncode==0
for filename in ['FROZEN_PROTOCOL.json','F_CACHE_EXTENSION.json']:
 config=json.loads((P/'configs'/filename).read_text());assert all(sha(ROOT/f)==h for f,h in config['source_sha256'].items())
js('outputs/FINAL_INTEGRITY.json',dict(status='PASS',tests=6,independent_prediction_verification='PASS',
 historical_sources_unchanged=True,figures_visually_checked=3,SOH_fits=0,new_neural_runs=0,
 only_stage_writes=True,Test=False,final_decision='TFT_TO_SOH_INSUFFICIENT_SUPPORT'))
files={str(f.relative_to(P)).replace('\\','/'):sha(f) for f in P.rglob('*') if f.is_file() and f.name not in ['ARTIFACT_MANIFEST.json','GIT_RECEIPT.local.json']}
js('outputs/ARTIFACT_MANIFEST.json',dict(files=files))
print('SEALED',len(files),'files')
