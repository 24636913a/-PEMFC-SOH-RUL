"""Offline byte-integrity audit of delivered artifacts; no Torch/GPU/data read."""
from pathlib import Path
import csv,hashlib,json,sys
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 receipt=json.loads((P/'outputs/SEAL_RECEIPT.json').read_text(encoding='utf-8'));manifest=P/'outputs/ARTIFACT_SHA256.csv'
 assert sha(manifest)==receipt['manifest_sha256'],'MANIFEST_IDENTITY_MISMATCH'
 with manifest.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
 for row in rows:
  f=(P/row['path']).resolve();assert f.is_relative_to(P) and f.is_file(),row['path']
  assert f.stat().st_size==int(row['bytes']) and sha(f)==row['sha256'],row['path']
 assert len(rows)==receipt['artifacts']
 local=0
 if '--verify-local-journals' in sys.argv:
  with (P/'outputs/LOCAL_ARTIFACT_INVENTORY.csv').open(encoding='utf-8',newline='') as stream:
   for row in csv.DictReader(stream):
    if row['Git_delivery_planned']=='False':
     f=P/row['path'];assert f.is_file() and sha(f)==row['sha256'],row['path'];local+=1
 print(json.dumps(dict(status='PASS',delivered_artifacts=len(rows),local_journals_checked=local,optimizer_steps=0,GPU_used=False),ensure_ascii=False))
if __name__=='__main__':main()
