"""Checkpoint byte identity vs starting Git; never deserialize old weights."""
from screen_common import *
def main():
 dest=P/'outputs/audits/HISTORICAL_CHECKPOINT_PROTECTION.csv'
 if dest.exists():
  table=rd(dest)
  for row in table.itertuples():assert sha(REPO/row.path)==row.sha256,('HISTORICAL_WEIGHT_CHANGED',row.path)
  print(f'HISTORICAL_WEIGHTS_UNCHANGED {len(table)}');return
 raw=subprocess.check_output(['git','ls-tree','-r','-z',BASE],cwd=REPO);rows=[]
 for block in raw.split(b'\0'):
  if not block:continue
  header,path=block.split(b'\t',1);name=path.decode('utf-8')
  if Path(name).suffix.lower() not in ['.pt','.pth','.ckpt']:continue
  f=REPO/name;bits=f.read_bytes();blob=hashlib.sha1(b'blob '+str(len(bits)).encode()+b'\0'+bits).hexdigest();expected=header.decode().split()[-1]
  assert blob==expected,('OLD_CHECKPOINT_GIT_MISMATCH',name)
  rows.append(dict(path=name,bytes=len(bits),git_blob=blob,sha256=hashlib.sha256(bits).hexdigest()))
 assert rows;cs(dest.relative_to(P),pd.DataFrame(rows));print(f'HISTORICAL_WEIGHT_BASELINE_PASS {len(rows)}')
if __name__=='__main__':main()
