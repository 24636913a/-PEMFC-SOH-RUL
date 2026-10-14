import hashlib,json,os,sys
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parents[1];PROJECT=P.parent;REPO=PROJECT.parent
BASE=REPO/'数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def js(rel,obj):
 q=P/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def cs(rel,d):d.to_csv(P/rel,index=False,float_format='%.17g',lineterminator='\n')
def md(rel,text):(P/rel).write_bytes((text.strip()+'\n').encode('utf-8'))
def safe_raw(path):
 q=Path(path).resolve();cfg=json.loads((P/'configs/SOURCE_PATH_WHITELIST.json').read_text(encoding='utf-8'))
 if q not in {(REPO/r['path']).resolve() for r in cfg['exact_paths']}:raise PermissionError('RAW_NOT_IN_EXACT_ALLOWLIST')
 if any(x in str(q).lower() for x in ['2300h','2000h']):raise PermissionError('LATE_TEST_SOURCE_DENIED')
 return q
def firewall(extra=()):
 cfg=json.loads((P/'configs/SOURCE_PATH_WHITELIST.json').read_text(encoding='utf-8'))
 allowed={(REPO/r['path']).resolve() for r in cfg['exact_paths']}|{Path(x).resolve() for x in extra}
 def hook(event,args):
  if event!='open' or isinstance(args[0],int):return
  q=Path(os.fsdecode(args[0])).resolve();m=args[1];fl=args[2]
  writing=(isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(fl,int) and fl&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if q.is_relative_to(REPO) and not q.is_relative_to(P) and (writing or q not in allowed):raise PermissionError('HISTORICAL_ACCESS_DENIED:'+str(q))
 sys.addaudithook(hook)
