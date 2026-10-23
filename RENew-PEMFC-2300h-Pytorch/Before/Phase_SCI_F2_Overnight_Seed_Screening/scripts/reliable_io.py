"""Bounded Windows atomic-rename retry. No model, label, optimizer or RNG changes."""
from screen_common import P,sha,os,json,time
def verify_IO_repair():
 f=P/'configs/WINDOWS_IO_REPAIR_AMENDMENT.json'
 assert sha(f)==f.with_suffix('.sha256').read_text(encoding='utf-8').strip()
 c=json.loads(f.read_text(encoding='utf-8'))
 for path,digest in c['code_sha256'].items():assert sha(P/path)==digest,('IO_REPAIR_CODE_IDENTITY_MISMATCH',path)
 assert c['failed_seed_not_retried']==151 and not c['scientific_configuration_changed']
 assert sha(P/'configs/frozen_seed_screening.json')==c['original_protocol_sha256']
 return c
def record_retry(record):
 f=P/'outputs/execution_ledger/IO_RENAME_RETRIES.jsonl'
 with f.open('a',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
def js(name,obj):
 f=(P/name).resolve();assert f.is_relative_to(P),'HISTORICAL_WRITE_PROHIBITED';f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.writing')
 with tmp.open('w',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
 delays=[.02,.05,.1,.2,.5,1.]
 for attempt in range(len(delays)+1):
  try:os.replace(tmp,f);return
  except PermissionError as exc:
   if attempt==len(delays):raise
   record_retry(dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),path=f.relative_to(P).as_posix(),attempt=attempt+1,wait_s=delays[attempt],exception=repr(exc),PID=os.getpid()))
   time.sleep(delays[attempt])
