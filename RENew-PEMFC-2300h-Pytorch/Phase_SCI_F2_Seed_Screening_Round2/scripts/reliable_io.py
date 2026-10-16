"""Bounded IO recovery only; no retry of a scientific training run."""
from screen_common import P,os,json,time

def atomic_replace(source,destination):
 destination=__import__('pathlib').Path(destination).resolve()
 assert destination.is_relative_to(P),'HISTORICAL_WRITE_PROHIBITED'
 delays=[.02,.05,.1,.2,.5,1.]
 for attempt in range(len(delays)+1):
  try:os.replace(source,destination);return
  except PermissionError as exc:
   if attempt==len(delays):raise
   f=P/'outputs/execution_ledger/IO_RENAME_RETRIES.jsonl'
   f.parent.mkdir(parents=True,exist_ok=True)
   with f.open('a',encoding='utf-8',newline='\n') as stream:
    stream.write(json.dumps(dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),path=destination.relative_to(P).as_posix(),attempt=attempt+1,wait_s=delays[attempt],exception=repr(exc),PID=os.getpid()))+'\n');stream.flush();os.fsync(stream.fileno())
   time.sleep(delays[attempt])

def js(name,obj):
 f=(P/name).resolve();assert f.is_relative_to(P),'HISTORICAL_WRITE_PROHIBITED'
 f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.writing')
 with tmp.open('w',encoding='utf-8',newline='\n') as s:
  s.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');s.flush();os.fsync(s.fileno())
 atomic_replace(tmp,f)
