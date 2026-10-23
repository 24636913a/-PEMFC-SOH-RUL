"""UTF8 status observer with Windows FILE_SHARE_DELETE; never import Torch."""
from pathlib import Path
import os,json,time,ctypes,argparse
P=Path(__file__).resolve().parents[1]
def read_shared(path):
 if os.name!='nt':return path.read_text(encoding='utf-8')
 from ctypes import wintypes
 k=ctypes.WinDLL('kernel32',use_last_error=True)
 k.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];k.CreateFileW.restype=wintypes.HANDLE
 k.ReadFile.argtypes=[wintypes.HANDLE,wintypes.LPVOID,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),wintypes.LPVOID]
 k.CloseHandle.argtypes=[wintypes.HANDLE]
 handle=k.CreateFileW(str(path),0x80000000,7,None,3,0x80,None)
 if handle==wintypes.HANDLE(-1).value:raise OSError(ctypes.get_last_error())
 try:
  chunks=[]
  while True:
   buffer=ctypes.create_string_buffer(65536);n=wintypes.DWORD();assert k.ReadFile(handle,buffer,len(buffer),ctypes.byref(n),None)
   if not n.value:break
   chunks.append(buffer.raw[:n.value])
  return b''.join(chunks).decode('utf-8')
 finally:k.CloseHandle(handle)

def snapshot():
 f=P/'outputs/execution_ledger/PROGRESS.json'
 if not f.exists():return dict(status='NOT_STARTED'),False
 state=json.loads(read_shared(f));active=[int(s) for s,r in state['seeds'].items() if r['status']=='STARTED']
 elapsed=state.get('elapsed_s',0)
 if state['status']=='RUNNING':
  from datetime import datetime
  elapsed=time.time()-datetime.fromisoformat(state['start_UTC'].replace('Z','+00:00')).timestamp()
 out=dict(status=state['status'],completed=state.get('completed_seeds',0),qualified=state.get('qualified_seeds',[]),elapsed_h=round(elapsed/3600,3))
 if active:
  seed=active[0];log=P/f'outputs/training_logs/seed_{seed}/console.log'
  lines=read_shared(log).splitlines() if log.exists() else []
  out['active_seed']=seed;out['latest_epoch']=next((x for x in reversed(lines) if x.startswith(f'Seed{seed} epoch=')),'initializing')
 receipt=P/'outputs/GIT_RECEIPT.local.json'
 if receipt.exists():out['publication']=json.loads(read_shared(receipt));return out,True
 if state.get('finalization_returncode',0):out['finalization_returncode']=state['finalization_returncode'];return out,True
 return out,False
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');a=parser.parse_args()
 while True:
  try:out,done=snapshot();print(json.dumps(out,ensure_ascii=False),flush=True)
  except (OSError,ValueError) as exc:print(json.dumps(dict(observer_error=repr(exc))),flush=True);done=False
  if a.once or done:break
  time.sleep(55)
