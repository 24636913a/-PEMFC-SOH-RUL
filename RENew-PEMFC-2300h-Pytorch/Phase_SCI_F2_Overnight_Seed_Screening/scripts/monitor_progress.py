"""Read-only observer: UTF-8 and FILE_SHARE_DELETE, no Torch/GPU or training."""
from pathlib import Path
import json,time,os,re
from execution_paths import progress_path,P
def read_shared(f):
 if os.name!='nt':return Path(f).read_text(encoding='utf-8')
 import ctypes
 from ctypes import wintypes as w
 k=ctypes.WinDLL('kernel32',use_last_error=True)
 k.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,w.LPVOID,w.DWORD,w.DWORD,w.HANDLE];k.CreateFileW.restype=w.HANDLE
 k.GetFileSizeEx.argtypes=[w.HANDLE,ctypes.POINTER(ctypes.c_longlong)];k.GetFileSizeEx.restype=w.BOOL
 k.ReadFile.argtypes=[w.HANDLE,w.LPVOID,w.DWORD,ctypes.POINTER(w.DWORD),w.LPVOID];k.ReadFile.restype=w.BOOL
 k.CloseHandle.argtypes=[w.HANDLE];k.CloseHandle.restype=w.BOOL
 handle=k.CreateFileW(str(Path(f).resolve()),0x80000000,7,None,3,0x80,None)
 if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
 try:
  size=ctypes.c_longlong();assert k.GetFileSizeEx(handle,ctypes.byref(size));buf=ctypes.create_string_buffer(size.value);count=w.DWORD()
  assert k.ReadFile(handle,buf,size.value,ctypes.byref(count),None)
  return buf.raw[:count.value].decode('utf-8')
 finally:k.CloseHandle(handle)
def once():
 f=progress_path();s=json.loads(read_shared(f));out=dict(controller=s['status'],completed=sum(v['status']=='COMPLETED' for v in s['seeds'].values()),progress=f.relative_to(P).as_posix())
 for seed,r in s['seeds'].items():
  if r['status']=='STARTED':
   console=P/f'outputs/training_logs/seed_{seed}/console.log';out['seed']=int(seed)
   if console.exists():
    lines=[line for line in read_shared(console).splitlines() if line.startswith('Seed')]
    if lines:out['last_epoch_message']=lines[-1]
 receipt=P/'outputs/GIT_RECEIPT.local.json'
 if receipt.exists():
  r=json.loads(read_shared(receipt))
  if r.get('execution_progress')==f.relative_to(P).as_posix():out['publication']=r
 return out,s
if __name__=='__main__':
 import sys
 while True:
  try:
   out,state=once();print(json.dumps(out,ensure_ascii=True),flush=True)
   if '--once' in sys.argv:break
   if out.get('publication',{}).get('status') in ['PUSH_VERIFIED','LOCAL_COMMIT_PUSH_FAILED','PUBLISH_FAILED_PRESERVED']:break
   if state.get('temporary_sleep_prevention_restored') and state.get('finalization_returncode',0)!=0:break
  except Exception as exc:print(type(exc).__name__+': '+str(exc),flush=True)
  time.sleep(55)
