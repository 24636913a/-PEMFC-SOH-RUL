"""Safe main delivery: ordinary push, then exact-commit Git Data API fallback.

No force, no Git/proxy/power configuration change, credentials stay in memory.
The fallback uploads the *same* local commit, verifies every Git object SHA,
and refuses any remote-parent mismatch. Run only after seal + local commit.
"""
from pathlib import Path
import base64, concurrent.futures, datetime, hashlib, json, os, re, subprocess, threading, time
import requests

P=Path(__file__).resolve().parents[1]; REPO=P.parent.parent
BASE='36572a6678a9e64df689c8c6ec282c7233b6ac36'
API='https://api.github.com/repos/24636913a/-PEMFC-SOH-RUL'

def git(*args,binary=False):
    q=subprocess.check_output(['git',*args],cwd=REPO)
    return q if binary else q.decode('utf-8').strip()

def receipt(value):
    p=P/'outputs/GIT_RECEIPT.local.json'
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    head=git('rev-parse','HEAD'); raw=git('cat-file','commit',head,binary=True)
    header,message=raw.split(b'\n\n',1); fields={}
    for line in header.decode().splitlines():
        key,value=line.split(' ',1); fields.setdefault(key,[]).append(value)
    assert fields['parent']==[BASE], 'UNEXPECTED_LOCAL_PARENT'
    names=git('diff-tree','--no-commit-id','--name-only','-r','-z','HEAD',binary=True).decode().rstrip('\0').split('\0')
    assert names and all((REPO/n).resolve().is_relative_to(P) for n in names),'OUTSIDE_NEW_STAGE'
    assert not git('status','--porcelain'), 'WORKTREE_NOT_CLEAN'
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    try:
        pushed=subprocess.run(['git','push','origin','main'],cwd=REPO,env=env,capture_output=True,text=True,timeout=100)
        ordinary=dict(returncode=pushed.returncode,stdout=pushed.stdout,stderr=pushed.stderr)
    except subprocess.TimeoutExpired:
        ordinary=dict(returncode=None,error='ordinary git push timed out at 100 seconds')
    receipt(dict(status='PUSH_IN_PROGRESS',sha=head,ordinary=ordinary))
    print('ORDINARY_PUSH_FINISHED',ordinary['returncode'],flush=True)
    # Even a successful push is verified independently below.
    credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',cwd=REPO,env=env,capture_output=True,text=True,timeout=20,check=True)
    secret=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line)['password']
    headers={'Authorization':'Bearer '+secret,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
    lock=threading.Lock(); last_write=[0.]
    def request(method,url,body=None):
        for i in range(5):
            if method!='GET':
                with lock:
                    time.sleep(max(0,.9-(time.monotonic()-last_write[0])))
                    last_write[0]=time.monotonic()
            q=requests.request(method,API+url,headers=headers,json=body,timeout=120)
            if q.status_code in [429,500,502,503,504] and i<4:
                time.sleep(min(10,2*(i+1))); continue
            if not q.ok: raise RuntimeError('GitHub API request failed: '+str(q.status_code)+' '+q.text[:500])
            return q.json()
        raise RuntimeError('API_RETRIES_EXHAUSTED')
    remote=request('GET','/git/ref/heads/main')['object']['sha']
    if remote==head:
        receipt(dict(status='PUSHED_VERIFIED',sha=head,ordinary=ordinary,method='ordinary git push',exact_remote_verified=True)); print('PUSHED_VERIFIED',head); return
    assert remote==BASE,'REMOTE_DIVERGED_NO_FORCE_OR_OVERWRITE'
    print('API_FALLBACK_REMOTE_PARENT_VERIFIED',remote,flush=True)
    objects={}; entries=[]
    for name in names:
        listing=git('ls-tree','HEAD','--',name).split('\t')[0].split()
        mode,kind,oid=listing; assert kind=='blob'
        blob=git('cat-file','blob',oid,binary=True)
        assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==oid
        objects[oid]=blob; entries.append(dict(path=name,mode=mode,type='blob',sha=oid))
    def upload(item):
        oid,blob=item; r=request('POST','/git/blobs',dict(content=base64.b64encode(blob).decode(),encoding='base64'))
        assert r['sha']==oid; return oid
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        uploaded=list(pool.map(upload,objects.items()))
    print('API_BLOBS_VERIFIED',len(uploaded),flush=True)
    tree=request('POST','/git/trees',dict(base_tree=git('rev-parse',BASE+'^{tree}'),tree=entries))
    assert tree['sha']==fields['tree'][0],'TREE_IDENTITY'
    def person(s):
        m=re.fullmatch(r'(.*) <([^<>]+)> (\d+) ([+-]\d{4})',s); assert m
        name,email,stamp,offset=m.groups(); sign=1 if offset[0]=='+' else -1
        zone=datetime.timezone(datetime.timedelta(minutes=sign*(int(offset[1:3])*60+int(offset[3:5]))))
        return dict(name=name,email=email,date=datetime.datetime.fromtimestamp(int(stamp),zone).isoformat())
    commit=request('POST','/git/commits',dict(message=message.decode(),tree=tree['sha'],parents=[BASE],author=person(fields['author'][0]),committer=person(fields['committer'][0])))
    assert commit['sha']==head,'EXACT_LOCAL_COMMIT_NOT_REPRODUCED'
    assert request('GET','/git/ref/heads/main')['object']['sha']==BASE,'REMOTE_CHANGED_DURING_UPLOAD'
    request('PATCH','/git/refs/heads/main',dict(sha=head,force=False))
    assert request('GET','/git/ref/heads/main')['object']['sha']==head,'REMOTE_VERIFY_FAILED'
    if git('rev-parse','refs/remotes/origin/main')==BASE: git('update-ref','refs/remotes/origin/main',head,BASE)
    receipt(dict(status='PUSHED_VERIFIED',sha=head,ordinary=ordinary,method='Git Data API exact local commit, force=false',
      blobs_verified=len(uploaded),tree_verified=True,commit_verified=True,exact_remote_verified=True))
    print('PUSHED_VERIFIED',head,'objects',len(uploaded))

if __name__=='__main__':
    try: main()
    except Exception as exc:
        p=P/'outputs/GIT_RECEIPT.local.json'
        previous=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        receipt(dict(previous,status='PUSH_FAILED_LOCAL_COMMIT_PRESERVED',error=repr(exc)))
        raise
