"""Recheck sealed historical identities, then seal this compact deliverable."""
from refine import *
from PIL import Image

def main():
    req(read('outputs/QA.json')['status']=='PASS','QA_NOT_PASSED')
    full=old.FULL_EVALUATION;diag=R/'Phase_SCI_F2_Late_Extreme_Cause_Audit';plus=R/'Phase_SCI_F2_RegimePlus_Development'
    rows=pd.read_csv(full/'outputs/audits/HISTORY_BEFORE.csv').to_dict('records')
    for base,manifest in [(full,full/'outputs/audits/ARTIFACT_MANIFEST.csv'),(R,R/'RealHyFC_F2_Main_Outputs/DELIVERY_MANIFEST.csv'),(diag,diag/'outputs/DELIVERY_MANIFEST.csv'),(plus,plus/'outputs/DELIVERY_MANIFEST.csv')]:
        for row in pd.read_csv(manifest).to_dict('records'):rows.append(dict(path=(base/row['path']).relative_to(REPO).as_posix(),sha256=row['sha256']))
    for f in [full/'outputs/audits/ARTIFACT_MANIFEST.csv',full/'outputs/audits/FINAL_QA.json',R/'RealHyFC_F2_Main_Outputs/DELIVERY_MANIFEST.csv',diag/'outputs/DELIVERY_MANIFEST.csv',diag/'outputs/FINAL_QA.json',plus/'outputs/DELIVERY_MANIFEST.csv',plus/'outputs/FINAL_QA.json']:
        name=f.relative_to(REPO).as_posix();blob=f.read_bytes();expected=subprocess.check_output(['git','rev-parse',BASE+':'+name],cwd=REPO,text=True).strip()
        req(hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==expected,'HISTORY_MANIFEST_CHANGED');rows.append(dict(path=name,sha256=sha(f)))
    req(len(rows)==len({r['path'] for r in rows})==65871,'HISTORY_INVENTORY')
    for i,row in enumerate(rows):
        req(sha(REPO/row['path'])==row['sha256'],'HISTORY_CHANGED:'+row['path'])
        if (i+1)%15000==0:print('HISTORY_VERIFIED',i+1,flush=True)
    js('outputs/HISTORY_QA.json',dict(status='PASS',files=len(rows),all_historical_SHA256_unchanged=True))
    with Image.open(path('figures/VOLTAGE_COMPARISON.png')) as image:req(min(image.info['dpi'])>299,'DPI')
    changed=subprocess.check_output(['git','diff','--name-only','-z'],cwd=REPO).decode().split('\0')
    req(all((REPO/name).resolve().is_relative_to(P) for name in changed if name),'OUTSIDE_PHASE_DIFF')
    inventory=[]
    for f in sorted(P.rglob('*')):
        if f.is_file() and f.name not in ['DELIVERY_MANIFEST.csv','FINAL_QA.json','GIT_RECEIPT.local.json'] and '__pycache__' not in f.parts and not f.name.endswith(('.writing','-wal','-shm')):
            req(f.stat().st_size<100*1024**2,'GIT_FILE_TOO_LARGE');inventory.append(dict(path=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
    csv('outputs/DELIVERY_MANIFEST.csv',inventory)
    js('outputs/FINAL_QA.json',dict(status='PASS',metrics=read('outputs/RESULT.json'),scientific_QA=read('outputs/QA.json'),history=read('outputs/HISTORY_QA.json'),manifest_files=len(inventory),
        figures=1,PNG_dpi=300,PDF=True,summary='SUMMARY.csv',no_historical_results_replaced=True))

if __name__=='__main__':main()
