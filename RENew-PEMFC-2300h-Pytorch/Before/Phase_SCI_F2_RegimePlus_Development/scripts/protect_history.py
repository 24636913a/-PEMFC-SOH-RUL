"""Compare all previously sealed artifacts, without opening research data values."""
from experiment import *

def main():
    repo=R.parent; full=old.FULL_EVALUATION; diag=R/'Phase_SCI_F2_Late_Extreme_Cause_Audit'
    rows=pd.read_csv(full/'outputs/audits/HISTORY_BEFORE.csv').to_dict('records')
    for base,manifest in [(full,full/'outputs/audits/ARTIFACT_MANIFEST.csv'),(R,R/'RealHyFC_F2_Main_Outputs/DELIVERY_MANIFEST.csv'),(diag,diag/'outputs/DELIVERY_MANIFEST.csv')]:
        for r in pd.read_csv(manifest).to_dict('records'): rows.append(dict(path=(base/r['path']).relative_to(repo).as_posix(),sha256=r['sha256']))
    for file in [full/'outputs/audits/ARTIFACT_MANIFEST.csv',full/'outputs/audits/FINAL_QA.json',R/'RealHyFC_F2_Main_Outputs/DELIVERY_MANIFEST.csv',diag/'outputs/DELIVERY_MANIFEST.csv',diag/'outputs/FINAL_QA.json']:
        name=file.relative_to(repo).as_posix(); blob=file.read_bytes(); expected=subprocess.check_output(['git','rev-parse',BASE+':'+name],cwd=repo,text=True).strip()
        req(hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==expected,'MANIFEST_WORKTREE_CHANGED')
        rows.append(dict(path=name,sha256=sha(file)))
    req(len(rows)==len({r['path'] for r in rows})==65762,'HISTORY_INVENTORY_COUNT')
    for i,r in enumerate(rows):
        req(sha(repo/r['path'])==r['sha256'],'HISTORY_CHANGED:'+r['path'])
        if (i+1)%15000==0: print('HISTORY_SHA_CHECKED',i+1,flush=True)
    js('outputs/HISTORICAL_PROTECTION.json',dict(status='PASS',historical_files=len(rows),all_SHA256_unchanged=True,model_weights_unchanged=True,
        historical_predictions_unchanged=True,papers_unchanged=True,comparison_base=BASE,verified_utc=now()))

if __name__=='__main__': main()
