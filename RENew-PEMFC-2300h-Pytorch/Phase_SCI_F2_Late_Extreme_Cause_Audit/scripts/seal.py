from audit_common import *
from PIL import Image
def main():
    check_protocol();assert read(P/'outputs/INDEPENDENT_NUMERIC_VERIFICATION.json')['status']=='PASS'
    assert read(P/'outputs/HISTORICAL_PROTECTION.json')['status']=='PASS'
    assert len(list((P/'figures').glob('*.png')))==len(list((P/'figures').glob('*.pdf')))==7
    for f in (P/'figures').glob('*.png'):
        with Image.open(f) as im:assert min(im.info['dpi'])>299
    tracked_changes=subprocess.check_output(['git','diff','--name-only'],cwd=REPO).decode().splitlines()
    assert all((REPO/f).is_relative_to(P) for f in tracked_changes),'HISTORICAL_CHANGE_AFTER_AUDIT'
    files=[]
    for f in sorted(P.rglob('*')):
        if f.is_file() and f.name not in ['DELIVERY_MANIFEST.csv','FINAL_QA.json','GIT_RECEIPT.local.json'] and '__pycache__' not in f.parts and not f.name.endswith('.writing'):
            assert f.stat().st_size<100*1024**2
            files.append(dict(path=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
    cs('outputs/DELIVERY_MANIFEST.csv',files)
    js('outputs/FINAL_QA.json',dict(status='F2_LATE_EXTREME_CAUSE_AUDIT_PASSED_EXPLORATORY',physical_events=2,original_models=5,seeds=SEEDS,original_late_queries_per_model_seed=13152,pointwise_error_rows=197280,figures=7,PNG_dpi=300,PDF=True,independent_numeric_verification=read(P/'outputs/INDEPENDENT_NUMERIC_VERIFICATION.json'),historical_protection=read(P/'outputs/HISTORICAL_PROTECTION.json'),formal_metrics_or_rankings_modified=False,training_or_tuning=False,gate_interventions=False,causal_gate_attribution='UNDETERMINED',manifest_files=len(files)))
if __name__=='__main__':main()
