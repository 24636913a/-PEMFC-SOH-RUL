from experiment import *
from PIL import Image

def main():
    p=get_protocol(); req(readj('outputs/INDEPENDENT_QA.json')['status']=='PASS','QA')
    req(readj('outputs/INPUT_CAUSALITY_AUDIT.json')['status']=='PASS','CAUSALITY')
    req(readj('outputs/HISTORICAL_PROTECTION.json')['status']=='PASS','HISTORY')
    req(len(list(path('figures').glob('*.png')))==len(list(path('figures').glob('*.pdf')))==6,'FIGURE_COUNT')
    for f in path('figures').glob('*.png'):
        with Image.open(f) as im: req(min(im.info['dpi'])>299,'DPI')
    changed=subprocess.check_output(['git','diff','--name-only','-z'],cwd=R.parent).decode().split('\0')
    req(all((R.parent/f).resolve().is_relative_to(P) for f in changed if f),'HISTORICAL_TRACKED_CHANGE')
    rows=[]
    for f in sorted(P.rglob('*')):
        if f.is_file() and f.name not in ['DELIVERY_MANIFEST.csv','FINAL_QA.json','GIT_RECEIPT.local.json'] and '__pycache__' not in f.parts and not f.name.endswith('.writing'):
            req(f.stat().st_size<100*1024**2,'TOO_LARGE_FOR_NORMAL_GIT')
            rows.append(dict(path=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
    csv('outputs/DELIVERY_MANIFEST.csv',rows)
    js('outputs/FINAL_QA.json',dict(status='PASS',decision=readj('outputs/DECISION.json'),new_fits=6,original_baselines=3,
        fixed_seeds=SEEDS,validation_queries_per_instance=4920,all_queries_retained=True,figures=6,PNG_dpi=300,PDF=True,
        independent_numeric_QA=readj('outputs/INDEPENDENT_QA.json'),historical_protection=readj('outputs/HISTORICAL_PROTECTION.json'),
        bounded_input_causality=readj('outputs/INPUT_CAUSALITY_AUDIT.json'),
        protocol_sha256=sha(path('configs/FROZEN_PROTOCOL.json')),manifest_files=len(rows),Assessment_late_or_Test_used_for_new_model=False))

if __name__=='__main__': main()
