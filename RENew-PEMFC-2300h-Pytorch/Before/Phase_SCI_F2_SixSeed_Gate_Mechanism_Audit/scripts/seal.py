"""Seal auditable delivery bytes after tests, scientific review and visual QA."""
from common import *
def main():
    cfg();assert read(P/'outputs/audits/INDEPENDENT_VERIFICATION.json')['pass_']
    assert read(P/'outputs/audits/TESTS.json')['pass']
    assert read(P/'outputs/audits/VISUAL_QA.json')['pass']
    paths=[p for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_MANIFEST.sha256','GIT_RECEIPT.local.json']]
    rows=[dict(path=str(f.relative_to(P)).replace('\\','/'),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(paths)]
    js('outputs/DELIVERY_MANIFEST.json',dict(base_commit=BASE,files=rows,scientific_status='SELECTED_INSTANCE_MECHANISM_DIAGNOSIS_NOT_INDEPENDENT_GAIN_PROOF'))
    txt('outputs/DELIVERY_MANIFEST.sha256',sha(P/'outputs/DELIVERY_MANIFEST.json')+'\n');print('DELIVERY_SEALED',len(rows))
if __name__=='__main__':main()
