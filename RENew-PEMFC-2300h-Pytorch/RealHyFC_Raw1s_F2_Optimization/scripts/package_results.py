"""Package completed, independently verified evidence; exclude local working caches."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import shutil
import zipfile
from pathlib import Path
from core import STUDY,OUT,load_cfg,assert_integrity,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',required=True);args=p.parse_args()
    cfg=load_cfg();assert_integrity(cfg,raw=True)
    summary=json.loads((OUT/'final_execution_summary.json').read_text(encoding='utf-8'))
    assert summary['trained_instances'] in [5,6]
    final_review=json.loads((OUT/'review_checkpoint3.json').read_text(encoding='utf-8'))
    assert final_review['decision']=='PASS'
    numeric=json.loads((OUT/'final_acceptance.json').read_text(encoding='utf-8'))
    assert numeric['pass'] and numeric['decision']=='PASS'
    assert numeric['trained_instance_count']==summary['trained_instances']
    diagnostics=json.loads((OUT/'independent_diagnostic_acceptance.json').read_text(encoding='utf-8'))
    assert diagnostics['pass']
    reports=[next(STUDY.glob(f'{i:02d}_*.md')) for i in range(6)]
    for f in reports:assert f.stat().st_size>100
    figures=list((STUDY/'figures').glob('*.png'))
    assert len(figures)==10
    assert {f.name[:2] for f in figures}=={f'{i:02d}' for i in range(1,11)}
    from PIL import Image
    for f in figures:
        assert f.with_suffix('.pdf').exists() and f.with_name(f.stem+'_plot_data.csv').exists()
        with Image.open(f) as im:assert min(im.info['dpi'])>=299
    save('outputs/environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ['torch','pytorch-forecasting','lightning','numpy','pandas','matplotlib']},
        raw_authority='17 original Git LFS Loadcycle CSV files',legacy_cache_access='read-only',
        frozen_protocol_sha256=sha(OUT/'frozen_protocol.json'),single_seed=202,single_GPU_serial=True,
        git_base_commit=cfg['base_commit'],training_implementation_sha256=cfg['implementation_sha256']))
    files=[f for f in STUDY.rglob('*') if f.is_file() and f!=OUT/'artifact_manifest.json'
        and not set(f.relative_to(STUDY).parts).intersection({'cache','work','__pycache__'})]
    manifest=[dict(path=f.relative_to(STUDY).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(files)]
    save('outputs/artifact_manifest.json',dict(excluded=['cache/','work/','__pycache__/'],files=manifest,
        note='Original17 CSVs remain in repository Git LFS, not duplicated into this archive. Existing previous-study memmaps are read-only runtime dependencies.'))
    destination=Path(args.destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    for f in STUDY.glob('*.md'):shutil.copy2(f,destination/f.name)
    for name in ['figures','outputs']:shutil.copytree(STUDY/name,destination/name,dirs_exist_ok=True)
    archive=destination/'RealHyFC_Raw1s_F2_Optimization_Seed202.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in STUDY.rglob('*'):
            if f.is_file() and not set(f.relative_to(STUDY).parts).intersection({'cache','work','__pycache__'}):
                z.write(f,'RealHyFC_Raw1s_F2_Optimization/'+f.relative_to(STUDY).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(destination=str(destination),zip=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
