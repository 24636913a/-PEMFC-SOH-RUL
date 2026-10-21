"""Publish a self-contained result bundle, excluding reproducible local memmaps."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import zipfile
from pathlib import Path
import torch
from audit_raw import STUDY,OUT,sha,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',required=True);args=p.parse_args()
    destination=Path(args.destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    summary=json.loads((OUT/'final_execution_summary.json').read_text(encoding='utf-8'))
    assert summary['trained_instances']==4
    for h in [30,120]: assert json.loads((OUT/f'H{h}_acceptance.json').read_text())['pass']
    packages=['torch','pytorch-forecasting','lightning','numpy','pandas','matplotlib']
    save('outputs/environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in packages},CUDA=torch.version.cuda,
        cuDNN=torch.backends.cudnn.version(),GPU=torch.cuda.get_device_name(0),
        GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        training_settings_authority='scripts/train_pilot.py configure(); frozen_protocol.json',
        serial_single_GPU=True))
    files=[f for f in STUDY.rglob('*') if f.is_file() and f!=OUT/'artifact_manifest.json' and not set(f.relative_to(STUDY).parts).intersection({'cache','work','__pycache__'})]
    manifest=[dict(path=f.relative_to(STUDY).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(files)]
    save('outputs/artifact_manifest.json',dict(excluded=['cache/','work/','__pycache__/'],files=manifest))
    for f in STUDY.glob('*.md'):shutil.copy2(f,destination/f.name)
    for name in ['figures','outputs']:
        shutil.copytree(STUDY/name,destination/name,dirs_exist_ok=True)
    archive=destination/'RealHyFC_Raw1s_Dynamic_Study_Seed202.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in STUDY.rglob('*'):
            if f.is_file() and not set(f.relative_to(STUDY).parts).intersection({'cache','work','__pycache__'}):
                z.write(f,'RealHyFC_Raw1s_Dynamic_Study/'+f.relative_to(STUDY).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(destination=str(destination),zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=sha(archive)),ensure_ascii=False))

if __name__=='__main__':main()
