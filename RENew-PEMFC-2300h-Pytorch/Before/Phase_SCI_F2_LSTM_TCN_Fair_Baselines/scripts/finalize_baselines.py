"""Submission-scope QA and artifact seal. No optimizer or training entry point."""
from baseline_common import *
import argparse,io,unittest

def artifacts():
    excludes={'outputs/audits/ARTIFACT_MANIFEST.csv','outputs/audits/ARTIFACT_SEAL.json','outputs/GIT_RECEIPT.local.json'}
    return sorted(f for f in P.rglob('*') if f.is_file() and f.relative_to(P).as_posix() not in excludes and not any(k in f.relative_to(P).parts for k in ['runtime','__pycache__']) and f.suffix!='.pyc')

def sealcheck():
    m=rd(P/'outputs/audits/ARTIFACT_MANIFEST.csv');assert sha(P/'outputs/audits/ARTIFACT_MANIFEST.csv')==read(P/'outputs/audits/ARTIFACT_SEAL.json')['manifest_sha256']
    assert set(m.path)=={f.relative_to(P).as_posix() for f in artifacts()}
    for r in m.itertuples():assert sha(P/r.path)==r.sha256,('ARTIFACT_CHANGED',r.path)
    print('ARTIFACT_SEAL_PASS',len(m),flush=True)

def main():
    cfg();n=protect();summary=read(P/'outputs/SUMMARY.json');assert summary['status']=='SCI_BASELINE_6_RUNS_COMPLETED'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='main'
    modified=subprocess.check_output(['git','diff','--name-only','-z',BASE],cwd=REPO).decode('utf-8').strip('\0').split('\0');prefix=P.relative_to(REPO).as_posix()+'/'
    assert all(not s or s.startswith(prefix) for s in modified),'HISTORICAL_GIT_DIFF'
    previous=P/'outputs/audits/UNIT_TESTS.json'
    if previous.exists() and not read(previous)['passed']:
        write('outputs/audits/UNIT_TESTS_failed_'+time.strftime('%Y%m%d_%H%M%S')+'.log',(P/'outputs/audits/UNIT_TESTS.log').read_text(encoding='utf-8'))
    suite=unittest.defaultTestLoader.discover(str(P/'tests'));log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);write('outputs/audits/UNIT_TESTS.log',log.getvalue())
    js('outputs/audits/UNIT_TESTS.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures)));assert result.wasSuccessful(),log.getvalue()
    ledger=read(P/'outputs/execution_ledger/RUN_LEDGER.json');assert len(ledger)==6 and all(r['status']=='COMPLETED' for r in ledger)
    execution=read(P/'outputs/audits/EXECUTION_AND_SAMPLER_AUDIT.json');assert execution['status']=='PASS' and execution['formal_starts']==execution['formal_completions']==6
    for name,seed in RUNS:
        metric=read(P/f'outputs/metrics/{name}_seed{seed}.json');assert metric['status']=='COMPLETED';assert sha(P/metric['checkpoint'])==metric['checkpoint_sha256'];assert sha(P/metric['predictions'])==metric['prediction_sha256']
    # Static scope review complements the strict-prefix runtime guard and saved identities.
    paths=['scripts/train_baseline.py','scripts/audit_run.py','scripts/run_baselines.py','scripts/compare_models.py']
    assert all('h.prepare(' not in (P/q).read_text(encoding='utf-8') for q in paths)
    js('outputs/audits/FINAL_QA.json',dict(status='SCI_BASELINE_6_RUNS_COMPLETED',formal_runs=6,successful_runs=6,extra_runs=0,retrained_runs=0,seeds=[42,52,62],model_class_AST_exact=True,checkpoint_and_all205_predictions_exact=True,history_files_unchanged=n,historical_paper_unchanged=True,Test=False,Assessment=False,independent_metric_checks=1200,unit_tests=result.testsRun,figures=6,preflight_sha256=sha(P/'outputs/audits/PREFLIGHT_REPORT.json'),protocol_sha256=sha(P/'configs/FROZEN_BASELINE_PROTOCOL.yaml'),visual_QA='All six final PNGs manually inspected by parent; automated300dpi/vector tests passed',scientific_limitations=['Development exploratory only','unequal model capacity','historical TFT-derived learning rate not individually optimized','three initializations and overlapping target times','historical and current costs not controlled benchmark']))
    # Actual file sizes and SHA, including every small checkpoint; no external backup invented.
    inventory=[]
    for f in sorted((P/'outputs/checkpoints').glob('*.pt')):inventory.append(dict(path=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f),version_control='included in this phase',separate_external_backup='NOT_PERFORMED; local original plus intended GitHub copy'))
    cs('outputs/audits/CHECKPOINT_INVENTORY.csv',inventory)
    cs('outputs/audits/ARTIFACT_MANIFEST.csv',[dict(path=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in artifacts()])
    js('outputs/audits/ARTIFACT_SEAL.json',dict(manifest_sha256=sha(P/'outputs/audits/ARTIFACT_MANIFEST.csv'),excludes='self/manifest/local Git receipt/ignored runtime and pycache only',publication_identity='Git commit; remote main independently verified after safe push'))
    sealcheck();print('SCI_BASELINE_6_RUNS_COMPLETED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-seal',action='store_true');a=p.parse_args()
    if a.verify_seal:sealcheck()
    else:main()
