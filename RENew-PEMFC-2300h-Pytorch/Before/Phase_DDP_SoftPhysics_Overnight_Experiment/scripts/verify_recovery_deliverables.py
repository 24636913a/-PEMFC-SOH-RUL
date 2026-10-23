"""Check saved evidence, protected hashes, report links and Git scope; no training."""
import csv, hashlib, json, re, subprocess
from pathlib import Path
from datetime import datetime, timezone

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parent.parent

def load(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()

def main():
    destination = PHASE / 'outputs/RECOVERY_FINAL_AUDIT.json'
    if destination.exists():
        raise RuntimeError('FINAL_AUDIT_ALREADY_RECORDED')
    cfgpath = PHASE / 'configs/DDP_SOFT_PHYSICS_FROZEN_PROTOCOL.json'
    cfg = load(cfgpath)
    assert sha(cfgpath) == cfgpath.with_suffix('.sha256').read_text().strip()
    manifests = [cfg['input_and_code_sha256'], load(PHASE / 'outputs/REPORT_SOURCE_MANIFEST.json')]
    for manifest in manifests:
        for name, expected in manifest.items():
            assert Path(name).suffix not in ('.parquet',), name
            assert Path(name).name != 'causal_features_30min.csv', 'NO_FULL_N1_HASH'
            assert sha(Path(name)) == expected, name
    decision = load(PHASE / 'outputs/QUALIFICATION_DECISION.json')
    assert decision['formal_runs_started'] == decision['epochs_trained'] == decision['neural_optimizer_steps'] == 0
    assert decision['protocol_sha256'] == sha(cfgpath)
    activity = load(PHASE / 'outputs/TRAIN_PHYSICS_ACTIVITY.json')
    assert {(r['seed'], r['role']) for r in activity} == {(s, r) for s in (42,52,62) for r in ('fresh_train_mode','best','last450')}
    assert sum(r['n_forecasts'] for r in activity) == 298080
    assert all(r['violations'] == r['physics_loss_dimensionless'] == r['activation_fraction'] == 0 for r in activity)
    access = load(PHASE / 'outputs/RUNTIME_DATA_ACCESS_AUDIT.json')
    assert access['bounded_reads'] == [{'read_nrows':3007, 'parsed_rows':3007}]
    assert not access['new_Test_access'] and not access['Assessment_windows_built']
    assert not access['Validation_windows_built_for_qualification']
    assert not access['new_polarization_or_health_target_read']
    denied = [e for e in access['events'] if e['action'] == 'DENIED']
    # Python may try a cached .pyc before falling back to the allowlisted .py.
    # Keep every denial, and allow this audit exception only for its verified source.
    for event in denied:
        p = Path(event['path'])
        assert not event['writing'] and p.suffix == '.pyc' and p.parent.name == '__pycache__'
        source = p.parent.parent / (p.name.split('.cpython-')[0] + '.py')
        assert str(source) in cfg['input_and_code_sha256'], str(source)
    tests = load(PHASE / 'outputs/SYNTHETIC_TEST_RESULTS.json')
    assert tests['returncode'] == 0 and 'Ran 14 tests' in tests['stderr']
    reports = sorted(PHASE.glob('[0-8][0-8]_*.md'))
    assert len(reports) == 9
    links = 0
    for p in reports:
        for url in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if not url.startswith(('http:', 'https:')):
                assert (p.parent / url).is_file(), url
                links += 1
    # Validate the extracted historical figures against their saved authorities.
    with (PHASE/'outputs/HISTORICAL_P3_G2_METRICS.csv').open(encoding='utf-8', newline='') as f:
        history = list(csv.DictReader(f))
    assert len(history) == 6
    for row in history:
        authority = load(PHASE.parent/'Phase_P3_Full450_Epoch_Sensitivity'/f"outputs/training_logs/{row['run_id']}/{row['checkpoint_role']}_result.json")
        for key in ('RMSE_mV','MAE_mV','endpoint_12h_RMSE_mV','P95_abs_error_mV','P99_abs_error_mV','max_abs_error_mV'):
            assert float(row[key]) == authority[key]
    with (PHASE/'outputs/HISTORICAL_P3_G2_24STEP_METRICS.csv').open(encoding='utf-8', newline='') as f:
        assert len(list(csv.DictReader(f))) == 144
    assert git('diff','--name-only') == '', 'HISTORICAL_TRACKED_CHANGES'
    assert git('diff','--cached','--name-only') == '', 'UNEXPECTED_PRESTAGED_FILES'
    subprocess.run(['git','diff','--check'],cwd=REPO,check=True)
    assets = {}
    for p in sorted(PHASE.rglob('*')):
        if p.is_file() and not p.is_relative_to(PHASE/'runtime') and p.name != 'GIT_TRANSPORT_RECEIPT.local.json':
            assets[str(p.relative_to(PHASE)).replace('\\','/')] = sha(p)
    record = {'passed':True, 'UTC':datetime.now(timezone.utc).isoformat(),
        'stage':'precommit', 'recovered_base_commit':git('rev-parse','HEAD'),
        'formal_training_runs':0, 'new_neural_optimizer_steps':0,
        'real_Train_forward_values':298080, 'zero_physics_loss_checks':9,
        'preserved_prior_synthetic_tests':14, 'reports':9, 'resolved_local_links':links,
        'historical_metrics_rows':6, 'historical_horizon_rows':144,
        'protected_hashes_verified':sum(map(len,manifests)), 'historical_tracked_diff':[],
        'unrelated_untracked_dependencies_preserved':True,
        'denied_historical_bytecode_cache_reads':len(denied),
        'report_validator_repair':'First check rejected all DENIED events; inspected nine read-only .pyc cache denials, then limited exception to verified .py sources. Qualification not rerun.',
        'new_Test_or_late_health_targets_read':False,
        'git_transport_receipt':'outputs/GIT_TRANSPORT_RECEIPT.local.json (local only, written after transport)',
        'phase_artifact_sha256_before_this_audit':assets}
    with destination.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(record,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in record.items() if k!='phase_artifact_sha256_before_this_audit'},ensure_ascii=False))

if __name__ == '__main__':
    main()
