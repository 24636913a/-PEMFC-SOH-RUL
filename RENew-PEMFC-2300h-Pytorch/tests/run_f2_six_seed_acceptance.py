"""Run the seven requested commands once, preserving stdout and old-file hashes."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'RealHyFC_F2_Main_Outputs'
SEEDS = [42, 103, 193, 202, 210, 212]
LEDGER = OUTPUTS / 'SIX_SEED_EXECUTION.json'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def save(value):
    LEDGER.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    assert not LEDGER.exists(), 'One-shot acceptance already started; inspect ledger rather than repeat inference'
    assets = json.loads((ROOT / 'RealHyFC_F2_Main_Seed_Assets.json').read_text(encoding='utf-8'))
    old = [path for path in OUTPUTS.rglob('*') if path.is_file()]
    for row in assets['models']:
        old.extend(ROOT.parent / row[key] for key in ['checkpoint', 'reference_prediction', 'training_history', 'protocol_source'])
    for seed in [52, 62]:
        old.extend([ROOT / f'Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F2-Seed{seed}_best.pt',
                    ROOT / f'Phase_SCI_RealHyFC_Full_Chronological_Evaluation/outputs/predictions/VALIDATION_F2_seed{seed}.csv'])
    before = {path.relative_to(ROOT.parent).as_posix(): sha(path) for path in sorted(set(old))}
    state = dict(status='RUNNING', program_sha256=sha(ROOT / 'RealHyFC_F2_Main.py'),
                 seeds=SEEDS, history_before=before, records=[], invalid_seeds=[])
    save(state)
    for seed in [*map(str, SEEDS), 'all']:
        directories = set(OUTPUTS.iterdir())
        command = [sys.executable, '-B', 'RealHyFC_F2_Main.py', '--seed', seed]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, encoding='utf-8', errors='strict')
        created = [path for path in set(OUTPUTS.iterdir()) - directories if path.is_dir()]
        record = dict(seed_argument=seed, command=command, returncode=result.returncode,
                      stdout=result.stdout, stderr=result.stderr,
                      runs=[path.name for path in sorted(created)])
        state['records'].append(record); save(state)
        assert result.returncode == 0 and len(created) == 1, record
        assert len(result.stdout.strip().splitlines()) == 3 and not result.stderr, record
        assert [line.split(':')[0].strip() for line in result.stdout.strip().splitlines()] == ['RMSE', 'MAE', 'R²']
        print('PASS', seed, flush=True)
    for seed in ['52', '62']:
        directories = set(OUTPUTS.iterdir())
        result = subprocess.run([sys.executable, '-B', 'RealHyFC_F2_Main.py', '--seed', seed], cwd=ROOT,
                                capture_output=True, encoding='utf-8')
        state['invalid_seeds'].append(dict(seed=seed, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        save(state)
        assert result.returncode == 2 and 'invalid choice' in result.stderr and not result.stdout
        assert set(OUTPUTS.iterdir()) == directories, 'Rejected seed created an output directory'
    for relative, digest in before.items():
        assert sha(ROOT.parent / relative) == digest, ('HISTORICAL_FILE_CHANGED', relative)
    state.update(status='PASS', historical_files_unchanged=len(before), new_training=False, hyperparameter_search=False)
    save(state)
    print('SEVEN_COMMANDS_AND_LEGACY_REJECTION_PASSED', flush=True)
