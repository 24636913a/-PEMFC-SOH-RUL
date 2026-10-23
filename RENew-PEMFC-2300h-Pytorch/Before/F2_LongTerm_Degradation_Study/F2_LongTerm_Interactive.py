# %% 1. Run control - VS Code: select the project Python 3.11 / CUDA kernel
"""Fresh long-trend pilot, or read-only display of this completed study.

Default is read-only to avoid unintentionally repeating a completed experiment.
MODE='pilot' genuinely starts four new seed202 GPU trainings, in a NEW folder;
it never loads historical checkpoints to continue training or replay old logs.
The frozen 48h pilot gate failed in the delivered run; no automatic six-seed
expansion or holdout evaluation is permitted by this entry point.
"""
MODE = 'analyze'  # 'analyze' or 'pilot'
from pathlib import Path
import json
import sys
HERE=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if not (HERE/'configs/PREREGISTERED_QUALIFICATION.json').is_file():
    raise RuntimeError('Open/run this file from F2_LongTerm_Degradation_Study')
sys.path.insert(0,str(HERE/'scripts'))
from IPython.display import display,Image,Markdown
import pandas as pd

# %% 2. Data continuity and frozen qualification (actual metadata audit)
print('Loading completed continuity qualification; no HOLDOUT outcome values')
audit=HERE/'outputs/qualification_20261020_v2'
display(pd.read_csv(audit/'WINDOW_COUNTS.csv'))
display(pd.read_csv(audit/'P0_QUALIFICATION.csv'))

# %% 3. Causal target definition and Train-only condition adjustment
print('Showing actual TRAIN + VALIDATION target diagnostics, not SOH labels')
display(pd.read_csv(audit/'TARGET_DIAGNOSTICS.csv'))
display(pd.read_csv(audit/'CONDITION_SUPPORT.csv'))
display(pd.read_csv(audit/'P1_QUALIFICATION.csv'))
display(Markdown((HERE/'literature/DESIGN_TRANSFER.md').read_text(encoding='utf-8')))

# %% 4. Explicit training policy; a new pilot is always from scratch
protocol=json.loads((HERE/'configs/FROZEN_TRAINING_PROTOCOL.json').read_text(encoding='utf-8'))
display(protocol)
if MODE=='pilot':
    from train_longterm import run_pilot
    print('Starting NEW seed202 LSTM/F2 trainings for 24h and 48h; live epochs and newly generated loss curves')
    run=run_pilot(live=True)
elif MODE=='analyze':
    run=Path(json.loads((HERE/'outputs/ACTIVE_PILOT.json').read_text(encoding='utf-8'))['path'])
    print('READ-ONLY analysis of actual completed run:',run)
else:raise ValueError("MODE must be 'analyze' or 'pilot'")

# %% 5. Actual run identity, logs, metrics and stop decision
for path in sorted(run.glob('*/COMPLETED.json')):
    result=json.loads(path.read_text(encoding='utf-8'))
    print(result['model'],result['seed'],result['horizon_steps']*.5,'h; best epoch',result['best_epoch'])
    display(pd.read_csv(path.parent/'METRICS.csv'))
    display(Image(filename=str(path.parent/'training_history.png')))
display(pd.read_csv(run/'PILOT_QUALIFICATION.csv'))
display(json.loads((run/'PILOT_DECISION.json').read_text(encoding='utf-8')))

# %% 6. Paper-candidate graphics (all real CSV; no prediction smoothing)
if MODE=='analyze':
    for path in sorted((HERE/'figures').glob('*.png')):
        display(Image(filename=str(path)))
    display(Markdown((HERE/'reports/04_长期预测最终研究总结.md').read_text(encoding='utf-8')))
else:
    print('New pilot complete. Preserve its run identity; independently verify before exporting new reports/figures.')
