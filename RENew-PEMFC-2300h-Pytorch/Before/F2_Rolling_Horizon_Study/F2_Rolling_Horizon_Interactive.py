# %% 1. Modes: show existing results, read-only analyze, or six fresh trainings
"""VS Code Python Interactive entry; no old training script import.

MODE='show' displays the delivered real CSV/PNG results, without training.
MODE='analyze' recomputes metrics read-only from saved prediction CSVs.
MODE='train' creates a NEW run, trains six Seed202 models from scratch, freezes
all best checkpoints, evaluates once, verifies, and saves new per-run figures.
Never use repeated new runs to select favorable CUDA numerical trajectories.
"""
MODE = 'show'  # exactly 'show', 'analyze', 'train'
SEED = 202    # this registered study supports no other seed
from pathlib import Path
from datetime import datetime,timezone
import sys
import json
if '__file__' in globals():HERE=Path(__file__).resolve().parent
else:
    candidates=[Path.cwd(),Path.cwd()/'F2_Rolling_Horizon_Study',Path.cwd()/'RENew-PEMFC-2300h-Pytorch/F2_Rolling_Horizon_Study']
    HERE=next((p for p in candidates if (p/'configs/FROZEN_PROTOCOL.json').is_file()),None)
    if HERE is None:raise RuntimeError('Open this file from the project workspace, or set the Interactive working directory to F2_Rolling_Horizon_Study')
assert MODE in ['show','analyze','train'] and SEED==202
sys.path.insert(0,str(HERE/'scripts'))
import pandas as pd
from IPython.display import display,Image,Markdown
from common import read,read_csv,metrics

# %% 2. Actual frozen configuration and stage progress
display(read(HERE/'configs/FROZEN_PROTOCOL.json'))
print('Seed202 selected using historical holdout; exploratory single-seed study, not a blind Test')
if MODE=='train':
    run=HERE/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_seed202_six_fresh')
    print('P1: read-only historical audit; creating new output:',run)
    from historical_audit import run as historical_audit
    historical_audit(run)
else:
    run=Path(read(HERE/'outputs/ACTIVE_RUN.json')['path'])
    print('Existing run (no training):',run)
display(read_csv(run/'WINDOW_COUNTS.csv'))
display(read_csv(run/'P1_HISTORICAL_PREFIX_METRICS.csv'))

# %% 3. Genuine fresh GPU training only in explicit train mode
if MODE=='train':
    print('P2/P3: six fresh models, real backward/optimizer updates, live epochs and real loss images')
    from train import run as train_cohort
    train_cohort(run,live=True)
else:
    print('No training in',MODE,'mode; showing actual stored best/stop epochs')
    for p in sorted((run/'models').glob('*/COMPLETED.json')):
        m=read(p);print(m['model'],'H',m['horizon_steps'],'best epoch',m['best_epoch'],'stop',m['stop_epoch'])

# %% 4. Unified evaluation and independent verification, or read-only recalculation
if MODE=='train':
    print('All6 best checkpoints frozen; one descriptive holdout pass, followed by independent verification')
    from evaluate import run as evaluate
    evaluate(run)
    from verify_independent import verify
    verify(run)
    from render_figures import run as render
    render(run,presentation_root=run/'presentation')
elif MODE=='analyze':
    result=[]
    for p in sorted((run/'models').glob('*/predictions_HOLDOUT.csv')):
        x=read_csv(p);result.append(dict(run=p.parent.name,**metrics(x)))
    display(pd.DataFrame(result))
    print('Read-only CSV recalculation; no inference, training or output overwrite')
else:print('Displaying already verified results; no model execution')

# %% 5. Actual final metrics and recorded scientific boundaries
display(read_csv(run/'FINAL_RESULT_TABLE.csv')[['model','source','history_steps','horizon_steps','RMSE_mV','MAE_mV','R2','queries','origins']])
display(read(run/'C1_VALIDATION_DECISION.json'))
display(read(run/'INDEPENDENT_VERIFICATION.json'))

# %% 6. Real saved plots displayed inside the Interactive Window
index=read(run/'FIGURE_INDEX.json')
for p in sorted(Path(index['figure_directory']).glob('*.png')):
    display(Image(filename=str(p)))
print('All images are real 300dpi PNGs; matching PDF and raw plotting CSV files are retained')
if MODE!='train':
    delivered=HERE/'outputs/DELIVERED_RUN.json'
    if delivered.exists() and run.resolve()==Path(read(delivered)['path']).resolve():
        display(Markdown((HERE/'reports/05_最终通俗总结.md').read_text(encoding='utf-8')))
    else:print('This additional run has its own metrics/figures; no report from a different cohort is displayed')
