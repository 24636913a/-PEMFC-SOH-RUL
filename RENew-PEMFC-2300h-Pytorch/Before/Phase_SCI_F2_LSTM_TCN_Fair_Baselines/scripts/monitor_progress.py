"""Read-only compact live progress; deliberately no torch or training imports."""
from pathlib import Path
import json,csv
P=Path(__file__).resolve().parents[1]
RUNS=[('LSTM',42),('TCN',42),('LSTM',52),('LSTM',62),('TCN',52),('TCN',62)]
if __name__=='__main__':
    completed=[];active=[];failed=[]
    for name,seed in RUNS:
        f=P/f'outputs/training_logs/{name}_seed{seed}/run_state.json'
        if not f.exists():continue
        state=json.loads(f.read_text(encoding='utf-8'))
        if state['status']=='COMPLETED':completed.append(f'{name}{seed}')
        elif state['status']=='FAILED':failed.append(f'{name}{seed}')
        else:
            row=dict(run=f'{name}{seed}',epoch=state['last_completed_epoch'],best_epoch=state.get('best_epoch'))
            log=f.parent/'epochs.csv'
            if log.exists():
                rows=list(csv.DictReader(log.open(encoding='utf-8')))
                if rows:row.update(current_RMSE_mV=round(float(rows[-1]['validation_RMSE_mV']),4),stale=int(rows[-1]['stale_epochs']))
            active.append(row)
    print(json.dumps(dict(completed=completed,active=active,failed=failed),ensure_ascii=False))
