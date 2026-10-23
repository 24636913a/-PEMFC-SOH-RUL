"""Read-only compact progress; never starts or resumes training."""
from pathlib import Path
import json
p=Path(__file__).resolve().parents[1]
ledger=json.loads((p/'outputs/EXECUTION_LEDGER.json').read_text(encoding='utf-8'))
active=ledger.get('active_run')
state=p/'outputs/training_logs'/str(active)/'run_state.json'
result={'status':ledger['status'],'completed':ledger['formal_completed'],'started':ledger['formal_started'],'active_run':active}
if state.exists():result['active_state']=json.loads(state.read_text(encoding='utf-8'))
print(json.dumps(result,ensure_ascii=False))
