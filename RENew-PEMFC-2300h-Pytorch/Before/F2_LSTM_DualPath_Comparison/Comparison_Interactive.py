# %% 1. Independent VS Code entry: fixed scientific protocol, real fresh training
# Run All starts 2 pilot + 12 formal models, serially. No historical weights loaded.
EXISTING_DATA_RUN = None  # acceptance may reuse THIS TASK's already qualified P0/P1

from pathlib import Path
import sys
if '__file__' in globals():FOLDER=Path(__file__).resolve().parent
else:
    options=[Path.cwd(),Path.cwd()/'F2_LSTM_DualPath_Comparison',Path.cwd()/'RENew-PEMFC-2300h-Pytorch/F2_LSTM_DualPath_Comparison']
    FOLDER=next(p for p in options if (p/'core.py').exists() and (p/'Comparison_Interactive.py').exists())
if str(FOLDER) not in sys.path:sys.path.insert(0,str(FOLDER))
import core as c
from stages import prepare_run,diagnose_existing
from training import pilot,formal_train,formal_evaluate
from io_utils import js
from IPython.display import display,Markdown

# %% 2. P0 then P1: no model training before identity and diagnosis pass
if EXISTING_DATA_RUN is None:
    run,frame,windows,scaler,prepared,protocol=prepare_run()
    diagnosis=diagnose_existing(run,frame,windows,protocol)
else:
    run=FOLDER/'outputs'/EXISTING_DATA_RUN
    c.require(not (run/'INVALIDATED_BEFORE_TRAINING.json').exists(),'INVALIDATED_DATA_RUN')
    c.require(c.read_json(run/'P0_ACCEPTANCE.json')['status']=='PASS' and c.read_json(run/'P1_ACCEPTANCE.json')['status']=='PASS','P0_P1_NOT_PASSED')
    c.require(not (run/'models').exists(),'REFUSE_REPEAT_OR_OVERWRITE_TRAINING')
    frame,windows,scaler,prepared,excluded=c.load_data();protocol=c.read_json(run/'FROZEN_PROTOCOL.json')
    c.require(c.sha(c.SOURCE)==protocol['data_sha256'] and c.sha(FOLDER/'core.py')==protocol['core_sha256'],'PREQUALIFIED_DATA_SOURCE_CHANGED')
    c.configure_environment()
display(Markdown('**P0/P1 passed.** Fixed Top10, 1750 Train /218 Validation /527 exploratory holdout origins. No strict CUDA mode.'))
display(c.read_csv(run/'PARTITIONS.csv'))

# %% 3. Two full Seed202 pilot trainings, Validation replay and update audit
pilot(run,frame,windows,scaler,prepared,protocol)

# %% 4. Twelve fresh formal instances: same policy, no holdout-based selection
ledgers=formal_train(run,windows,scaler,prepared,protocol)

# %% 5. Only now compute NEW Train/Validation/holdout predictions and metrics
table,means,paired=formal_evaluate(run,frame,windows,scaler,prepared,ledgers)

# %% 6. P3 figures and P4 recommendations: never execute additional optimization
from finish import finalize
finalize(run,frame,windows,scaler,prepared,table,means,paired)
print('COMPARISON_PIPELINE_COMPLETED',str(run),flush=True)
