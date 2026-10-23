# %% 1. Independent real-training research entry. No old interactive script is imported.
START_NEW_RUN = False  # False resumes this task's completed instances without training them twice.
RUN_ID = None
STAGE_LIMIT = 'A'  # A or B. This completed run did not qualify C/D; no unregistered extra training.
if STAGE_LIMIT not in ['A','B']:raise ValueError('本轮仅注册A/B；C/D未触发，禁止启动未登记训练。')

from pathlib import Path
import sys
if '__file__' in globals():FOLDER=Path(__file__).resolve().parent
else:
    options=[Path.cwd(),Path.cwd()/'F2_Performance_Improvement',Path.cwd()/'RENew-PEMFC-2300h-Pytorch/F2_Performance_Improvement']
    FOLDER=next(p for p in options if (p/'frozen_f2.py').exists())
if str(FOLDER) not in sys.path:sys.path.insert(0,str(FOLDER))
for module_name in ['frozen_f2','settings','diagnostics','train','io_utils','selection','stage_a','protocol',
    'history_diagnosis','report_a','verify_stage','stage_b','residual_fusion','report_b','repeat_diagnosis',
    'phase_flow','final_reports','artifact_index','final_integrity']:
    loaded=sys.modules.get(module_name)
    if loaded is not None and getattr(loaded,'__file__',None) and Path(loaded.__file__).resolve().parent!=FOLDER:
        raise RuntimeError('请重启VS Code Python交互内核：检测到其他目录的同名模块 '+module_name)
import frozen_f2 as f
from protocol import prepare
from history_diagnosis import analyze
from stage_a import run_stage
from IPython.display import display,Markdown

# %% 2. Data identity + historical read-only Train/Validation diagnosis
if RUN_ID is None and not START_NEW_RUN and (FOLDER/'ACTIVE_RUN.json').exists():RUN_ID=f.read_json(FOLDER/'ACTIVE_RUN.json')['run']
if RUN_ID is None:
    run,frame,windows,scaler,prepared=prepare()
else:
    run=FOLDER/'outputs'/RUN_ID
    f.require(f.read_json(run/'P0_ACCEPTANCE.json')['status']=='PASS','P0_REQUIRED')
    f.configure_environment();frame,windows,scaler,prepared,excluded=f.load_selection_data()
if not (run/'historical/IDENTITY_AUDIT.json').exists():analyze(run,frame,windows,scaler,prepared)
f.model_identity()
f.require(f.sha(f.SOURCE)==f.read_json(run/'PROTOCOL.json')['data_sha256'],'FROZEN_DATA_CHANGED')
for name,digest in f.read_json(run/'CODE_BEFORE_A.json').items():
    f.require(f.sha(FOLDER/name)==digest,'FROZEN_MATHEMATICAL_SOURCE_CHANGED:'+name)
display(Markdown('**Train/Validation identity passed.** Native F2, Top10, 1750/218 windows. No holdout windows or new scores.'))

# %% 3. Nine fresh Stage A trainings, then frozen Validation-only decision
a_result=run_stage(run,frame,windows,scaler,prepared)
display(f.read_csv(run/'tables/A_VALIDATION_PER_SEED.csv'))
display(f.read_csv(run/'tables/A_VALIDATION_MEAN_SD.csv'))
print('STAGE_A_FINISHED',run,flush=True)

# %% 4. Independent replay audit; no new training and still no holdout inference
from verify_stage import verify
if not (run/'audits/A_INDEPENDENT_AUDIT.json').exists():verify(run,'A')
else:f.require(f.read_json(run/'audits/A_INDEPENDENT_AUDIT.json')['status']=='PASS','A_AUDIT_NOT_PASS')

# %% 5. One registered residual-fusion hypothesis; six NEW trainings if requested
if STAGE_LIMIT=='B':
    from stage_b import run_stage as run_b
    b_result=run_b(run,frame,windows,scaler,prepared)
    display(f.read_csv(run/'tables/B_VALIDATION_PER_SEED.csv'))
    display(f.read_csv(run/'tables/B_VALIDATION_MEAN_SD.csv'))

# %% 6. Independent B audit; actual best weights, no optimizer updates
if STAGE_LIMIT=='B':
    if not (run/'audits/B_INDEPENDENT_AUDIT.json').exists():verify(run,'B')
    else:f.require(f.read_json(run/'audits/B_INDEPENDENT_AUDIT.json')['status']=='PASS','B_AUDIT_NOT_PASS')

# %% 7. Explicit conditional closure; completed models and figures are never overwritten
if STAGE_LIMIT=='B':
    from phase_flow import review_ab,stop_if_no_upgrade
    review=review_ab(run)
    if review['eligible_candidates']:
        display(Markdown('存在合格前置候选。必须先完成C/D独立预注册；本入口不启动未登记训练。'))
    else:
        stop_if_no_upgrade(run)
        if not (run/'04_最终研究总结.md').exists():
            from final_reports import write_stopped
            write_stopped(run)
        from IPython.display import Image
        display(f.read_csv(run/'tables/FINAL_VALIDATION_STUDY_SUMMARY.csv'))
        display(Image(filename=str(run/'figures/FINAL/01_Validation_study_summary.png')))
        print('NO_QUALIFIED_UPGRADE: A9+B6真实新训练已完成；C/D未触发，新后留指标NA。',flush=True)
