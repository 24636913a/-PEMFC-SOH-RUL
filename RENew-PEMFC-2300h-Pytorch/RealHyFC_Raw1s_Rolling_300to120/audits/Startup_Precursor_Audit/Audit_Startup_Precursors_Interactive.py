# %% 审计配置：仅在用户Run All Cells后执行，不接入F2训练流程
RUN_MODE = "AUDIT"
PRECURSOR_SECONDS = [1, 5, 10, 30, 60, 120]
PREDICTION_HORIZONS = [30, 60, 120]
MAKE_FIGURES = True

# %% 定位独立目录；不import正式F2或任何历史训练入口
import sys
from pathlib import Path
sys.dont_write_bytecode = True


def locate_audit():
    path = globals().get('__file__')
    if path and Path(path).name == 'Audit_Startup_Precursors_Interactive.py':
        return Path(path).resolve().parent
    relative = 'RealHyFC_Raw1s_Rolling_300to120/audits/Startup_Precursor_Audit'
    for parent in (Path.cwd(), *Path.cwd().parents):
        for candidate in (parent, parent/relative, parent/'RENew-PEMFC-2300h-Pytorch'/relative):
            if (candidate/'Audit_Startup_Precursors_Interactive.py').is_file() and (candidate/'startup_audit_core').is_dir():
                return candidate.resolve()
    raise RuntimeError('请在VS Code打开本项目或切换到Startup_Precursor_Audit目录')


AUDIT_DIRECTORY = locate_audit()
if str(AUDIT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIRECTORY))
from startup_audit_core.runner import run_audit

# %% 开始真实CPU审计：17个原始文件 → Train规则冻结 → 全量事件/固定规则检查 → CSV/图/报告
RESULT_DIRECTORY = run_audit(AUDIT_DIRECTORY, dict(
    run_mode=RUN_MODE, precursor_seconds=sorted(set(PRECURSOR_SECONDS+[0]), reverse=True),
    prediction_horizons=list(PREDICTION_HORIZONS), make_figures=MAKE_FIGURES,
))
