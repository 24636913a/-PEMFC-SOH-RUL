# %% [markdown]
# # 电压突变观测关联细化（只读缓存、CPU、无训练）
# 在VS Code中选择原项目Python 3.11环境，Run All Cells。
# 每次运行创建新的runs目录；不覆盖旧审计。

# %% 用户设置及定位（此单元不运行分析）
from pathlib import Path
import sys

# 默认沿用冻结run；仅当旧缓存已恢复到独立目录时填写完整路径。
SOURCE_RUN = None
STAGE_DIR = None  # 通常留空；如交互窗口位于其他工作目录，可填本脚本所在目录。


def locate_stage():
    if STAGE_DIR is not None:
        return Path(STAGE_DIR).resolve()
    if '__file__' in globals():
        return Path(__file__).resolve().parent
    relative = Path('RealHyFC_Raw1s_Rolling_300to120/audits/Voltage_Event_Classification_Refinement')
    for parent in (Path.cwd(), *Path.cwd().parents):
        for candidate in (parent, parent/relative, parent/'RENew-PEMFC-2300h-Pytorch'/relative):
            if (candidate/'association_refinement/pipeline.py').is_file():
                return candidate.resolve()
    raise FileNotFoundError('请设置STAGE_DIR为本脚本所在目录；不会自动查找或重建原始数据')


stage = locate_stage()
if str(stage) not in sys.path:
    sys.path.insert(0, str(stage))
print('入口：', stage)
print('只使用已保存缓存；事件前/瞬间/后分别统计；不加载F2、不读取原始Loadcycle CSV。')

# %% 执行用户授权的手动分析（每次新建输出目录；此单元才真正运行）
from association_refinement.pipeline import run

OUTPUT_DIR = run(stage, source_override=SOURCE_RUN)
print('结果位置：', OUTPUT_DIR)

# %% 显示本次真实结果摘要；不会重跑分析
from IPython.display import Markdown, display
import pandas as pd

display(Markdown((OUTPUT_DIR/'REPORT.md').read_text(encoding='utf-8')))
display(pd.read_csv(OUTPUT_DIR/'association_rates_by_phase.csv').query("criterion == 'family_background_unusual'"))
display(pd.read_csv(OUTPUT_DIR/'ordinary_control_rates.csv').query("criterion == 'family_background_unusual'"))
