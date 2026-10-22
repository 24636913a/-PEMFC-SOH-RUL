# %% 用户设置：本文件是唯一正式入口；Run All Cells才会开始真实新训练
MODEL = "F2"                 # "F2" 或 "LSTM"，每次只训练一个模型
SEED = 202
LOOKBACK = 300
HORIZON = 120
TARGET_MODE = "V"            # DeltaV仅预留，不属于本次登记运行
SPLIT = (0.7, 0.1, 0.2)
RUN_MODE = "TRAIN"           # 不提供Holdout评价入口

MAX_EPOCHS = 450
PATIENCE = 40
BATCH_SIZE = 16              # microbatch；两批梯度累计，有效batch=32
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 0.0009605449687795866
WEIGHT_DECAY = 6.95942232671908e-06
TRAIN_ORIGIN_STRIDE = 1       # 训练前固定；可增大以减少训练量，必须记录
FEATURE_COUNT = 10           # 六项必要状态/门控字段 + Train Spearman四项
PREPROCESS_ORIGINS = 10000   # Train合法起点均匀取样拟合筛选/scaler，不按误差选
PLOT_EVERY_EPOCHS = 5
PROGRESS_BATCHES = 500

# %% 运行环境及真实配置（不导入旧交互脚本，不读取旧权重）
import sys
from pathlib import Path
sys.dont_write_bytecode = True


def locate_entry():
    supplied = globals().get('__file__')
    if supplied and Path(supplied).name == 'F2_Train_Interactive.py':
        return Path(supplied).resolve()
    for parent in (Path.cwd(), *Path.cwd().parents):
        for relative in ('F2_Train_Interactive.py',
                         'REAL_Paper_File/F2主程序/F2_Train_Interactive.py',
                         'RENew-PEMFC-2300h-Pytorch/REAL_Paper_File/F2主程序/F2_Train_Interactive.py'):
            candidate = parent/relative
            if candidate.is_file() and (candidate.parent/'raw1s').is_dir():
                return candidate.resolve()
    raise RuntimeError('请先在VS Code打开本仓库，或将当前目录切换到F2主程序目录')


ENTRY = locate_entry()
if str(ENTRY.parent) not in sys.path:
    sys.path.insert(0, str(ENTRY.parent))
from raw1s.runner import run

SETTINGS = dict(model=MODEL, seed=SEED, lookback=LOOKBACK, horizon=HORIZON,
                target_mode=TARGET_MODE, split=SPLIT, run_mode=RUN_MODE,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
                gradient_accumulation=GRADIENT_ACCUMULATION,
                learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                gradient_clip=1.0, train_stride=TRAIN_ORIGIN_STRIDE,
                feature_count=FEATURE_COUNT, preprocess_origins=PREPROCESS_ORIGINS,
                plot_every_epochs=PLOT_EVERY_EPOCHS,
                progress_batches=PROGRESS_BATCHES, chunk_rows=50000)
print(f'所选模型：{MODEL}，Seed：{SEED}，原始秒级300→120，目标V；只训练及评价Validation。')

# %% 用户运行此单元：创建独立目录 → 原始CSV → 冻结边界/特征 → 新训练 → Validation/绘图
RESULT_DIRECTORY = run(SETTINGS, ENTRY)
