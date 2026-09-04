import os
import pandas as pd

# ==================================================
# 数据集根目录（已为你写好）
DATASET_ROOT = r"E:\作业\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\FC1_FC2_Excel\FC1_FC2_Excel"
# ==================================================

# 关键词字典：用于自动识别列含义
KEYWORDS = {
    "time": ["time", "t", "hour", "sec", "second", "cycle"],
    "voltage": ["voltage", "volt", "v"],
    "current": ["current", "amp", "i"],
    "temperature": ["temp", "temperature"],
    "pressure": ["pressure", "press"],
    "power": ["power"],
    "soh": ["soh", "health"],
    "rul": ["rul", "remaining", "life"],
}

def detect_columns(columns):
    """根据列名关键词判断物理含义"""
    found = {k: [] for k in KEYWORDS}
    for col in columns:
        col_lower = str(col).lower()
        for key, kws in KEYWORDS.items():
            if any(kw in col_lower for kw in kws):
                found[key].append(col)
    return found

def inspect_excel(file_path):
    """读取并分析单个 Excel"""
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        return None, f"❌ 无法读取（可能是格式或权限问题）：{e}"

    info = {
        "rows": len(df),
        "cols": list(df.columns),
        "detected": detect_columns(df.columns),
    }
    return info, None

def main():
    print("\n========== PEMFC 数据集自动体检 ==========\n")

    excel_files = []
    for root, _, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.endswith(".xlsx") or f.endswith(".xls"):
                excel_files.append(os.path.join(root, f))

    print(f"📁 在数据集中共发现 Excel 文件数量：{len(excel_files)}\n")

    summary = {k: 0 for k in KEYWORDS}

    for file in excel_files:
        print(f"📄 文件名：{os.path.basename(file)}")
        info, error = inspect_excel(file)

        if error:
            print(error)
            print("-" * 60)
            continue

        print(f"  - 数据行数：{info['rows']}")
        print(f"  - 列名：{info['cols']}")

        for k, v in info["detected"].items():
            if v:
                print(f"  ✔ 识别到【{k}】相关列：{v}")
                summary[k] += 1

        print("-" * 60)

    print("\n========== 数据集整体统计 ==========")
    for k, v in summary.items():
        if v > 0:
            print(f"【{k.upper()}】出现于 {v} 个 Excel 文件中")

    print("\n========== 科研视角初步判断 ==========")

    if summary["time"] > 0 and summary["voltage"] > 0:
        print("✅ 数据集中普遍包含【时间 + 电压】，适合做 PEMFC 退化建模")

    if summary["rul"] > 0:
        print("✅ 发现显式 RUL / 寿命相关字段，可直接做 RUL 预测")
    else:
        print("⚠️ 未发现显式 RUL，建议通过 EOL 阈值自行构造 RUL")

    print("\n👉 下一步可考虑：")
    print(" - LSTM / GRU（时间序列基线）")
    print(" - 加入物理约束的神经网络（PINN 或弱物理约束）")

    print("\n========== 体检完成 ==========\n")

if __name__ == "__main__":
    main()
