# -*- coding: utf-8 -*-
"""批量运行 FC2 消融实验，记录结果到 results.txt 和 summary.pkl"""
import subprocess, sys, re, pickle, time, os
from datetime import datetime
from pathlib import Path

BASE = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\New-PEMFC-Pytorch")
SCRIPTS_DIR = BASE / "消融实验代码"
OUT_DIR = BASE / "消融实验outputs"

# FC2 全部8个模型
FC2_SCRIPTS = [
    "消融实验代码-TFT-纯MSE-FC2.py",
    "消融实验代码-TFT-minus-GRN-FC2.py",
    "消融实验代码-TFT-minus-VS-FC2.py",
    "消融实验代码-PINN-TFT-MT-FC2.py",
    "消融实验代码-DLinear-FC2.py",
    "消融实验代码-ResMLP-FC2.py",
    "消融实验代码-LSTM-Baseline-FC2.py",
    "消融实验代码-TSMixer-FC2.py",
]

results = {}
results_file = OUT_DIR / "results.txt"

with open(results_file, 'a', encoding='utf-8') as log:
    log.write(f"\n{'='*60}\n")
    log.write(f"FC2 消融实验 — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    log.write(f"{'='*60}\n")

for script_name in FC2_SCRIPTS:
    script_path = SCRIPTS_DIR / script_name
    model_label = script_name.replace("消融实验代码-", "").replace("-FC2.py", "")
    print(f"\n[{datetime.now().strftime('%H:%M')}] 运行: {model_label}")

    start = time.time()
    try:
        r = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE),
            capture_output=True, text=True,
            encoding='utf-8', errors='replace',
            timeout=4 * 3600  # 4h 超时
        )
    except subprocess.TimeoutExpired:
        print(f"  FAIL 超时 (4h)")
        results[model_label] = {'error': 'timeout'}
        continue

    elapsed = time.time() - start
    output = r.stdout + r.stderr

    # 提取 Loop R²
    m = re.search(r'Loop R²:\s*Min=([\d\.\-]+)\s+Mean=([\d\.\-]+)\s+Max=([\d\.\-]+)', output)
    rmse_m = re.search(r'RMSE:\s*([\d\.\-]+)\s*V', output)
    top5_m = re.search(r'Top-?\d+ 集成 R²=([\d\.\-]+)', output)
    mc_m   = re.search(r'平均预测标准差:\s*([\d\.\-]+)\s*V', output)

    if m:
        min_r = float(m.group(1))
        mean_r = float(m.group(2))
        max_r = float(m.group(3))
        rmse_v = float(rmse_m.group(1)) if rmse_m else None
        top5_v = float(top5_m.group(1)) if top5_m else None
        mc_v   = float(mc_m.group(1)) if mc_m else None
        results[model_label] = {
            'min_r2': min_r, 'mean_r2': mean_r, 'max_r2': max_r,
            'top5_r2': top5_v, 'rmse': rmse_v, 'mc_sigma': mc_v,
            'elapsed_min': elapsed / 60,
        }
        print(f"  OK Min={min_r:.4f} Mean={mean_r:.4f} Max={max_r:.4f} ({elapsed/60:.0f}min)")

        with open(results_file, 'a', encoding='utf-8') as log:
            log.write(f"\n{model_label}: Min={min_r:.4f} Mean={mean_r:.4f} Max={max_r:.4f} "
                     f"(RMSE={rmse_v}V, Top5={top5_v}, MCσ={mc_v}V, {elapsed/60:.0f}min)\n")
    else:
        # 尝试找单次评估的 R²
        m2 = re.search(r'R²\s*:\s*([\d\.\-]+)', output)
        if m2:
            print(f"  WARN Loop 失败, 单次 R²={float(m2.group(1)):.4f}")
            results[model_label] = {'error': 'loop_failed', 'single_r2': float(m2.group(1))}
        else:
            print(f"  FAIL 无结果")
            results[model_label] = {'error': 'no_output'}
        # 保存调试输出
        err_path = OUT_DIR / f"debug_{model_label}.txt"
        with open(err_path, 'w', encoding='utf-8') as f:
            f.write(output[-8000:])

    # 每步保存中间结果
    with open(OUT_DIR / "summary.pkl", 'wb') as f:
        pickle.dump(results, f)

# 最终汇总
print("\n\n" + "="*70)
print("FC2 消融实验结果汇总")
print("="*70)
print(f"{'模型':<22s} {'Min R²':<10s} {'Mean R²':<10s} {'Max R²':<10s} {'Top-5':<10s}")
print("-"*62)
for model_label, r in results.items():
    if 'error' in r:
        print(f"{model_label:<22s} ERROR: {r['error']}")
    else:
        print(f"{model_label:<22s} {r['min_r2']:<10.4f} {r['mean_r2']:<10.4f} {r['max_r2']:<10.4f} {r.get('top5_r2','N/A')}")

with open(results_file, 'a', encoding='utf-8') as log:
    log.write(f"\n{'='*60}\n完成: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

print(f"\n结果: {OUT_DIR / 'summary.pkl'}")
print(f"日志: {results_file}")
