"""CPU-only frozen-origin coverage report and publication figure 10.

Reads frozen CSVs and historical memmaps mode=r, never trains or changes core,
protocol, source data or historical outputs. Outputs only report 01 and figure
10 PNG/PDF/plot-data CSV in this new study.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
OLD = STUDY.parent / "RealHyFC_Raw1s_Dynamic_Study"
OUT = STUDY / "outputs"
FIG = STUDY / "figures"
ZERO_A = 0.1
STATE_NAMES = {
    "stable_loaded": "全段保持带载代理",
    "stable_low_zero": "全段保持低/零流代理",
    "loaded_to_low_zero": "出现带载→低/零流",
    "low_zero_to_loaded": "出现低/零流→带载",
    "both_crossings": "同一窗口双向切换",
}
CATEGORY_NAMES = {
    "loaded_to_low_or_zero": "带载进入低/零流",
    "low_or_zero_to_loaded": "低/零流恢复带载",
    "current_or_command_change": "电流/命令/模式变化",
    "temperature_or_pressure_change": "温度或压力变化",
    "no_above_threshold_observed_control_change": "未触发已定义控制变化规则",
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            result.update(block)
    return result.hexdigest()


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def event_coverage(origins, horizon, trigger_rows, trigger_event):
    result = collections.Counter()
    event_origins = 0
    for origin in origins:
        left = np.searchsorted(trigger_rows, origin + 1)
        right = np.searchsorted(trigger_rows, origin + horizon + 1)
        found = np.unique(trigger_event[left:right])
        event_origins += int(len(found) > 0)
        result.update(found.tolist())
    return result, event_origins


def current_windows(origins, horizon, current):
    values = np.asarray(current[origins[:, None] + np.arange(horizon + 1)], dtype=float)
    low = np.abs(values) <= ZERO_A
    down = ((~low[:, :-1]) & low[:, 1:]).any(axis=1)
    up = (low[:, :-1] & (~low[:, 1:])).any(axis=1)
    return collections.Counter(np.select(
        [down & up, down, up, low.all(axis=1)],
        ["both_crossings", "loaded_to_low_zero", "low_zero_to_loaded", "stable_low_zero"],
        default="stable_loaded",
    ).tolist())


def mode_counts(values):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    counts = collections.Counter(values.tolist())
    return {format(key, ".8g"): value for key, value in sorted(counts.items())}


def main():
    cfg = json.loads((OUT / "frozen_protocol.json").read_text(encoding="utf-8"))
    sources = ["selected_train_origins.csv", "train_candidate_origins_60s.csv", "event_origin_coverage.csv", "event_catalog.csv", "event_triggers.csv"]
    for name in sources:
        expected = cfg["frozen_file_sha256"]["outputs/" + name]
        assert sha(OUT / name) == expected, name
    selected = pd.read_csv(OUT / sources[0])
    candidates = pd.read_csv(OUT / sources[1])
    declared = pd.read_csv(OUT / sources[2])
    catalog = pd.read_csv(OUT / sources[3])
    triggers = pd.read_csv(OUT / sources[4])
    train_catalog = catalog.loc[catalog.partition.eq("TRAIN")].copy()
    train_events = set(train_catalog.event_id.astype(int))
    train_triggers = triggers.loc[triggers.event_id.isin(train_events)]
    tr = train_triggers.trigger_row.to_numpy(dtype=np.int64)
    te = train_triggers.event_id.to_numpy(dtype=np.int64)
    assert np.all(np.diff(tr) > 0)
    new = selected.origin.to_numpy(dtype=np.int64)
    assert len(new) == cfg["sampling"]["actual_total"] and len(new) == len(np.unique(new))
    assert selected.legal_H30.all() and selected.legal_H120.all()
    common = candidates.loc[candidates.legal_H120].origin.to_numpy(dtype=np.int64)
    assert set(new).issubset(set(common))
    schema = json.loads((OLD / "outputs/cache_schema.json").read_text(encoding="utf-8"))
    n = schema["rows"]
    x = np.memmap(OLD / "cache/x.bin", mode="r", dtype="float32", shape=(n, 24))
    valid = np.memmap(OLD / "cache/valid.bin", mode="r", dtype="uint8", shape=(n,))
    feature = {name: idx for idx, name in enumerate(cfg["input_features"])}
    current = x[:, feature["actual_current_A"]]
    mode = x[:, feature["operation_point"]]
    cut = cfg["seams"][0]["after_raw_row"]
    train_rows = np.flatnonzero(np.asarray(valid[:cut], dtype=bool))
    assert len(train_rows) > 0
    old = {h: np.load(OLD / f"outputs/origins_H{h}.npz")["TRAIN"].astype(np.int64) for h in (30, 120)}
    origin_sets = {"TRAIN_VALID_ROWS": train_rows, "COMMON_LEGAL_60S": common, "NEW_8192": new,
                   "OLD_2048_H30": old[30], "OLD_2048_H120": old[120]}
    selection = selected.selection.value_counts().to_dict()
    coverage = {}
    table = []
    tidy = []

    def record(panel, metric, sampling, category, value, denominator=None, horizon=None, unit="count"):
        tidy.append(dict(panel=panel, metric=metric, horizon_seconds=horizon, sampling=sampling,
                         category=category, value=value, denominator=denominator, unit=unit))

    for h in (30, 120):
        for label, values, declared_label in (("OLD_2048", old[h], "OLD_2048"), ("NEW_8192", new, "NEW_TRAIN")):
            counts, hit_origins = event_coverage(values, h, tr, te)
            expected = declared.loc[(declared.horizon == h) & declared.sampling.eq(declared_label)]
            expected_map = dict(zip(expected.event_id.astype(int), expected.origins_covering_event.astype(int)))
            assert dict(counts) == expected_map, (h, label, "Frozen coverage disagrees with independently counted trigger hits")
            count_values = np.array(list(counts.values()), dtype=int)
            coverage[h, label] = dict(counts=counts, origins=len(values), hit_origins=hit_origins,
                                      covered=len(counts), pairs=int(count_values.sum()),
                                      repeat_median=float(np.median(count_values)) if len(count_values) else 0,
                                      repeat_p95=float(np.quantile(count_values, .95)) if len(count_values) else 0,
                                      repeat_max=int(count_values.max()) if len(count_values) else 0)
            item = coverage[h, label]
            table.append([h, label, len(values), hit_origins, len(counts), f"{100*len(counts)/len(train_events):.2f}%",
                          item["pairs"], f"{item['repeat_median']:.1f}", f"{item['repeat_p95']:.1f}", item["repeat_max"]])
            for key in ("covered", "origins", "hit_origins", "pairs", "repeat_median", "repeat_p95", "repeat_max"):
                record("a" if key == "covered" else "report", key, label, "all_train_episodes", item[key], len(train_events), h)
            for category in ("0", "1", "2", "3", "4", "5+"):
                repeats = len(train_events) - len(counts) if category == "0" else int((count_values >= 5).sum()) if category == "5+" else int((count_values == int(category)).sum())
                record("c" if h == 120 and category != "0" else "report", "episodes_by_origin_repetition", label, category, repeats, len(train_events), h)
            for name, count in current_windows(values, h, current).items():
                record("report", "origin_to_target_current_state", label, name, count, len(values), h)

    for name, count in selection.items():
        record("b", "selection_composition", "NEW_8192", name, int(count), len(new))
    modes = {}
    low_fractions = {}
    for label, values in origin_sets.items():
        low_n = int((np.abs(current[values]) <= ZERO_A).sum())
        low_fractions[label] = low_n / len(values)
        modes[label] = mode_counts(mode[values])
        for state, count in (("loaded", len(values) - low_n), ("low_zero", low_n)):
            record("d", "origin_current_state", label, state, count, len(values))
        for state, count in modes[label].items():
            record("report", "origin_operation_point", label, state, count, len(values))
    classified_path = OUT / "diagnostics/classified_events.csv"
    classes = None
    if classified_path.exists():
        classified = pd.read_csv(classified_path)
        classes = train_catalog[["event_id"]].merge(classified[["event_id", "category"]], on="event_id", how="left", validate="one_to_one")
        assert classes.category.notna().all()
        for category, group in classes.groupby("category"):
            group_events = set(group.event_id.astype(int))
            record("report", "episode_category_total", "ALL_TRAIN_EPISODES", category, len(group_events), len(train_events))
            for (h, label), item in coverage.items():
                record("report", "episode_category_covered", label, category, len(group_events & set(item["counts"])), len(group_events), h)

    FIG.mkdir(parents=True, exist_ok=True)
    plot_csv = FIG / "10_training_coverage_plot_data.csv"
    plot_frame = pd.DataFrame(tidy)
    plot_frame.to_csv(plot_csv, index=False, encoding="utf-8-sig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    fallback_fonts = [name for name in ("Arial", "Helvetica", "DejaVu Sans") if name in available_fonts]
    plt.rcParams.update({"font.family": fallback_fonts, "font.size": 13,
                         "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2,
                         "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.1))
    blue, gray, green = "#0F4D92", "#CFCECE", "#8BCF8B"
    ax = axes[0, 0]
    pos = np.arange(2)
    for offset, label, color, hatch in ((-.18, "OLD_2048", gray, "//"), (.18, "NEW_8192", blue, "")):
        bars = ax.bar(pos + offset, [coverage[h, label]["covered"] for h in (30, 120)], .34,
                      color=color, edgecolor="black", linewidth=1, hatch=hatch, label=label.replace("_", " "))
        ax.bar_label(bars, padding=4, fontsize=11)
    ax.set(xticks=pos, xticklabels=["30 s", "120 s"], ylabel="Distinct TRAIN episodes covered",
           title="(a) Recorded episode coverage", ylim=(0, max(c["covered"] for c in coverage.values()) * 1.24))
    ax.legend(loc="upper left", fontsize=10)
    ax = axes[0, 1]
    labels = ["UNIFORM_75_PERCENT", "EVENT_SUPPLEMENT", "UNIFORM_SHORTFALL_FALLBACK"]
    values = [selection.get(name, 0) for name in labels]
    bottom = 0
    for label, value, color, hatch in zip(["Uniform coverage", "Event supplement", "Uniform fallback"], values, [blue, green, gray], ["", "//", ".."]):
        if value:
            ax.barh([0], [value], left=bottom, color=color, edgecolor="black", hatch=hatch, label=label)
            direct_name = {"Uniform coverage": "Uniform", "Event supplement": "Event", "Uniform fallback": "Fallback"}[label]
            ax.text(bottom + value/2, 0, f"{direct_name}\n{value} ({100*value/len(new):.0f}%)", ha="center", va="center", fontsize=12,
                    color="white" if color == blue else "black")
            bottom += value
    ax.set(yticks=[0], yticklabels=["New origins"], xlabel="Fixed selected TRAIN origins", title="(b) Frozen sampling composition", xlim=(0, len(new)*1.02))
    ax = axes[1, 0]
    bins = ["1", "2", "3", "4", "5+"]
    for offset, label, color, hatch in ((-.18, "OLD_2048", gray, "//"), (.18, "NEW_8192", blue, "")):
        items = plot_frame[(plot_frame.panel == "c") & plot_frame.sampling.eq(label)].set_index("category")
        ax.bar(np.arange(5)+offset, [items.loc[name, "value"] for name in bins], .34, color=color, edgecolor="black", linewidth=1, hatch=hatch, label=label.replace("_", " "))
    ax.set(xticks=np.arange(5), xticklabels=bins, xlabel="Origins covering the same episode", ylabel="Number of covered TRAIN episodes",
           title="(c) Repeated coverage at 120 s")
    ax.legend(fontsize=10)
    ax = axes[1, 1]
    labels = ["TRAIN_VALID_ROWS", "COMMON_LEGAL_60S", "OLD_2048_H120", "NEW_8192"]
    zero_pct = np.array([100*low_fractions[name] for name in labels])
    ax.bar(np.arange(4), 100-zero_pct, color=blue, edgecolor="black", label="Loaded proxy")
    ax.bar(np.arange(4), zero_pct, bottom=100-zero_pct, color=gray, edgecolor="black", hatch="//", label="Low/zero-current proxy")
    for idx, pct in enumerate(zero_pct):
        if pct >= 7:
            ax.text(idx, 100-pct/2, ("Low/zero\n" if idx == 0 else "") + f"{pct:.1f}%", ha="center", va="center", fontsize=10)
        ax.text(idx, (100-pct)/2, ("Loaded\n" if idx == 0 else "") + f"{100-pct:.1f}%", ha="center", va="center", fontsize=10, color="white")
    ax.set(xticks=np.arange(4), xticklabels=["TRAIN\nvalid rows", "Legal\n60 s grid", "Old 2048\nH120", "New\n8192"],
           ylabel="Origin / row share (%)", ylim=(0, 100), title="(d) Observed current-state composition")
    for ax in axes.ravel():
        ax.tick_params(width=1.5, length=5)
    fig.tight_layout(pad=2, h_pad=2.7, w_pad=2)
    fig.savefig(FIG / "10_training_coverage.png", dpi=300, facecolor="white")
    fig.savefig(FIG / "10_training_coverage.pdf", dpi=300, facecolor="white")
    plt.close(fig)

    repeat_rows = []
    for h in (30, 120):
        for label in ("OLD_2048", "NEW_8192"):
            counts = np.array(list(coverage[h, label]["counts"].values()))
            repeat_rows.append([h, label, len(train_events)-len(counts), *[int((counts == k).sum()) for k in (1, 2, 3, 4)], int((counts >= 5).sum())])
    state_rows = []
    for label, values in origin_sets.items():
        state_rows.append([label, len(values), f"{100*(1-low_fractions[label]):.3f}%", f"{100*low_fractions[label]:.3f}%",
                           f"{100*(low_fractions[label]-low_fractions['COMMON_LEGAL_60S']):+.3f}"])
    mode_rows = []
    for state in sorted(set().union(*(set(value) for value in modes.values())), key=float):
        mode_rows.append([state, *[f"{100*modes[label].get(state, 0)/len(origin_sets[label]):.3f}%" for label in origin_sets]])
    window_rows = []
    for h in (30, 120):
        for label in ("OLD_2048", "NEW_8192"):
            counts = current_windows(old[h] if label == "OLD_2048" else new, h, current)
            denominator = coverage[h, label]["origins"]
            window_rows.append([h, label, *[f"{counts[name]} ({100*counts[name]/denominator:.2f}%)" for name in STATE_NAMES]])
    category_section = "分类诊断文件尚未生成。本报告不以占位类别代替实际结果；待 `outputs/diagnostics/classified_events.csv` 生成后可重运行本脚本补充。"
    if classes is not None:
        class_rows = []
        for category, group in classes.groupby("category"):
            group_events = set(group.event_id.astype(int))
            row = [CATEGORY_NAMES.get(category, category), len(group_events)]
            for h in (30, 120):
                for label in ("OLD_2048", "NEW_8192"):
                    hits = len(group_events & set(coverage[h, label]["counts"]))
                    row.append(f"{hits} ({100*hits/len(group_events):.2f}%)")
            class_rows.append(row)
        category_section = md_table(["冻结回顾性类别", "Train episode总数", "旧H30", "新H30", "旧H120", "新H120"], class_rows)
        category_section += "\n\n类别来自 `outputs/diagnostics/classified_events.csv` 的已冻结±30秒回顾性规则，只描述记录联系，不能作为推理输入，也不能视为硬件启停或因果机制真值。未触发已定义变化规则不等于没有物理变化。普通带载与稳定低/零流不是另建的电压episode类型，在前表单独统计。"
    report = f"""# 训练样本及突变覆盖分析

## 结论与统计对象

本轮实际得到{len(new):,}个共同合法Train起点，两跨度与两模型使用同一集合。其中均匀时间覆盖{selection.get('UNIFORM_75_PERCENT', 0):,}个，事件补充{selection.get('EVENT_SUPPLEMENT', 0):,}个，均匀补足{selection.get('UNIFORM_SHORTFALL_FALLBACK', 0):,}个。候选60秒网格有{len(candidates):,}个H30合法起点，H30/H120共同合法候选{len(common):,}个。没有补造样本或为成绩删除突变。

Train共记录{len(train_events):,}个电压episode、{len(train_triggers):,}个超过冻结阈值的原始单秒跳变。episode按同段、同分区、相邻触发间隔不超过120秒合并。它是观测代理，不能因命名“独立事件”就推断物理上独立或等同一次硬件启停。覆盖是预测起点后1..H目标内实际出现至少一次该episode触发；同一起点多次命中同一episode只计1次。

## 旧2048与新8192的实际覆盖

{md_table(['跨度/s', '起点方案', '起点数', '含事件的起点', '覆盖不同episode', '占全部Train episode', 'episode×起点命中数', '已覆盖episode重复P50', 'P95', '最大'], table)}

新旧episode命中已用原始触发清单独立重算，并逐项对照冻结 `event_origin_coverage.csv` 一致。覆盖更多不自动等于模型误差更低；训练和评价结果另报。

{md_table(['跨度/s', '起点方案', '未覆盖episode', '恰好1个起点', '2个', '3个', '4个', '至少5个'], repeat_rows)}

这些是重复窗口覆盖次数，不是独立试验重复。一次episode持续越长，可能对应更多合法起点。事件补充阶段每episode最多选一个代表起点，均匀阶段仍可能命中同一episode，因此最终重复数不能误写成全部等于1。

## 原始运行状态覆盖与偏差

低/零流代理固定为 `abs(I.S.act)<=0.1 A`，其余有限值称带载代理。第一行基线是Train内模型有效原始行，第二行是共同合法60秒候选起点，两者分母不同。偏差列相对于共同合法候选起点比例，单位是百分点。

{md_table(['集合', '行/起点数', '带载代理', '低/零流代理', '低/零流相对候选偏差/百分点'], state_rows)}

`operation_point` 为原始记录模式码，没有把未知码映射成未经验证的物理模式名称。各模式在起点处的比例为：

{md_table(['模式码', *origin_sets.keys()], mode_rows)}

窗口电流状态使用起点至最后目标的真实记录作事后覆盖统计。双向切换单独列出，不强行归入一个方向；此表不作为模型输入或重新挑选样本：

{md_table(['跨度/s', '起点方案', *STATE_NAMES.values()], window_rows)}

## 不同事件类型的覆盖

{category_section}

## 图、数据与解释边界

图10为 `figures/10_training_coverage.png`（300dpi）和同名PDF；原始绘图及本报告详细统计为 `figures/10_training_coverage_plot_data.csv`。图(a)比较事件覆盖，(b)展示固定采样组成，(c)显示120秒任务的重复覆盖，(d)展示观测电流状态偏差。图形采用scientific-figure-making规范的无衬线字体、蓝/灰/绿配色、黑边与纹理，并保留PDF矢量文字。

新均匀集合与事件补充相加后，不再保证最终分布与Train自然时间占比一致。事件采样和B阶段权重3共同强调事件，解释误差变化时须同时看全样本、普通区间和不同事件。所有候选/选中清单在新模型结果前冻结；本脚本只读这批记录和旧缓存，不作采样调整、调参或训练。约1Hz原始记录、逐秒目标与每60秒起点是三个不同概念。

本报告只说明Train覆盖，不能借此声称Validation有效样本彼此独立，也不解释已暴露后留案例以制定新规则。训练仍只有Seed202，结果不具有多种子稳定性证据。AI协助实现和撰写，表中数值由本脚本对冻结产物独立统计，研究者需结合原始记录复核。
"""
    (STUDY / "01_训练样本及突变覆盖分析.md").write_text(report, encoding="utf-8", newline="\n")
    print(json.dumps(dict(status="COVERAGE_REPORT_COMPLETE_NO_TRAINING", new_origins=len(new), train_episodes=len(train_events),
                          figure=str(FIG / "10_training_coverage.png"), report=str(STUDY / "01_训练样本及突变覆盖分析.md")), ensure_ascii=False))


if __name__ == "__main__":
    main()
