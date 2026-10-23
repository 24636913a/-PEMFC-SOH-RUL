"""Regenerate only Phase PI's corrected audit/report artifacts from committed text data.

This entry point is intentionally posthoc-only: it imports no model, loader,
or training code and reads no raw Development/Test data or prediction binary.
It validates frozen CSV/JSON facts before writing the corrected Markdown and
audit JSON displays.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from posthoc_report_audit import assess_fairness, research_conclusion_text, resource_display


PI = Path(__file__).resolve().parents[1]
BASE_CORRECTION_COMMIT = "cd006de9f25ca7c7b58bd5e56a0d74db980366ce"
FROZEN_EXPECTED = {
    "pdfnet_formal_rmse_mV": 32.21347458631837,
    "timesnet_seed42_formal_rmse_mV": 14.206557830029526,
    "fold_rmse_mV": {
        "fold_1": 28.58706028097499,
        "fold_2": 34.000880645419294,
        "fold_3": 34.052482832560834,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_frozen_decision(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _assert_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Frozen result mismatch for {name}: {actual} != {expected}")


def _markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [
            resource_display(row[column], model=str(row.get("model", "")), column=column)
            for column in frame.columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _activation_rows() -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for fold in ("fold_1", "fold_2", "fold_3"):
        payload = _read_json(PI / "outputs" / "activation" / f"pdfnet_inspired_{fold}_seed42.json")
        value = float(payload["more_than_one_patch_update_fraction"])
        rows.append((fold, value))
    return rows


def _local_binary_cache_status() -> str:
    predictions = list((PI / "outputs" / "predictions").glob("pdfnet_inspired_*_seed42.npz"))
    checkpoints = list((PI / "checkpoints").glob("pdfnet_inspired_*_seed42.pt"))
    if predictions and checkpoints:
        return "LOCAL_BINARY_CACHE_PRESENT_NOT_RECOMPUTED"
    return "RAW_BINARY_CACHE_NOT_AVAILABLE_FOR_INDEPENDENT_RECOMPUTATION"


def main() -> None:
    old_fairness = _read_json(PI / "outputs" / "FAIRNESS_AUDIT.json")
    identity = pd.read_csv(PI / "outputs" / "WINDOW_IDENTITY_AUDIT.csv")
    seed42_results = pd.read_csv(PI / "outputs" / "SEED42_RESULTS.csv")
    summary = pd.read_csv(PI / "outputs" / "SUMMARY_RESULTS.csv")
    seed42_gate = _read_json(PI / "outputs" / "SEED42_GATE.json")
    multiseed_gate = _read_json(PI / "outputs" / "MULTISEED_GATE.json")
    resource = _read_json(PI / "outputs" / "RESOURCE_AUDIT.json")
    decision = _read_frozen_decision(PI / "final" / "FINAL_EXPLORATORY_DECISION.yaml")

    pdfnet_summary = summary.loc[(summary["model"] == "PDFNetInspired_Exploratory") & (summary["seed"] == 42)]
    timesnet_summary = summary.loc[(summary["model"] == "TimesNet") & (summary["seed"] == 42)]
    if len(pdfnet_summary) != 1 or len(timesnet_summary) != 1:
        raise RuntimeError("Frozen summary must contain exactly one seed-42 row for each model")
    _assert_close("PDFNetInspired_Exploratory formal RMSE", pdfnet_summary.iloc[0]["formal_RMSE_mV"], FROZEN_EXPECTED["pdfnet_formal_rmse_mV"])
    _assert_close("TimesNet seed-42 formal RMSE", timesnet_summary.iloc[0]["formal_RMSE_mV"], FROZEN_EXPECTED["timesnet_seed42_formal_rmse_mV"])
    for fold, expected in FROZEN_EXPECTED["fold_rmse_mV"].items():
        row = seed42_results.loc[seed42_results["fold"] == fold]
        if len(row) != 1:
            raise RuntimeError(f"Frozen seed-42 results missing unique {fold}")
        _assert_close(f"PDFNetInspired_Exploratory {fold} RMSE", row.iloc[0]["rmse_mV"], expected)
    if seed42_gate.get("status") != "PDFNET_INSPIRED_NO_PROMISE":
        raise RuntimeError("Frozen seed-42 research status differs from the registered closed result")
    if multiseed_gate != {"status": "NOT_RUN", "reason": "seed42_no_promise", "seed42_status": "PDFNET_INSPIRED_NO_PROMISE"}:
        raise RuntimeError("Frozen multi-seed stop record differs from the registered closed result")
    expected_decision = {
        "status": "PDFNET_INSPIRED_NO_PROMISE",
        "formal_backbone": "TIMESNET_FREEZE",
        "stop_after_phase": "true",
    }
    if {key: decision.get(key) for key in expected_decision} != expected_decision:
        raise RuntimeError("FINAL_EXPLORATORY_DECISION.yaml is not the frozen closed decision")
    if resource.get("status") != "PASS" or resource.get("dimension") != 128 or resource.get("selected_microbatch") != 64:
        raise RuntimeError("Frozen D=128 resource record is inconsistent")

    fairness_checks = assess_fairness(
        identity,
        post_boundary_rows_entered_pipeline=int(old_fairness["post_boundary_rows_entered_pipeline"]),
        test_accessed=bool(old_fairness["test_accessed"]),
        future_covariates=bool(old_fairness["future_covariates"]),
        future_13_variable_supervision=bool(old_fairness["future_13_variable_supervision"]),
        external_decomposition=bool(old_fairness["external_decomposition"]),
    )
    corrected_fairness = {
        **{key: value for key, value in old_fairness.items() if key not in fairness_checks},
        **fairness_checks,
        "posthoc_correction_type": "POSTHOC_REPORT_AND_AUDIT_CORRECTION_ONLY",
        "posthoc_correction_base_commit": BASE_CORRECTION_COMMIT,
        "previous_reported_status": "PASS",
        "previous_reported_timesnet_cache_identity_pass": False,
        "correction_reason": "test_accessed=False was incorrectly included in a generic all-True boolean aggregation",
        "performance_data_regenerated": False,
        "model_training_performed": False,
        "test_accessed_during_correction": False,
    }
    if corrected_fairness["status"] != "PASS":
        raise RuntimeError(f"Posthoc audit is not clean: {corrected_fairness['failed_checks']}")

    activations = _activation_rows()
    activation_lines = "\n".join(
        f"- `{fold}`: more-than-one-inner-patch update fraction = `{fraction:.5f}`."
        for fold, fraction in activations
    )
    candidate = summary.loc[(summary["model"] == "PDFNetInspired_Exploratory") & (summary["seed"] == 42)]
    controls = summary.loc[summary["model"] == "TimesNet"]
    final_status = decision["status"]
    report = f"""# Phase PI final research report (posthoc correction)

## Frozen scope and result

Model: `PDFNetInspired_Exploratory`
Claim: **PDFNet-inspired exploratory architecture**
Formal backbone: **TIMESNET_FREEZE** (unchanged)
Backbone search: **CLOSED**

Phase PI evaluated one preregistered **D=128** PDFNet-inspired configuration. Stage 0 passed on an RTX 3070 Laptop GPU with `selected_microbatch=64` and effective batch 64; no `D=128 -> D=64` resource fallback occurred. This is not an official PDFNet implementation or author reproduction.

## Corrected fairness audit

`FAIRNESS_AUDIT.status = {corrected_fairness['status']}` is now derived from named checks rather than hard-coded. TimesNet cache identity checks all passed for every fold × seed: origin row, target row, truth, anchor, target end, and horizon. The window protocol, Test firewall, future-information firewall, and external-decomposition firewall also passed. The registered 3007-row prefix yielded 3005 Development rows; two boundary rows did not enter legal windows, scalers, fitting, prediction, or scoring. Test targets, predictions, and metrics were not accessed.

## Decision

Seed-42 status: `{seed42_gate['status']}`.
Multi-seed status: `{multiseed_gate['status']}` (`{multiseed_gate['reason']}`).

The preregistered PDFNet-inspired exploratory architecture {research_conclusion_text(final_status)}.

Therefore seeds 2024 and 3407 were not run. `TIMESNET_FREEZE` and `BACKBONE_SEARCH_CLOSED` remain unchanged. This result does not mean that official PDFNet, GPRU, periodic recurrent networks, or dynamic graphs failed on RealHyFC.

## Frozen seed-42 summaries (mV)

{_markdown_table(candidate)}

## Frozen TimesNet controls (mV)

{_markdown_table(controls)}

`peak_gpu_memory_MiB` and `total_inner_plus_refit_time_s` for TimesNet are displayed as **N/A** because Phase PI did not measure them. The historical performance CSV is preserved unchanged.

## Structure activation diagnostic (non-causal)

{activation_lines}

In this frozen D=128 implementation, most outer-period segments contained one inner patch, so continuous cross-patch GPRU recurrent updates were activated only for a limited fraction of recorded patch updates. This is a structural activation diagnostic only; it does not establish that this behavior caused the lower predictive performance and it is not a component-level causal attribution.

## Closure

No retraining, rescoring, Test access, architecture change, D/k/graph/fusion/loss/optimizer change, VMD/EMD decomposition, backbone promotion, or new backbone screening was performed for this correction. The next research boundary is a separately preregistered leakage-safe decomposition and degradation-feature study; it is not executed here.
"""
    note = f"""# Phase PI posthoc correction note

Correction baseline commit: `{BASE_CORRECTION_COMMIT}`
Correction type: `POSTHOC_REPORT_AND_AUDIT_CORRECTION`

## Historical inconsistency corrected

At the correction baseline, `outputs/FAIRNESS_AUDIT.json` reported `status = PASS` while reporting `timesnet_cache_identity_pass = false`. The false cache-identity flag was a reporting-logic error: `test_accessed=False` is a correct safety state, but it was included in a generic all-True boolean aggregation.

The corrected audit explicitly checks the six cache identity fields separately from window conditions and negative safety assertions. All six identity fields are true for every saved fold × seed record; `test_accessed`, future covariates, future 13-variable supervision, and external decomposition are correctly required to be false. Thus the corrected `FAIRNESS_AUDIT.status` remains `PASS`, now for a valid derived reason.

The prior final report also incorrectly said the exploratory architecture “showed sufficient potential” because it treated `multiseed=NOT_RUN` as a positive state. The corrected conclusion is `PDFNET_INSPIRED_NO_PROMISE`: the preregistered PDFNet-inspired exploratory architecture did not show sufficient potential under the frozen RealHyFC protocol.

## What did not change

- No model was trained, no seed-42 run was repeated, and seeds 2024/3407 were not run.
- No formal prediction metric, fold result, gate result, or `FINAL_EXPLORATORY_DECISION.yaml` research state was changed.
- No Test data, Test prediction, or Test metric was accessed.
- `TIMESNET_FREEZE` and `BACKBONE_SEARCH_CLOSED` remain in force.

The correction regenerated only the derived fairness display and final Markdown report. Local binary-cache status: `{_local_binary_cache_status()}`. No cache was recomputed or used to regenerate formal performance data.
"""
    (PI / "outputs" / "FAIRNESS_AUDIT.json").write_text(json.dumps(corrected_fairness, indent=2) + "\n", encoding="utf-8")
    (PI / "final" / "FINAL_RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    (PI / "final" / "POSTHOC_CORRECTION_NOTE.md").write_text(note, encoding="utf-8")
    print(json.dumps({"fairness_status": corrected_fairness["status"], "final_status": final_status, "binary_cache_status": _local_binary_cache_status()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
