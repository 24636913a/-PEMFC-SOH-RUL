"""Run Phase-F synthetic-only engineering audits.

This script intentionally never imports a RealHyFC table or a Phase-E cache.
It writes implementation evidence only after the test suite passes.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
import subprocess
import sys

import pytorch_forecasting as pf
import torch
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import pytorch_forecasting.models.temporal_fusion_transformer.sub_modules as sub_modules

PHASE_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PHASE_ROOT.parent
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))
if str(RESEARCH_ROOT / "Phase_TFT_Original_Baseline" / "models") not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT / "Phase_TFT_Original_Baseline" / "models"))

from canonical_tft import CanonicalStrictHistoryTFT, count_parameters  # noqa: E402
from models import DualPathStrictHistoryTFT, RegimeAwareDualPathTFT  # noqa: E402
from models.frozen_feature_schema import PHASE_E_PROTOCOL, protocol_sha256  # noqa: E402
from tests._support import MODEL_KWARGS, make_history, named_state_hash, tensor_hash  # noqa: E402
from tests._forward_parity import (  # noqa: E402
    ATOL, RTOL, OUTPUT_FIELDS, align_shared_state, compare_full_outputs,
    enabled_neutral_fusion_audit, initial_f1_f2_full_forward_audit,
)

OUTPUTS = PHASE_ROOT / "outputs" / "implementation_audit"
REPORTS = PHASE_ROOT / "reports"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, payload: dict) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_readme_status(passed: bool) -> None:
    """Synchronize status on success AND failure without changing history."""
    readme = PHASE_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    status = "PHASE_F_ENABLED_FORWARD_PARITY_PASS" if passed else "PHASE_F_ENABLED_FORWARD_PARITY_BLOCKED"
    review = "READY_FOR_F0_F1_F2_TRAINING_AUTHORIZATION_REVIEW" if passed else status
    text = re.sub(r"Engineering implementation status: `[^`]+`\.",
                  f"Engineering implementation status: `{status}`.", text)
    text = re.sub(r"`[^`]+` is not a training authorization\.",
                  f"`{review}` is not a training authorization.", text)
    readme.write_text(text, encoding="utf-8")


def build_models():
    torch.manual_seed(101)
    f0 = CanonicalStrictHistoryTFT(**MODEL_KWARGS).eval()
    torch.manual_seed(101)
    f1 = DualPathStrictHistoryTFT(**MODEL_KWARGS, dual_path_enabled=True).eval()
    torch.manual_seed(101)
    f2 = RegimeAwareDualPathTFT(**MODEL_KWARGS, dual_path_enabled=True).eval()
    return f0, f1, f2


def run_unittest_suite() -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        cwd=PHASE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "command": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "synthetic_only": True,
        "real_data_accessed": False,
        "assessment_inference_performed": False,
        "test_accessed": False,
    }
    log = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests", log)
    payload["actual_test_count"] = int(match.group(1)) if match else 0
    payload["new_test_count"] = sum(
        bool(re.search(rf"^test_.* \(tests\.{module}\..*\) .*ok$", log, re.MULTILINE))
        for module in ("test_f1_enabled_forward_parity", "test_f2_full_forward_initial_parity")
    )
    payload["original_test_count"] = payload["actual_test_count"] - payload["new_test_count"]
    payload["pass"] = (completed.returncode == 0 and payload["actual_test_count"] == 15
                       and payload["new_test_count"] == 2)
    write_json("F_SYNTHETIC_TEST_RESULTS.json", payload)
    (OUTPUTS / "F_SYNTHETIC_TEST_LOG.txt").write_text(log, encoding="utf-8")
    if not payload["pass"]:
        raise RuntimeError("Phase-F synthetic unittest suite failed")
    return payload


def main() -> None:
    tests = run_unittest_suite()
    tft_path = Path(inspect.getsourcefile(TemporalFusionTransformer))
    sub_path = Path(inspect.getsourcefile(sub_modules.InterpretableMultiHeadAttention))
    expected = {
        "_tft.py": "6ad5f6c806d314cfec53851ce0914c6d3476c4b05bd136a2764f4a562f1b7343",
        "sub_modules.py": "9ef892c565f6891ddf6d21c52c4d7f90393ec731924851aee872468cffe2cf9e",
    }
    observed = {"_tft.py": sha256_file(tft_path), "sub_modules.py": sha256_file(sub_path)}
    if observed != expected or str(pf.__version__) != "1.8.0":
        raise RuntimeError("Frozen upstream TFT dependency identity mismatch")

    f0, f1, f2 = build_models()
    source = make_history()
    torch.manual_seed(101)
    native_for_parity = CanonicalStrictHistoryTFT(**MODEL_KWARGS).eval()
    torch.manual_seed(101)
    disabled = DualPathStrictHistoryTFT(**MODEL_KWARGS, dual_path_enabled=False).eval()
    disabled_state = align_shared_state(native_for_parity, disabled, "history_branch.")
    with torch.no_grad():
        native_output = native_for_parity.forward_history(source)
        disabled_output = disabled.forward_history(source)
    disabled_fields = compare_full_outputs(native_output, disabled_output)
    if not all(row["exact_equal"] for row in disabled_fields.values()):
        raise RuntimeError("Disabled path full outputs are not exactly equal")
    native_prediction = native_output.prediction
    disabled_prediction = disabled_output.prediction
    parity_max_abs = float((native_prediction - disabled_prediction).abs().max())
    if parity_max_abs != 0.0:
        raise RuntimeError(f"Native parity contract failed: {parity_max_abs}")

    counts = {
        "F0_CanonicalStrictHistoryTFT": count_parameters(f0),
        "F1_DualPathStrictHistoryTFT": count_parameters(f1),
        "F2_RegimeAwareDualPathTFT": count_parameters(f2),
        "F1_minus_F0": count_parameters(f1) - count_parameters(f0),
        "F2_minus_F1": count_parameters(f2) - count_parameters(f1),
        "synthetic_only": True,
    }
    if (count_parameters(f0), count_parameters(f1), count_parameters(f2)) != (323483, 332379, 332436):
        raise RuntimeError("Frozen parameter counts changed")
    write_json("F_MODEL_PARAMETER_COUNTS.json", counts)
    initialization = {
        "shared_tft_state_hash_f0": named_state_hash(f0, ""),
        "shared_tft_state_hash_f1_base_keys_match_f0": all(
            torch.equal(value, f1.state_dict()[name]) for name, value in f0.state_dict().items()
        ),
        "shared_tft_state_hash_f2_base_keys_match_f0": all(
            torch.equal(value, f2.state_dict()[name]) for name, value in f0.state_dict().items()
        ),
        "history_branch_hash_f1": named_state_hash(f1, "history_branch."),
        "history_branch_hash_f2": named_state_hash(f2, "history_branch."),
        "gate_output_weight_all_zero": bool(torch.equal(f2.regime_gate.output.weight, torch.zeros_like(f2.regime_gate.output.weight))),
        "gate_output_bias_all_zero": bool(torch.equal(f2.regime_gate.output.bias, torch.zeros_like(f2.regime_gate.output.bias))),
    }
    write_json("F_INITIALIZATION_AUDIT.json", initialization)
    parity = {
        "native_parity_mode": "DualPathStrictHistoryTFT(dual_path_enabled=False) delegates to super().forward",
        "input_shape": list(source.shape),
        "native_prediction_sha256": tensor_hash(native_prediction),
        "disabled_prediction_sha256": tensor_hash(disabled_prediction),
        "max_absolute_difference": parity_max_abs,
        "pass": True,
        "mode": "eval",
        "shared_state": disabled_state,
        "outputs": disabled_fields,
        "exact_equality_required": True,
        "enabled_path_coverage": False,
        "enabled_path_evidence": "F_ENABLED_FORWARD_PARITY_AUDIT.json",
    }
    write_json("F_NATIVE_PARITY_AUDIT.json", parity)
    enabled = {
        "status": "PHASE_F_ENABLED_FORWARD_PARITY_PASS",
        "atol": ATOL, "rtol": RTOL,
        "floating_exact_equality_checked_first": True,
        "integer_equality_required": "exact",
        "F1_enabled_neutral_fusion": enabled_neutral_fusion_audit(),
        "F1_F2_initial_full_forward": initial_f1_f2_full_forward_audit(),
        "actual_test_count": tests["actual_test_count"],
        "original_test_count": tests["original_test_count"],
        "new_test_count": tests["new_test_count"],
        "formal_model_code_changed": False,
        "synthetic_only": True,
        "realhyfc_training_performed": False,
        "assessment_inference_performed": False,
        "test_accessed": False,
        "pass": True,
    }
    write_json("F_ENABLED_FORWARD_PARITY_AUDIT.json", enabled)
    manifest = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "pytorch_forecasting": str(pf.__version__),
        "frozen_feature_schema_source": str(PHASE_E_PROTOCOL),
        "frozen_feature_schema_sha256": protocol_sha256(),
        "frozen_feature_schema_data_read": False,
        "upstream_source_paths": {"_tft.py": str(tft_path), "sub_modules.py": str(sub_path)},
        "upstream_source_sha256": observed,
        "expected_source_sha256": expected,
        "realhyfc_data_read": False,
        "assessment_model_inference": False,
        "test_accessed": False,
        "formal_training_performed": False,
        "synthetic_checkpoint_only": True,
    }
    write_json("F_RUNTIME_DEPENDENCY_MANIFEST.json", manifest)

    REPORTS.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    for label, audit in (("Disabled path", {"cases": [{"input_shape": list(source.shape), "outputs": disabled_fields}]}),
                         ("Enabled neutral fusion", enabled["F1_enabled_neutral_fusion"]),
                         ("F1/F2 initial full forward", enabled["F1_F2_initial_full_forward"])):
        for case in audit["cases"]:
            values = [str(case["outputs"][field]["max_absolute_difference"]) for field in OUTPUT_FIELDS]
            comparison_rows.append(f"|{label}|{case['input_shape']}|" + "|".join(values) + "|PASS|")
    parity_table = ("|Comparison (eval)|Input shape|" + "|".join(OUTPUT_FIELDS) + "|Result|\n"
                    + "|---|---|" + "|".join(["---:"] * len(OUTPUT_FIELDS)) + "|---|\n"
                    + "\n".join(comparison_rows) + "\n\n")
    all_exact = all(row["exact_equal"]
                    for audit in (enabled["F1_enabled_neutral_fusion"], enabled["F1_F2_initial_full_forward"])
                    for case in audit["cases"] for row in case["outputs"].values())
    exact_statement = ("All measured full outputs are exactly equal" if all_exact
                       else "Some measured outputs differ within the predeclared tolerance")
    (REPORTS / "F_IMPLEMENTATION_AUDIT.md").write_text(
        "# Phase F Engineering Implementation Audit\n\n"
        "## Result\n\n`PHASE_F_ENABLED_FORWARD_PARITY_PASS`\n\n"
        "`READY_FOR_F0_F1_F2_TRAINING_AUTHORIZATION_REVIEW` is engineering review readiness, not training authorization.\n\n"
        f"Actual suite: {tests['actual_test_count']} tests passed ({tests['original_test_count']} original + {tests['new_test_count']} new). Full execution log: `outputs/implementation_audit/F_SYNTHETIC_TEST_LOG.txt`. No RealHyFC table, Assessment target, or formal Test source was opened. The original suite exercised disabled native parity, F1 shapes/causal convolution, F2 local gate/fusion gradients, decoder firewall, native state handoff, synthetic backward and checkpoint reload; it did not establish enabled full-forward parity. The two new tests close that coverage gap.\n\n"
        "## Three distinct forward comparisons\n\n"
        "Disabled parity delegates to native forward and is not evidence for custom forward. Enabled neutral-fusion parity uses a TEST-ONLY subclass returning the LSTM representation at fusion; the formal history branch still executes once per case. F1/F2 full parity runs both formal enabled models, including the gate once per case, with exact initial g=0.5. A patched inherited forward raises if either enabled candidate attempts native fallback. Each enabled comparison uses input seeds 219/731 and batch sizes 1/3.\n\n"
        "All shared named parameters and buffers are checked for identical initial values, names and shapes, explicitly copied and checked again. F0-to-F1 permits only history_branch extra state; F1-to-F2 permits only regime_gate extra state. VSN, LSTM encoder/decoder, GRN/GLU, attention, output head and shared convolution state are included. Complete key lists and hook counts are in `F_ENABLED_FORWARD_PARITY_AUDIT.json`. Formal model code and frozen architecture were unchanged.\n\n"
        f"Predeclared float tolerances: atol={ATOL}, rtol={RTOL}; exact equality checked first, integer/length fields require exact equality. {exact_statement}; empty static_variables tensors are structurally equal with recorded max difference 0. Eval-only: additional dropout RNG consumption precludes a claim of training-mode bitwise parity.\n\n"
        + parity_table +
        "## Runtime\n\n"
        f"- pytorch-forecasting: {pf.__version__}\n- PyTorch: {torch.__version__}\n- CUDA: {torch.version.cuda}\n"
        f"- `_tft.py` SHA-256: `{observed['_tft.py']}`\n- `sub_modules.py` SHA-256: `{observed['sub_modules.py']}`\n\n"
        "## Measured parameters\n\n"
        f"|Model|Trainable parameters|\n|---|---:|\n|F0|{counts['F0_CanonicalStrictHistoryTFT']}|\n|F1|{counts['F1_DualPathStrictHistoryTFT']}|\n|F2|{counts['F2_RegimeAwareDualPathTFT']}|\n\n"
        "F1 adds a causal multi-scale branch; F2 adds 57 gate parameters. The upstream `forward()` is monolithic, so the project-local F1 subclass mirrors the audited 1.8.0 flow and changes only the post-LSTM encoder representation prior to concatenation with the untouched decoder representation. This copy is version-sensitive and is protected by the dependency manifest.\n\n"
        "## Engineering risks\n\n"
        "The main risk is upstream-source drift. This generator fails closed on tests, dependency identity, parity or parameter-count failure and replaces its success status with BLOCKED. F1/F2 add parameters, so future differences cannot be assigned solely to a mechanism without the frozen parameter/compute ledger.\n\n"
        "Historical limits remain: `PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS`, `C1_D0_E0_REPRODUCIBILITY_UNRESOLVED`, and `PHASE_F_TFT_ARCHITECTURE_EXPLORATION_ONLY`. TimesNet remains frozen and backbone search closed. Synthetic parity does not resolve historical numerical reproducibility or authorize formal training.\n\n"
        "## Reproduction\n\nRun `python -B scripts/audit_f_model_implementation.py` from this phase directory. It executes the entire unittest suite, repeats all parity comparisons, and regenerates this report and JSON evidence. No dataset/cache loaders are called.\n",
        encoding="utf-8",
    )
    (REPORTS / "F_DESIGN_REVISION_REPORT.md").write_text(
        "# Phase F Design Revision Record\n\n"
        "|Issue|Revision|Reason and evidence|Structural effect|Conclusion effect|\n|---|---|---|---|---|\n"
        "|Inherited Assessment selection|`PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS`|Phase-E E2 was selected after exposed Assessment RMSE (7.785821 mV), despite its Validation RMSE (6.064596 mV) exceeding E0 Raw (4.797507 mV).|None|F0–F2 can only be Development exploratory.|\n"
        "|C1/D0/E0 disagreement|`C1_D0_E0_REPRODUCIBILITY_UNRESOLVED`|Known aligned keys/truths but divergent outputs: 11.445854, 13.169241, 19.318765 mV. No verified cause is asserted.|None|No cross-stage score claim or causal attribution.|\n"
        "|BiLSTM wording|Corrected|Bidirectional encoding within a wholly observed history window is not intrinsically future leakage. It is excluded here because F1 preserves the native encoder-to-decoder handoff, tests LSTM+causal-convolution complementarity, and avoids reopening encoder-replacement search.|None|No claim that BiLSTM is inherently invalid.|\n"
        "|TimesNet relationship|`PHASE_F_TFT_ARCHITECTURE_EXPLORATION_ONLY`|TimesNet freeze/backbone-search closure remains in force. This is a user-authorized TFT architecture branch only.|None|F0–F2 cannot automatically replace TimesNet or create a cross-protocol ranking.|\n"
        "|Gate interpretation|Data-driven fusion only|F2 gate is a scalar mixture based on observed history; it is not a physical constraint or mechanism detector.|F2 only|Gate values are diagnostics, not mechanism evidence.|\n\n"
        "No formal model was trained for this report; all evidence is code/source review plus synthetic tensors.\n",
        encoding="utf-8",
    )
    update_readme_status(passed=True)
    print("PHASE_F_ENABLED_FORWARD_PARITY_PASS")
    print("READY_FOR_F0_F1_F2_TRAINING_AUTHORIZATION_REVIEW")
    print(json.dumps({"parameters": counts, "tests": tests['actual_test_count'], "native_parity": "PASS", "enabled_parity": "PASS"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = {"status": "PHASE_F_ENABLED_FORWARD_PARITY_BLOCKED", "pass": False,
                   "error_type": type(exc).__name__, "error": str(exc),
                   "synthetic_only": True, "formal_training_performed": False,
                   "assessment_inference_performed": False, "test_accessed": False}
        write_json("F_ENABLED_FORWARD_PARITY_AUDIT.json", failure)
        write_json("F_NATIVE_PARITY_AUDIT.json", failure)
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "F_IMPLEMENTATION_AUDIT.md").write_text(
            "# Phase F Engineering Implementation Audit\n\n"
            "`PHASE_F_ENABLED_FORWARD_PARITY_BLOCKED`\n\n"
            f"{type(exc).__name__}: {exc}\n\nNo training readiness is established. "
            "See F_SYNTHETIC_TEST_RESULTS.json and F_SYNTHETIC_TEST_LOG.txt for test evidence.\n",
            encoding="utf-8")
        update_readme_status(passed=False)
        print("PHASE_F_ENABLED_FORWARD_PARITY_BLOCKED", file=sys.stderr)
        raise
