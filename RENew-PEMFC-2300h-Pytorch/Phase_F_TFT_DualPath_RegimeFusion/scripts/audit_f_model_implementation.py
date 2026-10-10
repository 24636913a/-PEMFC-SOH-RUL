"""Run Phase-F synthetic-only engineering audits.

This script intentionally never imports a RealHyFC table or a Phase-E cache.
It writes implementation evidence only after the test suite passes.
"""
from __future__ import annotations

import hashlib
import inspect
import json
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

OUTPUTS = PHASE_ROOT / "outputs" / "implementation_audit"
REPORTS = PHASE_ROOT / "reports"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, payload: dict) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    write_json("F_SYNTHETIC_TEST_RESULTS.json", payload)
    if completed.returncode != 0:
        raise RuntimeError("Phase-F synthetic unittest suite failed")
    return payload


def main() -> None:
    run_unittest_suite()
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
    with torch.no_grad():
        native_prediction = native_for_parity.predict_delta(source)
        disabled_prediction = disabled.predict_delta(source)
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
    }
    write_json("F_NATIVE_PARITY_AUDIT.json", parity)
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
    (REPORTS / "F_IMPLEMENTATION_AUDIT.md").write_text(
        "# Phase F Engineering Implementation Audit\n\n"
        "## Result\n\n`PHASE_F_IMPLEMENTATION_READY_FOR_TRAINING_REVIEW`\n\n"
        "All synthetic-only unit tests passed. No RealHyFC table, Assessment target, or formal Test source was opened. The suite exercised native parity, F1 shapes and causal convolution, F2 gate initialization/gradients, decoder placeholder firewall, native encoder-to-decoder state handoff, synthetic backward and synthetic checkpoint reload.\n\n"
        "## Runtime\n\n"
        f"- pytorch-forecasting: {pf.__version__}\n- PyTorch: {torch.__version__}\n- CUDA: {torch.version.cuda}\n"
        f"- `_tft.py` SHA-256: `{observed['_tft.py']}`\n- `sub_modules.py` SHA-256: `{observed['sub_modules.py']}`\n\n"
        "## Measured parameters\n\n"
        f"|Model|Trainable parameters|\n|---|---:|\n|F0|{counts['F0_CanonicalStrictHistoryTFT']}|\n|F1|{counts['F1_DualPathStrictHistoryTFT']}|\n|F2|{counts['F2_RegimeAwareDualPathTFT']}|\n\n"
        "F1 adds a causal multi-scale branch; F2 adds 57 gate parameters. The upstream `forward()` is monolithic, so the project-local F1 subclass mirrors the audited 1.8.0 flow and changes only the post-LSTM encoder representation prior to concatenation with the untouched decoder representation. This copy is version-sensitive and is protected by the dependency manifest.\n\n"
        "## Engineering risks\n\n"
        "The main risk is upstream-source drift. A future run must fail closed if the recorded PTF hashes differ. F1/F2 also add parameters, so future RMSE differences cannot be assigned solely to a mechanism without the frozen parameter/compute ledger. READY means engineering review may be requested; it is not training authorization.\n",
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
    print("PHASE_F_IMPLEMENTATION_READY_FOR_TRAINING_REVIEW")
    print(json.dumps({"parameters": counts, "tests": "PASS", "native_parity": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
