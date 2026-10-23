import fs from "node:fs/promises";
import { Workbook } from "file:///C:/Users/1/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = "E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N2_7R_Horizon_Aware_Gating";
const relativePaths = [
  "outputs/integrity_audit/paired_expert_prediction_audit.csv",
  "outputs/stability/sg_lstm_stability_bootstrap.csv",
  "outputs/gate_diagnostics/pointwise_winner_by_horizon.csv",
  "outputs/gate_diagnostics/expert_disagreement_analysis.csv",
  "outputs/gate_diagnostics/gate_feature_definition.csv",
  "outputs/gate_diagnostics/gate_logistic_state_horizon.csv",
  "outputs/gate_diagnostics/gate_logistic_disagreement.csv",
  "outputs/gate_diagnostics/gate_continuous_mlp.csv",
  "outputs/gate_diagnostics/fixed_fusion_meta_oos.csv",
  "outputs/gate_diagnostics/horizon_static_fusion.csv",
  "outputs/gate_diagnostics/adaptive_gate_meta_oos.csv",
  "outputs/gate_diagnostics/gate_weight_by_horizon.csv",
  "outputs/gate_diagnostics/gate_weight_by_ageing_stage.csv",
  "outputs/gate_diagnostics/gate_weight_by_disagreement.csv",
  "outputs/gate_diagnostics/timesnet_tail_error_gate_analysis.csv",
  "outputs/gate_diagnostics/ophrs2_ablation.csv",
  "outputs/gate_diagnostics/gate_multiseed_summary.csv",
  "outputs/stability/gate_bootstrap_comparison.csv",
];
const rows = [];
for (const relativePath of relativePaths) {
  const fullPath = `${root}/${relativePath}`;
  const text = await fs.readFile(fullPath, "utf8");
  const workbook = await Workbook.fromCSV(text, { sheetName: "Audit" });
  const inspection = await workbook.inspect({
    kind: "sheet,region", sheetId: "Audit", range: "A1:Z8", maxChars: 2500,
    tableMaxRows: 8, tableMaxCols: 26, tableMaxCellChars: 100,
  });
  const inspected = inspection?.ndjson ?? String(inspection);
  rows.push({ path: relativePath, status: text.trim().split(/\r?\n/).length >= 2 && inspected.length > 0 ? "PASS" : "FAIL", byteCount: Buffer.byteLength(text, "utf8"), inspectedRange: "A1:Z8", inspectionCharacters: inspected.length });
}
const payload = { status: rows.length === 18 && rows.every((row) => row.status === "PASS") ? "PASS" : "FAIL", library: "@oai/artifact-tool", method: "Workbook.fromCSV plus bounded workbook.inspect", csvCount: rows.length, rows };
await fs.writeFile(`${root}/outputs/integrity_audit/artifact_tool_csv_audit.json`, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ status: payload.status, csvCount: payload.csvCount }));
