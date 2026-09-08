import fs from "node:fs/promises";
import { Workbook } from "file:///C:/Users/1/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = "E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N2_7_Complementarity_and_Robustness";
const relativePaths = [
  "outputs/integrity_audit/paired_prediction_integrity_audit.csv",
  "outputs/error_complementarity/timesnet_win_rate.csv",
  "outputs/error_complementarity/error_correlation.csv",
  "outputs/error_complementarity/large_error_overlap.csv",
  "outputs/fusion_diagnostics/oracle_fusion_upper_bound.csv",
  "outputs/fusion_diagnostics/fixed_fusion_baseline.csv",
  "outputs/shift_diagnostics/feature_shift_smd_psi.csv",
  "outputs/shift_diagnostics/window_shift_score.csv",
  "outputs/shift_diagnostics/shift_conditioned_model_performance.csv",
  "outputs/shift_diagnostics/expert_regime_analysis.csv",
  "outputs/shift_diagnostics/winner_predictability.csv",
  "outputs/timesnet_tuning/timesnet_robustness_tuning.csv",
  "outputs/preprocessing/preprocessing_screening.csv",
  "outputs/multiseed/multiseed_robustness.csv",
  "outputs/error_complementarity/final_complementarity_analysis.csv",
];

const rows = [];
for (const relativePath of relativePaths) {
  const fullPath = `${root}/${relativePath}`;
  const csvText = await fs.readFile(fullPath, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Audit" });
  const inspection = await workbook.inspect({
    kind: "sheet,region",
    sheetId: "Audit",
    range: "A1:Z8",
    maxChars: 2500,
    tableMaxRows: 8,
    tableMaxCols: 26,
    tableMaxCellChars: 100,
  });
  const ndjson = inspection?.ndjson ?? String(inspection);
  rows.push({
    path: relativePath,
    status: ndjson && ndjson.length > 0 ? "PASS" : "FAIL",
    byteCount: Buffer.byteLength(csvText, "utf8"),
    inspectionCharacters: ndjson.length,
    inspectedRange: "A1:Z8",
  });
}

const payload = {
  status: rows.every((row) => row.status === "PASS") && rows.length === 15 ? "PASS" : "FAIL",
  library: "@oai/artifact-tool",
  libraryVersion: "2.8.59",
  csvCount: rows.length,
  method: "Workbook.fromCSV plus bounded workbook.inspect",
  rows,
};
await fs.writeFile(`${root}/outputs/integrity_audit/artifact_tool_csv_audit.json`, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ status: payload.status, csvCount: payload.csvCount }));
