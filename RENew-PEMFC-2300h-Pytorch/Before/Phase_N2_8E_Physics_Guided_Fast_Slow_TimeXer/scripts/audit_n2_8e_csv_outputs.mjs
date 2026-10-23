import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const root = String.raw`E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8E_Physics_Guided_Fast_Slow_TimeXer`;
const files = [
  "outputs/model_screening/N2_8E_seed42_comparison.csv",
  "outputs/ablation/N2_8E_physics_ablation.csv",
  "outputs/horizon_analysis/N2_8E_per_horizon.csv",
  "outputs/model_screening/N2_8E_fold_metrics.csv",
  "outputs/tail_analysis/N2_8E_tail_metrics.csv",
  "outputs/diagnostics/fast_slow_variance_audit.csv",
  "outputs/diagnostics/current_sensitivity_audit.csv",
  "outputs/diagnostics/current_sensitivity_summary.csv",
  "outputs/diagnostics/slow_monotonicity_audit.csv",
  "outputs/model_screening/N2_8E_multiseed_comparison.csv",
  "outputs/diagnostics/physics_loss_training_history.csv",
  "outputs/diagnostics/step_alignment_derivative_audit.csv",
  "outputs/diagnostics/physics_loss_dominance_audit.csv",
];

const results = [];
for (const relative of files) {
  const path = `${root}\\${relative.replaceAll("/", "\\")}`;
  const text = await fs.readFile(path, "utf8");
  const workbook = await Workbook.fromCSV(text, { sheetName: "Audit" });
  const sheet = workbook.worksheets.getItemAt(0);
  const used = sheet.getUsedRange();
  const values = used.values;
  const headers = values[0].map(String);
  const badHeaders = headers.length === 0 || headers.some(x => x.trim() === "") || new Set(headers).size !== headers.length;
  const inspection = await workbook.inspect({ kind: "table", range: "Audit!A1:Z8", maxChars: 1800, tableMaxRows: 8, tableMaxCols: 26 });
  results.push({
    file: relative,
    row_count: Math.max(0, values.length - 1),
    column_count: headers.length,
    header_unique_and_nonempty: !badHeaders,
    artifact_tool_inspected: Boolean(inspection.ndjson),
    parse_status: !badHeaders && values.length >= 2 ? "PASS" : "FAIL",
  });
}
const status = results.every(x => x.parse_status === "PASS") ? "PASS" : "FAIL";
await fs.writeFile(`${root}\\outputs\\diagnostics\\csv_artifact_audit.json`, JSON.stringify({ status, files: results }, null, 2) + "\n", "utf8");
console.log(JSON.stringify({ status, file_count: results.length, failed: results.filter(x => x.parse_status !== "PASS").map(x => x.file) }));
