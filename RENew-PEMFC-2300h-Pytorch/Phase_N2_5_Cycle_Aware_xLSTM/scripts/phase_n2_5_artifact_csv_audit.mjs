import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const root = String.raw`E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_5_Cycle_Aware_xLSTM`;
const requiredCsvFiles = [
  "outputs/xlstm_variant_screening/xlstm_variant_screening.csv",
  "outputs/cycle_aware_ablation/cycle_aware_ablation.csv",
  "outputs/cycle_aware_ablation/wrong_period_control.csv",
  "outputs/multi_seed_robustness/multi_seed_architecture_robustness.csv",
  "outputs/cycle_analysis/cycle_region_per_horizon_metrics.csv",
  "outputs/statistical_tests/bootstrap_architecture_comparison.csv",
  "outputs/cycle_analysis/cycle_attention_weight_summary.csv",
  "outputs/cycle_analysis/ageing_stage_model_comparison.csv",
  "outputs/model_comparison/model_efficiency_comparison.csv",
];

const errorPattern = /#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!)/g;
const audited = [];

for (const relativePath of requiredCsvFiles) {
  const absolutePath = path.join(root, ...relativePath.split("/"));
  const csvText = await fs.readFile(absolutePath, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Phase_N2_5_Result" });
  const inspection = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 5000,
    tableMaxRows: 8,
    tableMaxCols: 26,
    tableMaxCellChars: 100,
  });
  const inspectionText = inspection?.ndjson ?? JSON.stringify(inspection);
  const errors = [...new Set(csvText.match(errorPattern) ?? [])];
  const nonEmptyLineCount = csvText.split(/\r?\n/).filter((line) => line.trim().length > 0).length;
  audited.push({
    relative_path: relativePath,
    bytes: Buffer.byteLength(csvText, "utf8"),
    data_row_count: Math.max(0, nonEmptyLineCount - 1),
    artifact_tool_import: "PASS",
    formula_error_tokens: errors,
    inspection_preview: inspectionText.slice(0, 1600),
  });
}

const allPassed = audited.every(
  (item) => item.artifact_tool_import === "PASS" && item.data_row_count > 0 && item.formula_error_tokens.length === 0,
);
const output = {
  status: allPassed ? "PASS" : "FAIL",
  tool: "@oai/artifact-tool Workbook.fromCSV + workbook.inspect",
  audited_file_count: audited.length,
  required_file_count: requiredCsvFiles.length,
  files: audited,
};

const outputPath = path.join(root, "outputs", "audits", "artifact_tool_csv_audit.json");
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");
process.stdout.write(`Phase N2.5 artifact CSV audit: ${output.status} (${audited.length}/${requiredCsvFiles.length})\n`);
if (!allPassed) process.exitCode = 1;
