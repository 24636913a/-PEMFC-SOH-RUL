import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "file:///C:/Users/1/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = String.raw`E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_6R_High_Fidelity_Revalidation`;
const requiredCsvFiles = [
  "outputs/implementation_audit/implementation_fidelity_audit.csv",
  "outputs/implementation_audit/lstm_reproduction_check.csv",
  "outputs/screening/high_fidelity_screening.csv",
  "outputs/multiseed/multiseed_comparison.csv",
  "outputs/per_horizon/per_horizon_rmse_r2_mae.csv",
  "outputs/per_horizon/cycle_region_rmse_r2.csv",
  "outputs/ageing_stage/ageing_stage_rmse_r2_mae.csv",
  "outputs/statistical_tests/bootstrap_rmse_difference.csv",
  "outputs/screening/model_efficiency.csv",
];

const errorPattern = /#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!|SPILL!|CALC!)/g;
const audited = [];

for (const relativePath of requiredCsvFiles) {
  const absolutePath = path.join(root, ...relativePath.split("/"));
  const csvText = await fs.readFile(absolutePath, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Phase_N2_6R_Result" });
  const inspection = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 5000,
    tableMaxRows: 8,
    tableMaxCols: 30,
    tableMaxCellChars: 120,
  });
  const errorInspection = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "Phase N2.6R CSV error scan",
  });
  const errors = [...new Set(csvText.match(errorPattern) ?? [])];
  const nonEmptyLineCount = csvText.split(/\r?\n/).filter((line) => line.trim().length > 0).length;
  audited.push({
    relative_path: relativePath,
    bytes: Buffer.byteLength(csvText, "utf8"),
    data_row_count: Math.max(0, nonEmptyLineCount - 1),
    artifact_tool_import: "PASS",
    formula_error_tokens: errors,
    inspection_preview: (inspection?.ndjson ?? JSON.stringify(inspection)).slice(0, 1600),
    error_scan_preview: (errorInspection?.ndjson ?? JSON.stringify(errorInspection)).slice(0, 800),
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

const outputPath = path.join(root, "outputs", "implementation_audit", "artifact_tool_csv_audit.json");
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");
process.stdout.write(`Phase N2.6R artifact CSV audit: ${output.status} (${audited.length}/${requiredCsvFiles.length})\n`);
if (!allPassed) process.exitCode = 1;
