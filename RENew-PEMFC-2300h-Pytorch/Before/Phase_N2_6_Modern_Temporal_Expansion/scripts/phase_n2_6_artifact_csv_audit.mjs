import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const root = String.raw`E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_6_Modern_Temporal_Expansion`;
const requiredCsvFiles = [
  "outputs/screening_results/modern_architecture_screening.csv",
  "outputs/multi_seed_robustness/multi_seed_backbone_comparison.csv",
  "outputs/per_horizon_analysis/per_horizon_model_comparison.csv",
  "outputs/per_horizon_analysis/cycle_region_model_comparison.csv",
  "outputs/ageing_stage_analysis/ageing_stage_architecture_comparison.csv",
  "outputs/statistical_tests/bootstrap_vs_lstm.csv",
  "outputs/model_efficiency/model_accuracy_efficiency_comparison.csv",
  "outputs/implementation_audit/implementation_sanity_audit.csv",
];

const errorPattern = /#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!|SPILL!|CALC!)/g;
const audited = [];

for (const relativePath of requiredCsvFiles) {
  const absolutePath = path.join(root, ...relativePath.split("/"));
  const csvText = await fs.readFile(absolutePath, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Phase_N2_6_Result" });
  const inspection = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 5000,
    tableMaxRows: 8,
    tableMaxCols: 28,
    tableMaxCellChars: 100,
  });
  const errorInspection = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "Phase N2.6 CSV error scan",
  });
  const inspectionText = inspection?.ndjson ?? JSON.stringify(inspection);
  const errorInspectionText = errorInspection?.ndjson ?? JSON.stringify(errorInspection);
  const errors = [...new Set(csvText.match(errorPattern) ?? [])];
  const nonEmptyLineCount = csvText.split(/\r?\n/).filter((line) => line.trim().length > 0).length;
  audited.push({
    relative_path: relativePath,
    bytes: Buffer.byteLength(csvText, "utf8"),
    data_row_count: Math.max(0, nonEmptyLineCount - 1),
    artifact_tool_import: "PASS",
    formula_error_tokens: errors,
    inspection_preview: inspectionText.slice(0, 1600),
    error_scan_preview: errorInspectionText.slice(0, 800),
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
process.stdout.write(`Phase N2.6 artifact CSV audit: ${output.status} (${audited.length}/${requiredCsvFiles.length})\n`);
if (!allPassed) process.exitCode = 1;
