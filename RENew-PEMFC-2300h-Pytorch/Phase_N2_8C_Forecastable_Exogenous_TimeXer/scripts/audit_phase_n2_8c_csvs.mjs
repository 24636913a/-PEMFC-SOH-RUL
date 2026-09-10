import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const phaseRoot = "E:\\沈农信电\\ PEMFC-SOH+RUL\\RENew-PEMFC-2300h-Pytorch\\Phase_N2_8C_Forecastable_Exogenous_TimeXer";
const outputsRoot = path.join(phaseRoot, "outputs");
const requiredCsvs = [
  "exogenous_forecasting/future_exogenous_forecast_quality.csv",
  "exogenous_forecasting/future_exogenous_forecast_summary.csv",
  "oracle_analysis/timexer_oracle_future_exo_seed42.csv",
  "timexer_screening/timexer_cycle_exo_seed42.csv",
  "timexer_screening/timexer_ridge_exo_seed42.csv",
  "timexer_screening/timexer_exogenous_gain.csv",
  "per_horizon/timexer_future_exo_per_horizon.csv",
  "error_correlation/exo_error_vs_voltage_error.csv",
  "tail_analysis/exo_error_stratified_voltage_metrics.csv",
  "timexer_screening/N2_8C_model_comparison.csv",
  "timexer_screening/N2_8C_multiseed_comparison.csv",
  "timexer_screening/foldwise_timexer_future_exo_metrics.csv",
];

const audits = [];
for (const relativePath of requiredCsvs) {
  const csvPath = path.join(outputsRoot, relativePath);
  const text = await fs.readFile(csvPath, "utf8");
  const workbook = await Workbook.fromCSV(text, { sheetName: "Audit" });
  const sheet = workbook.worksheets.getItem("Audit");
  const usedRange = sheet.getUsedRange(true);
  const values = usedRange.values;
  const headers = (values[0] ?? []).map((value) => String(value));
  const dataRows = values.slice(1).filter((row) => row.some((value) => value !== null && value !== undefined && String(value).trim() !== ""));
  const invalidTokens = [];
  for (let rowIndex = 0; rowIndex < dataRows.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < dataRows[rowIndex].length; columnIndex += 1) {
      const value = dataRows[rowIndex][columnIndex];
      if (typeof value === "string" && /^(nan|inf|-inf)$/i.test(value.trim())) {
        invalidTokens.push({ row: rowIndex + 2, header: headers[columnIndex], value });
      }
    }
  }
  const preview = await workbook.inspect({
    kind: "table",
    range: `Audit!A1:${String.fromCharCode(64 + Math.min(26, headers.length))}${Math.min(6, dataRows.length + 1)}`,
    include: "values",
    tableMaxRows: 6,
    tableMaxCols: Math.min(26, headers.length),
  });
  audits.push({ relative_path: relativePath, column_count: headers.length, data_row_count: dataRows.length, headers, invalid_tokens: invalidTokens, preview: preview.ndjson });
}

const audit = {
  scope: "N2.8C required CSV artifacts, imported and inspected using @oai/artifact-tool Workbook.fromCSV",
  required_csv_count: requiredCsvs.length,
  all_files_present: audits.length === requiredCsvs.length,
  files_with_no_data_rows: audits.filter((entry) => entry.data_row_count === 0).map((entry) => entry.relative_path),
  files_with_invalid_tokens: audits.filter((entry) => entry.invalid_tokens.length > 0).map((entry) => ({ relative_path: entry.relative_path, tokens: entry.invalid_tokens })),
  artifacts: audits,
};
await fs.writeFile(path.join(outputsRoot, "csv_artifact_tool_audit.json"), `${JSON.stringify(audit, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ required_csv_count: audit.required_csv_count, all_files_present: audit.all_files_present, files_with_no_data_rows: audit.files_with_no_data_rows, files_with_invalid_tokens: audit.files_with_invalid_tokens }, null, 2));
