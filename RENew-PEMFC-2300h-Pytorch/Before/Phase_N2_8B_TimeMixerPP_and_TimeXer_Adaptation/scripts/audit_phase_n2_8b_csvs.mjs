import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const phaseRoot = "E:\\沈农信电\\ PEMFC-SOH+RUL\\RENew-PEMFC-2300h-Pytorch\\Phase_N2_8B_TimeMixerPP_and_TimeXer_Adaptation";
const outputsRoot = path.join(phaseRoot, "outputs");
const requiredCsvs = [
  "foldwise_delta_metrics.csv",
  "lifecycle_stability_comparison.csv",
  "N2_8B_multiseed_comparison.csv",
  "N2_8B_seed42_model_comparison.csv",
  "per_horizon_delta_metrics.csv",
  "tail_error_comparison.csv",
  "model_screening/timemixerpp_pypots_seed42.csv",
  "model_screening/timemixerpp_paperfaithful_seed42.csv",
  "model_screening/timexer_pemfc_a_seed42.csv",
  "model_screening/timexer_pemfc_life_seed42.csv",
];

function isBlank(value) {
  return value === null || value === undefined || String(value).trim() === "";
}

const audits = [];
for (const relativePath of requiredCsvs) {
  const csvPath = path.join(outputsRoot, relativePath);
  const source = await fs.readFile(csvPath, "utf8");
  const workbook = await Workbook.fromCSV(source, { sheetName: "Audit" });
  const sheet = workbook.worksheets.getItem("Audit");
  const used = sheet.getUsedRange(true);
  const values = used.values;
  const headers = (values[0] ?? []).map((value) => String(value));
  const body = values.slice(1);
  const nonEmptyRows = body.filter((row) => row.some((value) => !isBlank(value)));
  const invalidNumericTokens = [];
  for (let rowIndex = 0; rowIndex < body.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < body[rowIndex].length; columnIndex += 1) {
      const value = body[rowIndex][columnIndex];
      if (typeof value === "string" && /^(nan|inf|-inf)$/i.test(value.trim())) {
        invalidNumericTokens.push({ row: rowIndex + 2, column: headers[columnIndex], value });
      }
    }
  }
  const preview = await workbook.inspect({
    kind: "table",
    range: `Audit!A1:${String.fromCharCode(64 + Math.min(headers.length, 26))}${Math.min(nonEmptyRows.length + 1, 6)}`,
    include: "values",
    tableMaxRows: 6,
    tableMaxCols: Math.min(headers.length, 26),
  });
  audits.push({
    relative_path: relativePath,
    column_count: headers.length,
    data_row_count: nonEmptyRows.length,
    headers,
    invalid_numeric_tokens: invalidNumericTokens,
    preview: preview.ndjson,
  });
}

const result = {
  audit_scope: "N2.8B required CSV artifacts; structure/content read through @oai/artifact-tool Workbook.fromCSV",
  required_csv_count: requiredCsvs.length,
  all_files_present: audits.length === requiredCsvs.length,
  files_with_no_data_rows: audits.filter((entry) => entry.data_row_count === 0).map((entry) => entry.relative_path),
  files_with_invalid_numeric_tokens: audits
    .filter((entry) => entry.invalid_numeric_tokens.length > 0)
    .map((entry) => ({ relative_path: entry.relative_path, tokens: entry.invalid_numeric_tokens })),
  artifacts: audits,
};

await fs.writeFile(
  path.join(outputsRoot, "csv_artifact_tool_audit.json"),
  `${JSON.stringify(result, null, 2)}\n`,
  "utf8",
);
console.log(JSON.stringify({
  required_csv_count: result.required_csv_count,
  all_files_present: result.all_files_present,
  files_with_no_data_rows: result.files_with_no_data_rows,
  files_with_invalid_numeric_tokens: result.files_with_invalid_numeric_tokens,
}, null, 2));
