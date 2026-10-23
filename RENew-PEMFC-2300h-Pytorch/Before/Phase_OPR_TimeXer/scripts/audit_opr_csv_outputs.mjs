import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const root = path.resolve(import.meta.dirname, "..");
const files = [
  "outputs/reproduction/baseline_reproduction_fold_metrics.csv",
  "outputs/reproduction/baseline_reproduction_summary.csv",
  "outputs/decomposition/prediction_decomposition.csv",
  "outputs/error_attribution/p1_p2_error_difference.csv",
  "outputs/error_attribution/fold3_error_attribution.csv",
  "outputs/horizon/horizon_error_attribution.csv",
  "outputs/reference_condition/reference_condition_diagnostics.csv",
  "outputs/reference_condition/reference_condition_summary.csv",
  "outputs/extreme_errors/extreme_error_unique_targets.csv"
];
const results = [];
for (const relative of files) {
  const content = await fs.readFile(path.join(root, relative), "utf8");
  let end = 0;
  let rows = 0;
  while (end < content.length && rows < 51) {
    const next = content.indexOf("\n", end);
    end = next < 0 ? content.length : next + 1;
    rows += 1;
  }
  const preview = content.slice(0, end);
  const wb = await Workbook.fromCSV(preview, { sheetName: "Audit" });
  const check = await wb.inspect({ kind: "table", range: "Audit!A1:Z7", include: "values", tableMaxRows: 7, tableMaxCols: 26, maxChars: 6000 });
  const dataRows = Math.max(0, (content.match(/\n/g) || []).length - 1);
  results.push({ file: relative, data_rows: dataRows, audited_preview_data_rows: Math.max(0, rows - 1), inspection: check.ndjson });
}
await fs.writeFile(path.join(root, "logs", "artifact_tool_csv_audit.json"), JSON.stringify(results, null, 2));
console.log(JSON.stringify({ status: "PASS", audited_csv_count: results.length }));
