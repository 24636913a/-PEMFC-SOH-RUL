import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const root = path.resolve(import.meta.dirname, "..");
const files = [
  "outputs/selector/selector_by_fold.csv",
  "outputs/selector/selector_by_horizon.csv",
  "outputs/selector/selector_confusion_matrix.csv",
  "outputs/selector/selector_input_disagreement.csv",
  "outputs/frozen_model/physical_violation_metrics.csv",
  "outputs/frozen_model/frozen_model_fold_metrics.csv",
  "outputs/frozen_model/frozen_model_summary.csv",
  "outputs/horizon/frozen_model_per_horizon.csv",
  "outputs/regime/frozen_model_regime_metrics.csv",
  "outputs/tail/frozen_model_tail_metrics.csv",
  "outputs/bootstrap/bootstrap_main_metric.csv",
  "outputs/bootstrap/bootstrap_block_sensitivity.csv",
  "outputs/retraining/retraining_fold_metrics.csv",
  "outputs/retraining/retraining_summary.csv",
  "outputs/multiseed/multiseed_summary.csv"
];
const results = [];
for (const relative of files) {
  const csv = await fs.readFile(path.join(root, relative), "utf8");
  const workbook = await Workbook.fromCSV(csv, { sheetName: "Audit" });
  const inspect = await workbook.inspect({ kind: "table", range: "Audit!A1:Z8", include: "values", tableMaxRows: 8, tableMaxCols: 26, maxChars: 8000 });
  const rows = csv.trimEnd().split(/\r?\n/);
  results.push({ file: relative, row_count_including_header: rows.length, inspection: inspect.ndjson });
}
await fs.writeFile(path.join(root, "logs", "artifact_tool_csv_audit.json"), JSON.stringify(results, null, 2));
console.log(JSON.stringify({ status: "PASS", audited_csv_count: results.length }));
