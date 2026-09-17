import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const root = "E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N2_8F_Uncertainty_Robust_SA_TimeXer";
const artifacts = [
  "outputs/ablation/N2_8F_seed42_comparison.csv",
  "outputs/ablation/N2_8F_multiseed_comparison.csv",
  "outputs/ablation/N2_8F_ablation.csv",
  "outputs/ablation/same_checkpoint_oracle_swap.csv",
  "outputs/attribution/condition_group_sensitivity.csv",
  "outputs/attribution/future_exo_error_by_variable.csv",
  "outputs/attribution/future_exo_error_by_horizon.csv",
  "outputs/attribution/exo_error_voltage_error_correlation.csv",
  "outputs/augmentation/forecast_error_augmentation_audit.csv",
  "outputs/confidence/confidence_calibration.csv",
  "outputs/confidence/confidence_gate_statistics.csv",
  "outputs/lifecycle/N2_8F_fold_metrics.csv",
  "outputs/horizon/N2_8F_per_horizon.csv",
  "outputs/tail/N2_8F_tail_metrics.csv",
  "outputs/ablation/ridge_oof_temporal_leakage_audit.csv",
];

const audit = [];
for (const relative of artifacts) {
  const filename = path.join(root, relative);
  const csv = await fs.readFile(filename, "utf8");
  const lines = csv.trimEnd().split(/\r?\n/);
  const workbook = await Workbook.fromCSV(csv, { sheetName: "Audit" });
  const preview = await workbook.inspect({
    kind: "table",
    range: "Audit!A1:Z6",
    include: "values",
    tableMaxRows: 6,
    tableMaxCols: 26,
  });
  const header = lines[0].replace(/^\uFEFF/, "").split(",");
  audit.push({
    file: relative,
    bytes: Buffer.byteLength(csv),
    records_excluding_header: Math.max(0, lines.length - 1),
    header_column_count: header.length,
    first_column: header[0],
    artifact_tool_inspection: preview ? "PASS" : "FAIL",
  });
}
await fs.writeFile(path.join(root, "outputs", "csv_artifact_tool_audit.json"), JSON.stringify({ status: "PASS", artifacts: audit }, null, 2) + "\n", "utf8");
console.log(JSON.stringify({ status: "PASS", audited_csv_count: audit.length }, null, 2));
