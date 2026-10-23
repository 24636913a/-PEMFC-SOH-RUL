import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const phaseRoot = path.resolve(process.argv[2]);
const outputRoot = path.join(phaseRoot, "outputs");

async function collect(dir) {
  const result = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const target = path.join(dir, entry.name);
    if (entry.isDirectory()) result.push(...await collect(target));
    else if (entry.isFile() && entry.name.toLowerCase().endsWith(".csv")) result.push(target);
  }
  return result;
}

const files = (await collect(outputRoot)).sort();
const records = [];
for (const file of files) {
  const csvText = await fs.readFile(file, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Data" });
  const sheet = workbook.worksheets.getItem("Data");
  const used = sheet.getUsedRange(true);
  const values = used?.values ?? [];
  const rows = values.length;
  const columns = rows > 0 ? Math.max(...values.map((row) => row.length)) : 0;
  const inspect = await workbook.inspect({
    kind: "sheet,region",
    sheetId: "Data",
    range: rows > 0 && columns > 0 ? `A1:${columnName(Math.min(columns, 26))}${Math.min(rows, 6)}` : "A1",
    maxChars: 2500,
    tableMaxRows: 6,
    tableMaxCols: 26,
  });
  const errorTokens = ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"];
  const formulaErrors = errorTokens.filter((token) => inspect.ndjson.includes(token));
  records.push({
    file: path.relative(phaseRoot, file).replaceAll("\\", "/"),
    rows,
    dataRows: Math.max(rows - 1, 0),
    columns,
    empty: rows <= 1 || columns === 0,
    duplicateHeaders: rows ? values[0].filter((value, index, row) => row.indexOf(value) !== index) : [],
    formulaErrors,
    inspectPreview: inspect.ndjson.slice(0, 1200),
  });
}

function columnName(column) {
  let value = column;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

const failures = records.filter((item) => item.empty || item.duplicateHeaders.length || item.formulaErrors.length);
const payload = {
  tool: "@oai/artifact-tool Workbook.fromCSV + inspect",
  csvCount: records.length,
  passedCount: records.length - failures.length,
  failedCount: failures.length,
  status: failures.length ? "FAIL" : "PASS",
  records,
};
await fs.writeFile(path.join(outputRoot, "csv_artifact_audit.json"), JSON.stringify(payload, null, 2) + "\n", "utf8");
console.log(JSON.stringify({ status: payload.status, csvCount: payload.csvCount, failedCount: payload.failedCount }, null, 2));
if (failures.length) process.exitCode = 1;
