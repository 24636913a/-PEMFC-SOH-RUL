import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const root = "E:\\沈农信电\\ PEMFC-SOH+RUL\\RENew-PEMFC-2300h-Pytorch\\Phase_N2_8D_Future_Conditioned_TimeXer";
const required = [
  "outputs/model_screening/N2_8D_seed42_comparison.csv",
  "outputs/ablation/N2_8D_ablation.csv",
  "outputs/horizon_analysis/N2_8D_per_horizon.csv",
  "outputs/tail_analysis/N2_8D_tail_error.csv",
  "outputs/ablation/future_condition_gain.csv",
  "outputs/model_screening/N2_8D_multiseed_comparison.csv",
  "outputs/model_screening/foldwise_fc_timexer_metrics.csv",
];
const artifacts=[];
for(const relative of required){
  const wb=await Workbook.fromCSV(await fs.readFile(path.join(root,relative),"utf8"),{sheetName:"Audit"});
  const values=wb.worksheets.getItem("Audit").getUsedRange(true).values;
  const headers=(values[0]??[]).map(String);const rows=values.slice(1).filter(r=>r.some(v=>v!==null&&v!==undefined&&String(v).trim()!==""));const invalid=[];
  rows.forEach((row,ri)=>row.forEach((v,ci)=>{if(typeof v==="string"&&/^(nan|inf|-inf)$/i.test(v.trim()))invalid.push({row:ri+2,header:headers[ci],value:v})}));
  const preview=await wb.inspect({kind:"table",range:`Audit!A1:${String.fromCharCode(64+Math.min(26,headers.length))}${Math.min(6,rows.length+1)}`,include:"values",tableMaxRows:6,tableMaxCols:Math.min(26,headers.length)});
  artifacts.push({relative_path:relative,column_count:headers.length,data_row_count:rows.length,headers,invalid_tokens:invalid,preview:preview.ndjson});
}
const audit={scope:"N2.8D required CSV artifacts inspected through @oai/artifact-tool Workbook.fromCSV",required_csv_count:required.length,all_files_present:artifacts.length===required.length,files_with_no_data_rows:artifacts.filter(x=>x.data_row_count===0).map(x=>x.relative_path),files_with_invalid_tokens:artifacts.filter(x=>x.invalid_tokens.length).map(x=>({relative_path:x.relative_path,tokens:x.invalid_tokens})),artifacts};
await fs.writeFile(path.join(root,"outputs","csv_artifact_tool_audit.json"),JSON.stringify(audit,null,2)+"\n","utf8");
console.log(JSON.stringify({required_csv_count:audit.required_csv_count,all_files_present:audit.all_files_present,files_with_no_data_rows:audit.files_with_no_data_rows,files_with_invalid_tokens:audit.files_with_invalid_tokens},null,2));
