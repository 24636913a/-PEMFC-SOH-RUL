import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook } from '@oai/artifact-tool';

const root = 'E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_R0_Rare_State_Identifiability/outputs';
async function listCsvs(dir) {
  const found=[];
  for (const entry of await fs.readdir(dir,{withFileTypes:true})) {
    const full=path.join(dir,entry.name);
    if (entry.isDirectory()) found.push(...await listCsvs(full));
    else if (entry.name.endsWith('.csv')) found.push(full);
  }
  return found;
}
const files=await listCsvs(root), audit=[];
for (const file of files) {
  const text=await fs.readFile(file,'utf8');
  const wb=await Workbook.fromCSV(text,{sheetName:'audit'});
  const view=await wb.inspect({kind:'table',range:'audit!A1:ZZ6',include:'values',tableMaxRows:6,tableMaxCols:702,maxChars:4000});
  const rows=text.trimEnd().split(/\r?\n/), header=rows[0]||'';
  audit.push({file:path.relative(root,file).replaceAll('\\','/'),bytes:Buffer.byteLength(text),row_count:Math.max(0,rows.length-1),header_column_count:header.split(',').length,artifact_tool_inspected:Boolean(view.ndjson),status:header.length?'PASS':'FAIL_EMPTY_HEADER'});
}
audit.sort((a,b)=>a.file.localeCompare(b.file));
await fs.writeFile(path.join(root,'audit','csv_artifact_audit.json'),JSON.stringify({csv_count:audit.length,all_pass:audit.every(x=>x.status==='PASS'),files:audit},null,2)+'\n','utf8');
