import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook } from '@oai/artifact-tool';
const root='E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_R1_Calibrated_Two_Scenario/outputs';
async function files(dir){const out=[];for(const e of await fs.readdir(dir,{withFileTypes:true})){const f=path.join(dir,e.name);if(e.isDirectory())out.push(...await files(f));else if(e.name.endsWith('.csv'))out.push(f)}return out}
const audit=[];for(const file of await files(root)){const text=await fs.readFile(file,'utf8');const wb=await Workbook.fromCSV(text,{sheetName:'audit'});const view=await wb.inspect({kind:'table',range:'audit!A1:ZZ6',include:'values',tableMaxRows:6,tableMaxCols:702,maxChars:4000});const rows=text.trimEnd().split(/\r?\n/),head=rows[0]||'';audit.push({file:path.relative(root,file).replaceAll('\\','/'),bytes:Buffer.byteLength(text),row_count:Math.max(0,rows.length-1),header_column_count:head.split(',').length,artifact_tool_inspected:Boolean(view.ndjson),status:head?'PASS':'FAIL_EMPTY_HEADER'})}
audit.sort((a,b)=>a.file.localeCompare(b.file));await fs.writeFile(path.join(root,'audit','csv_artifact_audit.json'),JSON.stringify({csv_count:audit.length,all_pass:audit.every(x=>x.status==='PASS'),files:audit},null,2)+'\n','utf8');
