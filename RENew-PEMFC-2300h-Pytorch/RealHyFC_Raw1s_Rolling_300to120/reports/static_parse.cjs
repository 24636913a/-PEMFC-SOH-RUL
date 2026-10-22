const fs=require('fs'),path=require('path');
const {parser}=require('@lezer/python');
const root=process.argv[2], out=process.argv[3];
function files(p){return fs.readdirSync(p,{withFileTypes:true}).flatMap(e=>e.isDirectory()?files(path.join(p,e.name)):[path.join(p,e.name)]);}
const results=files(root).filter(f=>f.endsWith('.py')).map(file=>{
 const source=fs.readFileSync(file,'utf8'), errors=[];
 parser.parse(source).iterate({enter(node){if(node.type.isError)errors.push({from:node.from,to:node.to,line:source.slice(0,node.from).split('\n').length,text:source.slice(node.from,node.to)});}});
 return {file,lines:source.split('\n').length,syntax_errors:errors};
});
fs.writeFileSync(out,JSON.stringify({method:'Lezer Python grammar; source text parsed only; no Python interpreter/import/runtime execution',results},null,2));
console.log(JSON.stringify(results.map(r=>({file:path.basename(r.file),errors:r.syntax_errors})),null,2));
if(results.some(r=>r.syntax_errors.length))process.exitCode=1;
