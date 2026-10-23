"""Execute real VS Code cells in Python3.11/CUDA ipykernel; preserve all evidence."""
from pathlib import Path
import ctypes
import json
import queue
import re
import subprocess
import sys
import time
import nbformat
from jupyter_client import KernelManager
import core as c
from io_utils import js

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    run=c.HERE/'outputs'/c.read_json(c.HERE/'ACTIVE_DATA_RUN.json')['run']
    c.require(not (run/'EXECUTION.json').exists(),'REFUSE_DUPLICATE_ACCEPTANCE')
    c.require(c.read_json(run/'P1_ACCEPTANCE.json')['status']=='PASS' and not (run/'INVALIDATED_BEFORE_TRAINING.json').exists(),'P1_NOT_QUALIFIED')
    repo=c.ROOT.parent
    names=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode('utf-8').rstrip('\0').split('\0')
    selected=[name for name in names if name.startswith('RENew-PEMFC-2300h-Pytorch/') and (repo/name).is_file()]
    protected={};large={}
    for index,name in enumerate(selected):
        path=repo/name;stat=path.stat()
        if stat.st_size<=64*1024*1024:protected[name]=c.sha(path)
        else:large[name]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        if index%10000==0:print('HISTORY_PROTECTION',index,'/',len(selected),flush=True)
    js(run/'HISTORY_BEFORE.json',protected);js(run/'HISTORY_LARGE_FILE_STATS.json',large)
    notebook=nbformat.v4.new_notebook()
    for block in re.split(r'^# %%',(c.HERE/'Comparison_Interactive.py').read_text(encoding='utf-8'),flags=re.MULTILINE)[1:]:
        _,_,body=block.partition('\n');notebook.cells.append(nbformat.v4.new_code_cell(body.strip()))
    first=notebook.cells[0];c.require(first.source.count('EXISTING_DATA_RUN = None')==1,'CELL_CONFIG')
    first.source=first.source.replace('EXISTING_DATA_RUN = None','EXISTING_DATA_RUN = '+repr(run.name),1)
    notebook.metadata.kernelspec=dict(name='python3',display_name='RealHyFC Python3.11 CUDA',language='python')
    output=run/'Comparison.executed.ipynb';report=dict(status='RUNNING',expected_pilot=2,expected_formal=12,
        Python=sys.executable,data_run=run.name,remote_base_sha='66b843ab1c53577d9fdb1071f27c5d8510b9d5c2',
        real_ipython=True,VSCode_GUI_clicked=False,training_from_scratch=True,displayed_PNG=0,loss_curve_refreshes=0)
    js(run/'EXECUTION.json',report)
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}']
    km.start_kernel(cwd=str(c.HERE));client=km.client();client.start_channels();locations={};lastsave=time.monotonic()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    try:
        client.wait_for_ready(timeout=60)
        for number,cell in enumerate(notebook.cells):
            request=client.execute(cell.source,store_history=True,stop_on_error=True);cell.outputs=[]
            while True:
                try:message=client.get_iopub_msg(timeout=2)
                except queue.Empty:continue
                if message['parent_header'].get('msg_id')!=request:continue
                kind,content=message['msg_type'],message['content']
                if kind=='execute_input':cell.execution_count=content['execution_count']
                elif kind in ['display_data','execute_result','update_display_data']:
                    display_id=content.get('transient',{}).get('display_id')
                    if kind=='update_display_data':
                        if display_id in locations:
                            old,index=locations[display_id];old.outputs[index].data=content['data'];old.outputs[index].metadata=content.get('metadata',{})
                        report['loss_curve_refreshes']+=1
                    else:
                        fields=dict(data=content['data'],metadata=content.get('metadata',{}))
                        if kind=='execute_result':fields['execution_count']=content['execution_count']
                        cell.outputs.append(nbformat.v4.new_output(kind,**fields))
                        if display_id:locations[display_id]=(cell,len(cell.outputs)-1)
                        if 'image/png' in content['data']:report['displayed_PNG']+=1
                elif kind=='stream':
                    value=content['text']
                    if cell.outputs and cell.outputs[-1].output_type=='stream' and cell.outputs[-1].name==content['name']:cell.outputs[-1].text+=value
                    else:cell.outputs.append(nbformat.v4.new_output('stream',name=content['name'],text=value))
                    for line in value.splitlines():
                        match=re.search(r'Epoch (\d+)/450',line)
                        if not match or int(match.group(1))==1 or int(match.group(1))%10==0:print(line,flush=True)
                    report['last_progress']=dict(cell=number,text=value[-1000:])
                elif kind=='error':
                    cell.outputs.append(nbformat.v4.new_output('error',ename=content['ename'],evalue=content['evalue'],traceback=content['traceback']))
                    raise RuntimeError('\n'.join(content['traceback']))
                elif kind=='status' and content['execution_state']=='idle':break
                if time.monotonic()-lastsave>25:
                    nbformat.write(notebook,output);js(run/'EXECUTION.json',report);lastsave=time.monotonic()
            nbformat.write(notebook,output);js(run/'EXECUTION.json',report);print('CELL_COMPLETE',number,flush=True)
        for name,digest in protected.items():c.require(c.sha(repo/name)==digest,'HISTORY_CHANGED:'+name)
        for name,stat in large.items():
            actual=(repo/name).stat();c.require(actual.st_size==stat['bytes'] and actual.st_mtime_ns==stat['mtime_ns'],'LARGE_HISTORY_CHANGED')
        report.update(status='PIPELINE_COMPLETED_HISTORY_UNCHANGED',historical_files_sha_unchanged=len(protected),historical_large_stats_unchanged=len(large))
    except Exception as error:
        report.update(status='FAILURE_PRESERVED',error=repr(error));raise
    finally:
        nbformat.write(notebook,output);client.stop_channels();km.shutdown_kernel(now=True)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);js(run/'EXECUTION.json',report)
    print('REAL_TRAINING_AND_PIPELINE_ACCEPTANCE_COMPLETED',flush=True)
