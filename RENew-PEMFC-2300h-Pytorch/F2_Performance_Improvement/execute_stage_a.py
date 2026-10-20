"""Execute Stage A cells in actual Python3.11/CUDA IPython. No simulated progress."""
from pathlib import Path
from datetime import datetime,timezone
import ctypes
import queue
import re
import sys
import time
import nbformat
from jupyter_client import KernelManager
import frozen_f2 as f
from io_utils import js


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    accepted=f.read_json(f.HERE/'preflight/CONTRACT_TESTS.json')
    f.require(accepted['status']=='PASS','CONTRACT_TESTS_REQUIRED')
    for name,digest in accepted['tested_source_sha256'].items():f.require(f.sha(f.HERE/name)==digest,'TESTED_SOURCE_CHANGED:'+name)
    notebook=nbformat.v4.new_notebook()
    for block in re.split(r'^# %%',(f.HERE/'Performance_Interactive.py').read_text(encoding='utf-8'),flags=re.MULTILINE)[1:]:
        _,_,body=block.partition('\n');notebook.cells.append(nbformat.v4.new_code_cell(body.strip()))
    notebook.metadata.kernelspec=dict(name='python3',display_name='RealHyFC Python3.11 CUDA',language='python')
    notebook.cells[0].source+="\nSTAGE_LIMIT='A'\n"
    attempt='attempt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    started=time.monotonic();lastsave=started;report=dict(status='RUNNING',stage='A',real_training=False,
        Python=sys.executable,real_IPython=True,VSCode_GUI_clicked=False,PNG_displays=0,loss_refreshes=0,
        fresh_instances_started=0,controller_sha256=f.sha(f.HERE/'execute_stage_a.py'))
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}']
    km.start_kernel(cwd=str(f.HERE));client=km.client();client.start_channels();locations={};run=None
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
                        report['loss_refreshes']+=1
                    else:
                        fields=dict(data=content['data'],metadata=content.get('metadata',{}))
                        if kind=='execute_result':fields['execution_count']=content['execution_count']
                        cell.outputs.append(nbformat.v4.new_output(kind,**fields))
                        if display_id:locations[display_id]=(cell,len(cell.outputs)-1)
                        if 'image/png' in content['data']:report['PNG_displays']+=1
                elif kind=='stream':
                    value=content['text']
                    if 'FRESH GPU training' in value:
                        report['real_training']=True;report['fresh_instances_started']+=value.count('FRESH GPU training')
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
                if (f.HERE/'ACTIVE_RUN.json').exists():run=f.HERE/'outputs'/f.read_json(f.HERE/'ACTIVE_RUN.json')['run']
                if time.monotonic()-lastsave>25:
                    target=(run if run is not None else f.HERE/'preflight')/'execution_attempts'/attempt;target.mkdir(parents=True,exist_ok=True)
                    nbformat.write(notebook,target/'Stage_A.executed.ipynb');js(target/'STAGE_A_EXECUTION.json',report);lastsave=time.monotonic()
            if (f.HERE/'ACTIVE_RUN.json').exists():run=f.HERE/'outputs'/f.read_json(f.HERE/'ACTIVE_RUN.json')['run']
            if run is not None:
                target=run/'execution_attempts'/attempt;target.mkdir(parents=True,exist_ok=True)
                nbformat.write(notebook,target/'Stage_A.executed.ipynb');js(target/'STAGE_A_EXECUTION.json',report)
            print('STAGE_A_CELL_COMPLETED',number,flush=True)
        f.require(run is not None and f.read_json(run/'A_RESULT.json')['status']=='A_COMPLETE','A_NOT_COMPLETE')
        report.update(status='A_COMPLETE',seconds=time.monotonic()-started)
    except Exception as error:
        report.update(status='FAILURE_PRESERVED',error=repr(error));raise
    finally:
        target=(run if run is not None else f.HERE/'preflight')/'execution_attempts'/attempt;target.mkdir(parents=True,exist_ok=True)
        nbformat.write(notebook,target/'Stage_A.executed.ipynb');js(target/'STAGE_A_EXECUTION.json',report)
        if run is not None:js(run/'LATEST_A_EXECUTION.json',dict(path=(target/'STAGE_A_EXECUTION.json').relative_to(run).as_posix(),status=report['status']))
        client.stop_channels();km.shutdown_kernel(now=True);ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print('STAGE_A_REAL_GPU_EXECUTION_FINISHED',flush=True)
