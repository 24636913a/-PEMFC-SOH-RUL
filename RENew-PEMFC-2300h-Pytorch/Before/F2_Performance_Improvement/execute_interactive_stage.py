"""Execute the same # %% entry in a real Python3.11/CUDA IPython kernel."""
from datetime import datetime,timezone
import argparse
import ctypes
import queue
import re
import sys
import time
import nbformat
from jupyter_client import KernelManager
import frozen_f2 as f
from io_utils import js


def execute(stage):
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    f.require(stage=='B','ONLY_REGISTERED_B_SUPPORTED_C_D_NOT_TRIGGERED')
    source=(f.HERE/'Performance_Interactive.py').read_text(encoding='utf-8')
    notebook=nbformat.v4.new_notebook()
    for block in re.split(r'^# %%',source,flags=re.MULTILINE)[1:]:
        _,_,body=block.partition('\n');notebook.cells.append(nbformat.v4.new_code_cell(body.strip()))
    notebook.cells[0].source+=f"\nSTAGE_LIMIT={stage!r}\n"
    notebook.metadata.kernelspec=dict(name='python3',display_name='RealHyFC Python3.11 CUDA',language='python')
    attempt=f'{stage}_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run=f.HERE/'outputs'/f.read_json(f.HERE/'ACTIVE_RUN.json')['run']
    # This driver resumes the explicitly registered ACTIVE run. A new study is
    # created from the interactive entry, not silently inside this controller.
    notebook.cells[0].source+=f"\nSTART_NEW_RUN=False\nRUN_ID={run.name!r}\n"
    f.require((run/f'{stage}_REGISTRATION.json').exists(),'UNREGISTERED_STAGE')
    def training_snapshot():
        values=[f.read_json(p) for p in (run/'models').glob('*/*/TRAINING_LEDGER.json')]
        return len(values),sum(v.get('optimizer_steps',0) for v in values)
    before_models,before_updates=training_snapshot()
    target=run/'execution_attempts'/attempt;target.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',stage=stage,real_training=False,Python=sys.executable,
        real_IPython=True,VSCode_GUI_clicked=False,PNG_displays=0,loss_refreshes=0,
        entry_sha256=f.sha(f.HERE/'Performance_Interactive.py'),controller_sha256=f.sha(f.HERE/'execute_interactive_stage.py'),
        explicit_run_id=run.name)
    started=lastsave=time.monotonic();locations={}
    def save():
        nbformat.write(notebook,target/f'Stage_{stage}.executed.ipynb')
        js(target/f'STAGE_{stage}_EXECUTION.json',report)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}']
    km.start_kernel(cwd=str(f.HERE));client=km.client();client.start_channels()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    try:
        client.wait_for_ready(timeout=60)
        for number,cell in enumerate(notebook.cells):
            request=client.execute(cell.source,store_history=True,stop_on_error=True);cell.outputs=[]
            while True:
                try:message=client.get_iopub_msg(timeout=2)
                except queue.Empty:continue
                if message['parent_header'].get('msg_id')!=request:continue
                kind,c=message['msg_type'],message['content']
                if kind=='execute_input':cell.execution_count=c['execution_count']
                elif kind in ['display_data','execute_result','update_display_data']:
                    ident=c.get('transient',{}).get('display_id')
                    if kind=='update_display_data':
                        if ident in locations:
                            old,index=locations[ident];old.outputs[index].data=c['data'];old.outputs[index].metadata=c.get('metadata',{})
                        report['loss_refreshes']+=1
                    else:
                        fields=dict(data=c['data'],metadata=c.get('metadata',{}))
                        if kind=='execute_result':fields['execution_count']=c['execution_count']
                        cell.outputs.append(nbformat.v4.new_output(kind,**fields))
                        if ident:locations[ident]=(cell,len(cell.outputs)-1)
                        if 'image/png' in c['data']:report['PNG_displays']+=1
                elif kind=='stream':
                    value=c['text']
                    if 'FRESH GPU training' in value:report['real_training']=True
                    if cell.outputs and cell.outputs[-1].output_type=='stream' and cell.outputs[-1].name==c['name']:cell.outputs[-1].text+=value
                    else:cell.outputs.append(nbformat.v4.new_output('stream',name=c['name'],text=value))
                    for line in value.splitlines():
                        match=re.search(r'Epoch (\d+)/450',line)
                        if not match or int(match.group(1))==1 or int(match.group(1))%10==0:print(line,flush=True)
                    report['last_progress']=dict(cell=number,text=value[-1000:])
                elif kind=='error':
                    cell.outputs.append(nbformat.v4.new_output('error',ename=c['ename'],evalue=c['evalue'],traceback=c['traceback']))
                    raise RuntimeError('\n'.join(c['traceback']))
                elif kind=='status' and c['execution_state']=='idle':break
                if time.monotonic()-lastsave>25:save();lastsave=time.monotonic()
            save();print(f'STAGE_{stage}_CELL_COMPLETED',number,flush=True)
        f.require(f.read_json(run/f'{stage}_RESULT.json')['status']==f'{stage}_COMPLETE','STAGE_NOT_COMPLETE')
        f.require(f.read_json(run/f'audits/{stage}_INDEPENDENT_AUDIT.json')['status']=='PASS','INDEPENDENT_AUDIT_NOT_PASS')
        after_models,after_updates=training_snapshot()
        report.update(status=f'{stage}_COMPLETE',seconds=time.monotonic()-started,
            new_model_instances_this_execution=after_models-before_models,
            optimizer_updates_this_execution=after_updates-before_updates,
            execution_kind='Fresh training' if report['real_training'] else 'Explicit resume/acceptance; no repeated training')
    except Exception as error:
        report.update(status='FAILURE_PRESERVED',error=repr(error));raise
    finally:
        save();js(run/f'LATEST_{stage}_EXECUTION.json',dict(path=(target/f'STAGE_{stage}_EXECUTION.json').relative_to(run).as_posix(),status=report['status']))
        client.stop_channels();km.shutdown_kernel(now=True);ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print(f'STAGE_{stage}_IPYTHON_EXECUTION_FINISHED',report['execution_kind'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['B'],required=True)
    execute(parser.parse_args().stage)
