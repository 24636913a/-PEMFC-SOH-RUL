"""Re-render real CSV figures and audit saved new weights; NEVER trains a model."""
import json
import queue
import shutil
import sys
import nbformat
from jupyter_client import KernelManager
import core as c
from io_utils import js

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    run=c.HERE/'outputs'/c.read_json(c.HERE/'ACTIVE_DATA_RUN.json')['run']
    c.require(c.read_json(run/'EXECUTION.json')['status']=='PIPELINE_COMPLETED_HISTORY_UNCHANGED','ORIGINAL_PIPELINE_NOT_COMPLETE')
    receipt=run/'audits/P3_POSTPROCESSING_QA.json'
    c.require(not receipt.exists(),'REFUSE_DUPLICATE_QA_EXECUTION')
    initial=run/'audits/initial_completed_pipeline';initial.mkdir(exist_ok=False)
    for name in ['P3_ACCEPTANCE.json','P5_AUDIT.json']:shutil.copyfile(run/name,initial/name)
    protected={p.relative_to(c.HERE).as_posix():c.sha(p) for p in (run/'models').rglob('*') if p.is_file()}
    code="""from pathlib import Path
import core as c
from finish import finalize
c.configure_environment()
run=c.HERE/'outputs'/c.read_json(c.HERE/'ACTIVE_DATA_RUN.json')['run']
frame,windows,scaler,prepared,excluded=c.load_data()
table=c.read_csv(run/'tables/PARTITION_METRICS.csv')
means=c.read_csv(run/'tables/MODEL_MEAN_SD.csv')
paired=c.read_csv(run/'tables/PAIRED_DIFFERENCES.csv')
finalize(run,frame,windows,scaler,prepared,table,means,paired)
print('POSTPROCESSING_AND_REPLAY_ONLY: no model training, no changed predictions')
"""
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)])
    notebook.metadata.kernelspec=dict(name='python3',display_name='RealHyFC Python3.11 CUDA',language='python')
    output=run/'Postprocessing_QA.executed.ipynb';cell=notebook.cells[0]
    report=dict(status='RUNNING',no_training=True,reason='Show both models worst-query windows explicitly; preserve fixed-Step24 figures',displayed_PNG=0)
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}']
    km.start_kernel(cwd=str(c.HERE));client=km.client();client.start_channels()
    try:
        client.wait_for_ready(timeout=60);request=client.execute(code,store_history=True,stop_on_error=True)
        while True:
            try:message=client.get_iopub_msg(timeout=2)
            except queue.Empty:continue
            if message['parent_header'].get('msg_id')!=request:continue
            kind,content=message['msg_type'],message['content']
            if kind=='execute_input':cell.execution_count=content['execution_count']
            elif kind in ['display_data','execute_result']:
                fields=dict(data=content['data'],metadata=content.get('metadata',{}))
                if kind=='execute_result':fields['execution_count']=content['execution_count']
                cell.outputs.append(nbformat.v4.new_output(kind,**fields))
                if 'image/png' in content['data']:report['displayed_PNG']+=1
            elif kind=='stream':
                cell.outputs.append(nbformat.v4.new_output('stream',name=content['name'],text=content['text']));print(content['text'],end='',flush=True)
            elif kind=='error':
                cell.outputs.append(nbformat.v4.new_output('error',ename=content['ename'],evalue=content['evalue'],traceback=content['traceback']))
                raise RuntimeError('\n'.join(content['traceback']))
            elif kind=='status' and content['execution_state']=='idle':break
        for name,digest in protected.items():c.require(c.sha(c.HERE/name)==digest,'TRAINING_ARTIFACT_CHANGED_IN_QA:'+name)
        report.update(status='PASS',model_predictions_weights_and_logs_unchanged=len(protected),P5=c.read_json(run/'P5_AUDIT.json')['status'],formal_P3_figures=12,VSCode_GUI_clicked=False,real_IPython_execution=True)
    except Exception as error:
        report.update(status='FAILURE_PRESERVED',error=repr(error));raise
    finally:
        nbformat.write(notebook,output);client.stop_channels();km.shutdown_kernel(now=True);js(receipt,report)
    print('POSTPROCESSING_QA_PASS',flush=True)
