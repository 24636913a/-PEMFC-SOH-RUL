"""Execute every # %% cell through IPython display hooks, read-only mode.

Checks actual PNG display payloads, not screenshots of a VS Code GUI. No train.
"""
import json
from IPython.core.interactiveshell import InteractiveShell
from audit_data import HERE,dump,sha

def run():
    path=HERE/'F2_LongTerm_Interactive.py';text=path.read_text(encoding='utf-8')
    shell=InteractiveShell.instance();shell.user_ns['__file__']=str(path)
    displays=[]
    def capture(data,metadata=None,**kwargs):
        displays.append(dict(mime=list(data),PNG_base64_length=len(data.get('image/png',''))))
    shell.display_pub.publish=capture
    cells=text.split('# %%')[1:];results=[]
    for i,cell in enumerate(cells,1):
        code=cell.split('\n',1)[1]
        result=shell.run_cell(code,store_history=False)
        assert result.success,result.error_in_exec or result.error_before_exec
        results.append(dict(cell=i,success=True))
    n=sum('image/png' in d['mime'] for d in displays)
    assert n>=16,n
    dump(HERE/'outputs/INTERACTIVE_ACCEPTANCE.json',dict(PASS=True,script_sha256=sha(path),cells=results,
        actual_PNG_display_payloads=n,display_records=displays,mode='analyze',training_started=False,
        verification_scope='Real IPython cell execution/display payloads; VS Code GUI not opened or visually certified'))
    print('IPython acceptance PASS:',len(cells),'cells;',n,'real PNG displays; no new training')

if __name__=='__main__':run()
