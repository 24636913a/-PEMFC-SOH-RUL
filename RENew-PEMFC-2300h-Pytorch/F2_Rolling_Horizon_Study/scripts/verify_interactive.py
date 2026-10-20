"""Actual IPython execution/display-channel checks for show and analyze modes.

No GUI screenshot claim. Never execute train mode in this acceptance script.
"""
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell
from common import HERE,read,dump,sha

def run():
    file=HERE/'F2_Rolling_Horizon_Interactive.py';text=file.read_text(encoding='utf-8')
    results=[]
    for mode in ['show','analyze']:
        shell=InteractiveShell.instance();shell.user_ns['__file__']=str(file)
        displays=[]
        def capture(data,metadata=None,**kwargs):displays.append(dict(mime=list(data),PNG_base64_length=len(data.get('image/png',''))))
        shell.display_pub.publish=capture
        code=text.replace("MODE = 'show'",f"MODE = '{mode}'",1)
        cells=code.split('# %%')[1:]
        for i,cell in enumerate(cells,1):
            result=shell.run_cell(cell.split('\n',1)[1],store_history=False)
            assert result.success,result.error_before_exec or result.error_in_exec
        pngs=sum('image/png' in d['mime'] for d in displays)
        assert pngs==12,(mode,pngs)
        results.append(dict(mode=mode,cells=len(cells),real_PNG_displays=pngs,training_started=False,PASS=True))
    out=Path(read(HERE/'outputs/ACTIVE_RUN.json')['path'])
    dump(out/'INTERACTIVE_ACCEPTANCE.json',dict(PASS=True,script_sha256=sha(file),modes=results,
        verification_scope='Real IPython code-cell and PNG display payload execution. VS Code GUI was not opened/certified.',
        no_extra_training=True))
    print('IPython acceptance PASS:show/analyze,12 cells,24 actual PNG displays, no training')

if __name__=='__main__':run()
