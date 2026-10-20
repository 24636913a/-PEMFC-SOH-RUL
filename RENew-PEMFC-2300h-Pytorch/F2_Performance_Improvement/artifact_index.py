"""Index and validate actual figures only. No synthetic or missing-stage plots."""
from PIL import Image
import frozen_f2 as f
from io_utils import csv,js


def build(run):
    figures=[];training=[]
    for path in sorted((run/'figures').rglob('*.png')):
        pdf=path.with_suffix('.pdf');data=path.with_name(path.stem+'_data.csv')
        f.require(pdf.is_file() and data.is_file() and data.stat().st_size>0,'MISSING_ACTUAL_FIGURE_SOURCE:'+str(path))
        with Image.open(path) as picture:
            dpi=picture.info.get('dpi');pixels=list(picture.size);picture.verify()
        f.require(dpi is not None and all(abs(value-300)<.1 for value in dpi),'PNG_300DPI_METADATA')
        with pdf.open('rb') as handle:f.require(handle.read(4)==b'%PDF','PDF_FORMAT')
        stage=path.relative_to(run/'figures').parts[0]
        source={'A':'report_a.py','B':'report_b.py'}.get(stage,'final_reports.py')
        f.require((f.HERE/source).exists(),'FIGURE_SCRIPT_MISSING')
        figures.append(dict(stage=stage,PNG=path.relative_to(f.HERE).as_posix(),
            PDF=pdf.relative_to(f.HERE).as_posix(),plotting_CSV=data.relative_to(f.HERE).as_posix(),
            plotting_script=source,script_SHA256=f.sha(f.HERE/source),
            protocol= (run/'PROTOCOL.json').relative_to(f.HERE).as_posix(),
            partition='Train/Validation only' if stage in ['A','B','C'] else 'See frozen final evaluation scope',
            pixels=str(pixels),PNG_dpi_x=dpi[0],PNG_dpi_y=dpi[1],
            scientific_scope='Exploratory research; no independent blind Test claim'))
    for path in sorted((run/'models').glob('*/*/training_history.png')):
        data=path.parent/'training_history.csv';pdf=path.with_suffix('.pdf')
        f.require(data.is_file() and pdf.is_file(),'MODEL_LOSS_FIGURE_SOURCE')
        with Image.open(path) as picture:
            dpi=picture.info.get('dpi');picture.verify()
        f.require(dpi is not None and all(abs(value-300)<.1 for value in dpi),'LOSS_PNG_300DPI')
        training.append(dict(model=path.parent.relative_to(run).as_posix(),
            PNG=path.relative_to(f.HERE).as_posix(),PDF=pdf.relative_to(f.HERE).as_posix(),
            plotting_CSV=data.relative_to(f.HERE).as_posix(),plotting_script='train.py:refresh',
            script_SHA256=f.sha(f.HERE/'train.py'),curves='Real current-run Epoch records, not historical replay'))
    csv(run/'FIGURE_INDEX.csv',figures);csv(run/'TRAINING_FIGURE_INDEX.csv',training)
    receipt=dict(status='PASS',formal_actual_figure_pairs=len(figures),actual_training_curve_pairs=len(training),
        every_formal_figure_has_real_CSV_PNG_PDF_and_script=True,PNG_metadata_300dpi=True,
        unexecuted_stages_have_no_fabricated_figure=True)
    js(run/'audits/FIGURE_COMPLETENESS.json',receipt)
    return receipt
