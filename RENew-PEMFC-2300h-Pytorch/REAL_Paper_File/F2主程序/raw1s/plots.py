"""Three truthful 300dpi figures and underlying plotting CSVs; display in IPython."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def emit(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
        if get_ipython() is not None:
            display(Image(filename=str(path)))
    except ImportError:
        pass


def loss_plot(history, run, final=False):
    table = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table.epoch, table.train_loss, label='Train', color='#2878B5')
    ax.plot(table.epoch, table.validation_loss, label='Validation', color='#D95319')
    ax.set(xlabel='Epoch', ylabel='Standardized target MSE', title='Fresh training history')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    emit(fig, run / 'training_history.png')


def final_plots(metrics, horizon, example, cfg, run):
    table = pd.DataFrame(dict(step=np.arange(1, cfg['horizon']+1), RMSE_mV=horizon))
    table.to_csv(run/'rmse_by_horizon.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table.step, table.RMSE_mV, color='#2878B5')
    ax.set(xlabel='Forecast step (nominal second)', ylabel='RMSE (mV)',
           title=f"{cfg['model']} | Validation | {metrics['origins']:,} origins")
    ax.spines[['top', 'right']].set_visible(False)
    emit(fig, run/'rmse_curve.png')
    # Exactly the FIRST legal Validation origin, fixed before seeing errors.
    # This is a single 120-step forecast, never a stitch of overlapping windows.
    example.to_csv(run/'voltage_prediction.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(example.elapsed_seconds, example.true_V, label='Observed', color='black')
    ax.plot(example.elapsed_seconds, example.predicted_V, label=cfg['model'], color='#2878B5')
    ax.set(xlabel='Actual elapsed seconds from origin', ylabel='Mean cell voltage (V)',
           title='Validation: first legal origin, one 120-step forecast')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    emit(fig, run/'voltage_prediction.png')
