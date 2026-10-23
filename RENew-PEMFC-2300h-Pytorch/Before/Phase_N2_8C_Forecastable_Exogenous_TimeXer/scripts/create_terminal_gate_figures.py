"""Create transparent placeholders for analyses prohibited by the Oracle terminal gate."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8C_Forecastable_Exogenous_TimeXer")
OUT = ROOT / "outputs" / "figures"

for filename, title, detail in (
    ("exo_error_vs_voltage_error.png", "Exogenous error versus voltage error", "Not evaluated\nOracle Future Exo mean RMSEΔV = 23.268 mV > 20 mV terminal gate\nNo formal CycleExo or RidgeExo predictions were permitted."),
    ("tail_error_vs_exo_error.png", "Tail error by exogenous-error stratum", "Not evaluated\nOracle terminal stop prohibits formal forecastable-exogenous TimeXer runs.\nNo correlation or tail-error inference is claimed."),
):
    figure, axis = plt.subplots(figsize=(7, 4), dpi=150)
    axis.text(0.5, 0.55, detail, ha="center", va="center", fontsize=11)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#808080")
    figure.tight_layout()
    figure.savefig(OUT / filename)
    plt.close(figure)
