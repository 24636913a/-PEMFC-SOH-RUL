"""Generate synthetic observations and separate, evaluation-only latent truth."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def generate_data(config, output_dir):
    c = config["generator"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(c["random_seed"])
    t = np.arange(0, c["end_time_h"] + c["sample_interval_h"] / 2, c["sample_interval_h"])
    end = c["end_time_h"]
    ref = c["reference_current_density_a_cm2"]
    # Dynamic load combines deterministic operating phases and stochastic demand.
    j = np.clip(0.79 + .29*np.sin(2*np.pi*t/85) + .19*np.sin(2*np.pi*t/31)
                + rng.normal(0, .10, len(t)), .22, 1.38)
    temp = (c["reference_temperature_c"] + 4.0*np.sin(2*np.pi*t/210)
            + 3.0*(j-ref) + rng.normal(0, .35, len(t)))
    dc = c["loss_linear_v_per_h"]*t + c["loss_quadratic_v_per_h2"]*t*t
    dr = c["slope_increment_at_end_ohm_cm2"]*(t/end)**1.15
    # This slowly varying residual is deliberately NOT supplied to the estimator.
    bias = c["residual_bias_at_end_v"]*(t/end) + .0018*np.sin(2*np.pi*t/1100)
    v0 = lambda current: (c["initial_voltage_at_reference_v"]
            - c["initial_polarization_slope_ohm_cm2"]*(current-ref)
            - c["initial_polarization_curvature"]*(current-ref)**2)
    latent_ref = c["initial_voltage_at_reference_v"] - dc
    operating_true = v0(j) - dc - dr*(j-ref) + c["temperature_coefficient_v_per_c"]*(temp-c["reference_temperature_c"]) + bias
    obs = pd.DataFrame({"time_h":t,"current_density_a_cm2":j,
         "temperature_c":temp,"voltage_cell_v":operating_true+rng.normal(0,c["operation_noise_sd_v"],len(t))})
    truth = pd.DataFrame({"time_h":t,"true_dc_v":dc,"true_dr_ohm_cm2":dr,
         "true_residual_bias_v":bias,"true_reference_voltage_v":latent_ref,
         "true_soh":latent_ref/c["initial_voltage_at_reference_v"]})
    rows=[]
    for k, time in enumerate(c["diagnostic_times_h"]):
        currents=np.linspace(.2,1.4,c["diagnostic_points_per_session"])
        loss=c["loss_linear_v_per_h"]*time+c["loss_quadratic_v_per_h2"]*time*time
        slope=c["slope_increment_at_end_ohm_cm2"]*(time/end)**1.15
        for current in currents:
            rows.append({"session_id":f"D{k:02d}","time_h":float(time),
                  "current_density_a_cm2":float(current),"temperature_c":c["reference_temperature_c"],
                  "voltage_cell_v":float(v0(current)-loss-slope*(current-ref)
                                            +rng.normal(0,c["diagnostic_noise_sd_v"]))})
    obs.to_csv(out/"operation.csv",index=False,float_format="%.10g")
    pd.DataFrame(rows).to_csv(out/"diagnostics.csv",index=False,float_format="%.10g")
    truth.to_csv(out/"truth_for_evaluation_only.csv",index=False,float_format="%.10g")
    metadata={"kind":"SYNTHETIC_DATA","not_a_real_dataset":True,
        "scenario":"Dynamic-load PEMFC teaching example; no measured RealHyFC/FC1 samples",
        "voltage_basis":"average single-cell voltage; controlled diagnostic voltage",
        "operating_samples":len(obs),"diagnostic_sessions":len(c["diagnostic_times_h"]),
        "points_per_diagnostic":c["diagnostic_points_per_session"],
        "random_seed":c["random_seed"],
        "diagnostic_anchor_semantics":"Eight longitudinal sessions, not eight current points",
        "generator_config":c}
    (out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
