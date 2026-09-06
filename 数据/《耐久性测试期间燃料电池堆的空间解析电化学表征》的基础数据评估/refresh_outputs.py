from pathlib import Path
import json
import pandas as pd
import audit_2300h as a

root = Path(__file__).resolve().parent
load_hourly = pd.read_csv(root / "loadcycle_hourly_features.csv")
pol = pd.read_csv(root / "polarization_ageing_summary.csv")
eis_records = []
spatial_records = []
sens = pd.read_csv(root / "sensitivity_ageing_summary.csv")
load_summary = pd.read_csv(root / "loadcycle_file_summary.csv")
inventory = pd.read_csv(root / "dataset_file_inventory.csv")
timeline = pd.read_csv(root / "loadcycle_timeline_summary.csv")
var_dict = pd.read_csv(root / "variable_dictionary.csv")
signal = pd.read_csv(root / "signal_quality_summary.csv")
sampling = pd.read_csv(root / "sampling_rate_analysis.csv")
degradation = a.build_degradation_summary(load_hourly, pol)
degradation.to_csv(root / "degradation_signal_summary.csv", index=False, encoding="utf-8-sig")
causal = pd.read_csv(root / "causal_feature_availability.csv")

points = []
for path in sorted((a.SRC_DIR / "EIS").rglob("*.txt")):
    meta, data = a.parse_eis_file(path, spatial=False)
    eis_records.append(meta)
    if meta.get("condition_token") == "nominal" and meta.get("stage") in {"BoL", "1150h", "2300h"} and len(data):
        for rec in data[["frequency_Hz", "z_real_Ohm", "z_imag_Ohm"]].to_dict("records"):
            rec.update({"stage": meta.get("stage"), "source_relative_path": meta.get("source_relative_path")})
            points.append(rec)
for path in sorted((a.SRC_DIR / "spatiallyResolvedEIS").rglob("*.txt")):
    meta, _ = a.parse_eis_file(path, spatial=True)
    spatial_records.append(meta)
eis = pd.DataFrame(eis_records)
spatial = pd.DataFrame(spatial_records)
eis.to_csv(root / "eis_ageing_summary.csv", index=False, encoding="utf-8-sig")
spatial.to_csv(root / "spatial_eis_summary.csv", index=False, encoding="utf-8-sig")
hi = a.build_hi_candidates(pol, eis, spatial)
hi.to_csv(root / "health_indicator_candidates.csv", index=False, encoding="utf-8-sig")

figures = a.make_figures(load_hourly, pol, eis, spatial, sens, load_summary, a.Counter(), points)
a.write_report(inventory, load_summary, timeline, var_dict, signal, sampling, degradation, pol, eis, spatial, sens, hi, causal, figures, a.read_paper_notes())
manifest_path = root / "audit_run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
manifest["generated_files"] = sorted(p.name for p in root.iterdir() if p.is_file())
manifest["refresh_script"] = str(root / "refresh_outputs.py")
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"refreshed {len(figures)} figures and report")
