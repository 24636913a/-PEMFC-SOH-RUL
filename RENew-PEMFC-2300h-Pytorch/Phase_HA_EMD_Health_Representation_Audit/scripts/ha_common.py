"""Read-only provenance, firewall, cache, matching, and bootstrap guards for HA V2."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DF_BASE_COMMIT = "6efb15bb03af1a8999ccf0b4ff2f75b13f9cd10a"
BASE_COMMIT = DF_BASE_COMMIT  # Compatibility name retained for the historical V1 stopped runner.
V1_BLOCKER_COMMIT = "edb22bc5490b362e4b1936e388a75e270a5def2b"
V1_PROTOCOL_SHA256 = "4ec5863046d72ac4598cb1633620ade478bc0deb8a675c3b0d9be3ffa0ae087b"
DEVELOPMENT_END_H = 1819.997738
CONTEXT_STEPS, RESOLUTION_H, RAW_PREFIX_ROWS, EXPECTED_DEVELOPMENT_ROWS = 48, 0.5, 3007, 3005
HA_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = HA_DIR.parent
DF_DIR = REPO_DIR / "Phase_DF_Leakage_Safe_Decomposition_Features"
N1_TABLE = REPO_DIR / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
BS_PROTOCOL = REPO_DIR / "Phase_BS_Final_Backbone_Screening" / "configs" / "PREREGISTERED_PROTOCOL.yaml"

SAFE_N1_COLUMNS = [
    "bin_end_ophrs2_h", "last_datetime", "temporal_segment_id", "cell_voltage_mean", "cell_voltage_std", "cell_voltage_range",
    "actual_current_A_mean", "actual_current_A_std", "actual_current_A_peak_to_peak",
    "actual_current_density_A_cm2_mean", "actual_current_density_A_cm2_std", "actual_current_density_A_cm2_peak_to_peak",
    "load_transition_count", "operation_point_transitions", "zero_current_fraction", "startup_flag", "shutdown_flag",
    "gap_before_flag", "gap_before_duration_s", "gap_after_flag",
]
REQUIRED_DF_FILES = [
    "configs/PREREGISTERED_DF_PROTOCOL.yaml", "configs/PREREGISTERED_DF_PROTOCOL.sha256", "configs/RESOLVED_VMD_PARAMETERS.yaml",
    "scripts/decomposition_features.py", "scripts/run_stage0.py", "scripts/run_stage1.py", "docs/CAUSALITY_CONTRACT.md",
    "docs/FEATURE_DICTIONARY.md", "outputs/CACHE_PROVENANCE_MANIFEST.csv", "outputs/ENDPOINT_AUDIT.csv",
    "outputs/STAGE0_SUMMARY.json", "outputs/LOCAL_ARTIFACT_MANIFEST_AUDIT.json", "outputs/LOCAL_ARTIFACT_MANIFEST.csv",
    "final/FINAL_DECISION.yaml", "final/FINAL_RESEARCH_REPORT.md",
]
CONDITION_COLUMNS = ["current_density", "current_density_std", "load_transition_count", "zero_current_fraction"]


class AuditFailure(RuntimeError):
    """Fail closed before a health conclusion can be issued."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(features: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_DIR, text=True).strip()


def git_head() -> str: return git_output("rev-parse", "HEAD")
def git_branch() -> str: return git_output("branch", "--show-current")


def ensure_base_commit() -> None:
    """Historical V1 exact-HEAD guard, retained for its stopped runner only."""
    if git_head() != DF_BASE_COMMIT:
        raise AuditFailure(f"HEAD mismatch: expected {DF_BASE_COMMIT}, found {git_head()}")


def assert_history_provenance() -> None:
    subprocess.check_call(["git", "merge-base", "--is-ancestor", DF_BASE_COMMIT, "HEAD"], cwd=REPO_DIR)
    subprocess.check_call(["git", "merge-base", "--is-ancestor", V1_BLOCKER_COMMIT, "HEAD"], cwd=REPO_DIR)
    v1 = HA_DIR / "configs" / "PREREGISTERED_HA_PROTOCOL.yaml"
    if not v1.exists() or sha256_file(v1) != V1_PROTOCOL_SHA256:
        raise AuditFailure("v1 Phase HA protocol is missing or its preserved SHA256 changed")
    if not (HA_DIR / "TECHNICAL_BLOCKER.md").exists():
        raise AuditFailure("v1 TECHNICAL_BLOCKER.md must remain preserved")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_protocol() -> dict:
    """Read the preserved V1 protocol only; V2 uses its own amendment file."""
    path = HA_DIR / "configs" / "PREREGISTERED_HA_PROTOCOL.yaml"
    return json.loads(path.read_text(encoding="utf-8"))


def assert_protocol_sources(_protocol: dict) -> None:
    """Compatibility guard for V1 code; V2 performs the stronger registered audit."""
    for relative in REQUIRED_DF_FILES:
        if not (DF_DIR / relative).exists():
            raise AuditFailure(f"missing frozen Phase DF file: {relative}")


def allowed_prefix_fingerprint(frame: pd.DataFrame) -> str:
    """Canonical fingerprint of only already-authorized logical N1 prefix rows."""
    if list(frame.columns) != SAFE_N1_COLUMNS or len(frame) != RAW_PREFIX_ROWS or not frame.index.equals(pd.RangeIndex(RAW_PREFIX_ROWS)):
        raise AuditFailure("prefix fingerprint requires exact SAFE columns and 0..3006 authorized rows")
    stream = io.StringIO(newline="")
    stream.write("schema=" + json.dumps(SAFE_N1_COLUMNS, ensure_ascii=False, separators=(",", ":")) + "\n")
    frame.to_csv(stream, index=True, index_label="origin_row", columns=SAFE_N1_COLUMNS, sep=",", quoting=csv.QUOTE_MINIMAL,
                 na_rep="<NA>", float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest()


def read_development_prefix() -> tuple[pd.DataFrame, dict]:
    """The sole N1 access point. Never hashes or reads N1 after row 3007."""
    if not N1_TABLE.exists(): raise AuditFailure(f"missing N1 table: {N1_TABLE}")
    frame = pd.read_csv(N1_TABLE, usecols=SAFE_N1_COLUMNS, nrows=RAW_PREFIX_ROWS)
    if len(frame) != RAW_PREFIX_ROWS: raise AuditFailure("N1 prefix row count differs from frozen 3007")
    frame = frame.loc[:, SAFE_N1_COLUMNS]
    frame.index = pd.RangeIndex(RAW_PREFIX_ROWS, name="origin_row")
    fingerprint = allowed_prefix_fingerprint(frame)
    development = frame.loc[frame.bin_end_ophrs2_h < DEVELOPMENT_END_H].copy()
    if len(development) != EXPECTED_DEVELOPMENT_ROWS or len(frame) - len(development) != 2:
        raise AuditFailure("frozen Development 3005/2 boundary counts do not reproduce")
    if (development.bin_end_ophrs2_h >= DEVELOPMENT_END_H).any(): raise AuditFailure("post-boundary N1 row entered Development")
    return development, {
        "source_relative_path": str(N1_TABLE.relative_to(REPO_DIR)).replace("\\", "/"), "pandas_read_csv_nrows": RAW_PREFIX_ROWS,
        "raw_prefix_rows_read": len(frame), "development_rows_after_filter": len(development), "boundary_rows_filtered": len(frame)-len(development),
        "post_boundary_rows_entered_pipeline": 0, "allowed_logical_prefix_fingerprint_sha256": fingerprint,
        "full_N1_file_hash_performed": False, "post_prefix_data_read": False, "test_accessed": False, "future_target_loaded": False,
        "loaded_columns": SAFE_N1_COLUMNS,
    }


def source_hashes() -> dict[str, str]:
    """Only Phase DF/BS files. N1 is represented solely by prefix fingerprint."""
    output = {}
    for relative in REQUIRED_DF_FILES:
        path = DF_DIR / relative
        if not path.exists(): raise AuditFailure(f"missing frozen Phase DF file: {relative}")
        output[f"Phase_DF_Leakage_Safe_Decomposition_Features/{relative}"] = sha256_file(path)
    output["Phase_BS_Final_Backbone_Screening/configs/PREREGISTERED_PROTOCOL.yaml"] = sha256_file(BS_PROTOCOL)
    return output


def execution_code_hashes() -> dict[str, str]:
    return {str(p.relative_to(HA_DIR)).replace("\\", "/"): sha256_file(p) for p in sorted((HA_DIR/"scripts").glob("*.py"))}


def validate_df_status() -> dict:
    text = (DF_DIR / "final" / "FINAL_DECISION.yaml").read_text(encoding="utf-8")
    for marker in ("EMD_SPECIFIC_ADVANCE", "TIMESNET_FREEZE", "CLOSED"):
        if marker not in text: raise AuditFailure(f"required inherited status absent: {marker}")
    return {"EMD_SPECIFIC_ADVANCE": True, "TIMESNET_FREEZE": True, "BACKBONE_SEARCH_CLOSED": True}


def read_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = pd.read_csv(DF_DIR / "outputs" / "CACHE_PROVENANCE_MANIFEST.csv")
    local = pd.read_csv(DF_DIR / "outputs" / "LOCAL_ARTIFACT_MANIFEST.csv")
    needed_cache = {"origin_row","origin_age_h","temporal_segment_id","method","cache_path","cache_sha256","input_voltage_sha256","method_parameter_sha256","feature_shape","fold_membership"}
    needed_local = {"artifact_kind","local_absolute_path","relative_phase_path","sha256","method"}
    if needed_cache.difference(cache.columns): raise AuditFailure("cache provenance manifest schema insufficient")
    if needed_local.difference(local.columns): raise AuditFailure("local artifact manifest schema insufficient")
    return cache, local


def cache_universe(cache: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    simple, emd = cache.loc[cache.method.eq("SimpleTrend")].copy(), cache.loc[cache.method.eq("EMD")].copy()
    for name, part in (("SimpleTrend",simple),("EMD",emd)):
        if part.origin_row.duplicated().any() or not part.feature_shape.eq("[48,2]").all(): raise AuditFailure(f"invalid {name} cache universe")
    joined = simple.merge(emd,on="origin_row",suffixes=("_simple","_emd"),validate="one_to_one")
    for field in ("origin_age_h","temporal_segment_id","input_voltage_sha256","fold_membership"):
        if not joined[f"{field}_simple"].astype(str).eq(joined[f"{field}_emd"].astype(str)).all(): raise AuditFailure(f"cross-method cache mismatch: {field}")
    if len(joined) != 1992: raise AuditFailure(f"expected 1992 cache origins, found {len(joined)}")
    return simple, emd, joined


def _artifact_row(local: pd.DataFrame, cache_path: Path, method: str) -> pd.Series:
    normalized = str(cache_path.resolve()).lower()
    candidates = local.loc[(local.artifact_kind.eq("origin_window_feature_cache")) & (local.method.eq(method)) & (local.local_absolute_path.map(lambda x: str(Path(x).resolve()).lower()).eq(normalized))]
    if len(candidates) != 1: raise AuditFailure(f"artifact file-hash reference unavailable/nonunique: {cache_path.name}")
    return candidates.iloc[0]


def validate_cache_entry(row: pd.Series, local: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Validate payload and artifact hashes separately; never compare them to each other."""
    path = Path(row.cache_path)
    if not path.exists(): raise AuditFailure(f"missing cache: {path}")
    artifact = _artifact_row(local, path, str(row.method))
    with np.load(path, allow_pickle=False) as archive:
        required={"features","origin_row","input_voltage_sha256","method_parameter_sha256"}
        if not required.issubset(archive.files): raise AuditFailure(f"archive metadata missing: {path.name}")
        feature, origin = archive["features"], int(archive["origin_row"].item())
        input_hash, parameter_hash = str(archive["input_voltage_sha256"].item()), str(archive["method_parameter_sha256"].item())
        metadata_keys = sorted(archive.files)
    payload_expected, payload_observed = str(row.cache_sha256), sha256_payload(feature)
    artifact_expected, artifact_observed = str(artifact.sha256), sha256_file(path)
    audit = {
        "method": str(row.method), "origin_row": int(row.origin_row), "cache_path": str(path),
        "expected_feature_payload_sha256": payload_expected, "observed_feature_payload_sha256": payload_observed,
        "expected_artifact_file_sha256": artifact_expected, "observed_artifact_file_sha256": artifact_observed,
        "feature_payload_hash_pass": payload_expected == payload_observed, "artifact_file_hash_pass": artifact_expected == artifact_observed,
        "archive_metadata_pass": origin == int(row.origin_row), "input_voltage_hash_pass": input_hash == str(row.input_voltage_sha256),
        "method_parameter_hash_pass": parameter_hash == str(row.method_parameter_sha256),
        "shape_dtype_finite_pass": feature.shape == (48,2) and feature.dtype == np.float32 and bool(np.isfinite(feature).all()),
        "archive_keys": ";".join(metadata_keys),
    }
    audit["CACHE_INTEGRITY_PASS"] = all(audit[k] for k in ("feature_payload_hash_pass","artifact_file_hash_pass","archive_metadata_pass","input_voltage_hash_pass","method_parameter_hash_pass","shape_dtype_finite_pass"))
    if not audit["CACHE_INTEGRITY_PASS"]: raise AuditFailure(f"cache integrity failed: {path.name}")
    return feature, audit


def load_npz_checked(row: pd.Series, expected_method_parameter: str) -> np.ndarray:
    """Historical runner compatibility, without reviving its erroneous hash comparison."""
    path = Path(row.cache_path)
    with np.load(path, allow_pickle=False) as archive:
        feature = archive["features"]
        if str(archive["method_parameter_sha256"].item()) != expected_method_parameter:
            raise AuditFailure("cache parameter mismatch")
    return feature


def p33_p67_and_stable_p50(development: pd.DataFrame, common: pd.DataFrame) -> dict:
    universe = common[["origin_row","origin_age_h_simple"]].rename(columns={"origin_age_h_simple":"origin_age_h"}).merge(development, left_on="origin_row", right_index=True, validate="one_to_one")
    fold1=universe.loc[(universe.origin_age_h>=72.788690)&(universe.origin_age_h<946.393214)]
    if len(fold1)!=1048: raise AuditFailure("Frozen Fold1 cache-universe count is not 1048")
    positive=fold1.loc[fold1.actual_current_density_A_cm2_mean>0]
    if len(positive)<30: raise AuditFailure("not enough Fold1 positive current origins")
    return {"P33_A_cm2":float(positive.actual_current_density_A_cm2_mean.quantile(.33)),"P67_A_cm2":float(positive.actual_current_density_A_cm2_mean.quantile(.67)),
            "stable_current_density_std_P50_A_cm2":float(positive.actual_current_density_A_cm2_std.quantile(.5)),"stable_current_density_peak_to_peak_P50_A_cm2":float(positive.actual_current_density_A_cm2_peak_to_peak.quantile(.5)),"positive_origin_count":int(len(positive))}


def p33_p67_from_fold1_training(development: pd.DataFrame, common: pd.DataFrame) -> tuple[float, float, int]:
    """Historical V1 helper retained; V2 uses the richer frozen-threshold record."""
    values = p33_p67_and_stable_p50(development, common)
    return values["P33_A_cm2"], values["P67_A_cm2"], values["positive_origin_count"]


def chronological_thirds(ages: pd.Series) -> tuple[float,float]:
    lo,hi=float(ages.min()),float(ages.max()); return lo+(hi-lo)/3,lo+2*(hi-lo)/3


def assign_third(age: float, lower: float, upper: float) -> str:
    return "EARLY" if age<=lower else "MIDDLE" if age<=upper else "LATE"


def assign_stratum(value: float,p33:float,p67:float)->str:
    if not np.isfinite(value) or value<=0: return "ZERO_CURRENT"
    return "LOW" if value<=p33 else "MID" if value<=p67 else "HIGH"


def valid_runs(frame: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    runs=[]
    for _,part in frame.groupby(["chronological_third","temporal_segment_id"],sort=True):
        part=part.sort_values("origin_row"); start=0
        while start+size<=len(part):
            cand=part.iloc[start:start+size]
            if np.all(np.diff(cand.origin_row.to_numpy())==1) and np.allclose(np.diff(cand.origin_age_h.to_numpy(float)),RESOLUTION_H,atol=1e-9): runs.append(cand); start+=size
            else: start+=1
    return runs


def consecutive_runs(frame: pd.DataFrame, n_origins: int) -> list[pd.DataFrame]:
    """Historical V1 name retained; V2 uses ``valid_runs``."""
    return valid_runs(frame, n_origins)


def matching_scales(fold1_blocks: pd.DataFrame) -> tuple[dict[str,float],list[str]]:
    scales,constants={},[]
    for col in CONDITION_COLUMNS:
        q=float(fold1_blocks[col].quantile(.75)-fold1_blocks[col].quantile(.25))
        if q==0: q=float(fold1_blocks[col].std(ddof=1))
        if not np.isfinite(q) or q==0: constants.append(col)
        else: scales[col]=q
    return scales,constants


def joint_match(early: pd.DataFrame, late: pd.DataFrame, scales: dict[str,float], minimum_pairs: int=5) -> tuple[pd.DataFrame,dict]:
    """Joint rectangle support then deterministic greedy 1:1 condition match."""
    bounds={}
    for col in CONDITION_COLUMNS:
        low=max(float(early[col].min()),float(late[col].min())); high=min(float(early[col].max()),float(late[col].max())); bounds[col]=(low,high)
    e=early.copy(); l=late.copy()
    for col,(lo,hi) in bounds.items(): e=e.loc[e[col].between(lo,hi)]; l=l.loc[l[col].between(lo,hi)]
    candidates=[]
    for ei,erow in e.iterrows():
        for li,lrow in l.iterrows():
            standardized={c:abs(float(erow[c])-float(lrow[c]))/scale for c,scale in scales.items()}
            if standardized and (max(standardized.values())>.5 or np.sqrt(sum(x*x for x in standardized.values()))>1.0): continue
            candidates.append((np.sqrt(sum(x*x for x in standardized.values())),int(erow.origin_start_row),int(lrow.origin_start_row),ei,li,standardized))
    selected=[]; used_e=set(); used_l=set()
    for distance,e_row,l_row,ei,li,standardized in sorted(candidates,key=lambda x:(x[0],x[1],x[2])):
        if ei in used_e or li in used_l: continue
        used_e.add(ei);used_l.add(li); selected.append({"early_index":ei,"late_index":li,"early_origin_start_row":e_row,"late_origin_start_row":l_row,"condition_distance":distance,**{f"standardized_absdiff_{c}":standardized.get(c,0.0) for c in CONDITION_COLUMNS}})
    pairs=pd.DataFrame(selected)
    info={"early_raw_blocks":len(early),"late_raw_blocks":len(late),"early_joint_support_blocks":len(e),"late_joint_support_blocks":len(l),"matched_pair_count":len(pairs),"unmatched_early":len(e)-len(pairs),"unmatched_late":len(l)-len(pairs),"joint_bounds":bounds,"MATCHED_COMMON_SUPPORT_PASS":len(early)>=5 and len(late)>=5 and len(pairs)>=minimum_pairs}
    return pairs,info


def moving_block_indices(frame: pd.DataFrame, length: int, rng: np.random.Generator) -> list[int]:
    """Resample contiguous analysis-block chunks separately inside each segment/third."""
    selected=[]
    for _,part in frame.groupby(["chronological_third","temporal_segment_id"],sort=True):
        indices=list(part.sort_values("origin_start_row").index)
        chunks=[indices[i:i+length] for i in range(len(indices)-length+1)]
        if not chunks: raise AuditFailure("time-dependent moving bootstrap has no legal contiguous superblock")
        sampled=[]
        while len(sampled)<len(indices): sampled.extend(chunks[int(rng.choice(len(chunks)))])
        selected.extend(sampled[:len(indices)])
    return selected


def quantile_summary(values: Iterable[float]) -> dict[str,float]:
    array=np.asarray(list(values),dtype=float); array=array[np.isfinite(array)]
    if not len(array): return {k:np.nan for k in ("signed_mean","signed_median","abs_median","p75_abs","p90_abs","p95_abs","p99_abs","max_abs","n")}
    ab=np.abs(array); return {"signed_mean":float(array.mean()),"signed_median":float(np.median(array)),"abs_median":float(np.median(ab)),"p75_abs":float(np.quantile(ab,.75)),"p90_abs":float(np.quantile(ab,.90)),"p95_abs":float(np.quantile(ab,.95)),"p99_abs":float(np.quantile(ab,.99)),"max_abs":float(ab.max()),"n":int(len(array))}

