from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chm_final_preexecution_audit.py"
SPEC = importlib.util.spec_from_file_location("chm_final_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CHMFinalPreexecutionTests(unittest.TestCase):
    def test_01_final_protocol_is_nonexecuting_and_has_three_profiles(self) -> None:
        protocol = MODULE.final_protocol()
        self.assertEqual(protocol["status"], "FINAL_PROTOCOL_FROZEN_AWAITING_EXECUTION_COMMAND")
        self.assertFalse(protocol["execution_authorized"])
        self.assertFalse(protocol["training_executed"])
        self.assertEqual(set(protocol["loss"]["profiles"]), {"A_BALANCED", "B_ANCHOR_EMPHASIS", "C_OBSERVATION_EMPHASIS"})
        self.assertEqual(protocol["retrospective_LOAO"]["inner_selection_fits"], 270)

    def test_02_registered_artifact_accepts_project_parent_with_2300h(self) -> None:
        _, path = MODULE._artifact_entry("rcr_cycle_catalog")
        self.assertIn("PEMFC-2300h-Pytorch", str(path))
        frame = MODULE.safe_read_frozen_artifact("rcr_cycle_catalog", path)
        self.assertEqual(len(frame), 156)

    def test_03_exact_development_prefix_is_real_reader_path(self) -> None:
        frame = MODULE.safe_read_development_raw_prefix(usecols=["OpHrs2 [h]"])
        self.assertEqual(len(frame), MODULE.PREFIX_ROWS_151_162)
        self.assertLess(float(frame["OpHrs2 [h]"].max()), MODULE.BOUNDARY_H)

    def test_04_raw_row_268685_is_rejected_before_parse(self) -> None:
        with self.assertRaises(MODULE.CHMFirewallFailure):
            MODULE.safe_read_development_raw_prefix(requested_rows=MODULE.PREFIX_ROWS_151_162 + 1)

    def test_05_only_registered_polarization_basenames_are_allowed(self) -> None:
        for name in MODULE.allowlist_config()["development_polarization"]["allowed_files"]:
            frame, _, _ = MODULE.safe_read_development_polarization(name, usecols=["DateTime", "j.S.act [A/cm2]"])
            self.assertGreater(len(frame), 0)
        for forbidden in (
            "Polcurve_Nominal_RHFC_2000h.csv",
            "Polcurve_Nominal_RHFC_2300h.csv",
            "Polcurve_Nominal_RHFC_1500h_afterAirAirStart.csv",
            "EIS_RHFC_2300h.csv",
        ):
            with self.assertRaises(MODULE.CHMFirewallFailure):
                MODULE.safe_read_development_polarization(forbidden)

    def test_06_unregistered_raw_source_is_rejected_by_production_reader(self) -> None:
        raw = MODULE._resolved(MODULE.allowlist_config()["development_raw_source"]["path"])
        with self.assertRaises(MODULE.CHMFirewallFailure):
            MODULE.safe_read_development_raw_prefix(raw.with_name("Loadcycle_134-150.csv"))

    def test_07_polarization_audit_uses_local_proxy_not_replicate_variance(self) -> None:
        audit = MODULE.audit_polarization_structure()
        self.assertEqual(len(audit), 7)
        self.assertFalse(audit["explicit_replicate_id_present"].any())
        self.assertFalse(audit["explicit_sweep_id_present"].any())
        self.assertFalse(audit["independently_identifiable_replicates"].any())
        self.assertTrue(audit["j1_local_row_count"].ge(2).all())
        self.assertTrue(audit["variance_semantics"].eq("LOCAL_WINDOW_SCATTER_PROXY").all())

    def test_08_hou_table_contains_every_complete_cycle_and_partial(self) -> None:
        table = pd.read_csv(MODULE.INPUTS / "CHM_HOU_TABLE.csv")
        catalog = MODULE.safe_read_frozen_artifact("rcr_cycle_catalog")
        alignment = MODULE.safe_read_frozen_artifact("paa_anchor_alignment")
        MODULE.validate_hou_table(table, catalog, alignment)
        self.assertEqual(len(table), 152)
        self.assertEqual(int(table["hou_kind"].eq("COMPLETE_CYCLE_HOU").sum()), 146)
        self.assertEqual(int(table["hou_kind"].eq("PREANCHOR_PARTIAL_HOU").sum()), 6)
        self.assertTrue(set(MODULE.NEW_CYCLE_IDS).issubset(set(table["raw_cycle_id"].dropna().astype(int))))

    def test_09_hou_integrity_gate_fails_for_duplicate_hou_id(self) -> None:
        table = pd.read_csv(MODULE.INPUTS / "CHM_HOU_TABLE.csv")
        corrupt = table.copy()
        corrupt.loc[1, "hou_id"] = corrupt.loc[0, "hou_id"]
        with self.assertRaises(MODULE.CHMPreexecutionFailure):
            MODULE.validate_hou_table(
                corrupt,
                MODULE.safe_read_frozen_artifact("rcr_cycle_catalog"),
                MODULE.safe_read_frozen_artifact("paa_anchor_alignment"),
            )

    def test_10_written_audits_confirm_no_test_or_fit(self) -> None:
        result = json.loads((MODULE.OUTPUTS / "FINAL_PREEXECUTION_AUDIT.json").read_text(encoding="utf-8"))
        firewall = json.loads((MODULE.OUTPUTS / "CHM_FIREWALL_INTEGRATION_AUDIT.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "FINAL_PROTOCOL_FROZEN_AWAITING_EXECUTION_COMMAND")
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["LOAO_executed"])
        self.assertFalse(result["bootstrap_model_fitting_executed"])
        self.assertFalse(result["test_polarization_accessed"])
        self.assertFalse(result["test_EIS_accessed"])
        self.assertTrue(all(firewall.values()))

    def test_11_frozen_input_hashes_are_verified_by_production_function(self) -> None:
        manifest = MODULE.verify_frozen_input_hashes()
        self.assertEqual(len(manifest["records"]), 13)
        self.assertIn("newly_completed_HOU_152_156", manifest["records"])


if __name__ == "__main__":
    unittest.main()
