"""S3.10 (roadmap S3.10, ARGON_DECISIONS item 39): Pleiades transit-fit failure and negative flags.

39.1 A failed fit is not an operational error: SUCCESS, fit_status FAILED, disposition
     INCOMPLETE_FIT_FAILED, vps_score None; merge_shards does not stop on it. The fallback to the
     initial guesses and TIER-2 HEURISTIC SCREEN PASS stays forbidden.
39.2 A photometric negative flag found before the failure (SECONDARY-ECLIPSE FLAG) stays the
     rejecting disposition, with fit_status FAILED.
39.3 A catalog query failure stays ERROR; fit_status and the secondary flag stay in the record.
Antares Stage 3A takes fit-failed SUCCESS rows out before the VPS contract and never drills them.
Every test runs deep_vet_target, run_stage_3a_filter or orchestrate_downstream_stages itself.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import antares_packager
import merge_shards
import pleiades
import test_char_pleiades_deep_vet as vet
import test_s3_9_c1fg as s39
from vps_contract import VPSContractError, validate_and_extract_vps_score

FIT_ERROR = RuntimeError("Optimal parameters not found: Number of calls to function has reached maxfev = 6000.")
VPS_FIELDS = ("vps_score", "fp_risk_percent", "vps_unclipped_total")


def failing_fit():
    return mock.patch.object(pleiades, "curve_fit", side_effect=FIT_ERROR)


def run(search=None, cand=None, catalog=None, fit_fails=True, tic=123456789):
    search = search or vet.two_sectors()
    cand = dict(cand or vet.candidate(), tic_id=tic)
    if fit_fails:
        with failing_fit():
            return vet.run_vet(search, cand, catalog)[0]
    return vet.run_vet(search, cand, catalog)[0]


class TestFitFailedRecord(unittest.TestCase):
    def assert_fit_failed(self, result, disposition):
        self.assertEqual(result["operational_status"], "SUCCESS")
        self.assertEqual(result["error_message"], "")
        self.assertEqual(result["fit_status"], "FAILED")
        self.assertFalse(result["fit_converged"])
        self.assertEqual(result["fit_error"], "RuntimeError: " + str(FIT_ERROR))
        self.assertEqual(result["disposition"], disposition)
        for key in VPS_FIELDS:
            self.assertIsNone(result[key], key)
        self.assertEqual(result["vps_ledger_json"], "")
        self.assertEqual(result["vps_version"], "")
        for key in pleiades.FIT_DERIVED_FIELDS:
            self.assertTrue(np.isnan(result[key]), key)

    def test_valid_control_converged_fit_unchanged(self):
        result = run(fit_fails=False)
        self.assertEqual(result["fit_status"], "CONVERGED")
        self.assertTrue(result["disposition"].startswith("TIER-2 HEURISTIC SCREEN PASS"))
        self.assertAlmostEqual(result["vps_score"], 0.2, delta=0.3)
        self.assertEqual(validate_and_extract_vps_score(result), result["vps_score"])

    def test_plain_fit_failure_is_incomplete(self):
        result = run()
        self.assert_fit_failed(result, "INCOMPLETE_FIT_FAILED")
        self.assertFalse(result["sec_cp_is_eb"])
        self.assertFalse(result["is_known_tess_eb"])

    def test_bound_violation_fit_failure_is_incomplete(self):
        # The S3.9 fixture: the radius-ratio guess is above the fit bound, curve_fit raises ValueError.
        result = run(cand=vet.candidate(depth_ppm=150000.0), fit_fails=False)
        self.assertEqual(result["disposition"], "INCOMPLETE_FIT_FAILED")
        self.assertEqual(result["fit_status"], "FAILED")
        self.assertIsNone(result["vps_score"])

    def test_duration_flags_not_published_after_failed_fit(self):
        # Converged control: the narrow-duration host is SUSPECT. After a failed fit the duration
        # ratio comes from the BLS duration (the forbidden fallback), so no SUSPECT is published.
        narrow = vet.candidate(r_star_sun=6.0, m_star_sun=1.2)
        self.assertEqual(run(cand=narrow, fit_fails=False)["disposition"], "SUSPECT / UNPHYSICALLY NARROW TRANSIT")
        self.assert_fit_failed(run(cand=narrow), "INCOMPLETE_FIT_FAILED")
        long_host = vet.candidate(r_star_sun=0.15, m_star_sun=0.15)
        self.assertTrue(run(cand=long_host, fit_fails=False)["disposition"].startswith("SUSPECT / UNPHYSICALLY LONG"))
        self.assert_fit_failed(run(cand=long_host), "INCOMPLETE_FIT_FAILED")

    def test_grazing_or_contaminated_not_published_after_failed_fit(self):
        catalog = vet.Catalog(neighbours=[(11.2, 6.0)])
        self.assertEqual(run(catalog=catalog, fit_fails=False)["disposition"], "SUSPECT / GRAZING OR CONTAMINATED")
        result = run(catalog=vet.Catalog(neighbours=[(11.2, 6.0)]))
        self.assert_fit_failed(result, "INCOMPLETE_FIT_FAILED")
        self.assertGreater(result["flux_contam_percent"], 0.0)  # catalog measurement kept

    def test_secondary_flag_kept_after_failed_fit(self):
        # The Cp scan after a failed fit uses the BLS duration (as before S3.9), so its SNR differs from
        # the converged run; the text is the one the record's own Cp fields give.
        converged = run(search=vet.two_sectors(sec_depth=900e-6), fit_fails=False)
        self.assertTrue(converged["disposition"].startswith("SECONDARY-ECLIPSE FLAG: possible circular EB"))
        result = run(search=vet.two_sectors(sec_depth=900e-6))
        self.assertTrue(result["sec_cp_is_eb"])
        self.assert_fit_failed(result, pleiades.secondary_disposition(result["sec_cp_phase"], result["sec_cp_snr_sigma"]))
        self.assertTrue(result["disposition"].startswith("SECONDARY-ECLIPSE FLAG: possible circular EB"))

    def test_catalog_match_after_failed_fit_both_answers(self):
        # Owner decision 2026-11-13: option A (keep). Both values of the switch stay tested.
        cases = {
            "tess_eb": (vet.Catalog(tess_eb=True), "CATALOG MATCH: TESS EB"),
            "nss": (vet.Catalog(nss="SB1"), "GAIA NSS MATCH / DUAL-STAR SYSTEM FLAG (SB1)"),
        }
        for name, (catalog, prefix) in cases.items():
            for keep in (True, False):
                with self.subTest(match=name, keep=keep), \
                     mock.patch.object(pleiades, "FIT_FAILED_KEEPS_CATALOG_FLAGS", keep):
                    result = run(catalog=catalog)
                    self.assert_fit_failed(result, result["disposition"])
                    if keep:
                        self.assertTrue(result["disposition"].startswith(prefix), result["disposition"])
                    else:
                        self.assertEqual(result["disposition"], "INCOMPLETE_FIT_FAILED")
                    # The match itself stays in the diagnostic columns under either answer.
                    if name == "tess_eb":
                        self.assertTrue(result["is_known_tess_eb"])
                    else:
                        self.assertEqual(result["gaia_nss_type"], "SB1")

    def test_catalog_match_kept_by_default(self):
        # Owner decision 2026-11-13, option A, on the production default (no patch).
        result = run(catalog=vet.Catalog(tess_eb=True))
        self.assert_fit_failed(result, "CATALOG MATCH: TESS EB (possible false-positive flag; TIC 987654321)")
        result = run(catalog=vet.Catalog(nss="SB1"))
        self.assert_fit_failed(result, "GAIA NSS MATCH / DUAL-STAR SYSTEM FLAG (SB1)")
        # TESS EB first, then the secondary flag, as in the converged branch.
        result = run(search=vet.two_sectors(sec_depth=900e-6), catalog=vet.Catalog(tess_eb=True))
        self.assertTrue(result["disposition"].startswith("CATALOG MATCH: TESS EB"))

    def test_catalog_failure_with_failed_fit_and_secondary_flag_stays_error(self):
        result = run(search=vet.two_sectors(sec_depth=900e-6), catalog=s39.FailingCatalog("vizier"))
        self.assertEqual(result["operational_status"], "ERROR")
        self.assertIsNone(result["disposition"])
        self.assertEqual(result["error_message"], "NOT_EVALUATED: TESS_EB_QUERY: TimeoutError: VizieR timeout")
        self.assertEqual(result["fit_status"], "FAILED")
        self.assertTrue(result["sec_cp_is_eb"])
        self.assertTrue(np.isnan(result["vps_score"]))
        with self.assertRaises(VPSContractError):
            validate_and_extract_vps_score(result)

    def test_catalog_failure_with_converged_fit_unchanged(self):
        result = run(catalog=s39.FailingCatalog("nss"), fit_fails=False)
        self.assertEqual(result["operational_status"], "ERROR")
        self.assertEqual(result["fit_status"], "CONVERGED")
        self.assertEqual(result["error_message"], "NOT_EVALUATED: GAIA_NSS_QUERY: OSError: Gaia TAP 500")

    def test_vps_contract_still_refuses_fit_failed_record(self):
        with self.assertRaises(VPSContractError):
            validate_and_extract_vps_score(run())

    def test_record_columns_follow_summary_schema(self):
        self.assertEqual(list(run()), list(pleiades.PLEIADES_SUMMARY_COLS))

    def test_display_names_failed_fit(self):
        with failing_fit():
            result, dossier, console = s39.run_vet_rendered(vet.two_sectors(), vet.candidate())
        self.assertEqual(result["disposition"], "INCOMPLETE_FIT_FAILED")
        self.assertIn("Disposition:       INCOMPLETE_FIT_FAILED", dossier)
        self.assertIn("Fit converged:   False (FAILED)", dossier)
        self.assertIn("Vetting Score VPS: NOT_EVALUATED / 100", dossier)
        self.assertIn("b = NOT_EVALUATED (initial guess 0.500)", dossier)
        self.assertIn("Vetting Penalty Score (VPS): NOT_EVALUATED (Pleiades", console)
        self.assertIn("-> INCOMPLETE_FIT_FAILED", console)
        self.assertNotIn("HEURISTIC", dossier + console)


def write_tiers(tmp, records):
    """Tier-1 report and the Pleiades summary written the way the Pleiades CLI writes it."""
    t1 = os.path.join(tmp, "candidates_report.csv")
    t2_dir = os.path.join(tmp, "deep_vetting")
    os.makedirs(t2_dir, exist_ok=True)
    t2 = os.path.join(t2_dir, "scientific_dossier_summary.csv")
    rows = [dict(vet.candidate(), tic_id=r["tic_id"], sde=12.0) for r in records]
    pd.DataFrame(rows).to_csv(t1, index=False)
    pd.DataFrame(records).to_csv(t2, index=False)
    return t1, t2


class TestStage3AFitFailed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = run(fit_fails=False, tic=1001)
        cls.incomplete = run(tic=1002)
        cls.secondary = run(search=vet.two_sectors(sec_depth=900e-6), tic=1003)
        cls.error = run(catalog=s39.FailingCatalog("vizier"), fit_fails=False, tic=1004)

    def stage_3a(self, records):
        with tempfile.TemporaryDirectory() as tmp:
            t1, t2 = write_tiers(tmp, records)
            return antares_packager.run_stage_3a_filter(tier1_csv=t1, tier2_csv=t2)

    def test_valid_control_clean_row_survives(self):
        survivors, rejected = self.stage_3a([self.clean])
        self.assertEqual(list(survivors["tic_id"]), ["1001"])
        self.assertEqual(len(rejected), 0)

    def test_fit_failed_rows_leave_before_contract(self):
        survivors, rejected = self.stage_3a([self.clean, self.incomplete, self.secondary])
        self.assertEqual(list(survivors["tic_id"]), ["1001"])
        by_tic = rejected.set_index("tic_id")
        self.assertEqual(by_tic.loc["1002", "stage_3a_status"], "INCOMPLETE_FIT_FAILED")
        self.assertEqual(by_tic.loc["1003", "stage_3a_status"], "REJECT")
        self.assertEqual(by_tic.loc["1002", "stage_3a_reason"], "Pleiades transit fit FAILED: INCOMPLETE_FIT_FAILED")
        self.assertTrue(by_tic.loc["1003", "stage_3a_reason"].startswith(
            "Pleiades transit fit FAILED: SECONDARY-ECLIPSE FLAG"))

    def test_only_fit_failed_rows(self):
        survivors, rejected = self.stage_3a([self.incomplete])
        self.assertEqual(len(survivors), 0)
        self.assertEqual(list(rejected["stage_3a_status"]), ["INCOMPLETE_FIT_FAILED"])

    def test_error_row_still_refused(self):
        with self.assertRaises(VPSContractError):
            self.stage_3a([self.clean, self.incomplete, self.error])

    def test_fit_failed_status_needs_success(self):
        # A FAILED fit inside an ERROR record is not routed around the contract.
        self.assertEqual(self.error["fit_status"], "CONVERGED")
        forged = dict(self.error, fit_status="FAILED")
        with self.assertRaises(VPSContractError):
            self.stage_3a([self.clean, forged])

    def test_packager_never_drills_and_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tiers(tmp, [self.clean, self.incomplete, self.secondary])
            drilled = []

            def drill(candidate, **kwargs):
                drilled.append(str(candidate["tic_id"]))
                return {"status": "ARCHIVAL_REJECT", "reason": "stub", "products_failed": []}
            with mock.patch.object(antares_packager, "execute_archival_drill", side_effect=drill), \
                 mock.patch.object(antares_packager, "run_stage_3c_packaging",
                                   return_value=(pd.DataFrame(), pd.DataFrame())):
                antares_packager.run_antares_packager(results_dir=tmp)
            audit = pd.read_csv(os.path.join(tmp, "stage3_vetting_audit.csv"), dtype={"tic_id": str})
        self.assertEqual(drilled, ["1001"])
        rows = audit.set_index("tic_id")
        self.assertEqual(rows.loc["1001", "stage_3a"], "PASS")
        self.assertEqual(rows.loc["1002", "stage_3a"], "INCOMPLETE_FIT_FAILED")
        self.assertEqual(rows.loc["1002", "stage_3b_status"], "SKIPPED")
        self.assertEqual(rows.loc["1002", "decision_reason"], "Pleiades transit fit FAILED: INCOMPLETE_FIT_FAILED")
        self.assertEqual(rows.loc["1003", "stage_3a"], "REJECT")

    def test_explicit_tic_does_not_drill_fit_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tiers(tmp, [self.clean, self.incomplete])
            with mock.patch.object(antares_packager, "execute_archival_drill") as drill:
                antares_packager.run_antares_packager(results_dir=tmp, target_tic="1002")
        drill.assert_not_called()


class TestMergeShardsFitFailed(unittest.TestCase):
    """merge_shards Pleiades and Antares stages with the real summary rows and the real packager."""

    @classmethod
    def setUpClass(cls):
        cls.clean = run(fit_fails=False, tic=1001)
        cls.incomplete = run(tic=1002)
        cls.error = run(catalog=s39.FailingCatalog("vizier"), tic=1004)

    def orchestrate(self, records):
        with tempfile.TemporaryDirectory() as tmp:
            write_tiers(tmp, records)
            consolidated = pd.read_csv(os.path.join(tmp, "candidates_report.csv"), dtype={"tic_id": str})

            def fake_run(cmd, **kwargs):
                if cmd[1] == "antares_packager.py":
                    with mock.patch.object(antares_packager, "execute_archival_drill",
                                           return_value={"status": "ARCHIVAL_REJECT", "reason": "stub",
                                                         "products_failed": []}):
                        antares_packager.run_antares_packager(results_dir=cmd[3])
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            with mock.patch.object(merge_shards.subprocess, "run", side_effect=fake_run):
                return merge_shards.orchestrate_downstream_stages(
                    consolidated, tmp, run_pleiades=True, run_antares=True)

    def test_fit_failed_does_not_stop_merge(self):
        stages = self.orchestrate([self.clean, self.incomplete])
        status = {s["stage"]: s["status"] for s in stages if "status" in s}
        self.assertEqual(status.get("pleiades"), "COMPLETED")
        self.assertEqual(status.get("antares"), "COMPLETED")

    def test_catalog_error_still_stops_merge(self):
        with self.assertRaises(merge_shards.StageExecutionError):
            self.orchestrate([self.clean, self.error])


if __name__ == "__main__":
    unittest.main()
