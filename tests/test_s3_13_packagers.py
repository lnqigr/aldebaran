"""S3.13: Stage 3B centroid cap (ARGON_DECISIONS item 34.1) and the ctoi_packager candidate gate.

Stage 3B runs the production execute_archival_drill on the seeded synthetic light curves of the
CHAR-TESTS-01 harness (only the MAST search and downloads are mocked) and run_antares_packager end to end.
The candidate record carries `centroid_state` (aldebaran.py since S3.8) and `tpf_spatial_pass` (the
explicit TPF result; aldebaran.py writes it from S3.12 on). The ctoi tests call run_ctoi_packager.
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import antares_packager as pack
import ctoi_packager as ctoi
from tests.test_char_antares_archival_drill import PERIOD, T0, item, candidate, run_drill

NE = "CENTROID_NOT_EVALUATED"
DECISION_KEYS = ("p_ref", "t0_ref", "dp_sec", "depth_ppm", "err_white_ppm", "snr_white", "snr_red",
                 "beta_red", "n_transits", "stack_author", "n_sectors_stack", "blend_check", "author_stats")


def spoc_only():
    """Blend check NOT_EVALUATED (no QLP/TGLC); passes on SNR."""
    return [item("SPOC", 5, 2000e-6, 21), item("SPOC", 6, 2000e-6, 22)]


def spoc_qlp():
    """Blend check EVALUATED and not blended."""
    return [item("SPOC", 5, 2000e-6, 21), item("SPOC", 6, 2000e-6, 22),
            item("QLP", 5, 1900e-6, 23), item("QLP", 6, 1900e-6, 24)]


def blended():
    return [item("SPOC", 5, 2000e-6, 31), item("SPOC", 6, 2000e-6, 32),
            item("QLP", 5, 100e-6, 33), item("QLP", 6, 100e-6, 34)]


def low_snr():
    return [item("SPOC", 5, 0.0, 41), item("SPOC", 6, 0.0, 42)]


def cand(**fields):
    return candidate(**fields)


class TestStage3BCentroidCap(unittest.TestCase):
    """Fixed fixtures through execute_archival_drill: intended deltas and identity elsewhere."""

    @classmethod
    def setUpClass(cls):
        cls.ref = {name: run_drill(search())[0] for name, search in
                   (("spoc_only", spoc_only), ("spoc_qlp", spoc_qlp), ("blended", blended), ("low_snr", low_snr))}

    def drill(self, search, **fields):
        return run_drill(search(), cand(**fields))[0]

    def assert_identical_numbers(self, name, result):
        for key in DECISION_KEYS:
            with self.subTest(case=name, key=key):
                self.assertEqual(result[key], self.ref[name][key])

    def test_valid_control_legacy_record_not_recorded_passes(self):
        result = self.ref["spoc_only"]
        self.assertEqual(result["status"], "ARCHIVAL_PASS")
        self.assertEqual(result["centroid_state"], "NOT_RECORDED")
        self.assertEqual(result["spatial_check"], "NONE")

    def test_evaluated_centroid_passes_unchanged(self):
        for search, name in ((spoc_only, "spoc_only"), (spoc_qlp, "spoc_qlp")):
            result = self.drill(search, centroid_state="EVALUATED")
            self.assertEqual((result["status"], result["reason"]), (self.ref[name]["status"], self.ref[name]["reason"]))
            self.assertEqual(result["centroid_state"], "EVALUATED")
            self.assert_identical_numbers(name, result)

    def test_not_evaluated_without_spatial_pass_is_review(self):
        # Intended delta: tpf_spatial_pass missing, False or a false string, blend check NOT_EVALUATED.
        for tpf in ({}, {"tpf_spatial_pass": False}, {"tpf_spatial_pass": "False"},
                    {"tpf_spatial_pass": np.nan}, {"tpf_spatial_pass": ""}):
            with self.subTest(tpf=tpf):
                result = self.drill(spoc_only, centroid_state=NE, **tpf)
                self.assertEqual(result["status"], "ARCHIVAL_REVIEW")
                self.assertTrue(result["reason"].startswith(NE + ": no pixel-level spatial check passed"))
                self.assertIn("Stage 3B blend check NOT_EVALUATED", result["reason"])
                self.assertEqual((result["centroid_state"], result["spatial_check"]), (NE, "NONE"))
                self.assert_identical_numbers("spoc_only", result)

    def test_not_evaluated_with_tpf_pass_stays_pass(self):
        for value in (True, np.bool_(True), "True", "TRUE", 1.0, "1"):
            with self.subTest(value=value):
                result = self.drill(spoc_only, centroid_state=NE, tpf_spatial_pass=value)
                self.assertEqual((result["status"], result["reason"]),
                                 (self.ref["spoc_only"]["status"], self.ref["spoc_only"]["reason"]))
                self.assertEqual(result["spatial_check"], "TPF_PASS")

    def test_not_evaluated_with_blend_pass_stays_pass(self):
        for tpf in ({}, {"tpf_spatial_pass": False}):
            with self.subTest(tpf=tpf):
                result = self.drill(spoc_qlp, centroid_state=NE, **tpf)
                self.assertEqual((result["status"], result["reason"]),
                                 (self.ref["spoc_qlp"]["status"], self.ref["spoc_qlp"]["reason"]))
                self.assertEqual(result["spatial_check"], "STAGE_3B_BLEND_PASS")
                self.assert_identical_numbers("spoc_qlp", result)

    def test_rejects_take_precedence_over_cap(self):
        for name, search in (("blended", blended), ("low_snr", low_snr)):
            with self.subTest(case=name):
                result = self.drill(search, centroid_state=NE)
                self.assertEqual((result["status"], result["reason"]),
                                 (self.ref[name]["status"], self.ref[name]["reason"]))
                self.assertEqual(result["status"], "ARCHIVAL_REJECT")

    def test_unknown_recorded_state_is_capped(self):
        result = self.drill(spoc_only, centroid_state="garbage")
        self.assertEqual(result["status"], "ARCHIVAL_REVIEW")
        self.assertEqual(result["centroid_state"], "GARBAGE")

    def test_tpf_field_csv_round_trip(self):
        # S3.12 contract: save_candidates_to_report writes tpf_spatial_pass as bool after centroid_state; rows
        # without the key are blank (NaN), and older reports have no column. Read back as Stage 3A reads it.
        cases = {
            "mixed": ([{"tic_id": "1", "centroid_state": NE, "tpf_spatial_pass": True},
                       {"tic_id": "2", "centroid_state": NE, "tpf_spatial_pass": False},
                       {"tic_id": "3", "centroid_state": NE}],
                      {"1": (False, NE, "TPF_PASS"), "2": (True, NE, "NONE"), "3": (True, NE, "NONE")}),
            "all_true": ([{"tic_id": "4", "centroid_state": NE, "tpf_spatial_pass": True},
                          {"tic_id": "5", "centroid_state": "EVALUATED", "tpf_spatial_pass": True}],
                         {"4": (False, NE, "TPF_PASS"), "5": (False, "EVALUATED", "TPF_PASS")}),
            "no_column": ([{"tic_id": "6", "centroid_state": NE}, {"tic_id": "7", "centroid_state": "EVALUATED"}],
                          {"6": (True, NE, "NONE"), "7": (False, "EVALUATED", "NONE")}),
            "legacy": ([{"tic_id": "8"}], {"8": (False, "NOT_RECORDED", "NONE")}),
        }
        for name, (records, expected) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "candidates_report.csv")
                cols = ["tic_id"] + [c for c in ("centroid_state", "tpf_spatial_pass") if any(c in r for r in records)]
                pd.DataFrame(records, columns=cols).to_csv(path, index=False)
                rows = pd.read_csv(path, dtype={"tic_id": str}).to_dict("records")
                got = {r["tic_id"]: pack.stage_3b_centroid_cap(r, "NOT_EVALUATED", False) for r in rows}
                self.assertEqual(got, expected)

    def test_cap_helper_truth_table(self):
        cases = [
            ({}, "NOT_EVALUATED", False, (False, "NOT_RECORDED", "NONE")),
            ({"centroid_state": "EVALUATED"}, "NOT_EVALUATED", False, (False, "EVALUATED", "NONE")),
            ({"centroid_state": NE}, "NOT_EVALUATED", False, (True, NE, "NONE")),
            ({"centroid_state": NE}, "EVALUATED", True, (True, NE, "NONE")),
            ({"centroid_state": NE}, "EVALUATED", False, (False, NE, "STAGE_3B_BLEND_PASS")),
            ({"centroid_state": NE, "tpf_spatial_pass": True}, "NOT_EVALUATED", False, (False, NE, "TPF_PASS")),
            ({"centroid_state": NE, "tpf_spatial_pass": False}, "NOT_EVALUATED", False, (True, NE, "NONE")),
            ({"centroid_state": float("nan")}, "NOT_EVALUATED", False, (False, "NOT_RECORDED", "NONE")),
        ]
        for record, check, blend, expected in cases:
            with self.subTest(record=record, check=check, blended=blend):
                self.assertEqual(pack.stage_3b_centroid_cap(record, check, blend), expected)


def stage3c_candidate(tic, **fields):
    row = dict(tic_id=tic, period_d=PERIOD, t0_btjd=T0, duration_hours=2.4, contam_pct=10.0, corr_risk=5.0,
               neb_count=0, sectors_stitched="5,6", fitted_rp_earth=2.0, fit_converged=True,
               r_star_sun=1.0, m_star_sun=1.0, teff_k=5700.0, impact_param_b=0.2)
    row.update(fields)
    return row


class TestAntaresEndToEnd(unittest.TestCase):
    """run_antares_packager: a capped peak never reaches the ranking or the ExoFOP upload."""

    def run_packager(self, survivors, search):
        phot = {str(r["tic_id"]): {"r_star": 1., "m_star": 1., "teff": 5700., "Tmag": 10., "Jmag": 9., "Kmag": 8.8}
                for r in survivors}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(pack, "run_stage_3a_filter", return_value=(pd.DataFrame(survivors), pd.DataFrame())), \
             mock.patch.object(pack.lk, "search_lightcurve", side_effect=lambda *a, **k: search()), \
             mock.patch.object(pack, "batch_query_tic_photometry", return_value=phot), \
             mock.patch.object(pack, "extract_camera_ccd_from_cache", return_value=("1", "2")), \
             contextlib.redirect_stdout(io.StringIO()):
            pack.run_antares_packager(results_dir=tmp)
            audit = pd.read_csv(os.path.join(tmp, "stage3_vetting_audit.csv"), dtype={"tic_id": str})
            rank = pd.read_csv(os.path.join(tmp, "antares_prioritized_targets_ranking.csv"), dtype={"tic_id": str})
            upload = pd.read_csv(os.path.join(tmp, "antares_exofop_ctoi_upload.csv"), sep="|", dtype={"TIC ID": str})
        return audit.set_index("tic_id"), rank, upload

    def test_capped_peak_not_packaged_others_identical(self):
        legacy = stage3c_candidate("101")
        evaluated = stage3c_candidate("102", centroid_state="EVALUATED", tpf_spatial_pass=True)
        capped = stage3c_candidate("103", centroid_state=NE, tpf_spatial_pass=False)
        missing = stage3c_candidate("104", centroid_state=NE)
        tpf = stage3c_candidate("105", centroid_state=NE, tpf_spatial_pass=True)
        audit, rank, upload = self.run_packager([legacy, evaluated, capped, missing, tpf], spoc_only)
        self.assertEqual(sorted(rank["tic_id"]), ["101", "102", "105"])
        self.assertEqual(sorted(upload["TIC ID"]), ["101", "102", "105"])
        for tic in ("103", "104"):
            self.assertEqual(audit.loc[tic, "stage_3b_status"], "ARCHIVAL_REVIEW")
            self.assertEqual(audit.loc[tic, "centroid_state"], NE)
            self.assertEqual(audit.loc[tic, "spatial_check"], "NONE")
        self.assertEqual(audit.loc["101", "centroid_state"], "NOT_RECORDED")
        self.assertEqual(audit.loc["105", "spatial_check"], "TPF_PASS")
        comments = upload.set_index("TIC ID")["User Comments / Disposition"]
        self.assertTrue(comments["105"].endswith(f"Centroid test: {NE} (spatial check passed: TPF_PASS)."))
        self.assertNotIn("Centroid test", comments["101"] + comments["102"])
        # Identity: the unaffected rows equal a run without the capped ones.
        _, rank_ref, upload_ref = self.run_packager([legacy, evaluated], spoc_only)
        keep = rank[rank["tic_id"].isin(["101", "102"])].drop(columns="rank").reset_index(drop=True)
        pd.testing.assert_frame_equal(keep, rank_ref.drop(columns="rank"))
        pd.testing.assert_frame_equal(upload[upload["TIC ID"].isin(["101", "102"])].reset_index(drop=True), upload_ref)

    def test_all_capped_writes_empty_tables(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(pack, "run_stage_3a_filter",
                               return_value=(pd.DataFrame([stage3c_candidate("103", centroid_state=NE)]), pd.DataFrame())), \
             mock.patch.object(pack.lk, "search_lightcurve", side_effect=lambda *a, **k: spoc_only()), \
             mock.patch.object(pack, "batch_query_tic_photometry") as phot, \
             contextlib.redirect_stdout(io.StringIO()):
            pack.run_antares_packager(results_dir=tmp)
            upload = pd.read_csv(os.path.join(tmp, "antares_exofop_ctoi_upload.csv"), sep="|")
        phot.assert_not_called()
        self.assertEqual(len(upload), 0)

    def test_stage_3c_rechecks_cap_on_a_forged_pass(self):
        forged = {"status": "ARCHIVAL_PASS", "candidate": stage3c_candidate("103", centroid_state=NE),
                  "blend_check": "NOT_EVALUATED", "p_ref": PERIOD, "t0_ref": T0, "depth_ppm": 2000.,
                  "stack_author": "SPOC", "snr_red": 10., "snr_white": 10., "n_transits": 8}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(pack, "batch_query_tic_photometry") as phot, \
             contextlib.redirect_stdout(io.StringIO()):
            rank, upload = pack.run_stage_3c_packaging([forged], results_dir=tmp)
        phot.assert_not_called()
        self.assertEqual((len(rank), len(upload)), (0, 0))

    def test_rejected_rows_audit_carries_state(self):
        rejected = pd.DataFrame([dict(tic_id="201", corr_risk=50., dur_ratio=1., neb_count=0, period_d=PERIOD,
                                      centroid_state=NE)])
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(pack, "run_stage_3a_filter",
                               return_value=(pd.DataFrame([stage3c_candidate("101")]), rejected)), \
             mock.patch.object(pack.lk, "search_lightcurve", side_effect=lambda *a, **k: low_snr()), \
             contextlib.redirect_stdout(io.StringIO()):
            pack.run_antares_packager(results_dir=tmp)
            audit = pd.read_csv(os.path.join(tmp, "stage3_vetting_audit.csv"), dtype={"tic_id": str}).set_index("tic_id")
        self.assertEqual(audit.loc["201", "centroid_state"], NE)
        self.assertEqual(audit.loc["201", "spatial_check"], "SKIPPED")
        self.assertEqual(audit.loc["101", "stage_3b_status"], "ARCHIVAL_REJECT")


PASS_DISP = "TIER-2 HEURISTIC SCREEN PASS (candidate; no calibrated FP-risk estimate)"


def t1_row(tic):
    return dict(tic_id=tic, period_days=4., t0_btjd=2000., duration_hours=2.4, depth_ppm=1200.,
                r_star_sun=1., m_star_sun=1., teff_k=5700., r_planet_earth=2.0, sector=93, sde=9.)


def t2_row(tic, **fields):
    row = dict(tic_id=tic, operational_status="SUCCESS", disposition=PASS_DISP, fit_status="CONVERGED",
               fit_converged=True, fitted_rp_earth=2.2, fitted_depth_ppm=1300., measured_box_depth_ppm=1200.,
               fitted_duration_hours=2.4, impact_param_b=.2, gaia_nss_type=np.nan, is_known_tess_eb=False,
               secondary_snr_sigma=0., sec_eclipse_phase=.5, dur_to_kepler_max=1., flux_contam_percent=0.,
               neb_capable_neighbors=0, sectors_stitched="93,94", vps_score=1.)
    row.update(fields)
    return row


class TestCtoiCandidateGate(unittest.TestCase):

    def run_ctoi(self, d, tier1, tier2=None):
        d = Path(d)
        pd.DataFrame(tier1).to_csv(d / "t1.csv", index=False)
        t2 = d / "t2.csv"
        if tier2 is not None:
            pd.DataFrame(tier2).to_csv(t2, index=False)
        phot = {r["tic_id"]: {"r_star": 1., "m_star": 1., "teff": 5700., "Tmag": 10., "Jmag": 9., "Kmag": 8.8}
                for r in tier1}
        out = io.StringIO()
        with mock.patch.multiple(ctoi, RESULTS_DIR=str(d), TIER1_CSV=str(d / "t1.csv"), TIER2_CSV=str(t2),
                                 RANKING_OUT_CSV=str(d / "rank.csv"), EXOFOP_OUT_CSV=str(d / "exofop.csv")), \
             mock.patch.object(ctoi, "batch_query_tic_photometry", return_value=phot) as query, \
             mock.patch.object(ctoi, "extract_camera_ccd_from_cache", return_value=("1", "2")), \
             contextlib.redirect_stdout(out):
            ctoi.run_ctoi_packager()
        return query, out.getvalue()

    def read(self, d):
        rank = pd.read_csv(Path(d) / "rank.csv", dtype={"tic_id": str})
        upload = pd.read_csv(Path(d) / "exofop.csv", sep="|", dtype={"TIC ID": str})
        return rank, upload

    def test_valid_control_candidate_rows_exported(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_ctoi(d, [t1_row("1"), t1_row("2")],
                          [t2_row("1"), t2_row("2", disposition="TIER-2 VETTED CANDIDATE (Low FP Risk)")])
            rank, upload = self.read(d)
        self.assertEqual(sorted(rank["tic_id"]), ["1", "2"])
        self.assertEqual(sorted(upload["TIC ID"]), ["1", "2"])

    def test_non_candidate_rows_never_exported_and_others_identical(self):
        blocked = {
            "10": t2_row("10", disposition="INCOMPLETE_FIT_FAILED", fit_status="FAILED", fit_converged=False,
                         vps_score=np.nan, fitted_rp_earth=np.nan),
            "11": t2_row("11", fit_status="FAILED"),
            "12": t2_row("12", operational_status="ERROR"),
            "13": t2_row("13", disposition="SUSPECT / GRAZING OR CONTAMINATED"),
            "14": t2_row("14", disposition="SUSPECT / UNPHYSICALLY LONG TRANSIT (Subgiant/Blend)"),
        }
        tier1 = [t1_row("1")] + [t1_row(t) for t in blocked] + [t1_row("15")]  # 15: no Tier-2 record
        with tempfile.TemporaryDirectory() as d:
            query, out = self.run_ctoi(d, tier1, [t2_row("1")] + list(blocked.values()))
            rank, upload = self.read(d)
        self.assertEqual(list(rank["tic_id"]), ["1"])
        self.assertEqual(list(upload["TIC ID"]), ["1"])
        self.assertEqual(query.call_args.args[0], ["1"])
        for tic, reason in (("10", "fit_status FAILED"), ("11", "fit_status FAILED"),
                            ("12", "operational_status ERROR"),
                            ("13", "non-candidate disposition (SUSPECT / GRAZING OR CONTAMINATED)"),
                            ("15", "non-candidate disposition (no Tier-2 record)")):
            self.assertIn(f"Not exported TIC {tic}: {reason}", out)
        with tempfile.TemporaryDirectory() as d:
            self.run_ctoi(d, [t1_row("1")], [t2_row("1")])
            rank_ref, upload_ref = self.read(d)
        pd.testing.assert_frame_equal(rank, rank_ref)
        pd.testing.assert_frame_equal(upload, upload_ref)

    def test_all_blocked_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "rank.csv").write_text("historical\n42\n")
            query, _ = self.run_ctoi(d, [t1_row("10")],
                                     [t2_row("10", disposition="INCOMPLETE_FIT_FAILED", fit_status="FAILED")])
            self.assertEqual((Path(d) / "rank.csv").read_text(), "historical\n42\n")
            self.assertFalse((Path(d) / "exofop.csv").exists())
        query.assert_not_called()

    def test_tier1_only_mode_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_ctoi(d, [t1_row("1")])
            rank, _ = self.read(d)
        self.assertEqual(list(rank["tic_id"]), ["1"])
        self.assertEqual(rank.loc[0, "disposition"], "TIER-1 CANDIDATE")

    def test_gate_helper(self):
        self.assertIsNone(ctoi.ctoi_export_block_reason({"disposition": PASS_DISP}))
        self.assertIsNone(ctoi.ctoi_export_block_reason({"disposition": "TIER-1 CANDIDATE"}, tier2_present=False))
        self.assertEqual(ctoi.ctoi_export_block_reason({"disposition": "TIER-1 CANDIDATE"}),
                         "non-candidate disposition (TIER-1 CANDIDATE)")
        self.assertEqual(ctoi.ctoi_export_block_reason({"disposition": "INCOMPLETE_FIT_FAILED"}),
                         "non-candidate disposition (INCOMPLETE_FIT_FAILED)")
        self.assertEqual(ctoi.ctoi_export_block_reason({"disposition": PASS_DISP, "fit_status": " failed "}),
                         "fit_status FAILED")


if __name__ == "__main__":
    unittest.main()
