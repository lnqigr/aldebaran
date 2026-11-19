"""S3.11-MISSING-EVIDENCE-01: missing TPF product and secondary scan not evaluated (ARGON_DECISIONS items 37 and 38).

Item 37: a peak that reaches Filter 6 when a successful MAST query finds no 2-minute TPF for the discovery sector ends
INCOMPLETE_TPF_UNAVAILABLE (host too where REJECT_TPF_PIXEL used to be), with no other-sector substitute; the host is
written to the scanned history; a failed query or read stays ERROR. Item 38: a discovery secondary scan NOT_EVALUATED
holds HOLD_SECONDARY_NOT_EVALUATED and continues through the later gates; a peak that passes them all, or whose
archival secondary scan is NOT_EVALUATED, ends REVIEW (SECONDARY_NOT_EVALUATED or ARCHIVAL_SECONDARY_NOT_EVALUATED),
never CANDIDATE and never INCOMPLETE. Every case runs the production path (analyze_single_target, Cp) with a valid
control. Offline: every network call is patched.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import aldebaran as ald  # noqa: E402
from test_s3_1_c1_fixes import GOOD_STAR, P_FIX, T0_FIX, TPF_OK, arch_pass  # noqa: E402

SEC_PASS = (False, 0.0, 0.0, 0.5, 3.0)
SEC_EB = (True, 9.0, 400.0, 0.5, 3.0)
NEV = {"not_evaluated_reason": "ECLIPSE_ZONE_TOO_NARROW"}
TPF_MISSING = (False, "TPF unavailable on MAST", None, None)
TPF_REJECT = (False, "Off-target source! -> BEB", np.ones((5, 5)), np.ones((5, 5), dtype=bool))
TPF_EMPTY_MASK = (False, "Empty SPOC pipeline aperture mask", None, None)
TPF_HOLD = (True, "REVIEW_TPF_EDGE_OR_MARGINAL: border pixel (col=0, row=0) z=5.1, z_diff=3.2", np.ones((5, 5)),
            np.ones((5, 5), dtype=bool))
ARCH_OK = (True, "Confirmed_in_Sectors_80,85", None)


def arch(status="Confirmed_in_Sectors_80,85", ok=True, sec_nev=None):
    info = arch_pass()
    if sec_nev:
        info["vr"] = {"secondary_not_evaluated_reason": sec_nev}
    return (ok, status, info)


class Run:
    """End to end through analyze_single_target with per-rank secondary, TPF and archival results."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="s311-out-")
        self.out_dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def run_target(self, secondary=((SEC_PASS, None),), tpf=(TPF_OK,), archival=None, star=GOOD_STAR, peaks=(0,)):
        n = 19440
        rng = np.random.default_rng(42)
        t = np.linspace(2459000, 2459027, n)
        f = 1.0 + 1e-4 * rng.standard_normal(n)
        ph = ((t - T0_FIX + 0.5 * P_FIX) % P_FIX) - 0.5 * P_FIX
        f[np.abs(ph) < 0.04] -= 0.0010
        sec_calls = list(secondary)
        self.tpf_calls = []
        tpf_results = list(tpf)

        def sec(*args, details=None, **kw):
            result, det = sec_calls.pop(0)
            if details is not None and det:
                details.update(det)
            return result

        def tpf_check(tic, sector, *args):
            self.tpf_calls.append((tic, sector))
            return tpf_results.pop(0)
        archival = list(archival if archival is not None else [arch()] * len(peaks))
        patches = [
            patch("lightkurve.search_lightcurve", side_effect=[[MagicMock()]]),
            patch.object(ald, "safe_download", return_value=MagicMock()),
            patch.object(ald, "extract_and_stitch_arrays", return_value=(t, f, np.full(n, 1e-4), np.zeros(n), np.zeros(n), 0.95)),
            patch.object(ald, "clip_upper_outliers", side_effect=lambda a, b, c: (a, b, c)),
            patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda a, b, **kw: np.ones_like(b)),
            patch.object(ald, "compute_whitened_kovacs_sde", return_value=(np.full(9000, 8.5), None)),
            patch.object(ald, "get_top_distinct_bls_peaks", return_value=list(peaks)),
            patch.object(ald, "OUTPUT_DIR", self.out_dir),
            patch.object(ald, "ARCHIVAL_DIAGNOSTICS_FILE", os.path.join(self.out_dir, "archival_vetting_diagnostics.jsonl")),
            patch.object(ald, "get_stellar_and_gaia_params", return_value=dict(star)),
            patch.object(ald, "_scan_full_orbit_secondary_eclipse_cp", side_effect=sec),
            patch.object(ald, "_verify_and_filter_tpf_cp", side_effect=tpf_check),
            patch.object(ald, "verify_multi_epoch", side_effect=archival),
            # Centroids are zero (constant) here; the S3.8 cap is exercised separately, so the TPF pass decides it.
            patch.object(ald, "evaluate_centroid_gate", side_effect=self.centroid),
        ]
        for p in patches:
            p.start()
        try:
            with patch.object(ald, "BoxLeastSquares") as bls_cls:
                bls = MagicMock()
                bls_cls.return_value = bls
                bls.power.return_value = MagicMock(period=np.full(9000, P_FIX), power=np.full(9000, 100.0),
                                                   duration=np.full(9000, 0.08), transit_time=np.full(9000, T0_FIX),
                                                   depth=np.full(9000, 0.001))
                _, status, msg, cand = ald.analyze_single_target("12345678", 106)
        finally:
            for p in reversed(patches):
                p.stop()
        return status, msg, cand

    @staticmethod
    def centroid(*args, **kwargs):
        return dict(state=ald.CENTROID_EVALUATED, reason=None, shift_arcsec=0.1, p_x=0.5, p_y=0.5, f_in=1.0, n_in=50,
                    n_in_usable=50, m_in=50, f_out=1.0, n_out=500, m_out=500)

    def peaks(self):
        with open(os.path.join(self.out_dir, "experimental_v2_vetting.jsonl"), encoding="utf-8") as fh:
            return [json.loads(x) for x in fh if x.strip()][-1]["peaks"]

    def journal(self):
        with open(os.path.join(self.out_dir, ald.AR_JOURNAL_FILENAME), encoding="utf-8") as fh:
            return [r for r in (json.loads(x) for x in fh if x.strip()) if r["record"] == "rank"]

    def gate(self, name, rank=0):
        return next(g for g in self.peaks()[rank]["gates"] if g["gate"] == name)


class Item37TpfFunctionTests(unittest.TestCase):
    def test_empty_query_is_confirmed_absence_for_this_sector_only(self):
        for fn in (ald.verify_and_filter_tpf, ald._verify_and_filter_tpf_cp):
            search = MagicMock(return_value=[])
            with self.subTest(fn=fn.__name__), patch.object(ald.lk, "search_targetpixelfile", search):
                ok, reason, diff, ap = fn("12345678", 106, 3.0, 2000.0, 0.1)
            self.assertEqual((ok, reason, diff, ap), (False, ald.TPF_UNAVAILABLE_REASON, None, None))
            # Item 37.2: one query, for the discovery sector only; no other sector is tried.
            self.assertEqual(search.call_count, 1)
            self.assertEqual(search.call_args.kwargs["sector"], 106)

    def test_failed_query_is_not_absence(self):
        for fn in (ald.verify_and_filter_tpf, ald._verify_and_filter_tpf_cp):
            with self.subTest(fn=fn.__name__), \
                    patch.object(ald.lk, "search_targetpixelfile", side_effect=ConnectionError("MAST down")):
                ok, reason, _, _ = fn("12345678", 106, 3.0, 2000.0, 0.1)
            self.assertIsNone(ok)
            self.assertNotEqual(reason, ald.TPF_UNAVAILABLE_REASON)


class Item37TargetTests(Run, unittest.TestCase):
    def test_valid_control_is_candidate(self):
        status, _, cand = self.run_target()
        self.assertEqual(status, "CANDIDATE")
        self.assertIsNotNone(cand)

    def test_missing_tpf_ends_incomplete_tpf_unavailable(self):
        status, msg, cand = self.run_target(tpf=(TPF_MISSING,))
        self.assertEqual(status, "INCOMPLETE_TPF_UNAVAILABLE")
        self.assertIsNone(cand)
        self.assertIn("sector 106", msg)
        g = self.gate("TPF")
        self.assertEqual((g["status"], g["metrics"]["state"]), ("MISSING_INPUT", "INCOMPLETE_TPF_UNAVAILABLE"))
        self.assertEqual((self.peaks()[0]["disposition"], self.peaks()[0]["cause"]), ("INCOMPLETE", "INCOMPLETE_TPF_UNAVAILABLE"))
        self.assertEqual(self.journal()[0]["exit_code"], "INCOMPLETE_TPF_UNAVAILABLE")
        self.assertEqual(self.tpf_calls, [("12345678", 106)])  # discovery sector only

    def test_missing_tpf_does_not_reach_archival(self):
        status, _, _ = self.run_target(tpf=(TPF_MISSING,), archival=[])
        self.assertEqual(status, "INCOMPLETE_TPF_UNAVAILABLE")

    def test_pixel_reject_and_empty_mask_stay_reject(self):
        for tpf in (TPF_REJECT, TPF_EMPTY_MASK):
            with self.subTest(reason=tpf[1]):
                self.assertEqual(self.run_target(tpf=(tpf,))[0], "REJECT_TPF_PIXEL")

    def test_read_failure_stays_error(self):
        status, _, _ = self.run_target(tpf=((None, "TPF read error: x", None, None),))
        self.assertEqual(status, "ERROR")

    def test_earlier_scientific_reject_of_rank_one_stands(self):
        # Rank 1 rejected at the secondary gate; rank 2 reaches Filter 6 with no TPF: the host keeps rank 1's reject.
        status, _, _ = self.run_target(secondary=((SEC_EB, None), (SEC_PASS, None)), tpf=(TPF_MISSING,), peaks=(0, 1))
        self.assertEqual(status, "REJECT_SECONDARY_ECLIPSE")
        self.assertEqual([r["exit_code"] for r in self.journal()], ["REJECT_SECONDARY_ECLIPSE", "INCOMPLETE_TPF_UNAVAILABLE"])

    def test_rank_two_missing_tpf_does_not_hide_rank_one_review(self):
        status, _, _ = self.run_target(secondary=((SEC_PASS, None), (SEC_PASS, None)), tpf=(TPF_HOLD, TPF_MISSING),
                                       peaks=(0, 1))
        self.assertEqual(status, "REVIEW_VETTING_HOLD")


class Item37HistoryTests(unittest.TestCase):
    def test_written_to_scanned_history_while_errors_retry(self):
        with tempfile.TemporaryDirectory(prefix="s311-hist-") as d, \
                patch.object(ald, "HISTORY_FILE", os.path.join(d, "scanned_tics.txt")):
            self.assertTrue(ald.record_target_history("201", "INCOMPLETE_TPF_UNAVAILABLE"))
            self.assertFalse(ald.record_target_history("202", "ERROR"))
            self.assertFalse(ald.record_target_history("203", "INCOMPLETE_ARCHIVAL_UNAVAILABLE"))
            self.assertEqual(ald.load_scanned_history(), {"201"})


class Item38UnitTests(unittest.TestCase):
    def test_caps_in_gate_order(self):
        E, N = ald.CENTROID_EVALUATED, ald.CENTROID_NOT_EVALUATED
        self.assertEqual(ald.review_caps(False, False, E, True), [])
        self.assertEqual(ald.review_caps(False, False, N, True), [])
        self.assertEqual(ald.review_caps(True, False, E, True), ["SECONDARY_NOT_EVALUATED"])
        self.assertEqual(ald.review_caps(False, True, E, False), ["ARCHIVAL_SECONDARY_NOT_EVALUATED"])
        self.assertEqual(ald.review_caps(True, True, N, False),
                         ["SECONDARY_NOT_EVALUATED", "ARCHIVAL_SECONDARY_NOT_EVALUATED", "CENTROID_NOT_EVALUATED"])


    def test_host_status_is_earliest_cap_across_peaks(self):
        cen = {"cause": "CENTROID_NOT_EVALUATED_NO_SPATIAL_CHECK"}
        arc = {"cause": "ARCHIVAL_SECONDARY_NOT_EVALUATED"}
        sec = {"cause": "SECONDARY_NOT_EVALUATED"}
        other = {"cause": "ODD_EVEN_INCONCLUSIVE"}
        self.assertIsNone(ald.review_cap_status([other]))
        self.assertEqual(ald.review_cap_status([cen, other]), "REVIEW_CENTROID_NOT_EVALUATED")
        self.assertEqual(ald.review_cap_status([cen, arc]), "REVIEW_ARCHIVAL_SECONDARY_NOT_EVALUATED")
        self.assertEqual(ald.review_cap_status([cen, arc, sec]), "REVIEW_SECONDARY_NOT_EVALUATED")


class Item38TargetTests(Run, unittest.TestCase):
    def assert_capped(self, status, cand, code, cause):
        self.assertEqual(status, code)
        self.assertIsNone(cand)
        peak = self.peaks()[0]
        self.assertEqual((peak["disposition"], peak["cause"]), ("REVIEW", cause))
        self.assertEqual(self.journal()[0]["disposition"], "REVIEW")

    def test_valid_control_is_candidate_without_holds(self):
        status, _, cand = self.run_target()
        self.assertEqual(status, "CANDIDATE")
        self.assertNotIn("evidence_holds", self.peaks()[0])
        self.assertNotIn("review_caps", self.peaks()[0])
        self.assertEqual(self.gate("SECONDARY")["status"], "PASS")

    def test_discovery_not_evaluated_passing_peak_is_review(self):
        status, _, cand = self.run_target(secondary=((SEC_PASS, NEV),))
        self.assert_capped(status, cand, "REVIEW_SECONDARY_NOT_EVALUATED", "SECONDARY_NOT_EVALUATED")
        peak = self.peaks()[0]
        self.assertEqual(peak["evidence_holds"], ["HOLD_SECONDARY_NOT_EVALUATED"])
        self.assertEqual(peak["review_caps"], ["SECONDARY_NOT_EVALUATED"])
        self.assertEqual(self.gate("SECONDARY")["status"], "NOT_EVALUATED")
        # Item 38.2: the hold continued through the pixel and archival gates.
        self.assertEqual(self.gate("TPF")["status"], "PASS")
        self.assertEqual(self.gate("ARCHIVAL")["status"], "PASS")

    def test_later_pixel_reject_still_applies(self):
        status, _, _ = self.run_target(secondary=((SEC_PASS, NEV),), tpf=(TPF_REJECT,))
        self.assertEqual(status, "REJECT_TPF_PIXEL")
        self.assertEqual(self.peaks()[0]["disposition"], "REJECT")

    def test_later_archival_reject_still_applies(self):
        status, _, _ = self.run_target(secondary=((SEC_PASS, NEV),), archival=[arch("Archival failure", ok=False)])
        self.assertEqual(status, "REJECT_ARCHIVAL")

    def test_missing_tpf_after_hold_is_incomplete_tpf(self):
        status, _, _ = self.run_target(secondary=((SEC_PASS, NEV),), tpf=(TPF_MISSING,))
        self.assertEqual(status, "INCOMPLETE_TPF_UNAVAILABLE")

    def test_not_incomplete_when_a_gate_had_missing_input(self):
        # No Gaia source: RUWE MISSING_INPUT is a pass. The cap still gives REVIEW, not INCOMPLETE (item 38.1).
        status, _, cand = self.run_target(secondary=((SEC_PASS, NEV),), star=dict(GOOD_STAR, ruwe=float("nan")))
        self.assert_capped(status, cand, "REVIEW_SECONDARY_NOT_EVALUATED", "SECONDARY_NOT_EVALUATED")
        self.assertEqual(self.gate("RUWE")["status"], "MISSING_INPUT")

    def test_no_gaia_source_control_is_candidate(self):
        self.assertEqual(self.run_target(star=dict(GOOD_STAR, ruwe=float("nan")))[0], "CANDIDATE")

    def test_existing_review_hold_keeps_its_status_and_records_the_hold(self):
        status, _, _ = self.run_target(secondary=((SEC_PASS, NEV),), tpf=(TPF_HOLD,))
        self.assertEqual(status, "REVIEW_VETTING_HOLD")
        self.assertEqual(self.peaks()[0]["evidence_holds"], ["HOLD_SECONDARY_NOT_EVALUATED"])

    def test_archival_not_evaluated_is_review(self):
        status, _, cand = self.run_target(archival=[arch(sec_nev="TOO_FEW_OUT_OF_TRANSIT_POINTS")])
        self.assert_capped(status, cand, "REVIEW_ARCHIVAL_SECONDARY_NOT_EVALUATED", "ARCHIVAL_SECONDARY_NOT_EVALUATED")
        self.assertEqual(self.peaks()[0]["evidence_holds"], ["HOLD_ARCHIVAL_SECONDARY_NOT_EVALUATED"])

    def test_archival_reason_on_a_rejected_archival_gate_is_not_a_cap(self):
        status, _, _ = self.run_target(archival=[arch("Archival failure", ok=False, sec_nev="TOO_FEW_OUT_OF_TRANSIT_POINTS")])
        self.assertEqual(status, "REJECT_ARCHIVAL")
        self.assertNotIn("evidence_holds", self.peaks()[0])

    def test_all_caps_recorded_and_first_names_the_host(self):
        with patch.object(ald, "centroid_candidate_allowed", return_value=False):
            status, _, cand = self.run_target(secondary=((SEC_PASS, NEV),),
                                              archival=[arch(sec_nev="TOO_FEW_OUT_OF_TRANSIT_POINTS")])
        self.assert_capped(status, cand, "REVIEW_SECONDARY_NOT_EVALUATED", "SECONDARY_NOT_EVALUATED")
        self.assertEqual(self.peaks()[0]["review_caps"],
                         ["SECONDARY_NOT_EVALUATED", "ARCHIVAL_SECONDARY_NOT_EVALUATED", "CENTROID_NOT_EVALUATED"])

    def test_centroid_cap_alone_keeps_its_s38_status(self):
        with patch.object(ald, "centroid_candidate_allowed", return_value=False):
            status, _, cand = self.run_target()
        self.assert_capped(status, cand, "REVIEW_CENTROID_NOT_EVALUATED", "CENTROID_NOT_EVALUATED_NO_SPATIAL_CHECK")

    def test_secondary_eb_reject_unchanged(self):
        self.assertEqual(self.run_target(secondary=((SEC_EB, None),))[0], "REJECT_SECONDARY_ECLIPSE")


if __name__ == "__main__":
    unittest.main()
