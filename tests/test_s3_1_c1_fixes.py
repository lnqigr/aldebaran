"""S3.1-C1-FIXES-01: failure-injection tests for the C1 fixes (roadmap S3.1).

ARGON_DECISIONS item 29: a failed TIC or Gaia query, or a TPF read error, ends the host ERROR; an unparseable archival
product is recorded. Item 31.6: the SDE 6.5 single-sector rule applies only after a successful query that confirms
zero archival sectors; a failed query or all-failed products end INCOMPLETE_ARCHIVAL_UNAVAILABLE at every SDE; an
exception after archival data load ends ERROR. C1d / OPS-12: observability failures are counted. OPS-11: an
incomplete sector is not reported complete. SCI-04: a secondary scan that could not test the orbit is journaled
NOT_EVALUATED (decision unchanged). Every case has a valid control. Offline: every network call is patched.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import aldebaran as ald  # noqa: E402

GOOD_STAR = {"has_params": True, "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05,
             "teff": 5800, "ruwe": 1.05, "tic_query_error": None, "gaia_query_error": None}
TPF_OK = (True, "OK", np.ones((5, 5)), np.ones((5, 5), dtype=bool))
P_FIX, T0_FIX = 3.141592653589793, 2459000.123456789


def arch_pass():
    return {"n_archival_sectors": 2, "archival_sectors_str": "80,85", "refined_p": P_FIX, "refined_t0": T0_FIX,
            "archival_snr": 8.5, "est_multi_sde_proxy": 11.3, "multi_sde": 11.3,
            "arch_bin_centers": np.linspace(-0.1, 0.1, 10), "arch_bin_means": np.ones(10), "n_trials": 12,
            "snr_crit": 3.2, "is_projected_strict": False, "delta_t_arch_days": 700.0, "sigma_p_sec": 1.25,
            "sigma_t0_sec": 45.0, "archival_sectors_requested": [80, 85], "archival_sectors_attempted": [80, 85],
            "archival_sectors_used": [80, 85], "archival_products_requested": ["a", "b"],
            "archival_products_attempted": ["a", "b"], "archival_products_used": ["a", "b"],
            "archival_products_failed": [], "ephemeris_source": "REFINED_MULTI_SECTOR",
            "ephemeris_status": "REFINED_VALID", "ephemeris_fallback_reason": "NONE"}


def tic_table(**cols):
    row = dict(rad=1.0, Teff=5800.0, e_rad=0.05, mass=1.0, e_mass=0.05, ra=10.0, dec=-20.0, pmra=1.0, pmdec=-1.0)
    row.update(cols)

    class Row(dict):
        colnames = list(row)
    return [Row(row)]


def row(mission, obsid):
    return types.SimpleNamespace(mission=[mission], obsid=obsid)


T_CURR = np.linspace(1800.0, 1827.0, 2000)


def call_archival(sde=8.0, t=T_CURR):
    return ald.verify_multi_epoch("12345678", 106, t, np.ones_like(t), np.full_like(t, 1e-3),
                                  3.0, 1801.0, 0.1, 1000.0, sde)


class StellarQueryTests(unittest.TestCase):
    def run_query(self, tic=None, tic_exc=None, gaia=None, gaia_exc=None):
        q = MagicMock(side_effect=tic_exc) if tic_exc else MagicMock(return_value=tic)
        g = MagicMock(side_effect=gaia_exc) if gaia_exc else MagicMock(return_value=gaia)
        with patch.object(ald.Catalogs, "query_criteria", q), patch.object(ald, "fetch_csv_with_headers", g):
            return ald.get_stellar_and_gaia_params("12345678")

    def test_valid_control(self):
        r = self.run_query(tic=tic_table(), gaia=pd.DataFrame({"ruwe": [1.2], "dist_deg": [0.0]}))
        self.assertTrue(r["has_params"])
        self.assertEqual(r["ruwe"], 1.2)
        self.assertIsNone(r["tic_query_error"])
        self.assertIsNone(r["gaia_query_error"])

    def test_tic_query_failure_is_reported(self):
        r = self.run_query(tic_exc=ConnectionError("MAST down"))
        self.assertFalse(r["has_params"])
        self.assertIn("ConnectionError: MAST down", r["tic_query_error"])
        self.assertIsNone(r["gaia_query_error"])

    def test_empty_tic_result_is_not_an_error(self):
        r = self.run_query(tic=[])
        self.assertFalse(r["has_params"])
        self.assertIsNone(r["tic_query_error"])

    def test_gaia_query_failure_is_reported(self):
        r = self.run_query(tic=tic_table(), gaia_exc=TimeoutError("TAP timeout"))
        self.assertTrue(r["has_params"])
        self.assertTrue(np.isnan(r["ruwe"]))
        self.assertIn("TimeoutError: TAP timeout", r["gaia_query_error"])
        self.assertIsNone(r["tic_query_error"])

    def test_gaia_without_source_is_not_an_error(self):
        r = self.run_query(tic=tic_table(), gaia=pd.DataFrame({"ruwe": [], "dist_deg": []}))
        self.assertTrue(np.isnan(r["ruwe"]))
        self.assertIsNone(r["gaia_query_error"])

    def test_certificate_error_still_propagates(self):
        import requests
        with self.assertRaises(requests.exceptions.SSLError):
            self.run_query(tic_exc=requests.exceptions.SSLError("bad cert"))
        with self.assertRaises(requests.exceptions.SSLError):
            self.run_query(tic=tic_table(), gaia_exc=requests.exceptions.SSLError("bad cert"))


class TpfReadTests(unittest.TestCase):
    def test_read_exception_is_not_evaluated(self):
        for fn in (ald.verify_and_filter_tpf, ald._verify_and_filter_tpf_cp):
            with self.subTest(fn=fn.__name__), \
                    patch.object(ald.lk, "search_targetpixelfile", side_effect=OSError("corrupt FITS")):
                ok, reason, _, _ = fn("12345678", 10, 3.0, 2000.0, 0.1)
                self.assertIsNone(ok)
                self.assertTrue(reason.startswith("TPF read error"))

    def test_unavailable_tpf_keeps_false(self):
        # SCI-80 / item 29.4: unchanged in this task.
        for fn in (ald.verify_and_filter_tpf, ald._verify_and_filter_tpf_cp):
            with self.subTest(fn=fn.__name__), patch.object(ald.lk, "search_targetpixelfile", return_value=[]):
                ok, reason, _, _ = fn("12345678", 10, 3.0, 2000.0, 0.1)
                self.assertIs(ok, False)
                self.assertEqual(reason, "TPF unavailable on MAST")


class ArchivalItem316Tests(unittest.TestCase):
    """verify_multi_epoch on the production code, with the archival query and products injected."""

    def test_zero_archival_sectors_confirmed_keeps_single_sector_rule(self):
        # Valid control: a successful query with only the current sector.
        with patch.object(ald.lk, "search_lightcurve", return_value=[row("TESS Sector 106", "cur")]):
            ok, status, diag = call_archival(sde=7.0)
            self.assertIs(ok, True)
            self.assertEqual(status, "Single-Sector_Only")
            self.assertNotIn("archival_failure_state", diag)
            ok, status, _ = call_archival(sde=6.0)
            self.assertIs(ok, False)
            self.assertIn("borderline SDE", status)

    def test_query_failure_is_incomplete_at_every_sde(self):
        for sde in (5.0, 6.4, 6.5, 12.0):
            with self.subTest(sde=sde), \
                    patch.object(ald.lk, "search_lightcurve", side_effect=ConnectionError("DNS")):
                ok, status, diag = call_archival(sde=sde)
                self.assertIsNone(ok)
                self.assertEqual(diag["archival_failure_state"], "INCOMPLETE_ARCHIVAL_UNAVAILABLE")
                self.assertIn("archival query failed (ConnectionError)", status)
                self.assertEqual(diag["archival_products_failed"][0]["stage"], "QUERY")

    def test_all_products_failed_is_incomplete_at_every_sde(self):
        rows = [row("TESS Sector 80", "p80"), row("TESS Sector 85", "p85")]
        for sde in (5.0, 6.5, 12.0):
            with self.subTest(sde=sde), patch.object(ald.lk, "search_lightcurve", return_value=rows), \
                    patch.object(ald, "safe_download", return_value=None):
                ok, status, diag = call_archival(sde=sde)
                self.assertIsNone(ok)
                self.assertEqual(diag["archival_failure_state"], "INCOMPLETE_ARCHIVAL_UNAVAILABLE")
                self.assertEqual([f["stage"] for f in diag["archival_products_failed"]], ["DOWNLOAD", "DOWNLOAD"])
                self.assertIn("all 2 archival product(s) failed", status)

    def test_unparseable_only_product_is_not_a_confirmed_zero(self):
        rows = [row("garbage", "bad"), row("TESS Sector 106", "cur")]
        for sde in (5.0, 8.0):
            with self.subTest(sde=sde), patch.object(ald.lk, "search_lightcurve", return_value=rows):
                ok, status, diag = call_archival(sde=sde)
                self.assertIsNone(ok)
                self.assertEqual(diag["archival_failure_state"], "INCOMPLETE_ARCHIVAL_UNAVAILABLE")
                self.assertEqual(diag["ephemeris_fallback_reason"], "ARCHIVAL_SECTOR_UNCONFIRMED")
                failed = diag["archival_products_failed"]
                self.assertEqual([(f["stage"], f["product_id"], f["sector"]) for f in failed],
                                 [("SECTOR_PARSE", "bad", -1)])

    def prepared(self):
        t = np.linspace(1000.0, 1027.0, 2000)
        f = np.ones(2000)
        e = np.full(2000, 0.001)
        return [patch.object(ald, "safe_download", return_value=MagicMock()),
                patch.object(ald, "extract_and_stitch_arrays", return_value=(t, f, e, None, None, 0.99)),
                patch.object(ald, "clip_upper_outliers", side_effect=lambda a, b, c: (a, b, c))]

    def test_unparseable_product_beside_a_prior_sector_is_recorded_only(self):
        rows = [row("garbage", "bad"), row("TESS Sector 80", "p80")]
        p = self.prepared()
        with patch.object(ald.lk, "search_lightcurve", return_value=rows), p[0], p[1], p[2], \
                patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda a, b, **kw: np.ones_like(b)):
            ok, status, diag = call_archival(sde=8.0)
        self.assertIsNotNone(ok)
        self.assertNotIn("archival_failure_state", diag)
        self.assertEqual(diag["archival_sectors_used"], [80])
        self.assertEqual([f["stage"] for f in diag["archival_products_failed"]], ["SECTOR_PARSE"])

    def test_partial_product_failure_is_unchanged(self):
        rows = [row("TESS Sector 80", "p80"), row("TESS Sector 85", "p85")]
        t = np.linspace(1000.0, 1027.0, 2000)
        dl = MagicMock(side_effect=[MagicMock(), None])
        with patch.object(ald.lk, "search_lightcurve", return_value=rows), patch.object(ald, "safe_download", dl), \
                patch.object(ald, "extract_and_stitch_arrays", return_value=(t, np.ones(2000), np.full(2000, 1e-3), None, None, 0.99)), \
                patch.object(ald, "clip_upper_outliers", side_effect=lambda a, b, c: (a, b, c)), \
                patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda a, b, **kw: np.ones_like(b)):
            ok, status, diag = call_archival(sde=8.0)
        self.assertIsNotNone(ok)
        self.assertNotIn("archival_failure_state", diag)
        self.assertEqual(diag["archival_products_used"], ["p80"])

    def test_exception_after_data_load_is_error_at_every_sde(self):
        rows = [row("TESS Sector 80", "p80")]
        for sde in (5.0, 6.5, 12.0):
            p = self.prepared()
            with self.subTest(sde=sde), patch.object(ald.lk, "search_lightcurve", return_value=rows), p[0], p[1], p[2], \
                    patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda a, b, **kw: np.ones_like(b)), \
                    patch.object(ald, "vr_refine_p3", side_effect=RuntimeError("code error")):
                ok, status, diag = call_archival(sde=sde)
                self.assertIsNone(ok)
                self.assertEqual(diag["archival_failure_state"], "ERROR")
                self.assertIn("archival evaluation failed (REFINEMENT: RuntimeError", status)


class TargetRun:
    """End to end through analyze_single_target (Cp, the only production path), with a temporary OUTPUT_DIR."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="s31-out-")
        self.out_dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def run_target(self, star=GOOD_STAR, tpf=TPF_OK, arch=None, archival_search=None, peaks=(0,),
                   secondary=(False, 0.0, 0.0, 0.5, 3.0), secondary_details=None, ledger=None):
        n = 19440
        rng = np.random.default_rng(42)
        t = np.linspace(2459000, 2459027, n)
        f = 1.0 + 1e-4 * rng.standard_normal(n)
        ph = ((t - T0_FIX + 0.5 * P_FIX) % P_FIX) - 0.5 * P_FIX
        f[np.abs(ph) < 0.04] -= 0.0010
        search = [[MagicMock()]] + list(archival_search or [])
        if arch is None and archival_search is None:
            arch = [(True, "Confirmed_in_Sectors_80,85", arch_pass())] * len(peaks)

        def sec(*args, details=None, **kw):
            if details is not None and secondary_details:
                details.update(secondary_details)
            return secondary

        patches = [
            patch("lightkurve.search_lightcurve", side_effect=search),
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
            patch.object(ald, "_verify_and_filter_tpf_cp", return_value=tpf),
        ]
        if arch is not None:
            patches.append(patch.object(ald, "verify_multi_epoch", side_effect=list(arch)))
        bls_patch = patch.object(ald, "BoxLeastSquares")
        for p in patches:
            p.start()
        try:
            return self._run(bls_patch, ledger)
        finally:
            for p in reversed(patches):
                p.stop()

    def _run(self, bls_patch, ledger):
        with bls_patch as bls_cls:
            bls = MagicMock()
            bls_cls.return_value = bls
            bls.power.return_value = MagicMock(period=np.full(9000, P_FIX), power=np.full(9000, 100.0),
                                               duration=np.full(9000, 0.08), transit_time=np.full(9000, T0_FIX),
                                               depth=np.full(9000, 0.001))
            kw = {}
            if ledger is not None:
                kw["ledger"] = ledger
                _, status, msg, _ = ald._analyze_single_target_impl("12345678", 106, **kw)
            else:
                self.last_result = ald.analyze_single_target("12345678", 106)
                _, status, msg, _ = self.last_result
        return status, msg

    def vetting_records(self):
        with open(os.path.join(self.out_dir, "experimental_v2_vetting.jsonl"), encoding="utf-8") as fh:
            return [json.loads(x) for x in fh if x.strip()]


class TargetStatusItem29Tests(TargetRun, unittest.TestCase):
    def test_valid_control_is_candidate(self):
        self.assertEqual(self.run_target()[0], "CANDIDATE")

    def test_old_style_star_dict_without_error_keys_is_candidate(self):
        star = {k: v for k, v in GOOD_STAR.items() if not k.endswith("_query_error")}
        self.assertEqual(self.run_target(star=star)[0], "CANDIDATE")

    def test_tic_query_failure_ends_error(self):
        status, msg = self.run_target(star=dict(GOOD_STAR, has_params=False, tic_query_error="ConnectionError: x"))
        self.assertEqual(status, "ERROR")
        self.assertIn("TIC stellar parameter query failed", msg)

    def test_missing_params_without_failure_keeps_reject(self):
        self.assertEqual(self.run_target(star=dict(GOOD_STAR, has_params=False))[0], "REJECT_NO_STELLAR_PARAMS")

    def test_gaia_query_failure_ends_error(self):
        status, msg = self.run_target(star=dict(GOOD_STAR, ruwe=np.nan, gaia_query_error="TimeoutError: y"))
        self.assertEqual(status, "ERROR")
        self.assertIn("Gaia DR3 RUWE query failed", msg)

    def test_gaia_without_source_keeps_pass(self):
        self.assertEqual(self.run_target(star=dict(GOOD_STAR, ruwe=np.nan))[0], "CANDIDATE")

    def test_evolved_star_rejected_before_gaia_failure(self):
        status, _ = self.run_target(star=dict(GOOD_STAR, r_star=2.5, gaia_query_error="TimeoutError: y"))
        self.assertEqual(status, "REJECT_EVOLVED_STAR")

    def test_tpf_read_error_ends_error(self):
        status, msg = self.run_target(tpf=(None, "TPF read error: corrupt", None, None))
        self.assertEqual(status, "ERROR")
        self.assertIn("no pixel test ran", msg)

    def test_tpf_pixel_reject_unchanged(self):
        status, _ = self.run_target(tpf=(False, "Off-target source! -> BEB", np.ones((5, 5)), np.ones((5, 5), dtype=bool)))
        self.assertEqual(status, "REJECT_TPF_PIXEL")

    def test_tpf_unavailable_is_incomplete(self):
        # S3.11 (ARGON_DECISIONS item 37) replaced REJECT_TPF_PIXEL for a confirmed missing product.
        self.assertEqual(self.run_target(tpf=(False, "TPF unavailable on MAST", None, None))[0], "INCOMPLETE_TPF_UNAVAILABLE")


class TargetStatusItem316Tests(TargetRun, unittest.TestCase):
    def test_real_archival_query_failure_ends_incomplete(self):
        # Production verify_multi_epoch: the second search_lightcurve call (archival) raises.
        status, msg = self.run_target(archival_search=[ConnectionError("DNS")])
        self.assertEqual(status, "INCOMPLETE_ARCHIVAL_UNAVAILABLE")
        self.assertIn("archival query failed (ConnectionError)", msg)
        diag = ald.get_latest_archival_diagnostic("12345678", 106, output_dir=self.out_dir)
        self.assertIsNone(diag["archival_confirmation_label"])
        self.assertIsNone(diag["arch_ok"])  # S3.1 review P2-2: not evaluated stays None
        self.assertEqual(diag["archival_failure_state"], "INCOMPLETE_ARCHIVAL_UNAVAILABLE")
        rec = self.vetting_records()[-1]
        self.assertEqual(rec["final_status"], "INCOMPLETE_ARCHIVAL_UNAVAILABLE")
        gate = [g for g in rec["peaks"][0]["gates"] if g["gate"] == "ARCHIVAL"][0]
        self.assertEqual(gate["status"], "MISSING_INPUT")

    def test_real_archival_zero_sectors_keeps_single_sector_candidate(self):
        status, msg = self.run_target(archival_search=[[row("TESS Sector 106", "cur")]])
        self.assertEqual(status, "CANDIDATE")
        self.assertIn("Single-Sector_Only", msg)

    def test_archival_error_state_ends_error(self):
        arch = arch_pass()
        arch["archival_failure_state"] = "ERROR"
        status, msg = self.run_target(arch=[(None, "archival evaluation failed (EVALUATION: ValueError: x)", arch)])
        self.assertEqual(status, "ERROR")
        self.assertIn("archival evaluation failed", msg)

    def test_failure_on_a_later_rank_ends_the_host(self):
        # Fail closed like the S3.3 all-ranks rule: rank 1 is CANDIDATE, the archival query of rank 2 fails.
        arch_fail = dict(arch_pass(), archival_failure_state="INCOMPLETE_ARCHIVAL_UNAVAILABLE")
        status, _ = self.run_target(peaks=(0, 1), arch=[(True, "Confirmed_in_Sectors_80,85", arch_pass()),
                                                        (None, "archival query failed (ConnectionError)", arch_fail)])
        self.assertEqual(status, "INCOMPLETE_ARCHIVAL_UNAVAILABLE")


class ArchivalDiagnosticRoundTripTests(TargetRun, unittest.TestCase):
    """S3.1 review P2-2: persisted and returned diagnostics keep None and the explicit failure state."""

    def round_trip(self, **kw):
        status, _ = self.run_target(**kw)
        persisted = ald.load_archival_diagnostics(self.out_dir)
        self.assertEqual(len(persisted), 1)
        return status, persisted[0]

    def test_query_unavailable(self):
        status, diag = self.round_trip(archival_search=[ConnectionError("DNS")])
        self.assertEqual(status, "INCOMPLETE_ARCHIVAL_UNAVAILABLE")
        self.assertIsNone(diag["arch_ok"])
        self.assertEqual(diag["archival_failure_state"], "INCOMPLETE_ARCHIVAL_UNAVAILABLE")

    def test_post_load_error(self):
        rows = [row("TESS Sector 80", "p80")]
        with patch.object(ald, "vr_timing", side_effect=RuntimeError("code error")):
            status, diag = self.round_trip(archival_search=[rows])
        self.assertEqual(status, "ERROR")
        self.assertIsNone(diag["arch_ok"])
        self.assertEqual(diag["archival_failure_state"], "ERROR")

    def test_returned_result_diagnostics_match(self):
        status, diag = self.round_trip(archival_search=[ConnectionError("DNS")])
        returned = self.last_result.archival_diagnostics
        self.assertEqual(len(returned), 1)
        self.assertIsNone(returned[0]["arch_ok"])
        self.assertEqual(returned[0]["archival_failure_state"], "INCOMPLETE_ARCHIVAL_UNAVAILABLE")
        self.assertEqual(json.loads(json.dumps(returned[0]))["arch_ok"], diag["arch_ok"])

    def test_pass_control_unchanged(self):
        status, diag = self.round_trip()
        self.assertEqual(status, "CANDIDATE")
        self.assertIs(diag["arch_ok"], True)
        self.assertNotIn("archival_failure_state", diag)

    def test_reject_control_unchanged(self):
        status, diag = self.round_trip(arch=[(False, "Archival non-detection in sectors 80", arch_pass())])
        self.assertEqual(status, "REJECT_ARCHIVAL")
        self.assertIs(diag["arch_ok"], False)
        self.assertNotIn("archival_failure_state", diag)


class ResumeEligibilityTests(unittest.TestCase):
    """S3.1 review P2-1: INCOMPLETE_ARCHIVAL_UNAVAILABLE and ERROR hosts stay selectable after resume."""

    STATUSES = {"101": "ERROR", "102": "INCOMPLETE_ARCHIVAL_UNAVAILABLE", "103": "CANDIDATE",
                "104": "REJECT_ARCHIVAL", "105": "REJECT_NO_STELLAR_PARAMS", "106": "INCOMPLETE_ODD_EVEN_DOWNSTREAM",
                "107": "REVIEW_VETTING_HOLD"}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="s31-resume-")
        self.addCleanup(self._tmp.cleanup)
        d = self._tmp.name
        for p in (patch.object(ald, "HISTORY_FILE", os.path.join(d, "scanned_tics.txt")),
                  patch.object(ald, "OUTPUT_DIR", d), patch.object(ald, "LOG_FILE", os.path.join(d, "run.log")),
                  patch.object(ald, "EVENTS_LOG_FILE", os.path.join(d, "events.log")),
                  patch.object(ald, "PRIORITY_TICS", []), patch.object(ald, "SEARCH_TOI_COMPANIONS", False)):
            p.start()
            self.addCleanup(p.stop)

    def test_history_written_only_for_final_statuses(self):
        written = {tic: ald.record_target_history(tic, code) for tic, code in self.STATUSES.items()}
        self.assertEqual({t for t, w in written.items() if not w}, {"101", "102"})
        self.assertEqual(ald.load_scanned_history(), {"103", "104", "105", "106", "107"})

    def test_unavailable_target_is_selected_again_after_resume(self):
        for tic, code in self.STATUSES.items():
            ald.record_target_history(tic, code)
        obs = {"target_name": list(self.STATUSES)}
        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), {})), \
                patch.object(ald.Observations, "query_criteria", return_value=obs):
            sector, queue = ald.select_latest_sector_and_targets([80], limit=100)
        self.assertEqual(sector, 80)
        self.assertEqual(sorted(queue), ["101", "102"])


class SecondaryNotEvaluatedTests(TargetRun, unittest.TestCase):
    def test_scan_reports_reason_without_changing_result(self):
        t = np.linspace(0, 10, 40)
        d = {}
        res = ald._scan_full_orbit_secondary_eclipse_cp(t, np.ones_like(t), 3.0, 0.5, 0.1, 1000.0, details=d)
        self.assertEqual(res, (False, 0.0, 0.0, 0.5, 4.0))
        self.assertEqual(d["not_evaluated_reason"], "TOO_FEW_OUT_OF_TRANSIT_POINTS")
        d = {}
        ald._scan_full_orbit_secondary_eclipse_cp(t, np.ones_like(t), 1.0, 0.5, 0.5, 1000.0, details=d)
        self.assertEqual(d["not_evaluated_reason"], "ECLIPSE_ZONE_TOO_NARROW")

    def test_scan_on_good_data_has_no_reason(self):
        t = np.linspace(0, 27, 5000)
        d = {}
        ald._scan_full_orbit_secondary_eclipse_cp(t, 1 + 1e-4 * np.random.default_rng(1).standard_normal(5000),
                                                  3.0, 0.5, 0.1, 1000.0, details=d)
        self.assertNotIn("not_evaluated_reason", d)
        self.assertGreater(d["n_windows_evaluated"], 0)

    def test_scan_nan_flux_is_not_evaluated(self):
        t = np.linspace(0, 27, 5000)
        d = {}
        res = ald._scan_full_orbit_secondary_eclipse_cp(t, np.full(5000, np.nan), 3.0, 0.5, 0.1, 1000.0, details=d)
        self.assertFalse(res[0])
        self.assertEqual(d["not_evaluated_reason"], "NON_FINITE_OUT_OF_TRANSIT_FLUX")

    def test_discovery_gate_journals_not_evaluated_and_caps_at_review(self):
        # S3.11 (ARGON_DECISIONS item 38) replaced the kept CANDIDATE with the REVIEW cap.
        status, _ = self.run_target(secondary_details={"not_evaluated_reason": "ECLIPSE_ZONE_TOO_NARROW"})
        self.assertEqual(status, "REVIEW_SECONDARY_NOT_EVALUATED")
        gate = [g for g in self.vetting_records()[-1]["peaks"][0]["gates"] if g["gate"] == "SECONDARY"][0]
        self.assertEqual(gate["status"], "NOT_EVALUATED")
        self.assertEqual(gate["metrics"]["reason"], "ECLIPSE_ZONE_TOO_NARROW")

    def test_discovery_gate_control_is_pass(self):
        self.run_target()
        gate = [g for g in self.vetting_records()[-1]["peaks"][0]["gates"] if g["gate"] == "SECONDARY"][0]
        self.assertEqual(gate["status"], "PASS")


class SuppressedFailureTests(TargetRun, unittest.TestCase):
    def test_log_writer_failure_is_counted(self):
        bad = os.path.join(self.out_dir, "is_a_dir")
        os.makedirs(bad)
        with patch.object(ald, "OUTPUT_DIR", self.out_dir), patch.object(ald, "CURRENT_RUN_ID", "r1"), \
                patch.object(ald, "LOG_FILE", bad), patch.object(ald, "EVENTS_LOG_FILE", bad):
            ald.write_log("x", print_to_console=False)
            ald.write_event("y")
            counts = ald.count_suppressed_failures(self.out_dir, "r1")
        self.assertEqual(counts, {"write_log": 1, "write_event": 1})

    def test_log_writer_control_counts_nothing(self):
        with patch.object(ald, "OUTPUT_DIR", self.out_dir), patch.object(ald, "CURRENT_RUN_ID", "r1"), \
                patch.object(ald, "LOG_FILE", os.path.join(self.out_dir, "a.log")), \
                patch.object(ald, "EVENTS_LOG_FILE", os.path.join(self.out_dir, "e.log")):
            ald.write_log("x", print_to_console=False)
            ald.write_event("y")
            self.assertEqual(ald.count_suppressed_failures(self.out_dir, "r1"), {})

    def test_unwritable_sink_falls_back_to_stderr(self):
        bad = os.path.join(self.out_dir, "file")
        open(bad, "w").close()
        with patch.object(ald, "OUTPUT_DIR", os.path.join(bad, "sub")), patch("sys.stderr") as err:
            ald.record_suppressed_failure("site_x", OSError("disk"))
        self.assertIn("site_x", "".join(c.args[0] for c in err.write.call_args_list))

    def test_diagnostics_reader_counts_corrupt_line(self):
        path = os.path.join(self.out_dir, "archival_vetting_diagnostics.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"tic_id": "1", "sector": 5}\n{not json\n')
        with patch.object(ald, "OUTPUT_DIR", self.out_dir), patch.object(ald, "CURRENT_RUN_ID", "r2"):
            self.assertEqual(len(ald.load_archival_diagnostics(self.out_dir)), 1)
            self.assertIsNotNone(ald.get_latest_archival_diagnostic("1", 5, output_dir=self.out_dir))
            counts = ald.count_suppressed_failures(self.out_dir, "r2")
        self.assertEqual(counts, {"load_archival_diagnostics:line": 1, "get_latest_archival_diagnostic:line": 1})

    def test_ledger_failure_is_counted_and_status_unchanged(self):
        ledger = MagicMock()
        for name in ("record_target_result", "record_spoc_search_start", "record_spoc_search",
                     "record_crowding_gate", "record_baseline_gate", "record_bls_start", "record_bls_search",
                     "record_low_sde_gate"):
            getattr(ledger, name).side_effect = RuntimeError("ledger disk full")
        with patch.object(ald, "CURRENT_RUN_ID", "r3"):
            status, _ = self.run_target(ledger=ledger)
            counts = ald.count_suppressed_failures(self.out_dir, "r3")
        self.assertEqual(status, "CANDIDATE")
        self.assertGreaterEqual(counts.get("ledger.record_target_result", 0), 1)
        self.assertGreaterEqual(counts.get("ledger.record_spoc_search_start", 0), 1)


class SectorCompletionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="s31-ops11-")
        self.addCleanup(self._tmp.cleanup)
        self.ev = os.path.join(self._tmp.name, "events.log")
        for p in (patch.object(ald, "OUTPUT_DIR", self._tmp.name), patch.object(ald, "EVENTS_LOG_FILE", self.ev),
                  patch.object(ald, "LOG_FILE", os.path.join(self._tmp.name, "run.log")),
                  patch.object(ald, "CURRENT_RUN_ID", "r4")):
            p.start()
            self.addCleanup(p.stop)

    def events(self):
        with open(self.ev, encoding="utf-8") as fh:
            return fh.read()

    def test_complete_sector(self):
        self.assertEqual(ald.finish_sector_run(80, 10, 10, False, 2), 0)
        self.assertIn("Completed Sector 80", self.events())

    def test_partial_scan_is_incomplete(self):
        self.assertEqual(ald.finish_sector_run(80, 7, 10, False, 2), 3)
        self.assertIn("INCOMPLETE Sector 80: 7/10", self.events())
        self.assertNotIn("Completed Sector", self.events())

    def test_interrupt_is_incomplete_even_when_counts_match(self):
        self.assertEqual(ald.finish_sector_run(80, 10, 10, True, 0), 3)
        self.assertIn("(interrupted)", self.events())

    def test_suppressed_failures_are_listed(self):
        ald.record_suppressed_failure("ledger.record_bls_search", RuntimeError("x"))
        ald.finish_sector_run(80, 1, 1, False, 0)
        self.assertIn("Suppressed observability failures: ledger.record_bls_search: 1", self.events())


if __name__ == "__main__":
    unittest.main()
